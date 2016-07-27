package zju.dsse;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix2D;
import jpscpu.LinearSolver;
import no.uib.cipr.matrix.Vector;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;
import no.uib.cipr.matrix.sparse.SparseVector;
import org.apache.log4j.Logger;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsTopoIsland;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.measure.MeasTypeCons;
import zju.measure.MeasureInfo;
import zju.util.YMatrixGetter;

/**
 * 量测位置优化
 * @author Dong Shufeng
 * Date: 2016/7/12
 */
public class MeasPosOpt implements MeasTypeCons {

    private static Logger log = Logger.getLogger(MeasPosOpt.class);

    private DistriSys distriSys;

    private MeasureInfo[] existMeas;

    //可以布置的位置
    private int[] cand_pos;
    //每个位置可以布置的量测集合
    private int[][] meas_types;
    //量测的均方差
    private double[][] meas_sigma;

    public MeasPosOpt(DistriSys distriSys) {
        this.distriSys = distriSys;
    }

    private void build() {

    }

    public void doOpt() {
        //辅助变量
        int row_count = 0;
        int element_count = 0;
        //目前只考虑只有一个电气岛的请看
        DsTopoIsland island = distriSys.getActiveIslands()[0];
        island.buildDetailedGraph();
        int n = island.getTns().size();
        IEEEDataIsland ieeeIsland = island.toIeeeIsland();
        YMatrixGetter Y = new YMatrixGetter(ieeeIsland);
        Y.formYMatrix();
        ASparseMatrixLink2D bApos = Y.formBApostrophe(false);
        ASparseMatrixLink2D bAposTwo = Y.formBApostropheTwo(false);

        int i = 0;
        DoubleMatrix2D[] Hs = new MySparseDoubleMatrix2D[cand_pos.length];
        int size = 2 * n - 1;//状态变量的维数
        for(int pos : cand_pos) {
            ASparseMatrixLink2D H = new ASparseMatrixLink2D(meas_types[i].length, size);
            Vector[] cols = new SparseVector[size];
            for(int j = 0; j < cols.length; j++)
                cols[j] = new SparseVector(size);
            int[] measTypes = meas_types[i];
            int index = 0;
            for(int measType : measTypes) {
                switch (measType) {
                    case TYPE_BUS_ANGLE:
                        H.setValue(index++, pos + size - 1, 1.0);
                        break;
                    case TYPE_BUS_VOLOTAGE:
                        H.setValue(index++, pos - 1, 1.0);
                        break;
                    case TYPE_BUS_ACTIVE_POWER:
                        int k = bApos.getIA()[pos];
                        while (k != -1) {
                            int j = bApos.getJA().get(k);
                            H.setValue(index, j, bApos.getVA().get(k));
                            k = bApos.getLINK().get(k);
                        }
                        index++;
                        break;
                    case TYPE_BUS_REACTIVE_POWER:
                        k = bAposTwo.getIA()[pos];
                        while (k != -1) {
                            int j = bAposTwo.getJA().get(k);
                            H.setValue(index, j, bAposTwo.getVA().get(k));
                            k = bAposTwo.getLINK().get(k);
                        }
                        index++;
                        break;
                    case TYPE_LINE_FROM_ACTIVE:
                        BranchData branch = ieeeIsland.getId2branch().get(pos);
                        H.setValue(index, branch.getTapBusNumber() - 1, -1.0 / branch.getBranchX());
                        H.setValue(index++, branch.getZBusNumber() - 1, -1.0 / branch.getBranchX());
                        break;
                    case TYPE_LINE_FROM_REACTIVE:
                        branch = ieeeIsland.getId2branch().get(pos);
                        H.setValue(index, branch.getTapBusNumber() - 1, -1.0 / branch.getBranchX());
                        H.setValue(index++, branch.getZBusNumber() - 1, -1.0 / branch.getBranchX());
                        break;
                    case TYPE_LINE_TO_ACTIVE:
                        branch = ieeeIsland.getId2branch().get(pos);
                        H.setValue(index, branch.getTapBusNumber() - 1, -1.0 / branch.getBranchX());
                        H.setValue(index++, branch.getZBusNumber() - 1, -1.0 / branch.getBranchX());
                        break;
                    case TYPE_LINE_TO_REACTIVE:
                        branch = ieeeIsland.getId2branch().get(pos);
                        H.setValue(index, branch.getTapBusNumber() - 1, -1.0 / branch.getBranchX());
                        H.setValue(index++, branch.getZBusNumber() - 1, -1.0 / branch.getBranchX());
                        break;
                    case TYPE_LINE_FROM_CURRENT:
                        branch = ieeeIsland.getId2branch().get(pos);
                        H.setValue(index, branch.getTapBusNumber() - 1, -1.0 / branch.getBranchX());
                        H.setValue(index++, branch.getZBusNumber() - 1, -1.0 / branch.getBranchX());
                        break;
                    case TYPE_LINE_TO_CURRENT:
                        branch = ieeeIsland.getId2branch().get(pos);
                        H.setValue(index, branch.getTapBusNumber() - 1, -1.0 / branch.getBranchX());
                        H.setValue(index++, branch.getZBusNumber() - 1, -1.0 / branch.getBranchX());
                        break;
                    default:
                        break;
                }
            }

            //计算 HT*W*H
            Hs[i] = new MySparseDoubleMatrix2D(size, size);
            int k1, k2, i1, i2, extraVarNum = 0;
            double s;
            boolean isExist;
            for (int row = 0; row < H.getN(); row++) {
                k1 = H.getJA2()[row];
                s = 0;
                //计算对角元
                while (k1 != -1) {
                    i1 = H.getIA2().get(k1);
                    s += meas_sigma[i][i1] * H.getVA().get(k1) * H.getVA().get(k1);
                    k1 = H.getLINK2().get(k1);
                }
                Hs[i].set(row, row, s);
                extraVarNum++;
                //计算上三角元素
                for (int col = row + 1; col < H.getN(); col++) {
                    k1 = H.getJA2()[row];
                    k2 = H.getJA2()[col];
                    s = 0;
                    isExist = false;
                    while (true) {
                        i1 = H.getIA2().get(k1);
                        i2 = H.getIA2().get(k2);
                        if (i1 == i2) {
                            s += meas_sigma[i][i1] * H.getVA().get(k1) * H.getVA().get(k2);
                            k1 = H.getLINK2().get(k1);
                            k2 = H.getLINK2().get(k2);
                            isExist = true;
                        } else if (i1 < i2) {
                            k1 = H.getLINK2().get(k1);
                        } else {
                            k2 = H.getLINK2().get(k2);
                        }
                        if (k1 == -1 || k2 == -1)
                            break;
                    }
                    if(!isExist)
                        continue;
                    Hs[i].setQuick(row, col, s);
                    extraVarNum++;
                }
            }
            i++;
        }


        //开始开辟内存

        //objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数
        double objValue[] = new double[(size) * (size) + cand_pos.length];
        //状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double columnLower[] = new double[objValue.length];
        //状态变量上限
        double columnUpper[] = new double[objValue.length];
        //指明那些是整数
        int whichInt[] = new int[cand_pos.length];

        //约束下限
        double rowLower[] = new double[row_count];
        //约束上限
        double rowUpper[] = new double[row_count];
        //约束中非零元系数
        double element[] = new double[element_count];
        //上面系数对应的列
        int column[] = new int[element_count];
        //每一行起始位置
        int starts[] = new int[rowLower.length + 1];
        starts[0] = 0;

        int obj_count = 0, whichint_count = 0;

        int numberRows = rowLower.length;
        int numberColumns = columnLower.length;
        double result[] = new double[numberColumns];
        //进行求解
        LinearSolver solver = new LinearSolver();
        solver.setDrive(LinearSolver.MLP_DRIVE_CBC);
        int status = solver.solveMlp(numberColumns, numberRows, objValue,
                columnLower, columnUpper, rowLower, rowUpper, element, column, starts, whichInt, result);
        if (status < 0) {
            log.warn("计算不收敛.");
        } else { //状态位显示计算收敛
            log.info("计算结束.");
        }
    }
}
