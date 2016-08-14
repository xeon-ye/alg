package zju.se;

import jpscpu.LinearSolver;
import org.apache.log4j.Logger;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink2D;
import zju.measure.MeasTypeCons;
import zju.util.YMatrixGetter;

/**
 * 量测位置优化
 * @author Dong Shufeng
 * Date: 2016/7/12
 */
public class MeasPosOpt implements MeasTypeCons {

    private static Logger log = Logger.getLogger(MeasPosOpt.class);

    //已经安装的量测：类型，位置和权重
    private int[] existMeasTypes;
    private int[] existMeasPos;
    private double[] existMeasWeight;

    //可以布置的位置
    private int[] candPos;
    //每个位置可以布置的量测集合
    private int[][] measTypesPerPos;
    //量测的均方差
    private double[][] measWeight;

    private IEEEDataIsland island;

    public MeasPosOpt(IEEEDataIsland island) {
        this.island = island;
    }

    private ASparseMatrixLink2D formH(YMatrixGetter Y, ASparseMatrixLink2D bApos, int[] measTypes, int pos) {
        int[] posArray = new int[measTypes.length];
        for(int i = 0; i < posArray.length; i++)
            posArray[i] = pos;
        return formH(Y, bApos, measTypes, posArray);
    }

    private ASparseMatrixLink2D formH(YMatrixGetter Y, ASparseMatrixLink2D bApos, int[] measTypes, int[] pos) {
        int i = 0, n = island.getBuses().size();
        int size = 2 * n;//状态变量的维数
        ASparseMatrixLink2D H = new ASparseMatrixLink2D(measTypes.length, size);
        for(int measType : measTypes) {
            switch (measType) {
                case TYPE_BUS_ANGLE:
                    H.setValue(i, pos[i] + size - 1, 1.0);
                    break;
                case TYPE_BUS_VOLOTAGE:
                    H.setValue(i, pos[i] - 1, 1.0);
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    int k = bApos.getIA()[pos[i] - 1];
                    while (k != -1) {
                        int j = bApos.getJA().get(k);
                        H.setValue(i, j + n, bApos.getVA().get(k));
                        k = bApos.getLINK().get(k);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    k = Y.getAdmittance()[1].getIA()[pos[i] - 1];
                    while (k != -1) {
                        int j = Y.getAdmittance()[1].getJA().get(k);
                        H.setValue(i, j + n, Y.getAdmittance()[1].getVA().get(k));
                        k = Y.getAdmittance()[1].getLINK().get(k);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    BranchData branch = island.getId2branch().get(pos[i]);
                    H.setValue(i, branch.getTapBusNumber() + n - 1, 1.0 / branch.getBranchX());
                    H.setValue(i, branch.getZBusNumber() + n - 1, -1.0 / branch.getBranchX());
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    branch = island.getId2branch().get(pos[i]);
                    H.setValue(i, branch.getTapBusNumber() + n - 1, -1.0 / branch.getBranchX());
                    H.setValue(i, branch.getZBusNumber() + n - 1, 1.0 / branch.getBranchX());
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    branch = island.getId2branch().get(pos[i]);
                    double r = branch.getBranchR();
                    double x = branch.getBranchX();
                    double b = -x / (r * r + x * x);

                    //general procedure for branchType 0,1,2,3
                    if (branch.getType() != BranchData.BRANCH_TYPE_ACLINE) {
                        double c = 1 / branch.getTransformerRatio();
                        b = c * b;
                    }
                    H.setValue(i, branch.getTapBusNumber() - 1, b);
                    H.setValue(i, branch.getZBusNumber() - 1, b);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    branch = island.getId2branch().get(pos[i]);
                    r = branch.getBranchR();
                    x = branch.getBranchX();
                    b = -x / (r * r + x * x);

                    //general procedure for branchType 0,1,2,3
                    if (branch.getType() != BranchData.BRANCH_TYPE_ACLINE) {
                        double c = 1 / branch.getTransformerRatio();
                        b = c * b;
                    }
                    H.setValue(i, branch.getTapBusNumber() - 1, -b);
                    H.setValue(i, branch.getZBusNumber() - 1, -b);
                    break;
                case TYPE_LINE_FROM_CURRENT:
                    //todo
                    branch = island.getId2branch().get(pos[i]);
                    H.setValue(i, branch.getTapBusNumber() - 1, -1.0 / branch.getBranchX());
                    H.setValue(i, branch.getZBusNumber() - 1, -1.0 / branch.getBranchX());
                    break;
                case TYPE_LINE_TO_CURRENT:
                    //todo
                    branch = island.getId2branch().get(pos[i]);
                    H.setValue(i, branch.getTapBusNumber() - 1, -1.0 / branch.getBranchX());
                    H.setValue(i, branch.getZBusNumber() - 1, -1.0 / branch.getBranchX());
                    break;
                default:
                    break;
            }
            i++;
        }
        return H;
    }

    ASparseMatrixLink2D formHTWH(ASparseMatrixLink2D H, double[] weight, int[] elementCount) {
        //计算 HT*W*H
        ASparseMatrixLink2D r = new ASparseMatrixLink2D(H.getN(), H.getN());
        int k1, k2, i1, i2;
        double s;
        boolean isExist;
        for (int row = 0; row < H.getN(); row++) {
            k1 = H.getJA2()[row];
            s = 0;
            //计算对角元
            while (k1 != -1) {
                i1 = H.getIA2().get(k1);
                s += weight[i1] * H.getVA().get(k1) * H.getVA().get(k1);
                k1 = H.getLINK2().get(k1);
            }
            if(s != 0.0) {
                r.setValue(row, row, s);
                elementCount[0] += (H.getN() - row);
            }
            //计算上三角元素
            for (int col = row + 1; col < H.getN(); col++) {
                k1 = H.getJA2()[row];
                k2 = H.getJA2()[col];
                s = 0;
                isExist = false;
                while (k1 != -1 && k2 != -1) {
                    i1 = H.getIA2().get(k1);
                    i2 = H.getIA2().get(k2);
                    if (i1 == i2) {
                        s += weight[i1] * H.getVA().get(k1) * H.getVA().get(k2);
                        k1 = H.getLINK2().get(k1);
                        k2 = H.getLINK2().get(k2);
                        isExist = true;
                    } else if (i1 < i2) {
                        k1 = H.getLINK2().get(k1);
                    } else {
                        k2 = H.getLINK2().get(k2);
                    }
                }
                if(!isExist)
                    continue;
                r.setValue(row, col, s);
                r.setValue(col, row, s);
                //计算变量的个数
                elementCount[0] += (2 * H.getN() - col - row);
            }
        }
        return r;
    }

    public void doOpt() {
        int n = island.getBuses().size();
        int size = 2 * n;//状态变量的维数

        YMatrixGetter Y = new YMatrixGetter(island);
        Y.formYMatrix();
        ASparseMatrixLink2D bApos = Y.formBApostrophe(false, n);

        //约束中非零元的个数
        int[] element_count = new int[]{0};

        ASparseMatrixLink2D[] Ds = new ASparseMatrixLink2D[candPos.length + 1];
        ASparseMatrixLink2D H0 = formH(Y, bApos, existMeasTypes, existMeasPos);
        Ds[0] = formHTWH(H0, existMeasWeight, element_count);

        int i = 1;
        for(int pos : candPos) {
            int[] measTypes = measTypesPerPos[i - 1];
            ASparseMatrixLink2D H = formH(Y, bApos, measTypes, pos);
            //H.printOnScreen();
            Ds[i] = formHTWH(H, measWeight[i - 1], element_count);
            i++;
        }


        //开始开辟内存

        //objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数
        double objValue[] = new double[candPos.length + (candPos.length + 1) * (size * size + size) / 2];
        //状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double columnLower[] = new double[objValue.length];
        //状态变量上限
        double columnUpper[] = new double[objValue.length];
        //指明那些是整数
        int whichInt[] = new int[candPos.length];

        //约束下限
        double rowLower[] = new double[(4 * candPos.length + 1 ) * (size * size + size) / 2];
        //约束上限
        double rowUpper[] = new double[rowLower.length];
        //约束中非零元系数
        double element[] = new double[element_count[0] + 5 * candPos.length * size * (size + 1)];
        //上面系数对应的列
        int column[] = new int[element.length];
        //每一行起始位置
        int starts[] = new int[rowLower.length + 1];
        starts[0] = 0;

        //开始赋值

        //给目标函数中的系数赋值
        for(i = 0; i < objValue.length; i++) {
            if(i < candPos.length) {
                columnLower[i] = 0;
                columnUpper[i] = 1;
                objValue[i] = 0;
            } else {
                columnLower[i] = Double.MIN_VALUE;
                columnUpper[i] = Double.MAX_VALUE;
                objValue[i] = 0;
            }
        }
        //对于row, col的元素,在变量中位置为candPos.length + row * size - row * (row + 1)/2 + col
        for(int row = 0; row < size; row++)
            objValue[candPos.length + row * size - row * (row + 1)/2 + row] = 1.0;

        //对约束的参数赋值
        int k, col, index, count, rowInA = 1;
        int nonZeroOfRow = 0, nonZeroOfCol[] = new int[size], nonZeroOfCurrent;
        for (int row = 0; row < size; row++) {
            for(int j = row; j < size; j++, rowInA++) {
                //记录当前行的前j列共有多少个非零元
                nonZeroOfCol[j] = 0;
                for(ASparseMatrixLink2D m : Ds)
                    nonZeroOfCol[j] += m.getNA()[row] * (j - row + 1);
                if(j == row) {
                    rowUpper[rowInA - 1] = 1;
                    rowLower[rowInA - 1] = 1;
                } else {
                    rowUpper[rowInA - 1] = 0;
                    rowLower[rowInA - 1] = 0;
                }
                starts[rowInA] = nonZeroOfRow + nonZeroOfCol[j];
            }

            nonZeroOfCurrent = 0;//记录当前
            count = 0; //记录当前矩阵的位置
            for(ASparseMatrixLink2D m : Ds) {
                k = m.getIA()[row];
                while (k != -1) {
                    col = m.getJA().get(k);
                    for(int j = row; j < size; j++) {
                        index = nonZeroOfRow + nonZeroOfCurrent;
                        if(j > row)
                            index += nonZeroOfCol[j - 1];
                        element[index] = m.getVA().get(k);
                        if(col > j)
                            column[index] = candPos.length + count * (size - 1) * (size + 2) / 2 + j * size - j * (j + 1) / 2 + col;
                        else
                            column[index] = candPos.length + count * (size - 1) * (size + 2) / 2 + col * size - col * (col + 1) / 2 + j;
                    }
                    nonZeroOfCurrent++;
                    k = m.getLINK().get(k);
                }
                count++;
            }
            //记录前row行一共多少个非零元
            for(ASparseMatrixLink2D m : Ds)
                nonZeroOfRow += m.getNA()[row] * (size - row);
        }

        System.out.println("====================");
        for(ASparseMatrixLink2D m : Ds)
            m.printOnScreen2();
        System.out.println("====================");

        index = nonZeroOfRow;
        for(i = 1; i < Ds.length; i++) {
            for(int row = 0; row < size; row++) {
                for(col = row; col < size; col++) {
                    element[index] = 1;
                    column[index++] = candPos.length + i * (size - 1) * (size + 2) / 2 + row * size - row * (row + 1)/2 + col;
                    element[index] = 1000;
                    column[index++] = i - 1;
                    //约束上下限
                    rowUpper[rowInA - 1] = Double.MAX_VALUE;
                    rowLower[rowInA - 1] = 0;
                    starts[rowInA] = starts[rowInA - 1] + 2;
                    rowInA++;

                    element[index] = 1;
                    column[index++] = candPos.length + i * (size - 1) * (size + 2) / 2 + row * size - row * (row + 1)/2 + col;
                    element[index] = -1000;
                    column[index++] = i - 1;
                    rowUpper[rowInA - 1] = 0;
                    rowLower[rowInA - 1] = Double.MIN_VALUE;
                    starts[rowInA] = starts[rowInA - 1] + 2;
                    rowInA++;

                    element[index] = 1;
                    column[index++] = candPos.length + i * (size - 1) * (size + 2) / 2  + row * size - row * (row + 1)/2 + col;
                    element[index] = -1000;
                    column[index++] = i - 1;
                    element[index] = -1;
                    column[index++] = candPos.length + row * size - row * (row + 1)/2 + col;;
                    rowUpper[rowInA - 1] = Double.MAX_VALUE;
                    rowLower[rowInA - 1] = -1000;
                    starts[rowInA] = starts[rowInA - 1] + 3;
                    rowInA++;

                    element[index] = 1;
                    column[index++] = candPos.length + i * (size - 1) * (size + 2) / 2  + row * size - row * (row + 1)/2 + col;
                    element[index] = 1000;
                    column[index++] = i - 1;
                    element[index] = -1;
                    column[index++] = candPos.length + row * size - row * (row + 1)/2 + col;
                    rowUpper[rowInA - 1] = 1000;
                    rowLower[rowInA - 1] = Double.MIN_VALUE;
                    starts[rowInA] = starts[rowInA - 1] + 3;
                    rowInA++;
                }
            }
        }


        //01变量放在最前面的位置
        for(i = 0; i < candPos.length; i++)
            whichInt[i] = i;

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
            System.out.println("优化结果：");
        }
    }

    public void setExistMeasTypes(int[] existMeasTypes) {
        this.existMeasTypes = existMeasTypes;
    }

    public void setExistMeasPos(int[] existMeasPos) {
        this.existMeasPos = existMeasPos;
    }

    public double[] getExistMeasWeight() {
        return existMeasWeight;
    }

    public void setExistMeasWeight(double[] existMeasWeight) {
        this.existMeasWeight = existMeasWeight;
    }

    public void setCandPos(int[] candPos) {
        this.candPos = candPos;
    }

    public void setMeasTypesPerPos(int[][] measTypesPerPos) {
        this.measTypesPerPos = measTypesPerPos;
    }
    public void setMeasWeight(double[][] measWeight) {
        this.measWeight = measWeight;
    }
}
