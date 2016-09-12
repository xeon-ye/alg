package zju.dsntp;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import org.apache.log4j.Logger;
import zju.common.NewtonSolver;
import zju.common.NewtonWlsModel;
import zju.devmodel.MapObject;
import zju.dsmodel.DsTopoIsland;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.se.AbstractSeAlg;
import zju.util.JacobianMakerPC;

/**
 * Branch current based state estimation
 *
 * @author: Dong Shufeng
 * Date: 2014/5/6
 */
public class BcbNewtonSe extends AbstractSeAlg implements NewtonWlsModel {

    private static Logger log = Logger.getLogger(BcbNewtonSe.class);

    //电气岛
    protected DsTopoIsland dsIsland;

    protected int[] branchOffset;

    public double[][] vTemp;

    public double[][] iTemp;

    //Jacobian矩阵，量测对所有节点的电压，相角
    public SparseDoubleMatrix2D jacobian;
    //Jacobian矩阵，去掉了不能观测的节点
    public SparseDoubleMatrix2D reducedJac;
    //用于精简Jacobian矩阵的函数
    public IntIntDoubleFunction reduceFunc;

    @Override
    public AVector getFinalVTheta() {
        return null;
    }

    public void doSeAnalyse() {
        long start = System.currentTimeMillis();
        final int n = Y.getAdmittance()[0].getM();

        int nonzero = JacobianMakerPC.getNonZeroNum(meas, Y);
        jacobian = new MySparseDoubleMatrix2D(meas.getZ().getN(), 2 * n, nonzero, 0.9, 0.95);

        NewtonSolver solver = new NewtonSolver();
        solver.setModel(this);
        setConverged(solver.solveWls());
        setIterNum(solver.getIterNum());
        setTimeUsed(System.currentTimeMillis() - start);
    }

    @Override
    public AVector getWeight() {
        return meas.getWeight();
    }

    @Override
    public boolean isJacLinear() {
        return true;
    }

    @Override
    public boolean isTolSatisfied(double[] delta) {
        return false;
    }

    @Override
    public AVector getInitial() {
        return null;
    }

    @Override
    public DoubleMatrix2D getJocobian(AVector state) {
        return null;
    }

    @Override
    public ASparseMatrixLink2D getJacobianStruc() {
        return null;
    }

    @Override
    public AVector getZ() {
        return meas.getZ();
    }

    @Override
    public double[] getDeltaArray() {
        return null;
    }

    @Override
    public AVector calZ(AVector state) {
        int index = 0;
        AVector result = meas.getZ_estimate();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                case TYPE_BUS_VOLOTAGE:
                    log.warn("基于支路电流的估计方法不支持电压和相角量测.");
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        int phase = meas.getBus_p_phase()[i];
                        //result.setValue(index, calBusPQ(dsIsland.getBusNoToTn().get(num), phase)[0]);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];
                        int phase = meas.getLine_from_p_phase()[k];
                        MapObject obj = dsIsland.getIdToBranch().get(num);
                        //result.setValue(index, calLinePQFrom(obj, phase)[0]);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];
                        int phase = meas.getLine_to_p_phase()[k];
                        MapObject obj = dsIsland.getIdToBranch().get(num);
                        //result.setValue(index, calLinePQTo(obj, phase)[0]);
                    }
                    break;
                case TYPE_LINE_FROM_CURRENT:
                    for (int k = 0; k < meas.getLine_from_i_amp_pos().length; k++, index++) {
                        Integer num = meas.getLine_from_i_amp_pos()[k];//num starts from 1
                        int phase = meas.getLine_from_i_amp_phase()[k];//num starts from 1
                        MapObject obj = dsIsland.getIdToBranch().get(num);
                        double[] c = dsIsland.getBranchHeadI().get(obj)[phase];
                        if (dsIsland.isICartesian()) {
                            result.setValue(index, c[0] * c[0] + c[1] * c[1]);//todo:
                        } else
                            result.setValue(index, c[0]);
                    }
                    break;
                case TYPE_LINE_TO_CURRENT:
                    for (int k = 0; k < meas.getLine_to_i_amp_pos().length; k++, index++) {
                        Integer num = meas.getLine_to_i_amp_pos()[k];//num starts from 1
                        int phase = meas.getLine_to_i_amp_phase()[k];//num starts from 1
                        MapObject obj = dsIsland.getIdToBranch().get(num);
                        double[] c = dsIsland.getBranchTailI().get(obj)[phase];
                        if (dsIsland.isICartesian()) {
                            result.setValue(index, c[0] * c[0] + c[1] * c[1]);//todo:
                        } else
                            result.setValue(index, c[0]);
                    }
                    break;
                default:
                    break;
            }
        }
        return meas.getZ_estimate();
    }


    @Override
    public boolean isJacStrucChange() {
        return false;
    }

}
