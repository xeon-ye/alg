package zju.se;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import zju.common.NewtonSolver;
import zju.common.NewtonWlsModel;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.util.JacobianMakerPC;
import zju.util.StateCalByPolar;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-19
 */
public class NewtonWlsSe extends AbstractSeAlg implements NewtonWlsModel {
    //private static Logger log = Logger.getLogger(NewtonWlsSe.class);

    public int vCount;
    public int aCount;
    //长度为 vCount + aCount
    public AVector state;
    //长度为2n
    public AVector fullState;
    //Jacobian矩阵，量测对所有节点的电压，相角
    public SparseDoubleMatrix2D jacobian;
    //Jacobian矩阵，去掉了不能观测的节点
    public SparseDoubleMatrix2D reducedJac;
    //用于精简Jacobian矩阵的函数
    public IntIntDoubleFunction reduceFunc;

    @Override
    public AVector getFinalVTheta() {
        return createVTheta(state);
    }

    public void doSeAnalyse() {
        long start = System.currentTimeMillis();
        final int n = Y.getAdmittance()[0].getM();
        vCount = unObserveBuses != null ? n - unObserveBuses.length : n;
        aCount = unObserveBuses != null ? n - unObserveBuses.length : n;
        if (getSlackBusNum() > 0) {
            if (isSlackBusVoltageFixed())
                vCount--;
            aCount--;
        }
        fullState = new AVector(n * 2);
        if (getSlackBusNum() > 0) {
            if (isSlackBusVoltageFixed())
                fullState.setValue(getSlackBusNum() - 1, getSlackBusVoltage());
            fullState.setValue(getSlackBusNum() - 1 + n, getSlackBusAngle());
        }

        int nonzero = JacobianMakerPC.getNonZeroNum(meas, Y);
        jacobian = new MySparseDoubleMatrix2D(meas.getZ().getN(), 2 * n, nonzero, 0.9, 0.95);
        reducedJac = new MySparseDoubleMatrix2D(meas.getZ().getN(), aCount + vCount, nonzero, 0.9, 0.95);

        reduceFunc = new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int j, double v) {
                //System.out.println(i + "\t" + j + "\t" + v);
                int col = j;
                if (j < n) {
                    if (j > getSlackBusNum() - 1 && isSlackBusVoltageFixed())
                        col--;
                    else if (j == getSlackBusNum() - 1 && isSlackBusVoltageFixed())
                        return v;
                    if (unObserveBuses != null)
                        for (int busNum : unObserveBuses) {
                            if (j == busNum - 1)
                                return v;
                            else if (j < busNum - 1)
                                break;
                            else
                                col--;
                        }
                    reducedJac.setQuick(i, col, v);
                } else {
                    if (j > getSlackBusNum() + n - 1)
                        col--;
                    else if (j == getSlackBusNum() + n - 1)
                        return v;
                    if (unObserveBuses != null)
                        for (int busNum : unObserveBuses) {
                            if (j == busNum + n - 1)
                                return v;
                            else if (j < busNum + n - 1)
                                break;
                            else
                                col--;
                        }
                    reducedJac.setQuick(i, col - n + vCount, v);
                }
                return v;
            }
        };

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
        return false;
    }

    @Override
    public boolean isTolSatisfied(double[] delta) {
        return false;
    }

    @Override
    public AVector getInitial() {
        state = new AVector(aCount + vCount);
        for (int i = 0; i < vCount; i++)
            state.setValue(i, 1.0);
        for (int i = 0; i < aCount; i++)
            state.setValue(i + vCount, 0);
        return state;
    }

    @Override
    public DoubleMatrix2D getJocobian(AVector state) {
        JacobianMakerPC.getJacobianOfVTheta(meas, Y, createVTheta(state).getValues(), jacobian);
        jacobian.forEachNonZero(reduceFunc);
        return reducedJac;
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
        StateCalByPolar.getEstimatedZ(meas, Y, createVTheta(state).getValues(), meas.getZ_estimate());
        return meas.getZ_estimate();
    }

    private AVector createVTheta(AVector state) {
        int n = Y.getAdmittance()[0].getM();
        if (unObserveBuses != null && unObserveBuses.length > 0) {
            for (int busNum : unObserveBuses)
                fullState.setValue(busNum - 1, 0.0);
            int count = 0;
            for (int i = 0; i < vCount; i++) {
                if (i + 1 >= unObserveBuses[count])
                    count++;
                if (getSlackBusNum() > 0 && isSlackBusVoltageFixed() && i >= getSlackBusNum() - 1)
                    fullState.setValue(i + count + 1, state.getValue(i));
                else
                    fullState.setValue(i + count, state.getValue(i));
            }
            count = 0;
            for (int i = 0; i < aCount; i++) {
                if (i + 1 >= unObserveBuses[count])
                    count++;
                if (getSlackBusNum() > 0 && i >= getSlackBusNum() - 1)
                    fullState.setValue(i + count + 1 + n, state.getValue(i + vCount));
                else
                    fullState.setValue(i + count + n, state.getValue(i + vCount));
            }
        } else {
            int col;
            for (int i = 0; i < n; i++) {
                col = i;
                if (i > getSlackBusNum() - 1 && isSlackBusVoltageFixed())
                    col--;
                else if (i == getSlackBusNum() - 1 && isSlackBusVoltageFixed())
                    continue;
                fullState.setValue(i, state.getValue(col));
            }
            for (int i = 0; i < n; i++) {
                col = i;
                if (i > getSlackBusNum() - 1)
                    col--;
                else if (i == getSlackBusNum() - 1)
                    continue;
                fullState.setValue(i + n, state.getValue(col + vCount));
            }
        }
        return fullState;
    }

    @Override
    public boolean isJacStrucChange() {
        return false;
    }
}
