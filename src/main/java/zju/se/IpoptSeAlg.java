package zju.se;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix2D;
import zju.common.IpoptModel;
import zju.common.IpoptSolver;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.measure.MeasTypeCons;
import zju.util.*;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2009-3-15
 */
public class IpoptSeAlg extends AbstractSeAlg implements MeasTypeCons, IpoptModel {
    public static final int VARIABLE_VTHETA = 1;
    public static final int VARIABLE_U = 2;
    public static final int VARIABLE_VTHETA_PQ = 3;
    public static final int VARIABLE_UI = 4;

    private int variable_type = -1;
    // Problem sizes
    protected int m, n, nele_jac, nele_hess, dimension, busNumber;

    protected double[] measLowerLimit, measUpperLimit;

    protected int slackBusCol = -1;

    protected MySparseDoubleMatrix2D jacobian, hessian;

    protected DoubleMatrix2D G, B;

    protected JacobianMakerRC jacobianMaker;

    protected double[] rcCurrent;

    protected SeObjective objFunc = new SeObjective();

    public void initial() {
        iterNum = 0;
        if (getSlackBusNum() > 0)
            slackBusCol = getSlackBusNum() - 1; // 得到松弛节点的列号
        busNumber = Y.getAdmittance()[0].getM(); // 节点数量
        //初始化约束个数和状态变量个数
        switch (variable_type) {
            case VARIABLE_VTHETA:
            case VARIABLE_U:
                dimension = 2 * busNumber; // 状态变量个数
                m = getMeas().getZ().getN() + zeroPBuses.length + zeroQBuses.length; // 量测个数加零注入个数
                break;
            case VARIABLE_VTHETA_PQ:
                dimension = 4 * busNumber;
                m = busNumber * 2 + getMeas().getZ().getN();
                break;
            case VARIABLE_UI:
                dimension = 4 * busNumber;
                m = busNumber * 2 + getMeas().getZ().getN() + zeroPBuses.length + zeroQBuses.length;
                break;
            default:
                break;
        }
        if (variable_type == VARIABLE_UI || variable_type == VARIABLE_U) {
            if (slackBusCol >= 0)
                m++;
            if (isSlackBusVoltageFixed())
                m++;
        }
        n = dimension + getMeas().getZ().getN(); // 状态变量个数加量测个数

        //得到Jacobian矩阵的非零元个数
        if (variable_type == VARIABLE_UI || variable_type == VARIABLE_U) {
            //初始化直角坐标系下的Jacobian矩阵生成器
            if (variable_type == VARIABLE_UI)
                jacobianMaker = new JacobianMakerRC(JacobianMakerRC.MODE_UI);
            else
                jacobianMaker = new JacobianMakerRC(JacobianMakerRC.MODE_U_ONLY);
            jacobianMaker.setY(Y);
            G = jacobianMaker.getG();
            B = jacobianMaker.getB();
            nele_jac = G.cardinality() * 4 + 2 * busNumber + jacobianMaker.getNonZeroNum(meas);
            nele_jac += jacobianMaker.getNoneZeroNum(zeroPBuses, zeroQBuses);
            if (slackBusCol >= 0) {
                if (isSlackBusVoltageFixed())
                    nele_jac++;
                nele_jac += 2;
            }
        } else if (variable_type == VARIABLE_VTHETA) {
            nele_jac = JacobianMakerPC.getNonZeroNum(meas, Y);
            nele_jac += JacobianMakerPC.getNoneZeroNum(zeroPBuses, zeroQBuses, Y);
        } else if (variable_type == VARIABLE_VTHETA_PQ) {
            nele_jac = JacobianMakerPC.getNonZeroNumOfFullState(meas, Y);
        }
        jacobian = new MySparseDoubleMatrix2D(m, dimension, nele_jac, 0.2, 0.9);
        nele_jac += meas.getZ().getN();

        //初始化Jacobian不变的部分
        switch (variable_type) {
            case VARIABLE_VTHETA_PQ:
                JacobianMakerPC.getConstantPartOfFullState(meas, Y, jacobian, 0);
                for (int i = 0; i < busNumber; i++) {
                    jacobian.setQuick(i + meas.getZ().getN(), i + 2 * busNumber, -1.0);
                    jacobian.setQuick(i + busNumber + meas.getZ().getN(), i + 3 * busNumber, -1.0);
                }
                break;
            case VARIABLE_U:
                int rowCount = meas.getZ().getN() + zeroPBuses.length + zeroQBuses.length;
                if (slackBusCol >= 0) {
                    if (isSlackBusVoltageFixed())
                        jacobian.setQuick(rowCount++, slackBusCol, 1.0);
                    jacobian.setQuick(rowCount, slackBusCol, -Math.tan(getSlackBusAngle()));
                    jacobian.setQuick(rowCount, slackBusCol + busNumber, 1.0);
                }
                break;
            case VARIABLE_UI:
                //对Jacobian中不变的部分进行赋值
                G.forEachNonZero(new IntIntDoubleFunction() {
                    public double apply(int row, int col, double v) {
                        jacobian.setQuick(row + meas.getZ().getN(), col, v);
                        jacobian.setQuick(row + meas.getZ().getN(), col + G.columns(), -B.getQuick(row, col));
                        jacobian.setQuick(row + G.rows() + meas.getZ().getN(), col, B.getQuick(row, col));
                        jacobian.setQuick(row + G.rows() + meas.getZ().getN(), col + G.columns(), v);
                        return v;
                    }
                });
                for (int i = 0; i < 2 * busNumber; i++)
                    jacobian.setQuick(i + meas.getZ().getN(), G.columns() + B.columns() + i, -1.0);
                rowCount = 2 * busNumber + meas.getZ().getN() + zeroPBuses.length + zeroQBuses.length;
                if (slackBusCol >= 0) {
                    if (isSlackBusVoltageFixed())
                        jacobian.setQuick(rowCount++, slackBusCol, 1.0);
                    jacobian.setQuick(rowCount, slackBusCol, -Math.tan(getSlackBusAngle()));
                    jacobian.setQuick(rowCount, slackBusCol + busNumber, 1.0);
                }
                break;
            default:
                break;
        }

        //为Hessian开辟内存
        hessian = new MySparseDoubleMatrix2D(dimension, dimension);
        //初始化状态变量的值，并给Hessian矩阵的结构和非零元个数
        getInitialGuess(busNumber, dimension);
        switch (variable_type) {
            case VARIABLE_VTHETA:
                HessianMakerPC.getHessianStruc(meas, Y, hessian);
                HessianMakerPC.getHessianStruc(zeroPBuses, zeroQBuses, Y, hessian);
                break;
            case VARIABLE_U:
                HessianMakerRC.getStrucOfU(meas, Y, hessian);
                HessianMakerRC.getStrucOfU(zeroPBuses, zeroQBuses, Y, hessian);
                break;
            case VARIABLE_VTHETA_PQ:
                for (int i = 0; i < busNumber; i++) {
                    variableState[i + 2 * busNumber] = StateCalByPolar.calBusP(i + 1, Y, variableState);
                    variableState[i + 3 * busNumber] = StateCalByPolar.calBusQ(i + 1, Y, variableState);
                }
                for (int i = 0; i < busNumber; i++)
                    HessianMakerPC.getStrucBusPQ(i + 1, Y, hessian);
                HessianMakerPC.getHessianStruc(meas, Y, hessian);
                break;
            case VARIABLE_UI:
                StateCalByRC.calI(Y, variableState, variableState, 2 * busNumber);
                rcCurrent = new double[2 * busNumber];
                HessianMakerRC.getStrucOfUI(meas, Y, hessian);
                HessianMakerRC.getStrucOfUI(zeroPBuses, zeroQBuses, Y, hessian);
                break;
            default:
                break;
        }

        initialMeasInObjFunc();
        objFunc.getHessStruc(hessian);
        nele_hess = hessian.cardinality() + measInObjFunc.length;
    }

    @Override
    public void setStartingPoint(double[] x) {
        System.arraycopy(variableState, 0, x, 0, variableState.length);
        StateCalByPolar.getEstimatedZ(meas, Y, variableState);
        for (int i = 0; i < getMeas().getZ().getN(); i++)
            x[i + dimension] = meas.getZ().getValue(i) - meas.getZ_estimate().getValue(i);
    }

    public boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g) {
        switch (variable_type) {
            case VARIABLE_VTHETA:
                StateCalByPolar.getEstimatedZ(meas, Y, x);
                break;
            case VARIABLE_U:
                StateCalByRC.getEstimatedZ_U(meas, Y, x);
                if (slackBusCol >= 0) {
                    int i = meas.getZ().getN() + zeroPBuses.length + zeroQBuses.length;
                    if (isSlackBusVoltageFixed())
                        g[i++] = x[slackBusCol] - getSlackBusVoltage() * Math.cos(getSlackBusAngle());
                    g[i] = x[slackBusCol + busNumber] - Math.tan(getSlackBusAngle()) * x[slackBusCol];
                }
                break;
            case VARIABLE_VTHETA_PQ:
                StateCalByPolar.getEstZOfFullState(meas, Y, x);
                for (int j = 0, i = meas.getZ().getN(); j < busNumber; i++, j++) {
                    g[i] = StateCalByPolar.calBusP(j + 1, Y, x) - x[j + 2 * busNumber];
                    g[i + busNumber] = StateCalByPolar.calBusQ(j + 1, Y, x) - x[j + 3 * busNumber];
                }
                break;
            case VARIABLE_UI:
                StateCalByRC.getEstimatedZ_UI(meas, Y, x);
                StateCalByRC.calI(Y, x, rcCurrent);
                int i = meas.getZ().getN();
                for (int j = 0; j < rcCurrent.length; j++, i++)
                    g[i] = rcCurrent[j] - x[j + 2 * busNumber];
                i += (zeroPBuses.length + zeroQBuses.length);
                if (slackBusCol >= 0) {
                    if (isSlackBusVoltageFixed())
                        g[i++] = x[slackBusCol] - getSlackBusVoltage() * Math.cos(getSlackBusAngle());
                    g[i] = x[slackBusCol + busNumber] - Math.tan(getSlackBusAngle()) * x[slackBusCol];
                }
                break;
            default:
                break;
        }
        int i = 0;
        for (; i < meas.getZ_estimate().getN(); i++)
            g[i] = meas.getZ_estimate().getValue(i) + x[i + dimension] - meas.getZ().getValue(i);
        int offset = -1;
        if (variable_type == VARIABLE_VTHETA || variable_type == VARIABLE_U)
            offset = meas.getZ().getN();
        else if (variable_type == VARIABLE_UI)
            offset = meas.getZ().getN() + 2 * busNumber;
        if (offset == -1)
            return true;
        i = offset;
        switch (variable_type) {
            case VARIABLE_VTHETA:
                for (int j = 0; j < zeroPBuses.length; j++, i++)
                    g[i] = StateCalByPolar.calBusP(zeroPBuses[j], Y, x);
                for (int j = 0; j < zeroQBuses.length; j++, i++)
                    g[i] = StateCalByPolar.calBusQ(zeroQBuses[j], Y, x);
                break;
            case VARIABLE_U:
                for (int j = 0; j < zeroPBuses.length; j++, i++)
                    g[i] = StateCalByRC.calBusP_U(zeroPBuses[j], Y, x);
                for (int j = 0; j < zeroQBuses.length; j++, i++)
                    g[i] = StateCalByRC.calBusQ_U(zeroQBuses[j], Y, x);
                break;
            case VARIABLE_UI:
                for (int j = 0; j < zeroPBuses.length; j++, i++)
                    g[i] = StateCalByRC.calBusP_UI(zeroPBuses[j], busNumber, x);
                for (int j = 0; j < zeroQBuses.length; j++, i++)
                    g[i] = StateCalByRC.calBusQ_UI(zeroQBuses[j], busNumber, x);
                break;
            default:
                break;
        }
        return true;
    }

    protected void updateJacobian(double[] x, boolean isNewX) {
        switch (variable_type) {
            case VARIABLE_VTHETA:
                JacobianMakerPC.getJacobianOfVTheta(meas, Y, x, jacobian);
                JacobianMakerPC.getJacobianOfVTheta(zeroPBuses, zeroQBuses, Y, x, jacobian, meas.getZ().getN());
                break;
            case VARIABLE_VTHETA_PQ:
                JacobianMakerPC.getVariablePartOfFullState(meas, Y, x, jacobian, 0);
                for (int i = 0; i < busNumber; i++) {
                    JacobianMakerPC.fillJacobian_bus_p(i + 1, Y, x, jacobian, i + meas.getZ().getN());
                    JacobianMakerPC.fillJacobian_bus_q(i + 1, Y, x, jacobian, i + busNumber + meas.getZ().getN());
                }
                break;
            case VARIABLE_U:
            case VARIABLE_UI:
                jacobianMaker.setUI(x);
                jacobianMaker.getJacobian(meas, jacobian, 0);
                int offset = meas.getZ().getN();
                if (variable_type == VARIABLE_UI)
                    offset += 2 * busNumber;
                jacobianMaker.getJacobian(zeroPBuses, zeroQBuses, jacobian, offset);
                break;
            default:
                break;
        }
    }

    protected void updateHessian(double[] x, double obj_factor, double[] lambda) {
        hessian.forEachNonZero((i, i2, v) -> 0.0);
        switch (variable_type) {
            case VARIABLE_VTHETA:
                HessianMakerPC.getHessianOfVTheta(meas, Y, x, hessian, lambda, 0);
                HessianMakerPC.getHessian(zeroPBuses, zeroQBuses, Y, x, hessian, lambda, meas.getZ().getN());
                break;
            case VARIABLE_U:
                HessianMakerRC.getHessianOfU(meas, Y, hessian, lambda, 0);
                HessianMakerRC.getHessianOfU(zeroPBuses, zeroQBuses, Y, hessian, lambda, meas.getZ().getN());
                break;
            case VARIABLE_VTHETA_PQ:
                HessianMakerPC.getHessianOfFullState(meas, Y, x, hessian, lambda, 0);
                for (int i = 0; i < busNumber; i++) {
                    if (Math.abs(lambda[i + meas.getZ().getN()]) > 1e-10)
                        HessianMakerPC.getHessianBusP(i + 1, Y, x, hessian, lambda[i + meas.getZ().getN()]);
                    if (Math.abs(lambda[i + busNumber + meas.getZ().getN()]) > 1e-10)
                        HessianMakerPC.getHessianBusQ(i + 1, Y, x, hessian, lambda[i + busNumber + meas.getZ().getN()]);
                }
                break;
            case VARIABLE_UI:
                HessianMakerRC.getHessianOfUI(meas, Y, hessian, lambda, 0);
                int index = meas.getZ().getN() + 2 * busNumber;
                HessianMakerRC.getHessianOfUI(zeroPBuses, zeroQBuses, Y, hessian, lambda, index);
                break;
            default:
                break;
        }

    }

    public void setXLimit(double[] x_L, double[] x_U) {
        if (variable_type == VARIABLE_VTHETA
                || variable_type == VARIABLE_VTHETA_PQ) {
            for (int i = 0; i < busNumber; i++) {
                x_L[i] = 0.;
                x_U[i] = 2.0;
                x_L[i + busNumber] = -Math.PI;
                x_U[i + busNumber] = Math.PI;
            }
            if (slackBusCol >= 0) {
                x_L[slackBusCol + busNumber] = getSlackBusAngle();
                x_U[slackBusCol + busNumber] = getSlackBusAngle();
                if (isSlackBusVoltageFixed()) {
                    x_L[slackBusCol] = getSlackBusVoltage();
                    x_U[slackBusCol] = getSlackBusVoltage();
                }
            }
        } else {
            for (int i = 0; i < x_L.length; i++) {//todo: contraint is loose
                x_L[i] = -2e15;
                x_U[i] = 2e15;
            }
            //for (int i = 0; i < busNumber; i++) {
            //    x_L[i] = -2.0;
            //    x_U[i] = 2.0;
            //}
        }
        if (variable_type == VARIABLE_VTHETA_PQ) {
            for (int i = 0; i < busNumber; i++) {
                //if(getIsland().getBusMap().get(i + 1).getType() == BusData.BUS_TYPE_GEN_PV
                //        || getIsland().getBusMap().get(i + 1).getType() == BusData.BUS_TYPE_SLACK) {
                //    x_L[i + 2 * busNumber] = 0;
                //    x_U[i + 2 * busNumber] = 2e15;
                //    x_L[i + 3 * busNumber] = 0;
                //    x_U[i + 3 * busNumber] = 2e15;
                //} else {
                //    x_L[i + 2 * busNumber] = -2e15;
                //    x_U[i + 2 * busNumber] = 0;
                //    x_L[i + 3 * busNumber] = -2e15;
                //    x_U[i + 3 * busNumber] = 0;
                //}
                x_L[i + 2 * busNumber] = -2e15;
                x_U[i + 2 * busNumber] = 2e15;
                x_L[i + 3 * busNumber] = -2e15;
                x_U[i + 3 * busNumber] = 2e15;
            }
            for (int busNo : zeroPBuses) {
                //x_L[busNo + 2 * busNumber - 1] = -tol_p;
                //x_U[busNo + 2 * busNumber - 1] = tol_p;
                x_L[busNo + 2 * busNumber - 1] = 0.0;
                x_U[busNo + 2 * busNumber - 1] = 0.0;
            }
            for (int busNo : zeroQBuses) {
                //x_L[busNo + 3 * busNumber - 1] = -tol_q;
                //x_U[busNo + 3 * busNumber - 1] = tol_q;
                x_L[busNo + 3 * busNumber - 1] = 0.0;
                x_U[busNo + 3 * busNumber - 1] = 0.0;
            }
        }
        initialMeasLimit(dimension, x_L, x_U);
    }

    @Override
    public AVector getFinalVTheta() {
        switch (variable_type) {
            case VARIABLE_VTHETA:
                return new AVector(variableState);
            case VARIABLE_VTHETA_PQ:
                double[] vTheta = new double[2 * busNumber];
                System.arraycopy(variableState, 0, vTheta, 0, vTheta.length);
                return new AVector(vTheta);
            case VARIABLE_U:
            case VARIABLE_UI:
                double[] state = new double[2 * busNumber];
                for (int i = 0; i < busNumber; i++) {
                    double a = variableState[i];
                    double b = variableState[i + busNumber];
                    state[i] = Math.sqrt(a * a + b * b);
                    state[i + busNumber] = Math.atan2(b, a);
                }
                return new AVector(state);
            default:
                return null;
        }
    }

    public void initialMeasInObjFunc() {
        measInObjFunc = new int[meas.getZ().getN()];
        for (int i = 0; i < meas.getZ().getN(); i++)
            measInObjFunc[i] = i;
        if (objFunc != null)
            objFunc.setMeasInObjFunc(getMeasInObjFunc());
    }

    protected void initialMeasLimit(int offset, double[] x_L, double[] x_U) {
        if (measLowerLimit != null)
            System.arraycopy(measLowerLimit, 0, x_L, offset, meas.getZ().getN());
        else
            for (int i = 0; i < getMeas().getZ().getN(); i++)
                x_L[i + offset] = -2e15;
        if (measUpperLimit != null)
            System.arraycopy(measUpperLimit, 0, x_U, offset, meas.getZ().getN());
        else
            for (int i = 0; i < getMeas().getZ().getN(); i++)
                x_U[i + offset] = 2e15;
    }

    // 开始状态估计计算
    public void doSeAnalyse() {
        long start = System.currentTimeMillis();
        initial();
        IpoptSolver solver = new IpoptSolver(this);
        solver.solve(maxIter, tolerance, true);
        setConverged(solver.isConverged());
        objective = solver.getObjective();
        setTimeUsed(System.currentTimeMillis() - start);
    }

    public boolean eval_f(int n, double[] x, boolean new_x, double[] obj_value) {
        return objFunc.eval_f(n, x, new_x, obj_value, dimension);
    }

    public boolean eval_grad_f(int n, double[] x, boolean new_x, double[] grad_f) {
        return objFunc.eval_grad_f(n, x, new_x, grad_f, dimension);
    }

    public boolean eval_jac_g(int n, double[] x, boolean new_x, int m, int nele_jac, final int[] iRow, final int[] jCol, final double[] values) {
        updateJacobian(x, new_x);
        final int[] count = {0};
        if (values == null) {
            jacobian.forEachNonZero((row, col, v) -> {
                iRow[count[0]] = row;
                jCol[count[0]] = col;
                count[0]++;
                return v;
            });
            for (int i = 0; i < meas.getZ().getN(); i++) {
                iRow[count[0]] = i;
                jCol[count[0]] = i + dimension;
                count[0]++;
            }
        } else {
            jacobian.forEachNonZero((row, col, v) -> {
                values[count[0]] = v;
                count[0]++;
                return v;
            });
            for (int i = 0; i < meas.getZ().getN(); i++) {
                values[count[0]] = 1.0;
                count[0]++;
            }
        }
        return true;
    }

    public boolean eval_h(int n, double[] x, boolean new_x, double obj_factor, int m, double[] lambda, boolean new_lambda, int nele_hess, final int[] iRow, final int[] jCol, final double[] values) {
        this.setIterNum(this.getIterNum() + 1);//todo: this is not perfect
        final int[] idx = new int[]{0};
        if (values == null) {
            objFunc.getHessStruc(iRow, jCol, hessian, dimension);
            hessian.forEachNonZero((i, j, v) -> {
                iRow[idx[0]] = i;
                jCol[idx[0]] = j;
                idx[0]++;
                return v;
            });
        } else {
            updateHessian(x, obj_factor, lambda);
            objFunc.fillHessian(x, values, hessian, obj_factor, dimension);
            hessian.forEachNonZero((i, j, v) -> {
                values[idx[0]] = v;
                idx[0]++;
                return v;
            });
        }
        return true;
    }

    public void fillState(double[] x) {
        System.arraycopy(x, 0, getVariableState(), 0, getVariableState().length);
    }

    public SeObjective getObjFunc() {
        return objFunc;
    }

    public void setObjFunc(SeObjective objFunc) {
        this.objFunc = objFunc;
    }

    public double[] getMeasLowerLimit() {
        return measLowerLimit;
    }

    public void setMeasLowerLimit(double[] measLowerLimit) {
        this.measLowerLimit = measLowerLimit;
    }

    public double[] getMeasUpperLimit() {
        return measUpperLimit;
    }

    public void setMeasUpperLimit(double[] measUpperLimit) {
        this.measUpperLimit = measUpperLimit;
    }

    public void setGLimit(double[] g_L, double[] g_U) {
        /* set the values of the constraint bounds */
        for (int i = 0; i < m; i++) {
            g_L[i] = 0;
            g_U[i] = 0;
        }
        int offset = -1;
        if (variable_type == VARIABLE_VTHETA || variable_type == VARIABLE_U)
            offset = meas.getZ().getN();
        else if (variable_type == VARIABLE_UI)
            offset = meas.getZ().getN() + 2 * busNumber;
        if (offset != -1) {
            int i = offset;
            for (int j = 0; j < zeroPBuses.length; j++, i++) {
                //g_L[i] = -tol_p;
                //g_U[i] = tol_p;
                g_L[i] = 0.;
                g_U[i] = 0.;
            }
            for (int j = 0; j < zeroQBuses.length; j++, i++) {
                //g_L[i] = -tol_q;
                //g_U[i] = tol_q;
                g_L[i] = 0.;
                g_U[i] = 0.;
            }
        }
    }

    public int getN() {
        return n;
    }

    public int getM() {
        return m;
    }

    public int getNele_jac() {
        return nele_jac;
    }

    public int getNele_hess() {
        return nele_hess;
    }

    public int getVariable_type() {
        return variable_type;
    }

    public void setVariable_type(int variable_type) {
        this.variable_type = variable_type;
    }
}
