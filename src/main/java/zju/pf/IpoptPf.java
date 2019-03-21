package zju.pf;

import cern.colt.function.IntIntDoubleFunction;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.common.IpoptModel;
import zju.common.IpoptSolver;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.measure.MeasTypeCons;
import zju.util.HessianMakerPC;
import zju.util.JacobianMakerPC;
import zju.util.PfUtil;
import zju.util.StateCalByPolar;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-8-9
 */
public class IpoptPf extends PolarPf implements IpoptModel, MeasTypeCons {
    private static Logger log = LogManager.getLogger(IpoptPf.class);

    protected int iterNum;
    // roblem sizes
    protected int n, m, nele_jac, nele_hess;

    protected MySparseDoubleMatrix2D hessian;

    public IpoptPf() {
        setPfMethod(ALG_IPOPT);
    }

    @Override
    public void setOriIsland(IEEEDataIsland oriIsland) {
        super.setOriIsland(oriIsland);
        Y.formConnectedBusCount();
    }

    public void doPf() {
        iterNum = 0;
        if (!ALG_IPOPT.equals(getPfMethod())) {
            super.doPf();
            return;
        }
        if (clonedIsland == null) {
            log.warn("电气岛为NULL, 潮流计算中止.");
            return;
        }
        if (clonedIsland.getSlackBusSize() > 1) {
            log.warn("目前不支持平衡节点个数大于1的情况, 潮流计算中止.");
            return;
        }

        if (isOutagePf) {
            beforeOutagePf();
        } else {
            //形成量测
            setMeas(PfUtil.formPQMeasure(clonedIsland));
            meas.setZ_estimate(new AVector(meas.getZ().getN()));
            initial();
        }
        //如果处理PV节点转PQ的机制，事先将要用到的矩阵形成
        if (isHandleQLim)
            pfSens.formSubMarix(Y.getAdmittance()[1]);

        doPfOnce();

        if (!isConverged()) {
            log.warn("潮流计算不收敛.");
        } else if (isHandleQLim)
            handlePvToPq();

        if (isOutagePf)
            afterOutagePf();
    }

    protected void doPfOnce() {
        if (!ALG_IPOPT.equals(getPfMethod())) {
            super.doPfOnce();
            return;
        }
        IpoptSolver solver = new IpoptSolver(this);
        solver.solve(maxIter, tolerance, true);
        setConverged(solver.isConverged());
    }

    public void initial() {
        // Get the default initial guess
        initialState();

        m = getMeas().getZ().getN();
        n = clonedIsland.getPqBusSize() * 2 + clonedIsland.getPvBusSize();
        //jacobian = JacobianMakerPC.getJacobianOfVTheta(meas, Y, variableState);
        jacStructure = JacobianMakerPC.getJacStrucOfVTheta(meas, Y, clonedIsland.getPqBusSize(), clonedIsland.getPvBusSize());
        jacobian = new MySparseDoubleMatrix2D(m, n, jacStructure.getVA().size(), 0.9, 0.99);
        JacobianMakerPC.getJacobianOfVTheta(meas, Y, variableState,
                clonedIsland.getPqBusSize(), clonedIsland.getPvBusSize(), jacobian);
        nele_jac = jacobian.cardinality();


        hessian = new MySparseDoubleMatrix2D(n, n);
        double[] tmp = new double[m];
        for (int i = 0; i < tmp.length; i++)
            tmp[i] = 1.0;
        HessianMakerPC.getHessianOfVTheta(meas, Y, variableState,
                clonedIsland.getPqBusSize(), clonedIsland.getPvBusSize(), hessian, tmp, 0);
        nele_hess = hessian.cardinality();
    }

    public int getN() {
        return n;
    }

    @Override
    public void fillState(double[] x) {
        updateState(x);
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

    public void setStartingPoint(double[] x) {
        updateShortState(x);
    }

    public boolean eval_f(int n, double[] x, boolean new_x, double[] obj_value) {
        return true;
    }

    public boolean eval_grad_f(int n, double[] x, boolean new_x, double[] grad_f) {
        return true;
    }

    public boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g) {
        if (new_x)
            updateState(x);
        StateCalByPolar.getEstimatedZ(meas, Y, variableState);
        int i = 0;
        for (; i < meas.getZ_estimate().getN(); i++)
            g[i] = meas.getZ_estimate().getValue(i) - meas.getZ().getValue(i);
        return true;
    }

    public boolean eval_jac_g(int n, double[] x, boolean new_x, int m, int nele_jac, final int[] iRow, final int[] jCol, final double[] values) {
        final int[] count = {0};
        updateJacobian(x, new_x);
        if (values == null) {
            jacobian.forEachNonZero(new IntIntDoubleFunction() {
                @Override
                public double apply(int row, int col, double v) {
                    iRow[count[0]] = row;
                    jCol[count[0]] = col;
                    count[0]++;
                    return v;
                }
            });
        } else {
            jacobian.forEachNonZero(new IntIntDoubleFunction() {
                @Override
                public double apply(int row, int col, double v) {
                    values[count[0]] = v;
                    count[0]++;
                    return v;
                }
            });
        }
        return true;
    }

    public boolean eval_h(int n, double[] x, boolean new_x, double obj_factor, int m, final double[] lambda, boolean new_lambda, int nele_hess, final int[] iRow, final int[] jCol, final double[] values) {
        iterNum++; //todo: this is not perfect
        final int[] idx = new int[]{0};
        if (values == null) {
            /* return the structure. This is a symmetric matrix, fill the lower left triangle only. */
            hessian.forEachNonZero(new IntIntDoubleFunction() {
                @Override
                public double apply(int i, int j, double v) {
                    iRow[idx[0]] = i;
                    jCol[idx[0]] = j;
                    idx[0]++;
                    return v;
                }
            });
        } else {
            updateHessian(x, new_x, obj_factor, lambda);
            hessian.forEachNonZero(new IntIntDoubleFunction() {
                @Override
                public double apply(int i, int j, double v) {
                    values[idx[0]] = v;
                    idx[0]++;
                    return v;
                }
            });
        }
        return true;
    }

    @Override
    public void setXLimit(double[] x_L, double[] x_U) {
        int pqSize = clonedIsland.getPqBusSize();
        for (int i = 0; i < pqSize; i++) {
            x_L[i] = -2e15;
            x_U[i] = 2e15;
        }
        for (int i = 0; i < pqSize + clonedIsland.getPvBusSize(); i++) {
            x_L[i + pqSize] = -Math.PI;
            x_U[i + pqSize] = Math.PI;
        }
    }

    @Override
    public void setGLimit(double[] g_L, double[] g_U) {
        int index = 0;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++) {
                        g_L[index] = 0.;
                        g_U[index] = 0.;
                    }
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        g_L[index] = -tol_v;
                        g_U[index] = tol_v;
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        g_L[index] = -tol_p;
                        g_U[index] = tol_p;
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        g_L[index] = -tol_q;
                        g_U[index] = tol_q;
                    }
                    break;
                default:
                    log.warn("潮流方程不支持的类型:" + type);
                    break;
            }
        }
    }

    protected void updateJacobian(double[] x, boolean isNewX) {
        if (isNewX)
            updateState(x);
        JacobianMakerPC.getJacobianOfVTheta(meas, Y, variableState,
                clonedIsland.getPqBusSize(), clonedIsland.getPvBusSize(), jacobian);
    }

    protected void updateHessian(double[] x, boolean new_x, double obj_factor, double[] lambda) {
        if (new_x)
            updateState(x);
        //先置零
        hessian.forEachNonZero(new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int i2, double v) {
                return 0.0;
            }
        });
        HessianMakerPC.getHessianOfVTheta(meas, Y, variableState,
                clonedIsland.getPqBusSize(), clonedIsland.getPvBusSize(), hessian, lambda, 0);
    }
}
