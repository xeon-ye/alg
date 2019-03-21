package zju.ta;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.coinor.Ipopt;
import zju.common.IpoptModel;
import zju.common.IpoptSolver;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-18
 */
public class ExciterXcOpt implements IpoptModel {
    private static Logger log = LogManager.getLogger(ExciterXcOpt.class);
    public final static int OBJ_TS_MAX = 1;
    public final static int OBJ_TD_MAX = 2;
    public final static int OBJ_WEIGHTED_MAX = 3;
    public final static int CONSTRAINT_NULL = 0;
    public final static int CONSTRAINT_TS = 1;
    public final static int CONSTRAINT_TD = 2;

    private HeffronPhilipsSystem hpSys;
    private double xcMin;
    private double xcMax;
    private double[] variableState = new double[4];//xc, k5, k6, vh1Theta
    private boolean isConverged;
    private boolean isPrintPath;
    private double objective;
    private int obj_option = OBJ_TS_MAX;
    private int cons_option = CONSTRAINT_NULL;

    public void doOpt() {
        doOpt(0.0);
    }

    public void doOpt(double initialXc) {
        doOpt(initialXc, OBJ_TS_MAX);
    }

    public void doOpt(double initialXc, int obj_option) {
        doOpt(initialXc, obj_option, CONSTRAINT_NULL);
    }

    public void doOpt(double initialXc, int obj_option, int cons_option) {
        initialPara(initialXc);
        setObj_option(obj_option);
        setCons_option(cons_option);
        IpoptSolver solver = new IpoptSolver(this);
        solver.setOption(Ipopt.KEY_HESSIAN_APPROXIMATION, "limited-memory");
        solver.solve();
        isConverged = solver.isConverged();
        if (solver.isConverged())
            objective = solver.getObjective();
    }

    public void initialPara(double xc) {
        hpSys.setExciterXc(xc);
        variableState[0] = xc;
        variableState[1] = hpSys.getK5();
        variableState[2] = hpSys.getK6();
        variableState[3] = hpSys.getThetaUh1();
    }

    public boolean eval_f(int n, double[] x, boolean new_x, double[] obj_value) {
        double[] deltaTsTd = hpSys.setExciterXc(x[0]);
        switch (obj_option) {
            case OBJ_TS_MAX:
                obj_value[0] = -deltaTsTd[0];
                break;
            case OBJ_TD_MAX:
                obj_value[0] = -deltaTsTd[1];
                break;
            case OBJ_WEIGHTED_MAX:
                obj_value[0] = -deltaTsTd[0] - 1.0 * deltaTsTd[1];
                break;
            default:
                log.warn("Not supported objective option :" + obj_option);
        }
        return true;
    }

    public boolean eval_grad_f(int n, double[] x, boolean new_x, double[] grad_f) {
        double k2 = hpSys.getK2();
        double k3 = hpSys.getK3();
        double k5 = x[1];
        double k6 = x[2];
        double exciterK = hpSys.getExciterK();
        double exciterT = hpSys.getExciterT();
        double tdop = hpSys.getGen().getTdop();
        double tmp1 = (1.0 / k3) + k6 * exciterK - tdop * exciterT;
        double tmp2 = tdop + exciterT / k3;
        double tmp3 = tmp1 * tmp1 + tmp2 * tmp2;

        switch (obj_option) {
            case OBJ_TS_MAX:
                grad_f[0] = 0.0;
                grad_f[1] = (k2 * exciterK * tmp1) / tmp3;
                grad_f[2] = k2 * k5 * exciterK * exciterK * (tmp2 * tmp2 - tmp1 * tmp1) / (tmp3 * tmp3);
                grad_f[3] = 0.0;
                break;
            case OBJ_TD_MAX:
                grad_f[0] = 0.0;
                grad_f[1] = -(k2 * exciterK * tmp2) / tmp3;
                grad_f[2] = k2 * k5 * exciterK * exciterK * (2.0 * tmp1 * tmp2) / (tmp3 * tmp3);
                grad_f[3] = 0.0;
                break;
            case OBJ_WEIGHTED_MAX:
                grad_f[0] = 0.0;
                grad_f[1] = -(1.0 * k2 * exciterK * tmp2 - k2 * exciterK * tmp1) / tmp3;
                grad_f[2] = k2 * k5 * exciterK * exciterK * (tmp2 * tmp2 - tmp1 * tmp1 + 1.0 * 2.0 * tmp1 * tmp2) / (tmp3 * tmp3);
                grad_f[3] = 0.0;
                break;
            default:
                log.warn("Not supported objective option :" + obj_option);
        }
        return true;
    }

    public boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g) {
        hpSys.setExciterXc(x[0]);
        double uh1X = hpSys.getUh1X();
        double uh1Y = hpSys.getUh1Y();
        g[0] = hpSys.getK5() - x[1];
        g[1] = hpSys.getK6() - x[2];
        g[2] = uh1Y / uh1X - Math.tan(x[3]);
        double[] deltaTsTd = hpSys.setExciterXc(x[0]);
        switch (cons_option) {
            case CONSTRAINT_NULL:
                break;
            case CONSTRAINT_TS:
                g[3] = deltaTsTd[0];
                break;
            case CONSTRAINT_TD:
                g[3] = deltaTsTd[1];
                break;
            default:
                log.warn("Not supported constraint option :" + obj_option);
        }
        return true;
    }

    public boolean eval_jac_g(int n, double[] x, boolean new_x, int m, int nele_jac, int[] iRow, int[] jCol, double[] values) {
        if (values == null) {
            iRow[0] = 0;
            jCol[0] = 0;
            iRow[1] = 0;
            jCol[1] = 1;
            iRow[2] = 0;
            jCol[2] = 3;
            iRow[3] = 1;
            jCol[3] = 0;
            iRow[4] = 1;
            jCol[4] = 2;
            iRow[5] = 1;
            jCol[5] = 3;
            iRow[6] = 2;
            jCol[6] = 0;
            iRow[7] = 2;
            jCol[7] = 3;
            switch (cons_option) {
                case CONSTRAINT_NULL:
                    break;
                case CONSTRAINT_TS:
                case CONSTRAINT_TD:
                    iRow[8] = 3;
                    jCol[8] = 0;
                    iRow[9] = 3;
                    jCol[9] = 1;
                    iRow[10] = 3;
                    jCol[10] = 2;
                    break;
                default:
                    log.warn("Not supported constraint option :" + obj_option);
            }
        } else {
            double xc = x[0];
            hpSys.setExciterXc(xc);
            double delta = hpSys.getGenModel().getDelta();
            double thetaUh1q = delta - x[3];
            double us = hpSys.getUs();
            double xdSigma = hpSys.getXdSigma();
            double xqSigma = hpSys.getXqSigma();
            double xdpSigma = hpSys.getXdpSigma();
            values[0] = Math.sin(thetaUh1q) * us * Math.cos(delta) / xqSigma
                    - Math.cos(thetaUh1q) * us * Math.sin(delta) / xdpSigma;
            values[1] = -1;
            values[2] = -Math.cos(thetaUh1q) * us * (hpSys.getGenXq() + xc) * Math.cos(delta) / xqSigma
                    - Math.sin(thetaUh1q) * us * (hpSys.getGenXdp() + xc) * Math.sin(delta) / xdpSigma;
            values[3] = -Math.cos(thetaUh1q) / xdpSigma;
            values[4] = -1;
            values[5] = Math.sin(thetaUh1q) * (xdSigma - hpSys.getGenXd() - xc) / xdpSigma;

            double iX = hpSys.getiAmpl() * Math.cos(hpSys.getiAngleInArc());
            double iY = hpSys.getiAmpl() * Math.sin(hpSys.getiAngleInArc());
            double uh1X = hpSys.getUh1X();
            double uh1Y = hpSys.getUh1Y();
            values[6] = (-iX * uh1X - uh1Y * iY) / (uh1X * uh1X);
            values[7] = -1.0 / (Math.cos(x[3]) * Math.cos(x[3]));

            double k2 = hpSys.getK2();
            double k3 = hpSys.getK3();
            double k5 = x[1];
            double k6 = x[2];
            double exciterK = hpSys.getExciterK();
            double exciterT = hpSys.getExciterT();
            double tdop = hpSys.getGen().getTdop();
            double tmp1 = (1.0 / k3) + k6 * exciterK - tdop * exciterT;
            double tmp2 = tdop + exciterT / k3;
            double tmp3 = tmp1 * tmp1 + tmp2 * tmp2;
            switch (cons_option) {
                case CONSTRAINT_NULL:
                    break;
                case CONSTRAINT_TS:
                    values[8] = 0.0;
                    values[9] = (k2 * exciterK * tmp1) / tmp3;
                    values[10] = k2 * k5 * exciterK * exciterK * (tmp2 * tmp2 - tmp1 * tmp1) / (tmp3 * tmp3);
                    break;
                case CONSTRAINT_TD:
                    values[8] = 0.0;
                    values[9] = -(k2 * exciterK * tmp2) / tmp3;
                    values[10] = k2 * k5 * exciterK * exciterK * (2.0 * tmp1 * tmp2) / (tmp3 * tmp3);
                    break;
                default:
                    log.warn("Not supported constraint option :" + obj_option);
            }
        }
        return true;
    }

    public boolean eval_h(int n, double[] x, boolean new_x, double obj_factor, int m, double[] lambda, boolean new_lambda, int nele_hess, int[] iRow, int[] jCol, double[] values) {
        return false; //todo:
    }

    @Override
    public void setXLimit(double[] x_L, double[] x_U) {
        x_L[0] = xcMin;
        x_L[1] = -2e15;
        x_L[2] = -2e15;
        x_L[3] = -Math.PI;
        x_U[0] = xcMax;
        x_U[1] = -2e15;
        x_U[2] = -2e15;
        x_U[3] = Math.PI;
    }

    @Override
    public void setGLimit(double[] g_L, double[] g_U) {
        switch (cons_option) {
            case CONSTRAINT_NULL:
                break;
            case CONSTRAINT_TS:
            case CONSTRAINT_TD:
                g_L[0] = 0.0;
                g_L[1] = 0.0;
                g_L[2] = 0.0;
                g_L[3] = -2e14;
                g_U[0] = 0.0;
                g_U[1] = 0.0;
                g_U[2] = 0.0;
                g_U[3] = 2e14;
                break;
            default:
                log.warn("Not supported constraint option :" + obj_option);
        }
    }

    public int getNele_jac() {
        if (cons_option != CONSTRAINT_NULL)
            return 11;
        else
            return 8;
    }

    public int getNele_hess() {
        return 0; //todo:
    }

    public int getM() {
        switch (cons_option) {
            case CONSTRAINT_NULL:
                break;
            case CONSTRAINT_TS:
            case CONSTRAINT_TD:
                return 4;
            default:
                log.warn("Not supported constraint option :" + obj_option);
        }
        return 3;
    }

    @Override
    public int getN() {
        return 4;
    }

    @Override
    public void fillState(double[] x) {
        System.arraycopy(x, 0, getVariableState(), 0, getVariableState().length);
    }

    @Override
    public void setStartingPoint(double[] x) {
        System.arraycopy(this.variableState, 0, x, 0, x.length);
    }

    public boolean isPrintPath() {
        return isPrintPath;
    }

    public void setPrintPath(boolean printPath) {
        isPrintPath = printPath;
    }

    public double getXcMin() {
        return xcMin;
    }

    public void setXcMin(double xcMin) {
        this.xcMin = xcMin;
    }

    public double getXcMax() {
        return xcMax;
    }

    public void setXcMax(double xcMax) {
        this.xcMax = xcMax;
    }

    public boolean isConverged() {
        return isConverged;
    }

    public double getObjective() {
        return objective;
    }

    public HeffronPhilipsSystem getHpSys() {
        return hpSys;
    }

    public void setHpSys(HeffronPhilipsSystem hpSys) {
        this.hpSys = hpSys;
    }

    public double getOptXc() {
        return variableState[0];
    }

    public void setObj_option(int obj_option) {
        this.obj_option = obj_option;
    }

    public void setCons_option(int cons_option) {
        this.cons_option = cons_option;
    }

    public double[] getVariableState() {
        return variableState;
    }

    public void setVariableState(double[] variableState) {
        this.variableState = variableState;
    }
}
