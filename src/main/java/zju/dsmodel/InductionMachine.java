package zju.dsmodel;

import cern.colt.matrix.DoubleMatrix2D;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.common.NewtonModel;
import zju.common.NewtonSolver;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;

import java.io.Serializable;

/**
 * This class is used to describe the induction machine. Both generator and motor are modeled. The difference lies
 * in slip and P_KW. For generator P_KW is negative. So is the slip.
 * The unit of voltage is KV, the unit of current is Ampere. The unit of Power is KW.
 *
 * @author : Zhen Dai
 *         Date: 2011-4-29
 *         Time: 13:58:35
 */
public class InductionMachine implements ThreePhaseLoad, NewtonModel, Serializable {
    public static final int CONN_TYPE_Y = 1;
    public static final int CONN_TYPE_D = 2;
    //public static final int CAL_MODE_V_S = 1;
    //public static final int CAL_MODE_I_S = 2;

    private static Logger log = LogManager.getLogger(InductionMachine.class);

    //定义常量
    private static final double[][] T = {{1., 0}, {0.5, -0.2886751346}, {0.5, 0.2886751346}};
    public static final double[][] A_real = {{1, 1, 1}, {1, -0.5, -0.5}, {1, -0.5, -0.5}};
    public static final double[][] A_imag = {{0, 0, 0}, {0, -0.866025, 0.866025}, {0, 0.866025, -0.866025}};
    public static final double[][] A_inv_real = {{1 / 3., 1 / 3., 1 / 3.}, {1 / 3., -0.5 / 3, -0.5 / 3}, {1 / 3., -0.5 / 3, -0.5 / 3}};
    public static final double[][] A_inv_imag = {{0, 0, 0}, {0, 0.866025 / 3, -0.866025 / 3}, {0, -0.866025 / 3, 0.866025 / 3}};
    private static final double[][] T_A_inv_real = new double[3][3];
    private static final double[][] T_A_inv_image = new double[3][3];

    //private static final double[][] W = {{2 / 3, 1 / 3, 0}, {0, 2 / 3, 1 / 3}, {1 / 3, 0, 2 / 3}};
    static {
        for (int i = 0; i < A_real.length; i++) {
            for (int j = 0; j < T.length; j++) {
                T_A_inv_real[i][j] = A_inv_real[i][j] * T[i][0] - A_inv_imag[i][j] * T[i][1];
                T_A_inv_image[i][j] = A_inv_imag[i][j] * T[i][0] + A_inv_real[i][j] * T[i][1];
            }
        }
    }

    //连接方式
    int connType;
    //阻抗值
    private double Rs, Xs, Rr, Xr, Xm;
    //正序和负序电阻
    private double[] RL_12 = new double[2];
    //正序和负序转差率
    private double[] slip_12 = new double[]{0, 2};
    //序导纳，单位矩阵，只存储对角线元素
    private double[][] YM_012 = new double[3][2];
    //序阻抗，单位矩阵，只存储对角线元素
    private double[][] ZM_012 = new double[3][2];
    //相导纳实部，满阵
    private double[][] YM_abc_real = new double[3][3];
    //相导纳虚部，满阵
    private double[][] YM_abc_imag = new double[3][3];
    //线电压
    private double[][] VLL_abc = new double[3][2];
    //序电压
    private double[][] VLN_12 = new double[2][2];
    //临时变量
    private double[][] tmpYm_abc = new double[3][2];
    //输出或吸收的有功功率, 在潮流计算中认为是恒定值
    private double p;
    //正序电流、负序电流
    private AVector state = new AVector(5);
    private AVector z_est = new AVector(5);
    public double[] delta = new double[5];
    public ASparseMatrixLink2D jacStruc = new ASparseMatrixLink2D(5, 5);
    public DoubleMatrix2D jacobian = new MySparseDoubleMatrix2D(5, 5, 16, 0.95, 0.99);
    transient public NewtonSolver solver;

    public void setUp(double RRs, double XXs, double RRr, double XXr, double XXm) {
        this.Rs = RRs;
        this.Rr = RRr;
        this.Xs = XXs;
        this.Xr = XXr;
        this.Xm = XXm;

        for (int i = 0; i < 4; i++)
            jacStruc.setValue(4, i, 1.0);
        jacStruc.setValue(0, 0, 1.0);
        jacStruc.setValue(0, 2, 1.0);
        jacStruc.setValue(0, 4, 1.0);
        jacStruc.setValue(1, 0, 1.0);
        jacStruc.setValue(1, 2, 1.0);
        jacStruc.setValue(1, 4, 1.0);
        jacStruc.setValue(2, 1, 1.0);
        jacStruc.setValue(2, 3, 1.0);
        jacStruc.setValue(2, 4, 1.0);
        jacStruc.setValue(3, 1, 1.0);
        jacStruc.setValue(3, 3, 1.0);
        jacStruc.setValue(3, 4, 1.0);
    }

    /**
     * To calculate the stator current when line-to-neutral voltage rather than line-to-line voltage is given.
     *
     * @param v means the line-to-neutral voltage.
     */
    public void calI(double[][] v, double[][] c) {
        phaseToLine(v);
        if (calSlip()) {
            c[0][0] = A_real[0][1] * state.getValue(0) - A_imag[0][1] * state.getValue(2)
                    + A_real[0][2] * state.getValue(1) - A_imag[0][2] * state.getValue(3);
            c[0][1] = A_real[0][1] * state.getValue(2) + A_imag[0][1] * state.getValue(0)
                    + A_real[0][2] * state.getValue(3) + A_imag[0][2] * state.getValue(1);
            c[1][0] = A_real[1][1] * state.getValue(0) - A_imag[1][1] * state.getValue(2)
                    + A_real[1][2] * state.getValue(1) - A_imag[1][2] * state.getValue(3);
            c[1][1] = A_real[1][1] * state.getValue(2) + A_imag[1][1] * state.getValue(0)
                    + A_real[1][2] * state.getValue(3) + A_imag[1][2] * state.getValue(1);
            c[2][0] = A_real[2][1] * state.getValue(0) - A_imag[2][1] * state.getValue(2)
                    + A_real[2][2] * state.getValue(1) - A_imag[2][2] * state.getValue(3);
            c[2][1] = A_real[2][1] * state.getValue(2) + A_imag[2][1] * state.getValue(0)
                    + A_real[2][2] * state.getValue(3) + A_imag[2][2] * state.getValue(1);
        }
    }

    /**
     * 计算转差率，计算条件是：三相总有功和电压已知
     *
     * @return 是否收敛
     */
    private boolean calSlip() {
        if (solver == null) {
            solver = new NewtonSolver(this);
            solver.setJacStrucReuse(true);
        }
        if (Math.abs(getSlip()) < 1e-6) {
            state.setValue(0, 0.0);
            state.setValue(2, 0.0);
            state.setValue(1, 0.0);
            state.setValue(3, 0.0);
            state.setValue(4, 0.0);
        } else {
            //转差率初值由外部设定
            formPara(getSlip());
            //设置初值
            state.setValue(0, VLN_12[0][0] * YM_012[1][0] - VLN_12[0][1] * YM_012[1][1]);
            state.setValue(2, VLN_12[0][0] * YM_012[1][1] + VLN_12[0][1] * YM_012[1][0]);
            state.setValue(1, VLN_12[1][0] * YM_012[2][0] - VLN_12[1][1] * YM_012[2][1]);
            state.setValue(3, VLN_12[1][0] * YM_012[2][1] + VLN_12[1][1] * YM_012[2][0]);
            state.setValue(4, getSlip());
        }

        //填充Jacobian矩阵中不变的部分
        jacobian.set(4, 0, VLN_12[0][0]);
        jacobian.set(4, 1, VLN_12[1][0]);
        jacobian.set(4, 2, VLN_12[0][1]);
        jacobian.set(4, 3, VLN_12[1][1]);

        if (!solver.solve()) {
            log.warn("计算转差率不收敛!");
            return false;
        } else {
            log.debug("迭代" + solver.getIterNum() + "次后计算转差率收敛,值为 " + getSlip());
            setSlip(state.getValue(4));
            return true;
        }
    }

    /**
     * 该方法不重新计算调差率，认为调差率是固定的
     *
     * @param v LN电压
     * @param c 存储计算结果
     */
    public void calI2(double[][] v, double[][] c) {
        phaseToLine(v);
        calI(c);
    }

    public void calI(double[][] c) {
        for (int i = 0; i < 3; i++) {
            c[i][0] = 0;
            c[i][1] = 0;
            for (int k = 0; k < 3; k++) {
                c[i][0] += YM_abc_real[i][k] * VLL_abc[k][0] - YM_abc_imag[i][k] * VLL_abc[k][1];
                c[i][1] += YM_abc_imag[i][k] * VLL_abc[k][0] + YM_abc_real[i][k] * VLL_abc[k][1];
            }
        }
    }

    public double[] calI(int phase) {
        double[] c = new double[2];
        for (int k = 0; k < 3; k++) {
            c[0] += YM_abc_real[phase][k] * VLL_abc[k][0] - YM_abc_imag[phase][k] * VLL_abc[k][1];
            c[1] += YM_abc_imag[phase][k] * VLL_abc[k][0] + YM_abc_real[phase][k] * VLL_abc[k][1];
        }
        return c;
    }

    private void phaseToLine(double[][] phase) {
        for (int j = 0; j < 3; j++) {
            VLL_abc[j][0] = phase[j][0] - phase[(j + 1) % 3][0];
            VLL_abc[j][1] = phase[j][1] - phase[(j + 1) % 3][1];
        }
        formVLN_12();
    }

    /**
     * 计算正序和负序的阻抗
     *
     * @param s 转差
     */
    public void formPara(double s) {
        setSlip(s);

        for (int i = 0; i < 2; i++)
            RL_12[i] = (1 - slip_12[i]) * Rr / slip_12[i];

        double tmpR, tmpX, sum;
        YM_012[0][0] = 1.0;
        tmpX = Xm + Xr;
        for (int i = 0; i < 2; i++) {
            tmpR = Rr + RL_12[i];
            sum = tmpR * tmpR + tmpX * tmpX;
            ZM_012[i + 1][0] = Rs + (tmpR * tmpX * Xm - Xm * Xr * tmpR) / sum;
            ZM_012[i + 1][1] = Xs + (tmpX * Xm * Xr + tmpR * tmpR * Xm) / sum;

            sum = ZM_012[i + 1][0] * ZM_012[i + 1][0] + ZM_012[i + 1][1] * ZM_012[i + 1][1];
            YM_012[i + 1][0] = ZM_012[i + 1][0] / sum;
            YM_012[i + 1][1] = -ZM_012[i + 1][1] / sum;
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                YM_abc_real[i][j] = A_real[i][j] * YM_012[j][0] - A_imag[i][j] * YM_012[j][1];
                YM_abc_imag[i][j] = A_imag[i][j] * YM_012[j][0] + A_real[i][j] * YM_012[j][1];
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                tmpYm_abc[j][0] = 0.0;
                tmpYm_abc[j][1] = 0.0;
                for (int k = 0; k < 3; k++) {
                    tmpYm_abc[j][0] += YM_abc_real[i][k] * T_A_inv_real[k][j] - YM_abc_imag[i][k] * T_A_inv_image[k][j];
                    tmpYm_abc[j][1] += YM_abc_real[i][k] * T_A_inv_image[k][j] + YM_abc_imag[i][k] * T_A_inv_real[k][j];
                }
            }
            for (int j = 0; j < 3; j++) {
                YM_abc_real[i][j] = tmpYm_abc[j][0];
                YM_abc_imag[i][j] = tmpYm_abc[j][1];
            }
        }
    }

    public void calVLN(double[] v, int phase) {
        v[0] = A_real[phase][1] * VLN_12[0][0] - A_imag[phase][1] * VLN_12[0][1]
                + A_real[phase][2] * VLN_12[1][0] - A_imag[phase][2] * VLN_12[1][1];
        v[1] = A_real[phase][1] * VLN_12[0][1] + A_imag[phase][1] * VLN_12[0][0]
                + A_real[phase][2] * VLN_12[1][1] + A_imag[phase][2] * VLN_12[1][0];
    }

    public void setRs(double rs) {
        Rs = rs;
    }

    public void setRr(double rr) {
        Rr = rr;
    }

    public void setXs(double xs) {
        Xs = xs;
    }

    public void setXr(double xr) {
        Xr = xr;
    }


    public void setXm(double xm) {
        Xm = xm;
    }

    public void setSlip(double slip) {
        slip_12[0] = slip;
        slip_12[1] = 2.0 - slip;
    }

    public void setSlip_12(double[] slip_12) {
        this.slip_12 = slip_12;
    }

    public void setRL_12(double[] RL_12) {
        this.RL_12 = RL_12;
    }

    public void setYM_012(double[][] YM_012) {
        this.YM_012 = YM_012;
    }

    public void setZM_012(double[][] ZM_012) {
        this.ZM_012 = ZM_012;
    }

    public double[] getRL_12() {
        return RL_12;
    }

    public double getRs() {
        return Rs;
    }

    public double getRr() {
        return Rr;
    }

    public double getXs() {
        return Xs;
    }

    public double getXr() {
        return Xr;
    }

    public double getXm() {
        return Xm;
    }

    public double getSlip() {
        return slip_12[0];
    }

    public double[] getSlip_12() {
        return slip_12;
    }

    public double[][] getYM_abc_real() {
        return YM_abc_real;
    }

    public double[][] getYM_abc_imag() {
        return YM_abc_imag;
    }

    public double[][] getYM_012() {
        return YM_012;
    }

    public double[][] getZM_012() {
        return ZM_012;
    }

    public double[][] getVLL_abc() {
        return VLL_abc;
    }

    public int getConnType() {
        return connType;
    }

    public void setConnType(int connType) {
        this.connType = connType;
    }

    public void setVLL_abc(double[][] VLL_abc) {
        this.VLL_abc = VLL_abc;

        formVLN_12();
    }

    private void formVLN_12() {
        double tmp1, tmp2;
        for (int i = 1; i < 3; i++) {
            VLN_12[i - 1][0] = 0.0;
            VLN_12[i - 1][1] = 0.0;
            for (int j = 0; j < 3; j++) {
                tmp1 = A_inv_real[i][j] * VLL_abc[j][0] - A_inv_imag[i][j] * VLL_abc[j][1];
                tmp2 = A_inv_real[i][j] * VLL_abc[j][1] + A_inv_imag[i][j] * VLL_abc[j][0];
                VLN_12[i - 1][0] += tmp1 * T[i][0] - tmp2 * T[i][1];
                VLN_12[i - 1][1] += tmp1 * T[i][1] + tmp2 * T[i][0];
            }
        }
    }

    public double getP() {
        return p;
    }

    public void setP(double p) {
        this.p = p;
    }

    public AVector getState() {
        return state;
    }

    @Override
    public int getMaxIter() {
        return 50;
    }

    @Override
    public double getTolerance() {
        return 1e-1;
    }

    @Override
    public boolean isTolSatisfied(double[] delta) {
        return false;
    }

    @Override
    public AVector getInitial() {
        return state;
    }

    @Override
    public DoubleMatrix2D getJocobian(AVector state) {
        setSlip(state.getValue(4));

        jacobian.set(0, 0, -Rr * Rs + slip_12[0] * (Xs * Xr + Xs * Xm + Xr * Xm));
        jacobian.set(0, 2, (Rr * Xs + Rr * Xm) + slip_12[0] * (Rs * Xr + Rs * Xm));
        jacobian.set(0, 4, (Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(0) +
                (Rs * Xr + Rs * Xm) * state.getValue(2) - (Xr + Xm) * VLN_12[0][1]);
        jacobian.set(1, 0, -(Rr * Xs + Rr * Xm) - slip_12[0] * (Rs * Xr + Rs * Xm));
        jacobian.set(1, 2, Rr * Rs + slip_12[0] * (Xs * Xr + Xs * Xm + Xr * Xm));
        jacobian.set(1, 4, (Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(2) -
                (Rs * Xr + Rs * Xm) * state.getValue(0) + (Xr + Xm) * VLN_12[0][0]);

        jacobian.set(2, 1, -Rr * Rs + slip_12[1] * (Xs * Xr + Xs * Xm + Xr * Xm));
        jacobian.set(2, 3, (Rr * Xs + Rr * Xm) + slip_12[1] * (Rs * Xr + Rs * Xm));
        jacobian.set(2, 4, -(Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(1) -
                (Rs * Xr + Rs * Xm) * state.getValue(3) + (Xr + Xm) * VLN_12[1][1]);
        jacobian.set(3, 1, -(Rr * Xs + Rr * Xm) - slip_12[1] * (Rs * Xr + Rs * Xm));
        jacobian.set(3, 3, Rr * Rs + slip_12[1] * (Xs * Xr + Xs * Xm + Xr * Xm));
        jacobian.set(3, 4, -(Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(3) +
                (Rs * Xr + Rs * Xm) * state.getValue(1) - (Xr + Xm) * VLN_12[1][0]);
        return jacobian;
    }

    @Override
    public ASparseMatrixLink2D getJacobianStruc() {
        return jacStruc;
    }

    @Override
    public AVector getZ() {
        return null;
    }

    @Override
    public double[] getDeltaArray() {
        return delta;
    }

    public AVector calZ(AVector state) {
        return calZ(state, true);
    }

    public AVector calZ(AVector state, boolean isUpdateSlip) {
        //更新转差率
        if (isUpdateSlip)
            setSlip(state.getValue(4));

        z_est.setValue(0, Rr * VLN_12[0][0] - Rr * Rs * state.getValue(0) + (Rr * Xs + Rr * Xm) * state.getValue(2)
                + slip_12[0] * ((Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(0)
                + (Rs * Xr + Rs * Xm) * state.getValue(2) - (Xr + Xm) * VLN_12[0][1]));
        z_est.setValue(1, Rr * VLN_12[0][1] + Rr * Rs * state.getValue(2) - (Rr * Xs + Rr * Xm) * state.getValue(0)
                + slip_12[0] * ((Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(2)
                - (Rs * Xr + Rs * Xm) * state.getValue(0) + (Xr + Xm) * VLN_12[0][0]));
        z_est.setValue(2, Rr * VLN_12[1][0] - Rr * Rs * state.getValue(1) + (Rr * Xs + Rr * Xm) * state.getValue(3)
                + slip_12[1] * ((Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(1)
                + (Rs * Xr + Rs * Xm) * state.getValue(3) - (Xr + Xm) * VLN_12[1][1]));
        z_est.setValue(3, Rr * VLN_12[1][1] + Rr * Rs * state.getValue(3) - (Rr * Xs + Rr * Xm) * state.getValue(1)
                + slip_12[1] * ((Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(3)
                - (Rs * Xr + Rs * Xm) * state.getValue(1) + (Xr + Xm) * VLN_12[1][0]));

        z_est.setValue(4, VLN_12[0][0] * state.getValue(0) + VLN_12[0][1] * state.getValue(2) +
                VLN_12[1][0] * state.getValue(1) + VLN_12[1][1] * state.getValue(3) - p / 3.0);
        return z_est;
    }

    @Override
    public boolean isJacStrucChange() {
        return false;
    }

    public void calZ(boolean isUpdateSlip) {
        calZ(state, isUpdateSlip);
    }

    public AVector getZ_est() {
        return z_est;
    }

    public double[][] getVLN_12() {
        return VLN_12;
    }

    public void fillJacOfI(double[][] values, int row) {
        double v1 = -Rr * Rs + slip_12[0] * (Xs * Xr + Xs * Xm + Xr * Xm);
        double v2 = (Rr * Xs + Rr * Xm) + slip_12[0] * (Rs * Xr + Rs * Xm);
        values[row][0] = v1 * A_inv_real[1][0] + v2 * A_inv_imag[1][0];
        values[row][1] = v1 * A_inv_real[1][1] + v2 * A_inv_imag[1][1];
        values[row][2] = v1 * A_inv_real[1][2] + v2 * A_inv_imag[1][2];
        values[row][3] = -v1 * A_inv_imag[1][0] + v2 * A_inv_real[1][0];
        values[row][4] = -v1 * A_inv_imag[1][1] + v2 * A_inv_real[1][1];
        values[row][5] = -v1 * A_inv_imag[1][2] + v2 * A_inv_real[1][2];

        v1 = -(Rr * Xs + Rr * Xm) - slip_12[0] * (Rs * Xr + Rs * Xm);
        v2 = Rr * Rs + slip_12[0] * (Xs * Xr + Xs * Xm + Xr * Xm);
        values[row + 1][0] = v1 * A_inv_real[1][0] + v2 * A_inv_imag[1][0];
        values[row + 1][1] = v1 * A_inv_real[1][1] + v2 * A_inv_imag[1][1];
        values[row + 1][2] = v1 * A_inv_real[1][2] + v2 * A_inv_imag[1][2];
        values[row + 1][3] = -v1 * A_inv_imag[1][0] + v2 * A_inv_real[1][0];
        values[row + 1][4] = -v1 * A_inv_imag[1][1] + v2 * A_inv_real[1][1];
        values[row + 1][5] = -v1 * A_inv_imag[1][2] + v2 * A_inv_real[1][2];

        v1 = -Rr * Rs + slip_12[1] * (Xs * Xr + Xs * Xm + Xr * Xm);
        v2 = (Rr * Xs + Rr * Xm) + slip_12[1] * (Rs * Xr + Rs * Xm);
        values[row + 2][0] = v1 * A_inv_real[2][0] + v2 * A_inv_imag[2][0];
        values[row + 2][1] = v1 * A_inv_real[2][1] + v2 * A_inv_imag[2][1];
        values[row + 2][2] = v1 * A_inv_real[2][2] + v2 * A_inv_imag[2][2];
        values[row + 2][3] = -v1 * A_inv_imag[2][0] + v2 * A_inv_real[2][0];
        values[row + 2][4] = -v1 * A_inv_imag[2][1] + v2 * A_inv_real[2][1];
        values[row + 2][5] = -v1 * A_inv_imag[2][2] + v2 * A_inv_real[2][2];

        v1 = -(Rr * Xs + Rr * Xm) - slip_12[1] * (Rs * Xr + Rs * Xm);
        v2 = Rr * Rs + slip_12[1] * (Xs * Xr + Xs * Xm + Xr * Xm);
        values[row + 3][0] = v1 * A_inv_real[2][0] + v2 * A_inv_imag[2][0];
        values[row + 3][1] = v1 * A_inv_real[2][1] + v2 * A_inv_imag[2][1];
        values[row + 3][2] = v1 * A_inv_real[2][2] + v2 * A_inv_imag[2][2];
        values[row + 3][3] = -v1 * A_inv_imag[2][0] + v2 * A_inv_real[2][0];
        values[row + 3][4] = -v1 * A_inv_imag[2][1] + v2 * A_inv_real[2][1];
        values[row + 3][5] = -v1 * A_inv_imag[2][2] + v2 * A_inv_real[2][2];

        values[row + 4][0] = VLN_12[0][0] * A_inv_real[1][0] + VLN_12[0][1] * A_inv_imag[1][0]
                + VLN_12[1][0] * A_inv_real[2][0] + VLN_12[1][1] * A_inv_imag[2][0];
        values[row + 4][1] = VLN_12[0][0] * A_inv_real[1][1] + VLN_12[0][1] * A_inv_imag[1][1]
                + VLN_12[1][0] * A_inv_real[2][1] + VLN_12[1][1] * A_inv_imag[2][1];
        values[row + 4][2] = VLN_12[0][0] * A_inv_real[1][2] + VLN_12[0][1] * A_inv_imag[1][2]
                + VLN_12[1][0] * A_inv_real[2][2] + VLN_12[1][1] * A_inv_imag[2][2];
        values[row + 4][3] = -VLN_12[0][0] * A_inv_imag[1][0] + VLN_12[0][1] * A_inv_real[1][0]
                - VLN_12[1][0] * A_inv_imag[2][0] + VLN_12[1][1] * A_inv_real[2][0];
        values[row + 4][4] = -VLN_12[0][0] * A_inv_imag[1][1] + VLN_12[0][1] * A_inv_real[1][1]
                - VLN_12[1][0] * A_inv_imag[2][1] + VLN_12[1][1] * A_inv_real[2][1];
        values[row + 4][5] = -VLN_12[0][0] * A_inv_imag[1][2] + VLN_12[0][1] * A_inv_real[1][2]
                - VLN_12[1][0] * A_inv_imag[2][2] + VLN_12[1][1] * A_inv_real[2][2];
    }

    public void fillJacOfU12(DoubleMatrix2D jac, int pos, int dim, int row, int sPos) {
        jac.set(row, pos + dim, -slip_12[0] * (Xr + Xm));
        jac.set(row, sPos, (Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(0) +
                (Rs * Xr + Rs * Xm) * state.getValue(2) - (Xr + Xm) * VLN_12[0][1]);
        jac.set(row + 1, pos, slip_12[0] * (Xr + Xm));
        jac.set(row + 1, sPos, (Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(2) -
                (Rs * Xr + Rs * Xm) * state.getValue(0) + (Xr + Xm) * VLN_12[0][0]);

        jac.set(row + 2, pos + dim + 1, -slip_12[1] * (Xr + Xm));
        jac.set(row + 2, sPos, -(Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(1) -
                (Rs * Xr + Rs * Xm) * state.getValue(3) + (Xr + Xm) * VLN_12[1][1]);
        jac.set(row + 3, pos + 1, slip_12[1] * (Xr + Xm));
        jac.set(row + 3, sPos, -(Xs * Xr + Xs * Xm + Xr * Xm) * state.getValue(3) +
                (Rs * Xr + Rs * Xm) * state.getValue(1) - (Xr + Xm) * VLN_12[1][0]);

        jac.set(row + 4, pos, state.getValue(0));
        jac.set(row + 4, pos + dim, state.getValue(2));
        jac.set(row + 4, pos + 1, state.getValue(1));
        jac.set(row + 4, pos + dim + 1, state.getValue(3));
    }

    public void fillJacStrucOfU12(ASparseMatrixLink2D jac, int pos, int dim, int row, int sPos) {
        jac.setValue(row, pos, Rr);
        jac.setValue(row, pos + dim, 1.0);
        jac.setValue(row, sPos, 1.0);

        jac.setValue(row + 1, pos, 1.0);
        jac.setValue(row + 1, pos + dim, Rr);
        jac.setValue(row + 1, sPos, 1.0);

        jac.setValue(row + 2, pos + 1, Rr);
        jac.setValue(row + 2, pos + dim + 1, 1.0);
        jac.setValue(row + 2, sPos, 1.0);

        jac.setValue(row + 3, pos + 1, 1.0);
        jac.setValue(row + 3, pos + dim + 1, Rr);
        jac.setValue(row + 3, sPos, 1.0);

        jac.setValue(row + 4, pos, 1.0);
        jac.setValue(row + 4, pos + dim, 1.0);
        jac.setValue(row + 4, pos + 1, 1.0);
        jac.setValue(row + 4, pos + dim + 1, 1.0);
    }

    //给出状态变量的个数，应根据具体情况分别对待
    //todo:这里这是针对最常见的潮流计算模型给出
    public int getStateSize() {
        return 5;
    }
}
