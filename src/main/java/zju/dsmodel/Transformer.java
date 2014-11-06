package zju.dsmodel;

import cern.colt.matrix.DoubleMatrix2D;
import zju.util.DoubleMatrixToolkit;

/**
 * This module of transformer is base on the analysis of
 * "Distribution System Modeling and Analysis, Second Edition"
 * on page 243 to 250
 * class "TF_D_GrY" means step-down transformer with Delta-Grounded Wye connection
 * In the three methods for iteration: calHeadV, calHeadI and calTailV, voltage's unit is V,
 * while current is in A. The unit of resistance is in Ohm.
 */
public class Transformer implements GeneralBranch {

    public static final int CONN_TYPE_D_GrY = 1;
    public static final int CONN_TYPE_Y_D = 2;
    public static final int CONN_TYPE_GrY_GrY = 3;
    public static final int CONN_TYPE_D_D = 4;

    //接线方式
    private int connType;

    //declare the variables
    private double[] r_pu = new double[3];
    private double[] x_pu = new double[3];
    private double r[] = new double[3];
    private double x[] = new double[3];
    //单位是千伏kV
    private double VLL_high_rated;
    private double VLL_low_rated;
    //单位是kVA
    private double Sn[] = new double[3];
    private double nt;
    private double[][] at_D_D = null;
    private double[][] At_D_D = null;
    private double[][] Bt_real_D_D = null;
    private double[][] Bt_imag_D_D = null;

    public Transformer() {
    }

    public Transformer(double VLL_high_rated, double VLL_low_rated, double totalSn) {
        this.VLL_high_rated = VLL_high_rated;
        this.VLL_low_rated = VLL_low_rated;
        for (int i = 0; i < this.Sn.length; i++)
            this.Sn[i] = totalSn / 3.0;
    }

    public Transformer(double VLL_high_rated, double VLL_low_rated, double totalSn, double r_pu, double x_pu) {
        this.VLL_high_rated = VLL_high_rated;
        this.VLL_low_rated = VLL_low_rated;
        for (int i = 0; i < this.Sn.length; i++)
            Sn[i] = totalSn / 3.0;
        for (int i = 0; i < this.r_pu.length; i++) {
            this.r_pu[i] = r_pu;
            this.x_pu[i] = x_pu;
        }
    }

    public void formPara() {
        double baseZ;
        double tempV = 0.0;
        switch (connType) {
            case CONN_TYPE_D_GrY:
                nt = DsModelCons.sqrt3 * VLL_high_rated / VLL_low_rated;
                tempV = VLL_low_rated / DsModelCons.sqrt3;
                break;
            case CONN_TYPE_GrY_GrY:
                nt = VLL_high_rated / VLL_low_rated;
                tempV = VLL_low_rated / DsModelCons.sqrt3;
                break;
            case CONN_TYPE_Y_D:
                nt = VLL_high_rated / VLL_low_rated / DsModelCons.sqrt3;
                tempV = VLL_low_rated;
                break;
            case CONN_TYPE_D_D:
                nt = VLL_high_rated / VLL_low_rated;
                tempV = VLL_low_rated;
                break;
            default:
                break;
        }
        for (int i = 0; i < r.length; i++) {
            baseZ = tempV * tempV * 1000.0 / Sn[i];
            r[i] = r_pu[i] * baseZ;
            x[i] = x_pu[i] * baseZ;
        }

        if (connType == CONN_TYPE_D_D) {
            double[][] W = {{2.0 / 3.0, 1.0 / 3.0, 0}, {0, 2.0 / 3.0, 1.0 / 3.0}, {1.0 / 3.0, 0, 2.0 / 3.0}};
            At_D_D = new double[][]{
                    {2.0 / (3.0 * nt), -1.0 / (3.0 * nt), -1.0 / (3.0 * nt)},
                    {-1.0 / (3.0 * nt), 2.0 / (3.0 * nt), -1.0 / (3.0 * nt)},
                    {-1.0 / (3.0 * nt), -1.0 / (3.0 * nt), 2.0 / (3.0 * nt)}};
            at_D_D = new double[][]{
                    {2.0 * nt / 3.0, -nt / 3.0, -nt / 3.0},
                    {-nt / 3.0, 2.0 * nt / 3.0, -nt / 3.0},
                    {-nt / 3.0, -nt / 3.0, 2.0 * nt / 3.0}};
            double zt_r = r[0] + r[1] + r[2];
            double zt_i = x[0] + x[1] + x[2];
            Bt_real_D_D = new double[][]{
                    {r[2], -r[1]},
                    {r[2], r[0] + r[2]},
                    {-r[0] - r[1], -r[1]}
            };
            Bt_imag_D_D = new double[][]{
                    {x[2], -x[1]},
                    {x[2], x[0] + x[2]},
                    {-x[0] - x[1], -x[1]}
            };
            DoubleMatrixToolkit.selfMul(Bt_real_D_D, zt_r);
            DoubleMatrixToolkit.selfMul(Bt_imag_D_D, zt_i);
            DoubleMatrixToolkit.selfAdd(Bt_real_D_D, Bt_imag_D_D);
            DoubleMatrixToolkit.selfMul(Bt_real_D_D, 1.0 / (zt_r * zt_r + zt_i * zt_i));

            double[][] tmp1 = {
                    {r[2], -r[1]},
                    {r[2], r[0] + r[2]},
                    {-r[0] - r[1], -r[1]}
            };
            DoubleMatrixToolkit.selfMul(Bt_imag_D_D, zt_r / zt_i);
            DoubleMatrixToolkit.selfMul(tmp1, zt_i);
            DoubleMatrixToolkit.selfSub(Bt_imag_D_D, tmp1);
            DoubleMatrixToolkit.selfMul(Bt_imag_D_D, 1.0 / (zt_r * zt_r + zt_i * zt_i));

            for (int i = 0; i < Bt_real_D_D.length; i++) {
                for (int j = 0; j < Bt_real_D_D[0].length; j++) {
                    double bt1 = Bt_real_D_D[i][j];
                    double bt2 = Bt_imag_D_D[i][j];
                    Bt_real_D_D[i][j] = bt1 * r[i] - bt2 * x[i];
                    Bt_imag_D_D[i][j] = bt1 * x[i] + bt2 * r[i];
                }
            }
            tmp1 = DoubleMatrixToolkit.mul(W, Bt_real_D_D);
            for (int i = 0; i < Bt_real_D_D.length; i++)
                System.arraycopy(tmp1[i], 0, Bt_real_D_D[i], 0, Bt_real_D_D[i].length);
            tmp1 = DoubleMatrixToolkit.mul(W, Bt_imag_D_D);
            for (int i = 0; i < Bt_imag_D_D.length; i++)
                System.arraycopy(tmp1[i], 0, Bt_imag_D_D[i], 0, Bt_imag_D_D[i].length);
        }
    }


    @Override
    public void calHeadV(double[][] tailV, double[][] tailI, double[][] headV) {
        switch (connType) {
            case CONN_TYPE_D_GrY:
                calHeadV_D_GrY(tailV, tailI, headV);
                break;
            case CONN_TYPE_Y_D:
                calHeadV_Y_D(tailV, tailI, headV);
                break;
            case CONN_TYPE_GrY_GrY:
                calHeadV_GrY_GrY(tailV, tailI, headV);
                break;
            case CONN_TYPE_D_D:
                calHeadV_D_D(tailV, tailI, headV);
                break;
            default:
                break;
        }
    }

    @Override
    public void calHeadI(double[][] tailV, double[][] tailI, double[][] headI) {
        switch (connType) {
            case CONN_TYPE_D_GrY:
                calHeadI_D_GrY(tailI, headI);
                break;
            case CONN_TYPE_Y_D:
                calHeadI_Y_D(tailI, headI);
                break;
            case CONN_TYPE_GrY_GrY:
            case CONN_TYPE_D_D:
                calHeadI_YY_DD(tailI, headI);
                break;
            default:
                break;
        }
    }

    /**
     * 填充约束 VLNABC = a * LNabc + b * Iabc 的jacobian
     *
     * @param jac jacobian矩阵
     * @param pos <br>pos[0]：tailV[0][0]在状态向量中的位置</br>
     *            <br>pos[1]：tailV[0][1]在状态向量中的位置</br>
     *            <br>pos[2]：tailI[0][0]在状态向量中的位置</br>
     *            <br>pos[3]：tailI[0][1]在状态向量中的位置</br>
     *            <br>pos[4]：headV[0][0]在状态向量中的位置</br>
     *            <br>pos[5]：headV[0][1]在状态向量中的位置</br>
     * @param row 起始行，最多六个方程
     */
    public void fillJacOfHeadV(DoubleMatrix2D jac, int[] pos, int row) {
        switch (connType) {
            case CONN_TYPE_D_GrY:
                //double tmp1 = 2. * (r[1] * tailI[1][0] - x[1] * tailI[1][1]) + (r[2] * tailI[2][0] - x[2] * tailI[2][1]);
                //headV[0][0] = -((2. * tailV[1][0] + tailV[2][0]) + tmp1) * nt / 3.;
                jac.setQuick(row, pos[0] + 1, -2.0 * nt / 3.0);
                jac.setQuick(row, pos[0] + 2, -nt / 3.0);
                jac.setQuick(row, pos[2] + 1, -2.0 * r[1] * nt / 3.0);
                jac.setQuick(row, pos[3] + 1, 2.0 * x[1] * nt / 3.0);
                jac.setQuick(row, pos[2] + 2, -r[2] * nt / 3.0);
                jac.setQuick(row, pos[3] + 2, x[2] * nt / 3.0);
                jac.setQuick(row, pos[4], -1.0);

                //tmp1 = 2. * (r[1] * tailI[1][1] + x[1] * tailI[1][0]) + (r[2] * tailI[2][1] + x[2] * tailI[2][0]);
                //headV[0][1] = -((2. * tailV[1][1] + tailV[2][1]) + tmp1) * nt / 3.;
                jac.setQuick(row + 1, pos[1] + 1, -2.0 * nt / 3.0);
                jac.setQuick(row + 1, pos[1] + 2, -nt / 3.0);
                jac.setQuick(row + 1, pos[3] + 1, -2.0 * r[1] * nt / 3.0);
                jac.setQuick(row + 1, pos[2] + 1, -2.0 * x[1] * nt / 3.0);
                jac.setQuick(row + 1, pos[3] + 2, -r[2] * nt / 3.0);
                jac.setQuick(row + 1, pos[2] + 2, -x[2] * nt / 3.0);
                jac.setQuick(row + 1, pos[5], -1.0);

                //tmp1 = 2. * (r[2] * tailI[2][0] - x[2] * tailI[2][1]) + (r[0] * tailI[0][0] - x[0] * tailI[0][1]);
                //headV[1][0] = -((2. * tailV[2][0] + tailV[0][0]) + tmp1) * nt / 3.;
                jac.setQuick(row + 2, pos[0] + 2, -2.0 * nt / 3.0);
                jac.setQuick(row + 2, pos[0], -nt / 3.0);
                jac.setQuick(row + 2, pos[2] + 2, -2.0 * r[2] * nt / 3.0);
                jac.setQuick(row + 2, pos[3] + 2, 2.0 * x[2] * nt / 3.0);
                jac.setQuick(row + 2, pos[2], -r[0] * nt / 3.0);
                jac.setQuick(row + 2, pos[3], x[0] * nt / 3.0);
                jac.setQuick(row + 2, pos[4] + 1, -1.0);

                //tmp1 = 2. * (r[2] * tailI[2][1] + x[2] * tailI[2][0]) + (r[0] * tailI[0][1] + x[0] * tailI[0][0]);
                //headV[1][1] = -((2. * tailV[2][1] + tailV[0][1]) + tmp1) * nt / 3.;
                jac.setQuick(row + 3, pos[1] + 2, -2.0 * nt / 3.0);
                jac.setQuick(row + 3, pos[1], -nt / 3.0);
                jac.setQuick(row + 3, pos[3] + 2, -2.0 * r[2] * nt / 3.0);
                jac.setQuick(row + 3, pos[2] + 2, -2.0 * x[2] * nt / 3.0);
                jac.setQuick(row + 3, pos[3], -r[0] * nt / 3.0);
                jac.setQuick(row + 3, pos[2], -x[0] * nt / 3.0);
                jac.setQuick(row + 3, pos[5] + 1, -1.0);

                //tmp1 = 2. * (r[0] * tailI[0][0] - x[0] * tailI[0][1]) + (r[1] * tailI[1][0] - x[1] * tailI[1][1]);
                //headV[2][0] = -((2. * tailV[0][0] + tailV[1][0]) + tmp1) * nt / 3.;
                jac.setQuick(row + 4, pos[0], -2.0 * nt / 3.0);
                jac.setQuick(row + 4, pos[0] + 1, -nt / 3.0);
                jac.setQuick(row + 4, pos[2], -2.0 * r[0] * nt / 3.0);
                jac.setQuick(row + 4, pos[3], 2.0 * x[0] * nt / 3.0);
                jac.setQuick(row + 4, pos[2] + 1, -r[1] * nt / 3.0);
                jac.setQuick(row + 4, pos[3] + 1, x[1] * nt / 3.0);
                jac.setQuick(row + 4, pos[4] + 2, -1.0);

                //tmp1 = 2. * (r[0] * tailI[0][1] + x[0] * tailI[0][0]) + (r[1] * tailI[1][1] + x[1] * tailI[1][0]);
                //headV[2][1] = -((2. * tailV[0][1] + tailV[1][1]) + tmp1) * nt / 3.;
                jac.setQuick(row + 5, pos[1], -2.0 * nt / 3.0);
                jac.setQuick(row + 5, pos[1] + 1, -nt / 3.0);
                jac.setQuick(row + 5, pos[3], -2.0 * r[0] * nt / 3.0);
                jac.setQuick(row + 5, pos[2], -2.0 * x[0] * nt / 3.0);
                jac.setQuick(row + 5, pos[3] + 1, -r[1] * nt / 3.0);
                jac.setQuick(row + 5, pos[2] + 1, -x[1] * nt / 3.0);
                jac.setQuick(row + 5, pos[5] + 2, -1.0);
                break;
            case CONN_TYPE_Y_D:
                //todo:
                break;
            case CONN_TYPE_GrY_GrY:
                //headV[0][0] = (tailV[0][0] + r[0] * tailI[0][0] - x[0] * tailI[0][1]) * nt;
                jac.setQuick(row, pos[0], nt);
                jac.setQuick(row, pos[2], r[0] * nt);
                jac.setQuick(row, pos[3], -x[0] * nt);
                jac.setQuick(row, pos[4], -1.0);
                //headV[0][1] = (tailV[0][1] + r[0] * tailI[0][1] + x[0] * tailI[0][0]) * nt;
                jac.setQuick(row + 1, pos[1], nt);
                jac.setQuick(row + 1, pos[3], r[0] * nt);
                jac.setQuick(row + 1, pos[2], x[0] * nt);
                jac.setQuick(row + 1, pos[5], -1.0);
                //headV[1][0] = (tailV[1][0] + r[1] * tailI[1][0] - x[1] * tailI[1][1]) * nt;
                jac.setQuick(row + 2, pos[0] + 1, nt);
                jac.setQuick(row + 2, pos[2] + 1, r[1] * nt);
                jac.setQuick(row + 2, pos[3] + 1, -x[1] * nt);
                jac.setQuick(row + 2, pos[4] + 1, -1.0);
                //headV[1][1] = (tailV[1][1] + r[1] * tailI[1][1] + x[1] * tailI[1][0]) * nt;
                jac.setQuick(row + 3, pos[1] + 1, nt);
                jac.setQuick(row + 3, pos[3] + 1, r[1] * nt);
                jac.setQuick(row + 3, pos[2] + 1, x[1] * nt);
                jac.setQuick(row + 3, pos[5] + 1, -1.0);
                //headV[2][0] = (tailV[2][0] + r[2] * tailI[2][0] - x[2] * tailI[2][1]) * nt;
                jac.setQuick(row + 4, pos[0] + 2, nt);
                jac.setQuick(row + 4, pos[2] + 2, r[2] * nt);
                jac.setQuick(row + 4, pos[3] + 2, -x[2] * nt);
                jac.setQuick(row + 4, pos[4] + 2, -1.0);
                //headV[2][1] = (tailV[2][1] + r[2] * tailI[2][1] + x[2] * tailI[2][0]) * nt;
                jac.setQuick(row + 5, pos[1] + 2, nt);
                jac.setQuick(row + 5, pos[3] + 2, r[2] * nt);
                jac.setQuick(row + 5, pos[2] + 2, x[2] * nt);
                jac.setQuick(row + 5, pos[5] + 2, -1.0);
                break;
            case CONN_TYPE_D_D:
                for (int i = 0; i < at_D_D.length; i++, row += 2) {
                    for (int k = 0; k < at_D_D[i].length; k++) {
                        jac.setQuick(row, pos[0] + k, at_D_D[i][k]);
                        jac.setQuick(row + 1, pos[1] + k, at_D_D[i][k]);
                    }
                    for (int k = 0; k < Bt_real_D_D[i].length; k++) {
                        jac.setQuick(row, pos[2] + k, nt * Bt_real_D_D[i][k]);
                        jac.setQuick(row, pos[3] + k, -nt * Bt_imag_D_D[i][k]);
                        jac.setQuick(row + 1, pos[2] + k, nt * Bt_imag_D_D[i][k]);
                        jac.setQuick(row + 1, pos[3] + k, nt * Bt_real_D_D[i][k]);
                    }
                    jac.setQuick(row, pos[4] + i, -1.0);
                    jac.setQuick(row + 1, pos[5] + i, -1.0);
                }
                break;
            default:
                break;
        }
    }

    /**
     * 填充约束 IABC = c * LNabc + d * Iabc 的jacobian
     *
     * @param jac jacobian矩阵
     * @param pos <br>pos[0]：tailV[0][0]在状态向量中的位置</br>
     *             <br>pos[1]：tailV[0][1]在状态向量中的位置</br>
     *             <br>pos[2]：tailI[0][0]在状态向量中的位置</br>
     *             <br>pos[3]：tailI[0][1]在状态向量中的位置</br>
     *             <br>pos[4]：headI[0][0]在状态向量中的位置</br>
     *             <br>pos[5]：headI[0][1]在状态向量中的位置</br>
     * @param row 起始行，最多六个方程
     */
    public void fillJacOfHeadI(DoubleMatrix2D jac, int[] pos, int row) {
        switch (connType) {
            case CONN_TYPE_D_GrY:
                //headI[0][0] = (tailI[0][0] - tailI[1][0]) / nt;
                jac.setQuick(row, pos[2], 1.0 / nt);
                jac.setQuick(row, pos[2] + 1, -1.0 / nt);
                jac.setQuick(row, pos[4], -1.0);
                //headI[0][1] = (tailI[0][1] - tailI[1][1]) / nt;
                jac.setQuick(row + 1, pos[3], 1.0 / nt);
                jac.setQuick(row + 1, pos[3] + 1, -1.0 / nt);
                jac.setQuick(row + 1, pos[5], -1.0);
                //headI[1][0] = (tailI[1][0] - tailI[2][0]) / nt;
                jac.setQuick(row + 2, pos[2] + 1, 1.0 / nt);
                jac.setQuick(row + 2, pos[2] + 2, -1.0 / nt);
                jac.setQuick(row + 2, pos[4] + 1, -1.0);
                //headI[1][1] = (tailI[1][1] - tailI[2][1]) / nt;
                jac.setQuick(row + 3, pos[3] + 1, 1.0 / nt);
                jac.setQuick(row + 3, pos[3] + 2, -1.0 / nt);
                jac.setQuick(row + 3, pos[5] + 1, -1.0);
                //headI[2][0] = (tailI[2][0] - tailI[0][0]) / nt;
                jac.setQuick(row + 4, pos[2] + 2, 1.0 / nt);
                jac.setQuick(row + 4, pos[2], -1.0 / nt);
                jac.setQuick(row + 4, pos[4] + 2, -1.0);
                //headI[2][1] = (tailI[2][1] - tailI[0][1]) / nt;
                jac.setQuick(row + 5, pos[3] + 2, 1.0 / nt);
                jac.setQuick(row + 5, pos[3], -1.0 / nt);
                jac.setQuick(row + 5, pos[5] + 2, -1.0);
                break;
            case CONN_TYPE_Y_D:
                //todo:
                break;
            case CONN_TYPE_GrY_GrY:
            case CONN_TYPE_D_D:
                //headI[0][0] = tailI[0][0] / nt;
                jac.setQuick(row, pos[2], 1.0 / nt);
                jac.setQuick(row, pos[4], -1.0);
                //headI[0][1] = tailI[0][1] / nt;
                jac.setQuick(row + 1, pos[3], 1.0 / nt);
                jac.setQuick(row + 1, pos[5], -1.0);
                //headI[1][0] = tailI[1][0] / nt;
                jac.setQuick(row + 2, pos[2] + 1, 1.0 / nt);
                jac.setQuick(row + 2, pos[4] + 1, -1.0);
                //headI[1][1] = tailI[1][1] / nt;
                jac.setQuick(row + 3, pos[3] + 1, 1.0 / nt);
                jac.setQuick(row + 3, pos[5] + 1, -1.0);
                //headI[2][0] = tailI[2][0] / nt;
                jac.setQuick(row + 4, pos[2] + 2, 1.0 / nt);
                jac.setQuick(row + 4, pos[4] + 2, -1.0);
                //headI[2][1] = tailI[2][1] / nt;
                jac.setQuick(row + 5, pos[3] + 2, 1.0 / nt);
                jac.setQuick(row + 5, pos[5] + 2, -1.0);
                break;
            default:
                break;
        }
    }

    @Override
    public int getNonZeroNumOfJac() {
        switch (connType) {
            case CONN_TYPE_D_GrY:
                return 60;
            case CONN_TYPE_Y_D:
                //todo:
                break;
            case CONN_TYPE_GrY_GrY:
                return 36;
            case CONN_TYPE_D_D:
                return 60;
            default:
                break;
        }
        return 0;
    }

    @Override
    public boolean containsPhase(int phase) {
        return true;
    }

    @Override
    public void calTailV(double[][] headV, double[][] tailI, double[][] tailV) {
        switch (connType) {
            case CONN_TYPE_D_GrY:
                calTailV_D_GrY(headV, tailI, tailV);
                break;
            case CONN_TYPE_Y_D:
                calTailV_Y_D(headV, tailI, tailV);
                break;
            case CONN_TYPE_GrY_GrY:
                calTailV_GrY_GrY(headV, tailI, tailV);
                break;
            case CONN_TYPE_D_D:
                calTailV_D_D(headV, tailI, tailV);
                break;
            default:
                break;
        }
    }

    /**
     * calculate the final iterative form for the transformer;
     * VLNABC = at * VLGabc + bt * Iabc;
     * at = -nt / 3 [0  2   1; 1   0   2; 2   1   0]
     * bt =  at * diag(Ztabc, Ztabc, Ztabc)
     */
    public void calHeadV_D_GrY(double[][] tailV, double[][] tailI, double[][] headV) {
        double tmp1 = 2. * (r[1] * tailI[1][0] - x[1] * tailI[1][1]) + (r[2] * tailI[2][0] - x[2] * tailI[2][1]);
        headV[0][0] = -((2. * tailV[1][0] + tailV[2][0]) + tmp1) * nt / 3.;
        tmp1 = 2. * (r[1] * tailI[1][1] + x[1] * tailI[1][0]) + (r[2] * tailI[2][1] + x[2] * tailI[2][0]);
        headV[0][1] = -((2. * tailV[1][1] + tailV[2][1]) + tmp1) * nt / 3.;

        tmp1 = 2. * (r[2] * tailI[2][0] - x[2] * tailI[2][1]) + (r[0] * tailI[0][0] - x[0] * tailI[0][1]);
        headV[1][0] = -((2. * tailV[2][0] + tailV[0][0]) + tmp1) * nt / 3.;
        tmp1 = 2. * (r[2] * tailI[2][1] + x[2] * tailI[2][0]) + (r[0] * tailI[0][1] + x[0] * tailI[0][0]);
        headV[1][1] = -((2. * tailV[2][1] + tailV[0][1]) + tmp1) * nt / 3.;

        tmp1 = 2. * (r[0] * tailI[0][0] - x[0] * tailI[0][1]) + (r[1] * tailI[1][0] - x[1] * tailI[1][1]);
        headV[2][0] = -((2. * tailV[0][0] + tailV[1][0]) + tmp1) * nt / 3.;
        tmp1 = 2. * (r[0] * tailI[0][1] + x[0] * tailI[0][0]) + (r[1] * tailI[1][1] + x[1] * tailI[1][0]);
        headV[2][1] = -((2. * tailV[0][1] + tailV[1][1]) + tmp1) * nt / 3.;
    }

    /**
     * IABC = dt * Iabc
     * dt = 1 / nt [1  -1  0; 0   1   -1; -1  0   1]
     */
    public void calHeadI_D_GrY(double[][] tailI, double[][] headI) {
        headI[0][0] = (tailI[0][0] - tailI[1][0]) / nt;
        headI[0][1] = (tailI[0][1] - tailI[1][1]) / nt;
        headI[1][0] = (tailI[1][0] - tailI[2][0]) / nt;
        headI[1][1] = (tailI[1][1] - tailI[2][1]) / nt;
        headI[2][0] = (tailI[2][0] - tailI[0][0]) / nt;
        headI[2][1] = (tailI[2][1] - tailI[0][1]) / nt;
    }

    /**
     * VLNabc = At * VLNABC - Bt * Iabc
     * At = 1 / nt [1   0   -1; -1  1   0; 0   -1  1]
     * Bt =  diag(Ztabc, Ztabc, Ztabc)
     */
    public void calTailV_D_GrY(double[][] headV, double[][] tailI, double[][] tailV) {
        tailV[0][0] = (headV[0][0] - headV[2][0]) / nt - (r[0] * tailI[0][0] - x[0] * tailI[0][1]);
        tailV[0][1] = (headV[0][1] - headV[2][1]) / nt - (r[0] * tailI[0][1] + x[0] * tailI[0][0]);
        tailV[1][0] = (headV[1][0] - headV[0][0]) / nt - (r[0] * tailI[1][0] - x[0] * tailI[1][1]);
        tailV[1][1] = (headV[1][1] - headV[0][1]) / nt - (r[0] * tailI[1][1] + x[0] * tailI[1][0]);
        tailV[2][0] = (headV[2][0] - headV[1][0]) / nt - (r[0] * tailI[2][0] - x[0] * tailI[2][1]);
        tailV[2][1] = (headV[2][1] - headV[1][1]) / nt - (r[0] * tailI[2][1] + x[0] * tailI[2][0]);
    }

    /**
     * calculate the final iterative form for the transformer;
     * VLNABC = at * VLGabc + bt * Iabc;
     * at = nt * [1  -1   0; 0   1   -1; -1   0   1]
     * bt =  nt / 3 * [Ztab     -Ztab   0; Ztbc     2*Ztbc   0; -2*Ztca   -Zca    0]
     */
    public void calHeadV_Y_D(double[][] tailV, double[][] tailI, double[][] headV) {
        double tmp1 = (r[0] * tailI[0][0] - x[0] * tailI[0][1]) - (r[0] * tailI[1][0] - x[0] * tailI[1][1]);
        headV[0][0] = (tailV[0][0] - tailV[1][0] + tmp1 / 3.0) * nt;
        tmp1 = (r[0] * tailI[0][1] + x[0] * tailI[0][0]) - (r[0] * tailI[1][1] + x[0] * tailI[1][0]);
        headV[0][1] = (tailV[0][1] - tailV[1][1] + tmp1 / 3.0) * nt;

        tmp1 = (r[1] * tailI[0][0] - x[1] * tailI[0][1]) + 2.0 * (r[1] * tailI[1][0] - x[1] * tailI[1][1]);
        headV[1][0] = (tailV[1][0] - tailV[2][0] + tmp1 / 3.0) * nt;
        tmp1 = (r[1] * tailI[0][1] + x[1] * tailI[0][0]) + 2.0 * (r[1] * tailI[1][1] + x[1] * tailI[1][0]);
        headV[1][1] = (tailV[1][1] - tailV[2][1] + tmp1 / 3.0) * nt;

        tmp1 = -2.0 * (r[2] * tailI[0][0] - x[2] * tailI[0][1]) - (r[2] * tailI[1][0] - x[2] * tailI[1][1]);
        headV[2][0] = (tailV[2][0] - tailV[0][0] + tmp1 / 3.0) * nt;
        tmp1 = -2.0 * (r[2] * tailI[0][1] + x[2] * tailI[0][0]) - (r[2] * tailI[1][1] + x[2] * tailI[1][0]);
        headV[2][1] = (tailV[2][1] - tailV[0][1] + tmp1 / 3.0) * nt;
    }

    /**
     * IABC = dt * Iabc
     * dt =  [1  -1  0; 1   2   0; -2  -1   0] / (3 * nt)
     */
    public void calHeadI_Y_D(double[][] tailI, double[][] headI) {
        headI[0][0] = (tailI[0][0] - tailI[1][0]) / (3.0 * nt);
        headI[0][1] = (tailI[0][1] - tailI[1][1]) / (3.0 * nt);
        headI[1][0] = (tailI[0][0] + 2 * tailI[1][0]) / (3.0 * nt);
        headI[1][1] = (tailI[0][1] + 2 * tailI[1][1]) / (3.0 * nt);
        headI[2][0] = (-2.0 * tailI[0][0] - tailI[1][0]) / (3.0 * nt);
        headI[2][1] = (-2.0 * tailI[0][1] - tailI[1][1]) / (3.0 * nt);
    }

    /**
     * VLNabc = At * VLNABC - Bt * Iabc
     * At =  [2   1   0; 0  2   1; 1   0  2] / （3 * nt）
     * Bt =  [2Ztab+Ztbc  2Ztbc-2Ztab   0; 2Ztbc-2Ztca  4Ztbc-Ztca   0; Ztab-4Ztca  -Ztab-2Ztca   0] / 9
     */
    public void calTailV_Y_D(double[][] headV, double[][] tailI, double[][] tailV) {
        double tmp1 = 2.0 * (r[0] * tailI[0][0] - x[0] * tailI[0][1]) - 2.0 * (r[0] * tailI[1][0] - x[0] * tailI[1][1])
                + (r[1] * tailI[0][0] - x[1] * tailI[0][1]) + 2.0 * (r[1] * tailI[1][0] - x[1] * tailI[1][1]);
        tailV[0][0] = (2.0 * headV[0][0] + headV[1][0]) / (3.0 * nt) - tmp1 / 9.0;
        tmp1 = 2.0 * (r[0] * tailI[0][1] + x[0] * tailI[0][0]) - 2.0 * (r[0] * tailI[1][1] + x[0] * tailI[1][0])
                + (r[1] * tailI[0][1] + x[1] * tailI[0][0]) + 2.0 * (r[1] * tailI[1][1] + x[1] * tailI[1][0]);
        tailV[0][1] = (2.0 * headV[0][1] + headV[1][1]) / (3.0 * nt) - tmp1 / 9.0;

        tmp1 = 2.0 * (r[1] * tailI[0][0] - x[1] * tailI[0][1]) + 4.0 * (r[1] * tailI[1][0] - x[1] * tailI[1][1])
                - 2.0 * (r[2] * tailI[0][0] - x[2] * tailI[0][1]) - (r[2] * tailI[1][0] - x[2] * tailI[1][1]);
        tailV[1][0] = (2.0 * headV[1][0] + headV[2][0]) / (3.0 * nt) - tmp1 / 9.0;
        tmp1 = 2.0 * (r[1] * tailI[0][1] + x[1] * tailI[0][0]) + 4.0 * (r[1] * tailI[1][1] + x[1] * tailI[1][0])
                - 2.0 * (r[2] * tailI[0][1] + x[2] * tailI[0][0]) - (r[2] * tailI[1][1] + x[2] * tailI[1][0]);
        tailV[1][1] = (2.0 * headV[1][1] + headV[2][1]) / (3.0 * nt) - tmp1 / 9.0;

        tmp1 = (r[0] * tailI[0][0] - x[0] * tailI[0][1]) - (r[0] * tailI[1][0] - x[0] * tailI[1][1])
                - 4.0 * (r[2] * tailI[0][0] - x[2] * tailI[0][1]) - 2.0 * (r[2] * tailI[1][0] - x[2] * tailI[1][1]);
        tailV[2][0] = (2.0 * headV[2][0] + headV[0][0]) / (3.0 * nt) - tmp1 / 9.0;
        tmp1 = (r[0] * tailI[0][1] + x[0] * tailI[0][0]) - (r[0] * tailI[1][1] + x[0] * tailI[1][0])
                - 4.0 * (r[2] * tailI[0][1] + x[2] * tailI[0][0]) - 2.0 * (r[2] * tailI[1][1] + x[2] * tailI[1][0]);
        tailV[2][1] = (2.0 * headV[2][1] + headV[0][1]) / (3.0 * nt) - tmp1 / 9.0;
    }

    /**
     * calculate the final iterative form for the transformer;
     * VLNABC = at * VLGabc + bt * Iabc;
     * at = diag(nt,nt,nt);
     * bt = nt * diag(Zta, Ztb, Ztc);
     */
    public void calHeadV_GrY_GrY(double[][] tailV, double[][] tailI, double[][] headV) {
        headV[0][0] = (tailV[0][0] + r[0] * tailI[0][0] - x[0] * tailI[0][1]) * nt;
        headV[0][1] = (tailV[0][1] + r[0] * tailI[0][1] + x[0] * tailI[0][0]) * nt;
        headV[1][0] = (tailV[1][0] + r[1] * tailI[1][0] - x[1] * tailI[1][1]) * nt;
        headV[1][1] = (tailV[1][1] + r[1] * tailI[1][1] + x[1] * tailI[1][0]) * nt;
        headV[2][0] = (tailV[2][0] + r[2] * tailI[2][0] - x[2] * tailI[2][1]) * nt;
        headV[2][1] = (tailV[2][1] + r[2] * tailI[2][1] + x[2] * tailI[2][0]) * nt;
    }

    /**
     * IABC = dt * Iabc
     * dt =  diag(1/nt, 1/nt, 1/nt);
     */
    public void calHeadI_YY_DD(double[][] tailI, double[][] headI) {
        headI[0][0] = tailI[0][0] / nt;
        headI[0][1] = tailI[0][1] / nt;
        headI[1][0] = tailI[1][0] / nt;
        headI[1][1] = tailI[1][1] / nt;
        headI[2][0] = tailI[2][0] / nt;
        headI[2][1] = tailI[2][1] / nt;
    }

    /**
     * VLNabc = At * VLNABC - Bt * Iabc
     * At =  diag(1/nt, 1/nt, 1/nt);
     * Bt =  diag(Zta, Ztb, Ztc);
     */
    public void calTailV_GrY_GrY(double[][] headV, double[][] tailI, double[][] tailV) {
        tailV[0][0] = headV[0][0] / nt - (r[0] * tailI[0][0] - x[0] * tailI[0][1]);
        tailV[0][1] = headV[0][1] / nt - (r[0] * tailI[0][1] + x[0] * tailI[0][0]);
        tailV[1][0] = headV[1][0] / nt - (r[1] * tailI[1][0] - x[1] * tailI[1][1]);
        tailV[1][1] = headV[1][1] / nt - (r[1] * tailI[1][1] + x[1] * tailI[1][0]);
        tailV[2][0] = headV[2][0] / nt - (r[2] * tailI[2][0] - x[2] * tailI[2][1]);
        tailV[2][1] = headV[2][1] / nt - (r[2] * tailI[2][1] + x[2] * tailI[2][0]);
    }

    /**
     * calculate the final iterative form for the transformer;
     * VLNABC = at * VLGabc + bt * Iabc;
     */
    public void calHeadV_D_D(double[][] tailV, double[][] tailI, double[][] headV) {
        for (int i = 0; i < at_D_D.length; i++) {
            headV[i][0] = 0;
            headV[i][1] = 0;
            for (int k = 0; k < at_D_D[i].length; k++) {
                headV[i][0] += at_D_D[i][k] * tailV[k][0];
                headV[i][1] += at_D_D[i][k] * tailV[k][1];
            }
            for (int k = 0; k < Bt_real_D_D[i].length; k++) {
                headV[i][0] += nt * (Bt_real_D_D[i][k] * tailI[k][0] - Bt_imag_D_D[i][k] * tailI[k][1]);
                headV[i][1] += nt * (Bt_imag_D_D[i][k] * tailI[k][0] + Bt_real_D_D[i][k] * tailI[k][1]);
            }
        }
    }

    /**
     * VLNabc = At * VLNABC - Bt * Iabc
     */
    public void calTailV_D_D(double[][] headV, double[][] tailI, double[][] tailV) {
        for (int i = 0; i < At_D_D.length; i++) {
            tailV[i][0] = 0;
            tailV[i][1] = 0;
            for (int k = 0; k < At_D_D[i].length; k++) {
                tailV[i][0] += At_D_D[i][k] * headV[k][0];
                tailV[i][1] += At_D_D[i][k] * headV[k][1];
            }
            for (int k = 0; k < Bt_real_D_D[i].length; k++) {
                tailV[i][0] -= (Bt_real_D_D[i][k] * tailI[k][0] - Bt_imag_D_D[i][k] * tailI[k][1]);
                tailV[i][1] -= (Bt_imag_D_D[i][k] * tailI[k][0] + Bt_real_D_D[i][k] * tailI[k][1]);
            }
        }
    }


    public int getConnType() {
        return connType;
    }

    public void setConnType(int connType) {
        this.connType = connType;
    }

    public double[] getR_pu() {
        return r_pu;
    }

    public void setR_pu(double r_pu) {
        for (int i = 0; i < this.r_pu.length; i++)
            this.r_pu[i] = r_pu;
    }

    public double[] getX_pu() {
        return x_pu;
    }

    public void setX_pu(double x_pu) {
        for (int i = 0; i < this.x_pu.length; i++)
            this.x_pu[i] = x_pu;
    }

    public double[] getR() {
        return r;
    }

    public void setR(double r) {
        for (int i = 0; i < this.r.length; i++)
            this.r[i] = r;
    }

    public double[] getX() {
        return x;
    }

    public void setX(double x) {
        for (int i = 0; i < this.x.length; i++)
            this.x[i] = x;
    }

    public void setR_pu(double[] r_pu) {
        System.arraycopy(r_pu, 0, this.r_pu, 0, r_pu.length);
    }

    public void setX_pu(double[] x_pu) {
        System.arraycopy(x_pu, 0, this.x_pu, 0, x_pu.length);
    }

    public void setR(double[] r) {
        System.arraycopy(r, 0, this.r, 0, r.length);
    }

    public void setX(double[] x) {
        System.arraycopy(x, 0, this.x, 0, x.length);
    }

    public double getVLL_high_rated() {
        return VLL_high_rated;
    }

    public void setVLL_high_rated(double VLL_high_rated) {
        this.VLL_high_rated = VLL_high_rated;
    }

    public double getVLL_low_rated() {
        return VLL_low_rated;
    }

    public void setVLL_low_rated(double VLL_low_rated) {
        this.VLL_low_rated = VLL_low_rated;
    }

    public double[] getSn() {
        return Sn;
    }

    public void setTotalSn(double sn) {
        for (int i = 0; i < this.Sn.length; i++)
            Sn[i] = sn / 3;
    }

    public void setSn(double[] sn) {
        System.arraycopy(sn, 0, this.Sn, 0, sn.length);
    }

    public double getNt() {
        return nt;
    }

    public void setNt(double nt) {
        this.nt = nt;
    }
}


