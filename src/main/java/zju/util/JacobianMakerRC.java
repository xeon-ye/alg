package zju.util;

import cern.colt.matrix.DoubleMatrix2D;
import zju.matrix.ASparseMatrixLink;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-19
 */
public class JacobianMakerRC implements MeasTypeCons {
    public static final int MODE_U_ONLY = 0;
    public static final int MODE_UI = 1;

    private double[] x; //Ux, Uy, Ix, Iy;
    private DoubleMatrix2D G, B;
    private YMatrixGetter Y;
    int mode;
    int n;

    public JacobianMakerRC(int mode) {
        this.mode = mode;
    }

    public int getNonZeroNum(MeasVector meas) {
        switch (mode) {
            case MODE_UI:
                return 4 * (meas.getBus_p_pos().length + meas.getBus_q_pos().length
                        + meas.getLine_from_p_pos().length + meas.getLine_from_q_pos().length
                        + meas.getLine_to_p_pos().length + meas.getLine_to_q_pos().length) +
                        2 * meas.getBus_v_pos().length;
            case MODE_U_ONLY:
                int capacity =  4 * (meas.getLine_from_p_pos().length + meas.getLine_from_q_pos().length
                        + meas.getLine_to_p_pos().length + meas.getLine_to_q_pos().length) +
                        2 * meas.getBus_v_pos().length;
                if (Y.getConnectedBusCount() == null) {
                    capacity += 2 * (meas.getBus_p_pos().length + meas.getBus_q_pos().length);
                } else {
                    for (int busNum : meas.getBus_p_pos())
                        capacity += (2 * Y.getConnectedBusCount()[busNum - 1] + 2);
                    for (int busNum : meas.getBus_q_pos())
                        capacity += (2 * Y.getConnectedBusCount()[busNum - 1] + 2);
                }
                return capacity;
            default:
                return 0;
        }
    }

    public void fillJacobian_bus_v(int[] pos, int row, DoubleMatrix2D r) {
        for (int k = 0; k < pos.length; k++, row++) {
            r.setQuick(row, pos[k] - 1, 2.0 * x[pos[k] - 1]);
            r.setQuick(row, pos[k] - 1 + G.columns(), 2.0 * x[pos[k] - 1 + n]);
        }
    }

    public void fillJacobian_bus_p(int[] pos, int row, DoubleMatrix2D r) {
        if (mode == MODE_UI) {
            for (int k = 0; k < pos.length; k++, row++) {
                r.setQuick(row, pos[k] - 1, x[pos[k] - 1 + 2 * n]);
                r.setQuick(row, pos[k] - 1 + n, x[pos[k] - 1 + 3 * n]);
                r.setQuick(row, pos[k] - 1 + 2 * n, x[pos[k] - 1]);
                r.setQuick(row, pos[k] - 1 + 3 * n, x[pos[k] - 1 + n]);
            }
        } else {
            ASparseMatrixLink[] admittance = Y.getAdmittance();
            for (int a = 0; a < pos.length; a++, row++) {
                int num = pos[a] - 1;
                int k = admittance[0].getIA()[num];
                double tmp1 = 0;
                double tmp2 = 0;
                double ei = x[num];
                double fi = x[num + n];
                while (k != -1) {
                    int j = admittance[0].getJA().get(k);
                    double ej = x[j];
                    double fj = x[j + n];
                    double gij = admittance[0].getVA().get(k);
                    double bij = admittance[1].getVA().get(k);
                    if (num == j) {
                        tmp1 += gij * ei + bij * fi;
                        tmp2 += -bij * ei + gij * fi;
                    } else {
                        r.setQuick(row, j, gij * ei + bij * fi);
                        r.setQuick(row, j + n, -bij * ei + gij * fi);
                    }
                    tmp1 += gij * ej - bij * fj;
                    tmp2 += gij * fj + bij * ej;
                    k = admittance[0].getLINK().get(k);
                }
                r.setQuick(row, num, tmp1);
                r.setQuick(row, num + n, tmp2);
            }
        }
    }

    public void fillJacobian_bus_q(int[] pos, int row, DoubleMatrix2D r) {
        if (mode == MODE_UI) {
            for (int k = 0; k < pos.length; k++, row++) {
                r.setQuick(row, pos[k] - 1, -x[pos[k] - 1 + 3 * n]);
                r.setQuick(row, pos[k] - 1 + n, x[pos[k] - 1 + 2 * n]);
                r.setQuick(row, pos[k] - 1 + 2 * n, x[pos[k] - 1 + n]);
                r.setQuick(row, pos[k] - 1 + 3 * n, -x[pos[k] - 1]);
            }
        } else {
            ASparseMatrixLink[] admittance = Y.getAdmittance();
            int n = G.columns();
            for (int a = 0; a < pos.length; a++, row++) {
                int num = pos[a] - 1;
                int k = admittance[0].getIA()[num];
                double tmp1 = 0;
                double tmp2 = 0;
                double ei = x[num];
                double fi = x[num + n];
                while (k != -1) {
                    int j = admittance[0].getJA().get(k);
                    double ej = x[j];
                    double fj = x[j + n];
                    double gij = admittance[0].getVA().get(k);
                    double bij = admittance[1].getVA().get(k);
                    if (num == j) {
                        tmp1 += -bij * ei + gij * fi;
                        tmp2 += -gij * ej - bij * fj;
                    } else {
                        r.setQuick(row, j, -bij * ei + gij * fi);
                        r.setQuick(row, j + n, -gij * ei - bij * fi);
                    }
                    tmp1 += -gij * fj - bij * ej;
                    tmp2 += gij * ej - bij * fj;
                    k = admittance[0].getLINK().get(k);
                }
                r.setQuick(row, num, tmp1);
                r.setQuick(row, num + n, tmp2);
            }
        }
    }

    public void fillJacobian_line_from_p(int[] pos, int row, DoubleMatrix2D r) {
        for (int k = 0; k < pos.length; k++, row++) {
            double[] gbg1b1 = Y.getLineAdmittance(pos[k], YMatrixGetter.LINE_FROM);
            int[] ij = Y.getFromTo(pos[k]);
            int i = ij[0] - 1;
            int j = ij[1] - 1;
            r.setQuick(row, i, 2 * (gbg1b1[0] + gbg1b1[2]) * x[i] + gbg1b1[1] * x[j + n] - gbg1b1[0] * x[j]);
            r.setQuick(row, i + G.columns(), 2 * (gbg1b1[0] + gbg1b1[2]) * x[i + n] - gbg1b1[1] * x[j] - gbg1b1[0] * x[j + n]);
            r.setQuick(row, j, -gbg1b1[0] * x[i] - gbg1b1[1] * x[i + n]);
            r.setQuick(row, j + G.columns(), gbg1b1[1] * x[i] - gbg1b1[0] * x[i + n]);
        }
    }

    public void fillJacobian_line_from_q(int[] pos, int row, DoubleMatrix2D r) {
        for (int k = 0; k < pos.length; k++, row++) {
            double[] gbg1b1 = Y.getLineAdmittance(pos[k], YMatrixGetter.LINE_FROM);
            int[] ij = Y.getFromTo(pos[k]);
            int i = ij[0] - 1;
            int j = ij[1] - 1;
            r.setQuick(row, i, -2 * (gbg1b1[1] + gbg1b1[3]) * x[i] + gbg1b1[1] * x[j] + gbg1b1[0] * x[j + n]);
            r.setQuick(row, i + G.columns(), -2 * (gbg1b1[1] + gbg1b1[3]) * x[i + n] + gbg1b1[1] * x[j + n] - gbg1b1[0] * x[j]);
            r.setQuick(row, j, gbg1b1[1] * x[i] - gbg1b1[0] * x[i + n]);
            r.setQuick(row, j + G.columns(), gbg1b1[0] * x[i] + gbg1b1[1] * x[i + n]);
        }
    }

    public void fillJacobian_line_to_p(int[] pos, int row, DoubleMatrix2D r) {
        for (int k = 0; k < pos.length; k++, row++) {
            double[] gbg1b1 = Y.getLineAdmittance(pos[k], YMatrixGetter.LINE_TO);
            int[] ij = Y.getFromTo(pos[k]);
            int j = ij[0] - 1;
            int i = ij[1] - 1;
            r.setQuick(row, i, 2 * (gbg1b1[0] + gbg1b1[2]) * x[i] + gbg1b1[1] * x[j + n] - gbg1b1[0] * x[j]);
            r.setQuick(row, i + G.columns(), 2 * (gbg1b1[0] + gbg1b1[2]) * x[i + n] - gbg1b1[1] * x[j] - gbg1b1[0] * x[j + n]);
            r.setQuick(row, j, -gbg1b1[0] * x[i] - gbg1b1[1] * x[i + n]);
            r.setQuick(row, j + G.columns(), gbg1b1[1] * x[i] - gbg1b1[0] * x[i + n]);
        }
    }

    public DoubleMatrix2D fillJacobian_line_to_q(int[] pos, int row, DoubleMatrix2D r) {
        for (int k = 0; k < pos.length; k++, row++) {
            double[] gbg1b1 = Y.getLineAdmittance(pos[k], YMatrixGetter.LINE_TO);
            int[] ij = Y.getFromTo(pos[k]);
            int j = ij[0] - 1;
            int i = ij[1] - 1;
            r.setQuick(row, i, -2 * (gbg1b1[1] + gbg1b1[3]) * x[i] + gbg1b1[1] * x[j] + gbg1b1[0] * x[j + n]);
            r.setQuick(row, i + G.columns(), -2 * (gbg1b1[1] + gbg1b1[3]) * x[i + n] + gbg1b1[1] * x[j + n] - gbg1b1[0] * x[j]);
            r.setQuick(row, j, gbg1b1[1] * x[i] - gbg1b1[0] * x[i + n]);
            r.setQuick(row, j + G.columns(), gbg1b1[0] * x[i] + gbg1b1[1] * x[i + n]);
        }
        return r;
    }

    public void getJacobian(MeasVector meas, DoubleMatrix2D r, int index){
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    //todo:
                    break;
                case TYPE_BUS_VOLOTAGE:
                    fillJacobian_bus_v(meas.getBus_v_pos(), index, r);
                    index +=  meas.getBus_v_pos().length;
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    fillJacobian_bus_p(meas.getBus_p_pos(), index, r);
                    index +=  meas.getBus_p_pos().length;
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    fillJacobian_bus_q(meas.getBus_q_pos(),index, r);
                    index +=  meas.getBus_q_pos().length;
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    fillJacobian_line_from_p(meas.getLine_from_p_pos(), index, r);
                    index +=  meas.getLine_from_p_pos().length;
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    fillJacobian_line_from_q(meas.getLine_from_q_pos(), index, r);
                    index +=  meas.getLine_from_q_pos().length;
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    fillJacobian_line_to_p(meas.getLine_to_p_pos(), index, r);
                    index +=  meas.getLine_to_p_pos().length;
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    fillJacobian_line_to_q(meas.getLine_to_q_pos(), index, r);
                    index +=  meas.getLine_to_q_pos().length;
                    break;
                default:
                    break;
            }
        }
    }

    public int getNoneZeroNum(int[] zeroPBuses, int[] zeroQBuses) {
        switch (mode) {
            case MODE_UI:
                return 4 * (zeroPBuses.length + zeroQBuses.length);
            case MODE_U_ONLY:
                int capacity = 0;
                if (Y.getConnectedBusCount() == null) {
                    capacity += 2 * (zeroPBuses.length + zeroQBuses.length);
                } else {
                    for (int busNum : zeroPBuses)
                        capacity += (2 * Y.getConnectedBusCount()[busNum - 1] + 2);
                    for (int busNum : zeroQBuses)
                        capacity += (2 * Y.getConnectedBusCount()[busNum - 1] + 2);
                }
                return capacity;
            default:
                return 0;
        }
    }

    public void getJacobian(int[] zeroPBuses, int[] zeroQBuses, DoubleMatrix2D jac, int index) {
        fillJacobian_bus_p(zeroPBuses, index, jac);
        index += zeroPBuses.length;
        fillJacobian_bus_q(zeroQBuses, index, jac);
    }

    public void setY(YMatrixGetter y) {
        G = y.getAdmittance()[0].toColtSparseMatrix();
        B = y.getAdmittance()[1].toColtSparseMatrix();
        this.Y = y;
        this.n = G.columns();
    }

    public YMatrixGetter getY() {
        return Y;
    }

    public void setUI(double[] x) {
        this.x = x;
    }


    public DoubleMatrix2D getG() {
        return G;
    }

    public DoubleMatrix2D getB() {
        return B;
    }
}
