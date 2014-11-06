package zju.util;

import zju.matrix.ASparseMatrixLink;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;

/**
 * Created by IntelliJ IDEA.
 * store lower triangular matrix only.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-19
 */
public class HessianMakerRC implements MeasTypeCons {

    public static void getStrucOfU(MeasVector meas, YMatrixGetter y, MySparseDoubleMatrix2D hessian) {
        int n = y.getAdmittance()[0].getN();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++) {
                        int num = meas.getBus_v_pos()[i];//num starts from 1
                        getHessianBusV(num, n, hessian);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        getHessianBusP2(num, n, y, hessian);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++) {
                        int num = meas.getBus_q_pos()[i];//num starts from 1
                        getHessianBusQ2(num, n, y, hessian);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        getHessianLineFromP(num, n, y, hessian, 1.0);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        getHessianLineFromQ(num, n, y, hessian, 1.0);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        getHessianLineToP(num, n, y, hessian, 1.0);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        getHessianLineToQ(num, n, y, hessian, 1.0);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    public static void getStrucOfUI(MeasVector meas, YMatrixGetter y, MySparseDoubleMatrix2D hessian) {
        int n = y.getAdmittance()[0].getN();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++) {
                        int num = meas.getBus_v_pos()[i];//num starts from 1
                        getHessianBusV(num, n, hessian);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        getHessianBusP(num, n, hessian);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++) {
                        int num = meas.getBus_q_pos()[i];//num starts from 1
                        getHessianBusQ(num, n, hessian);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        getHessianLineFromP(num, n, y, hessian, 1.0);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        getHessianLineFromQ(num, n, y, hessian, 1.0);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        getHessianLineToP(num, n, y, hessian, 1.0);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        getHessianLineToQ(num, n, y, hessian, 1.0);
                    }
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
    }

    public static void getHessianOfU(MeasVector meas, YMatrixGetter y, MySparseDoubleMatrix2D hession, double[] lambda, int index) {
        int n = y.getAdmittance()[0].getN();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++)
                        index++;
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        int num = meas.getBus_v_pos()[i];//num starts from 1
                        getHessianBusV(num, n, hession, lambda[index]);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        getHessianBusP2(num, n, y, hession, lambda[index]);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int num = meas.getBus_q_pos()[i];//num starts from 1
                        getHessianBusQ2(num, n, y, hession, lambda[index]);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        getHessianLineFromP(num, n, y, hession, lambda[index]);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        getHessianLineFromQ(num, n, y, hession, lambda[index]);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        getHessianLineToP(num, n, y, hession, lambda[index]);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        getHessianLineToQ(num, n, y, hession, lambda[index]);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    public static void getHessianOfUI(MeasVector meas, YMatrixGetter y, MySparseDoubleMatrix2D hession, double[] lambda, int index) {
        int n = y.getAdmittance()[0].getN();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++)
                        index++;
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        int num = meas.getBus_v_pos()[i];//num starts from 1
                        getHessianBusV(num, n, hession, lambda[index]);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        getHessianBusP(num, n, hession, lambda[index]);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int num = meas.getBus_q_pos()[i];//num starts from 1
                        getHessianBusQ(num, n, hession, lambda[index]);
                    }

                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        getHessianLineFromP(num, n, y, hession, lambda[index]);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        getHessianLineFromQ(num, n, y, hession, lambda[index]);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        getHessianLineToP(num, n, y, hession, lambda[index]);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        getHessianLineToQ(num, n, y, hession, lambda[index]);
                    }
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
    }

    public static void getHessianBusV(int pos, int n, MySparseDoubleMatrix2D result) {
        getHessianBusV(pos, n, result, 1.0);
    }

    public static void getHessianBusV(int pos, int n, MySparseDoubleMatrix2D result, double lamda) {
        result.addQuick(pos - 1, pos - 1, lamda * 2.0);
        result.addQuick(pos - 1 + n, pos - 1 + n, lamda * 2.0);
    }

    public static void getHessianBusP2(int pos, int n, YMatrixGetter y, MySparseDoubleMatrix2D result) {
        getHessianBusP2(pos, n, y, result, 1.0);
    }

    public static void getHessianBusP2(int pos, int n, YMatrixGetter y, MySparseDoubleMatrix2D result, double lamda) {
        pos -= 1;
        ASparseMatrixLink[] admittance = y.getAdmittance();
        int k = admittance[0].getIA()[pos];
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double gij = admittance[0].getVA().get(k);
            double bij = admittance[1].getVA().get(k);
            k = admittance[0].getLINK().get(k);
            if (j == pos) {
                result.addQuick(pos, pos, lamda * 2.0 * gij);
                result.addQuick(pos + n, pos + n, lamda * 2.0 * gij);
            } else if (j > pos) {
                result.addQuick(j, pos, lamda * gij);
                result.addQuick(j + n, pos + n, lamda * gij);
                result.addQuick(pos + n, j, lamda * bij);
                result.addQuick(j + n, pos, lamda * -bij);
            } else {
                result.addQuick(pos, j, lamda * gij);
                result.addQuick(pos + n, j + n, lamda * gij);
                result.addQuick(pos + n, j, lamda * bij);
                result.addQuick(j + n, pos, lamda * -bij);
            }
        }
    }

    public static void getHessianBusQ2(int pos, int n, YMatrixGetter y, MySparseDoubleMatrix2D result) {
        getHessianBusQ2(pos, n, y, result, 1.0);
    }

    public static void getHessianBusQ2(int pos, int n, YMatrixGetter y, MySparseDoubleMatrix2D result, double lamda) {
        pos -= 1;
        ASparseMatrixLink[] admittance = y.getAdmittance();
        int k = admittance[0].getIA()[pos];
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double gij = admittance[0].getVA().get(k);
            double bij = admittance[1].getVA().get(k);
            k = admittance[0].getLINK().get(k);
            if (j == pos) {
                result.addQuick(pos, pos, lamda * -2 * bij);
                result.addQuick(pos + n, pos + n, lamda * -2 * bij);
            } else if (j > pos) {
                result.addQuick(j, pos, lamda * -bij);
                result.addQuick(j + n, pos + n, lamda * -bij);
                result.addQuick(pos + n, j, lamda * gij);
                result.addQuick(j + n, pos, lamda * -gij);
            } else {
                result.addQuick(pos, j, lamda * -bij);
                result.addQuick(pos + n, j + n, lamda * -bij);
                result.addQuick(pos + n, j, lamda * gij);
                result.addQuick(j + n, pos, lamda * -gij);
            }
        }
    }

    public static void getHessianBusP(int pos, int n, MySparseDoubleMatrix2D result) {
        getHessianBusP(pos, n, result, 1.0);
    }

    public static void getHessianBusP(int pos, int n, MySparseDoubleMatrix2D result, double lamda) {
        //result.setQuick(pos - 1, pos - 1 + 2 * n, 1);
        //result.setQuick(pos - 1 + n, pos - 1 + 3 * n, 1);
        result.addQuick(pos - 1 + 2 * n, pos - 1, lamda * 1.0);
        result.addQuick(pos - 1 + 3 * n, pos - 1 + n, lamda * 1.0);
    }

    public static void getHessianBusQ(int pos, int n, MySparseDoubleMatrix2D result) {
        getHessianBusQ(pos, n, result, 1.0);
    }

    public static void getHessianBusQ(int pos, int n, MySparseDoubleMatrix2D result, double lamda) {
        //result.setQuick(pos - 1, pos - 1 + 3 * n, -1);
        //result.setQuick(pos - 1 + n, pos - 1 + 2 * n, 1);
        result.addQuick(pos - 1 + 2 * n, pos - 1 + n, lamda * 1.0);
        result.addQuick(pos - 1 + 3 * n, pos - 1, lamda * -1.0);
    }

    public static void getHessianLineFromP(int num, int n, YMatrixGetter y, MySparseDoubleMatrix2D r, double lamda) {
        double[] gbg1b1 = y.getLineAdmittance(num, YMatrixGetter.LINE_FROM);
        int[] ij = y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        r.addQuick(i, i, lamda * 2 * (gbg1b1[0] + gbg1b1[2]));
        r.addQuick(i + n, i + n, lamda * 2 * (gbg1b1[0] + gbg1b1[2]));
        r.addQuick(i + n, j, lamda * -gbg1b1[1]);
        r.addQuick(j + n, i, lamda * gbg1b1[1]);
        if (i > j) {
            r.addQuick(i, j, lamda * -gbg1b1[0]);
            r.addQuick(i + n, j + n, lamda * -gbg1b1[0]);
        } else {
            r.addQuick(j, i, lamda * -gbg1b1[0]);
            r.addQuick(j + n, i + n, lamda * -gbg1b1[0]);
        }
    }

    public static void getHessianLineFromQ(int num, int n, YMatrixGetter y, MySparseDoubleMatrix2D r, double lamda) {
        double[] gbg1b1 = y.getLineAdmittance(num, YMatrixGetter.LINE_FROM);
        int[] ij = y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        r.addQuick(i, i, lamda * -2 * (gbg1b1[1] + gbg1b1[3]));
        r.addQuick(i + n, i + n, lamda * -2 * (gbg1b1[1] + gbg1b1[3]));
        r.addQuick(i + n, j, lamda * -gbg1b1[0]);
        r.addQuick(j + n, i, lamda * gbg1b1[0]);
        if (i > j) {
            r.addQuick(i, j, lamda * gbg1b1[1]);
            r.addQuick(i + n, j + n, lamda * gbg1b1[1]);
        } else {
            r.addQuick(j, i, lamda * gbg1b1[1]);
            r.addQuick(j + n, i + n, lamda * gbg1b1[1]);
        }
    }

    public static void getHessianLineToP(int num, int n, YMatrixGetter y, MySparseDoubleMatrix2D r, double lamda) {
        double[] gbg1b1 = y.getLineAdmittance(num, YMatrixGetter.LINE_TO);
        int[] ij = y.getFromTo(num);
        int j = ij[0] - 1;
        int i = ij[1] - 1;
        r.addQuick(i, i, lamda * 2.0 * (gbg1b1[0] + gbg1b1[2]));
        r.addQuick(i + n, i + n, lamda * 2.0 * (gbg1b1[0] + gbg1b1[2]));
        r.addQuick(i + n, j, lamda * -gbg1b1[1]);
        r.addQuick(j + n, i, lamda * gbg1b1[1]);
        if (i > j) {
            r.addQuick(i, j, lamda * -gbg1b1[0]);
            r.addQuick(i + n, j + n, lamda * -gbg1b1[0]);
        } else {
            r.addQuick(j, i, lamda * -gbg1b1[0]);
            r.addQuick(j + n, i + n, lamda * -gbg1b1[0]);
        }
    }

    public static void getHessianLineToQ(int num, int n, YMatrixGetter y, MySparseDoubleMatrix2D r, double lamda) {
        double[] gbg1b1 = y.getLineAdmittance(num, YMatrixGetter.LINE_TO);
        int[] ij = y.getFromTo(num);
        int j = ij[0] - 1;
        int i = ij[1] - 1;
        r.addQuick(i, i, lamda * -2 * (gbg1b1[1] + gbg1b1[3]));
        r.addQuick(i + n, i + n, lamda * -2 * (gbg1b1[1] + gbg1b1[3]));
        r.addQuick(i + n, j, lamda * -gbg1b1[0]);
        r.addQuick(j + n, i, lamda * gbg1b1[0]);
        if (i > j) {
            r.addQuick(i, j, lamda * gbg1b1[1]);
            r.addQuick(i + n, j + n, lamda * gbg1b1[1]);
        } else {
            r.addQuick(j, i, lamda * gbg1b1[1]);
            r.addQuick(j + n, i + n, lamda * gbg1b1[1]);
        }
    }

    public static void getHessianOfU(int[] zeroPBuses, int[] zeroQBuses, YMatrixGetter y,
                              MySparseDoubleMatrix2D hess, double[] lamda, int index) {
        int n = y.getAdmittance()[0].getN();
        for (int i = 0; i < zeroPBuses.length; i++, index++)
            getHessianBusP2(zeroPBuses[i], n, y, hess, lamda[index]);
        for (int i = 0; i < zeroQBuses.length; i++, index++)
            getHessianBusQ2(zeroQBuses[i], n, y, hess, lamda[index]);
    }

    public static void getStrucOfU(int[] zeroPBuses, int[] zeroQBuses, YMatrixGetter y, MySparseDoubleMatrix2D hess) {
        int n = y.getAdmittance()[0].getN();
        for (int bus : zeroPBuses)
            getHessianBusP2(bus, n, y, hess);
        for (int bus : zeroQBuses)
            getHessianBusQ2(bus, n, y, hess);
    }

    public static void getHessianOfUI(int[] zeroPBuses, int[] zeroQBuses, YMatrixGetter y,
                                      MySparseDoubleMatrix2D hess, double[] lamda, int index) {
        int n = y.getAdmittance()[0].getN();
        for (int i = 0; i < zeroPBuses.length; i++, index++)
            getHessianBusP(zeroPBuses[i], n, hess, lamda[index]);
        for (int i = 0; i < zeroQBuses.length; i++, index++)
            getHessianBusQ(zeroQBuses[i], n, hess, lamda[index]);
    }

    public static void getStrucOfUI(int[] zeroPBuses, int[] zeroQBuses, YMatrixGetter y, MySparseDoubleMatrix2D hess) {
        int n = y.getAdmittance()[0].getN();
        for (int bus : zeroPBuses)
            getHessianBusP(bus, n, hess);
        for (int bus : zeroQBuses)
            getHessianBusQ(bus, n, hess);
    }
}
