package zju.util;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import zju.matrix.ASparseMatrixLink;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-26
 */
public class HessianMakerPC implements MeasTypeCons {
    /**
     * @param meas   潮流计算用的量测，只包含P,Q量测
     * @param y      导纳阵
     * @param state  状态量，包含所有节点的电压幅值和相角
     * @param pqSize PQ节点个数
     * @param pvSize PV节点个数
     * @param result Hessian矩阵
     */
    public static void getHessianOfVTheta(MeasVector meas, YMatrixGetter y, double[] state,
                                          int pqSize, int pvSize, MySparseDoubleMatrix2D result,
                                          double[] lambda, int index) {
        int n = y.getAdmittance()[0].getM();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++) {
                        int pos = meas.getBus_p_pos()[i] - 1;
                        ASparseMatrixLink[] admittance = y.getAdmittance();
                        int k = admittance[0].getIA()[pos];
                        while (k != -1) {
                            int j = admittance[0].getJA().get(k);
                            double thetaIJ = state[pos + n] - state[j + n];
                            double gij = admittance[0].getValue(pos, j);
                            double bij = admittance[1].getValue(pos, j);
                            double sin = Math.sin(thetaIJ);
                            double cos = Math.cos(thetaIJ);
                            if (j == pos) {
                                if (pos < pqSize)
                                    result.addQuick(pos, pos, lambda[index] * 2 * admittance[0].getValue(pos, pos));
                                double mij = -state[pos] * state[j] * (gij * cos + bij * sin);
                                mij += StateCalByPolar.calBusP(pos + 1, y, state);

                                if (pos < pqSize) {
                                    double v = -admittance[1].getVA().get(k) * state[pos] * state[pos];
                                    v -= StateCalByPolar.calBusQ(pos + 1, y, state);
                                    result.addQuick(pos + pqSize, pos, lambda[index] * v / state[pos]);
                                }
                                result.addQuick(pos + pqSize, pos + pqSize, lambda[index] * -mij);
                            } else {
                                double v = state[pos] * (admittance[0].getVA().get(k) * cos + admittance[1].getVA().get(k) * sin);
                                double v1 = v / state[pos];
                                if (j > pos && j < pqSize) {
                                    result.addQuick(j, pos, lambda[index] * v1);
                                } else if (j < pos && pos < pqSize) {
                                    result.addQuick(pos, j, lambda[index] * v1);
                                }
                                double mij = -state[pos] * state[j] * (gij * cos + bij * sin);
                                v = state[pos] * state[j] * (admittance[0].getVA().get(k) * sin - admittance[1].getVA().get(k) * cos);
                                v1 = v / state[j];

                                if (j < pqSize) {
                                    result.addQuick(j + pqSize, j, lambda[index] * v1);
                                    result.addQuick(pos + pqSize, j, lambda[index] * -v1);
                                }
                                if (j < pqSize + pvSize && pos < pqSize)
                                    result.addQuick(j + pqSize, pos, lambda[index] * v / state[pos]);
                                if (j < pqSize + pvSize)
                                    result.addQuick(j + pqSize, j + pqSize, lambda[index] * mij);
                                if (pos > j)
                                    result.addQuick(pos + pqSize, j + pqSize, lambda[index] * -mij);
                                else if (j < pqSize + pvSize)
                                    result.addQuick(j + pqSize, pos + pqSize, lambda[index] * -mij);
                            }
                            k = admittance[0].getLINK().get(k);
                        }
                        index++;
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++) {
                        int pos = meas.getBus_q_pos()[i] - 1;
                        ASparseMatrixLink[] admittance = y.getAdmittance();
                        int k = admittance[0].getIA()[pos];
                        while (k != -1) {
                            int j = admittance[0].getJA().get(k);
                            double thetaIJ = state[pos + n] - state[j + n];
                            double gij = admittance[0].getValue(pos, j);
                            double bij = admittance[1].getValue(pos, j);
                            double sin = Math.sin(thetaIJ);
                            double cos = Math.cos(thetaIJ);
                            if (j == pos) {
                                result.addQuick(pos, pos, lambda[index] * -2 * admittance[1].getValue(pos, pos));
                                double hij = state[pos] * state[j] * (gij * sin - bij * cos);
                                double v = -admittance[0].getVA().get(k) * state[pos] * state[pos];
                                v += StateCalByPolar.calBusP(pos + 1, y, state);
                                hij -= StateCalByPolar.calBusQ(pos + 1, y, state);
                                result.addQuick(pos + pqSize, pos, lambda[index] * v / state[pos]);
                                result.addQuick(pos + pqSize, pos + pqSize, lambda[index] * hij);
                            } else {
                                double v = state[pos] * (admittance[0].getVA().get(k) * sin - admittance[1].getVA().get(k) * cos);
                                double v1 = v / state[pos];
                                if (j > pos && j < pqSize)
                                    result.addQuick(j, pos, lambda[index] * v1);
                                else if (j < pos)
                                    result.addQuick(pos, j, lambda[index] * v1);

                                double hij = state[pos] * state[j] * (gij * sin - bij * cos);
                                v = -state[pos] * state[j] * (admittance[0].getVA().get(k) * cos + admittance[1].getVA().get(k) * sin);
                                v1 = v / state[j];
                                if (j < pqSize) {
                                    result.addQuick(j + pqSize, j, lambda[index] * v1);
                                    result.addQuick(pos + pqSize, j, lambda[index] * -v1);
                                }
                                if (j < pqSize + pvSize) {
                                    result.addQuick(j + pqSize, pos, lambda[index] * v / state[pos]);
                                    result.addQuick(j + pqSize, j + pqSize, lambda[index] * -hij);
                                }
                                if (pos > j)
                                    result.addQuick(pos + pqSize, j + pqSize, lambda[index] * hij);
                                else if (j < pqSize + pvSize)
                                    result.addQuick(j + pqSize, pos + pqSize, lambda[index] * hij);
                            }
                            k = admittance[0].getLINK().get(k);
                        }
                        index++;
                    }
                    break;
            }
        }
    }

    public static void getHessianStruc(MeasVector meas, YMatrixGetter y, DoubleMatrix2D result) {
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    break;
                case TYPE_BUS_VOLOTAGE:
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++)
                        getStrucBusPQ(meas.getBus_p_pos()[i], y, result);
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++)
                        getStrucBusPQ(meas.getBus_q_pos()[i], y, result);
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++)
                        getStrucLineFromPQ(meas.getLine_from_p_pos()[k], y, result);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++)
                        getStrucLineFromPQ(meas.getLine_from_q_pos()[k], y, result);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++)
                        getStrucLineToPQ(meas.getLine_to_p_pos()[k], y, result);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++)
                        getStrucLineToPQ(meas.getLine_to_q_pos()[k], y, result);
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
    }

    public static void getStrucLineToPQ(int pos, YMatrixGetter y, DoubleMatrix2D result) {
        int n = y.getAdmittance()[0].getM();
        int[] ij = y.getFromTo(pos);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        result.setQuick(j, j, 1);
        result.setQuick(j + n, j, 1);
        result.setQuick(j + n, i, 1);
        result.setQuick(j + n, j + n, 1);
        result.setQuick(i + n, j, 1);
        result.setQuick(i + n, i, 1);
        result.setQuick(i + n, i + n, 1);
        if (i > j) {
            result.setQuick(i, j, 1);
            result.setQuick(i + n, j + n, 1);
        } else {
            result.setQuick(j, i, 1);
            result.setQuick(j + n, i + n, 1);
        }
    }

    public static void getStrucLineFromPQ(int pos, YMatrixGetter y, DoubleMatrix2D result) {
        int n = y.getAdmittance()[0].getM();
        int[] ij = y.getFromTo(pos);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        result.setQuick(i, i, 1);
        result.setQuick(i + n, i, 1);
        result.setQuick(i + n, j, 1);
        result.setQuick(i + n, i + n, 1);
        result.setQuick(j + n, i, 1);
        result.setQuick(j + n, j, 1);
        result.setQuick(j + n, j + n, 1);
        if (i > j) {
            result.setQuick(i, j, 1);
            result.setQuick(i + n, j + n, 1);
        } else {
            result.setQuick(j, i, 1);
            result.setQuick(j + n, i + n, 1);
        }
    }

    public static void getStrucBusPQ(int pos, YMatrixGetter y, DoubleMatrix2D result) {
        int n = y.getAdmittance()[0].getM();
        pos = pos - 1;
        ASparseMatrixLink[] admittance = y.getAdmittance();
        int k = admittance[0].getIA()[pos];
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            if (j == pos) {
                result.setQuick(pos, pos, 1);
                result.setQuick(pos + n, pos, 1);
                result.setQuick(pos + n, pos + n, 1);
            } else {
                if (j > pos)
                    result.setQuick(j, pos, 1);
                else
                    result.setQuick(pos, j, 1);
                result.setQuick(j + n, j, 1);
                result.setQuick(pos + n, j, 1);
                result.setQuick(j + n, pos, 1);
                result.setQuick(j + n, j + n, 1);
                if (pos > j)
                    result.setQuick(pos + n, j + n, 1);
                else
                    result.setQuick(j + n, pos + n, 1);
            }
            k = admittance[0].getLINK().get(k);
        }
    }

    public static void getHessianOfVTheta(MeasVector meas, YMatrixGetter y, double[] state,
                                          MySparseDoubleMatrix2D hession, double[] lambda, int start) {
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++)
                        start++;
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++)
                        start++;
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, start++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        getHessianBusP(num, y, state, hession, lambda[start]);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, start++) {
                        int num = meas.getBus_q_pos()[i];//num starts from 1
                        getHessianBusQ(num, y, state, hession, lambda[start]);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, start++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        getHessianLineFromP(num, y, state, hession, lambda[start]);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, start++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        getHessianLineFromQ(num, y, state, hession, lambda[start]);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, start++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        getHessianLineToP(num, y, state, hession, lambda[start]);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, start++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        getHessianLineToQ(num, y, state, hession, lambda[start]);
                    }
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
    }

    public static void getStrucOfFullState(MeasVector meas, YMatrixGetter y, DoubleMatrix2D result) {
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    break;
                case TYPE_BUS_VOLOTAGE:
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++)
                        getStrucLineFromPQ(meas.getLine_from_p_pos()[k], y, result);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++)
                        getStrucLineFromPQ(meas.getLine_from_q_pos()[k], y, result);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++)
                        getStrucLineToPQ(meas.getLine_to_p_pos()[k], y, result);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++)
                        getStrucLineToPQ(meas.getLine_to_q_pos()[k], y, result);
                    break;
                default:
                    break;
            }
        }
    }

    public static void getHessianOfFullState(MeasVector meas, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D hession, double[] lambda, int index) {
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++)
                        index++;
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++)
                        index++;
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++)
                        index++;
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++)
                        index++;
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        getHessianLineFromP(num, y, state, hession, lambda[index]);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        getHessianLineFromQ(num, y, state, hession, lambda[index]);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        getHessianLineToP(num, y, state, hession, lambda[index]);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        getHessianLineToQ(num, y, state, hession, lambda[index]);
                    }
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
    }

    public static SparseDoubleMatrix2D getHessianBusP(int pos, YMatrixGetter y, AVector state, int size) {
        return getHessianBusP(pos, y, state.getValues(), size);
    }

    public static SparseDoubleMatrix2D getHessianBusP(int pos, YMatrixGetter y, double[] state, int size) {
        int n = 2 * y.getAdmittance()[0].getN();
        MySparseDoubleMatrix2D result = new MySparseDoubleMatrix2D(n, n, size, 0.2, 0.5);
        getHessianBusP(pos, y, state, result);
        return result;
    }

    public static void getHessianBusP(int pos, YMatrixGetter y, AVector state, MySparseDoubleMatrix2D result) {
        getHessianBusP(pos, y, state.getValues(), result);
    }

    public static void getHessianBusP(int pos, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result) {
        getHessianBusP(pos, y, state, result, 1.0);
    }

    public static void getHessianBusP(int pos, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result, double lambda) {
        int n = y.getAdmittance()[0].getM();
        pos = pos - 1;
        ASparseMatrixLink[] admittance = y.getAdmittance();
        int k = admittance[0].getIA()[pos];
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double thetaIJ = state[pos + n] - state[j + n];
            double gij = admittance[0].getValue(pos, j);
            double bij = admittance[1].getValue(pos, j);
            double sin = Math.sin(thetaIJ);
            double cos = Math.cos(thetaIJ);
            if (j == pos) {
                result.addQuick(pos, pos, lambda * 2 * admittance[0].getValue(pos, pos));
                double mij = -state[pos] * state[j] * (gij * cos + bij * sin);
                double v = -admittance[1].getVA().get(k) * state[pos] * state[pos];
                v -= StateCalByPolar.calBusQ(pos + 1, y, state);
                mij += StateCalByPolar.calBusP(pos + 1, y, state);

                result.addQuick(pos + n, pos, lambda * v / state[pos]);
                result.addQuick(pos + n, pos + n, lambda * -mij);
            } else {
                double v = state[pos] * (admittance[0].getVA().get(k) * cos + admittance[1].getVA().get(k) * sin);
                double v1 = v / state[pos];
                if (j > pos) {
                    result.addQuick(j, pos, lambda * v1);
                } else {
                    result.addQuick(pos, j, lambda * v1);
                }
                double mij = -state[pos] * state[j] * (gij * cos + bij * sin);
                v = state[pos] * state[j] * (admittance[0].getVA().get(k) * sin - admittance[1].getVA().get(k) * cos);
                v1 = v / state[j];

                result.addQuick(j + n, j, lambda * v1);
                result.addQuick(pos + n, j, lambda * -v1);
                result.addQuick(j + n, pos, lambda * v / state[pos]);
                result.addQuick(j + n, j + n, lambda * mij);
                if (pos > j)
                    result.addQuick(pos + n, j + n, lambda * -mij);
                else
                    result.addQuick(j + n, pos + n, lambda * -mij);
            }
            k = admittance[0].getLINK().get(k);
        }
    }

    public static DoubleMatrix2D getHessianBusQ(int pos, YMatrixGetter y, AVector state) {
        return getHessianBusQ(pos, y, state.getValues());
    }

    public static DoubleMatrix2D getHessianBusQ(int pos, YMatrixGetter y, double[] state) {
        ASparseMatrixLink[] admittance = y.getAdmittance();
        int k = admittance[0].getIA()[pos - 1];
        int count = 0;
        while (k != -1) {
            count++;
            k = admittance[0].getLINK().get(k);
        }
        return getHessianBusQ(pos, y, state, 6 * (count - 1) + 3);
    }

    public static SparseDoubleMatrix2D getHessianBusQ(int pos, YMatrixGetter y, AVector state, int size) {
        return getHessianBusQ(pos, y, state.getValues(), size);
    }

    public static SparseDoubleMatrix2D getHessianBusQ(int pos, YMatrixGetter y, double[] state, int size) {
        int n = 2 * y.getAdmittance()[0].getM();
        MySparseDoubleMatrix2D result = new MySparseDoubleMatrix2D(n, n, size, 0.2, 0.5);
        getHessianBusQ(pos, y, state, result);
        return result;
    }

    public static void getHessianBusQ(int pos, YMatrixGetter y, AVector state, MySparseDoubleMatrix2D result) {
        getHessianBusQ(pos, y, state.getValues(), result);
    }

    public static void getHessianBusQ(int pos, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result) {
        getHessianBusQ(pos, y, state, result, 1.0);
    }

    public static void getHessianBusQ(int pos, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result, double lambda) {
        int n = y.getAdmittance()[0].getM();
        pos = pos - 1;
        ASparseMatrixLink[] admittance = y.getAdmittance();
        int k = admittance[0].getIA()[pos];
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double thetaIJ = state[pos + n] - state[j + n];
            double gij = admittance[0].getValue(pos, j);
            double bij = admittance[1].getValue(pos, j);
            double sin = Math.sin(thetaIJ);
            double cos = Math.cos(thetaIJ);
            if (j == pos) {
                result.addQuick(pos, pos, lambda * -2 * admittance[1].getValue(pos, pos));

                double hij = state[pos] * state[j] * (gij * sin - bij * cos);
                double v = -admittance[0].getVA().get(k) * state[pos] * state[pos];
                v += StateCalByPolar.calBusP(pos + 1, y, state);
                hij -= StateCalByPolar.calBusQ(pos + 1, y, state);
                result.addQuick(pos + n, pos, lambda * v / state[pos]);
                result.addQuick(pos + n, pos + n, lambda * hij);
            } else {
                double v = state[pos] * (admittance[0].getVA().get(k) * sin - admittance[1].getVA().get(k) * cos);
                double v1 = v / state[pos];
                if (j > pos)
                    result.addQuick(j, pos, lambda * v1);
                else
                    result.addQuick(pos, j, lambda * v1);

                double hij = state[pos] * state[j] * (gij * sin - bij * cos);
                v = -state[pos] * state[j] * (admittance[0].getVA().get(k) * cos + admittance[1].getVA().get(k) * sin);
                v1 = v / state[j];
                result.addQuick(j + n, j, lambda * v1);
                result.addQuick(pos + n, j, lambda * -v1);
                result.addQuick(j + n, pos, lambda * v / state[pos]);
                result.addQuick(j + n, j + n, lambda * -hij);
                if (pos > j)
                    result.addQuick(pos + n, j + n, lambda * hij);
                else
                    result.addQuick(j + n, pos + n, lambda * hij);
            }
            k = admittance[0].getLINK().get(k);
        }
    }


    public static DoubleMatrix2D getHessianLineFromP(int num, YMatrixGetter Y, AVector state) {
        return getHessianLineFromP(num, Y, state.getValues());
    }

    public static DoubleMatrix2D getHessianLineFromP(int num, YMatrixGetter Y, double[] state) {
        int size = 2 * Y.getAdmittance()[0].getN();
        MySparseDoubleMatrix2D result = new MySparseDoubleMatrix2D(size, size, 9, 0.2, 0.5);
        getHessianLineFromP(num, Y, state, result);
        return result;
    }

    public static void getHessianLineFromP(int num, YMatrixGetter y, AVector state, MySparseDoubleMatrix2D result) {
        getHessianLineFromP(num, y, state.getValues(), result);
    }

    public static void getHessianLineFromP(int num, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result) {
        getHessianLineFromP(num, y, state, result, 1.0);
    }

    public static void getHessianLineFromP(int num, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result, double lambda) {
        int n = y.getAdmittance()[0].getM();
        int[] ij = y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = y.getLineAdmittance(num, YMatrixGetter.LINE_FROM);
        double thetaIJ = state[i + n] - state[j + n];
        double cos = Math.cos(thetaIJ);
        double sin = Math.sin(thetaIJ);

        result.addQuick(i, i, lambda * 2 * (gbg1b1[0] + gbg1b1[2]));
        result.addQuick(i + n, i, lambda * state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        result.addQuick(i + n, j, lambda * state[i] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        result.addQuick(i + n, i + n, lambda * state[i] * state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        result.addQuick(j + n, i, lambda * -state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        result.addQuick(j + n, j, lambda * -state[i] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        result.addQuick(j + n, j + n, lambda * state[i] * state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        if (i > j) {
            result.addQuick(i, j, lambda * (-gbg1b1[0] * cos - gbg1b1[1] * sin));
            result.addQuick(i + n, j + n, lambda * -state[i] * state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        } else {
            result.addQuick(j, i, lambda * (-gbg1b1[0] * cos - gbg1b1[1] * sin));
            result.addQuick(j + n, i + n, lambda * -state[i] * state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        }
    }

    public static DoubleMatrix2D getHessianLineFromQ(int num, YMatrixGetter Y, AVector state) {
        return getHessianLineFromQ(num, Y, state.getValues());
    }

    public static DoubleMatrix2D getHessianLineFromQ(int num, YMatrixGetter Y, double[] state) {
        int size = 2 * Y.getAdmittance()[0].getN();
        MySparseDoubleMatrix2D result = new MySparseDoubleMatrix2D(size, size, 9, 0.2, 0.5);
        getHessianLineFromQ(num, Y, state, result);
        return result;
    }

    public static void getHessianLineFromQ(int num, YMatrixGetter y, AVector state, MySparseDoubleMatrix2D result) {
        getHessianLineFromQ(num, y, state.getValues(), result);
    }

    public static void getHessianLineFromQ(int num, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result) {
        getHessianLineFromQ(num, y, state, result, 1.0);
    }

    public static void getHessianLineFromQ(int num, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result, double lambda) {
        int n = y.getAdmittance()[0].getM();
        int[] ij = y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = y.getLineAdmittance(num, YMatrixGetter.LINE_FROM);
        double thetaIJ = state[i + n] - state[j + n];
        double sin = Math.sin(thetaIJ);
        double cos = Math.cos(thetaIJ);
        result.addQuick(i, i, lambda * -2 * (gbg1b1[1] + gbg1b1[3]));
        result.addQuick(i + n, i, lambda * -state[i] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        result.addQuick(i + n, j, lambda * -state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        result.addQuick(i + n, i + n, lambda * state[i] * state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        result.addQuick(j + n, i, lambda * state[i] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        result.addQuick(j + n, j, lambda * state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        result.addQuick(j + n, j + n, lambda * state[i] * state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        if (i > j) {
            result.addQuick(i, j, lambda * (-gbg1b1[0] * sin + gbg1b1[1] * cos));
            result.addQuick(i + n, j + n, lambda * -state[i] * state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        } else {
            result.addQuick(j, i, lambda * (-gbg1b1[0] * sin + gbg1b1[1] * cos));
            result.addQuick(j + n, i + n, lambda * -state[i] * state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        }
    }

    public static DoubleMatrix2D getHessianLineToP(int num, YMatrixGetter Y, AVector state) {
        return getHessianLineToP(num, Y, state.getValues());
    }

    public static DoubleMatrix2D getHessianLineToP(int num, YMatrixGetter Y, double[] state) {
        int size = 2 * Y.getAdmittance()[0].getN();
        MySparseDoubleMatrix2D result = new MySparseDoubleMatrix2D(size, size, 9, 0.2, 0.5);
        getHessianLineToP(num, Y, state, result);
        return result;
    }

    public static void getHessianLineToP(int num, YMatrixGetter y, AVector state, MySparseDoubleMatrix2D result) {
        getHessianLineToP(num, y, state.getValues(), result);
    }

    public static void getHessianLineToP(int num, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result) {
        getHessianLineToP(num, y, state, result, 1.0);
    }

    public static void getHessianLineToP(int num, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result, double lambda) {
        int n = y.getAdmittance()[0].getM();
        int[] ij = y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = y.getLineAdmittance(num, YMatrixGetter.LINE_TO);
        double thetaIJ = state[j + n] - state[i + n];
        double cos = Math.cos(thetaIJ);
        double sin = Math.sin(thetaIJ);
        result.addQuick(j, j, lambda * 2 * (gbg1b1[0] + gbg1b1[2]));
        result.addQuick(j + n, j, lambda * state[i] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        result.addQuick(j + n, i, lambda * state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        result.addQuick(j + n, j + n, lambda * state[i] * state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        result.addQuick(i + n, j, lambda * -state[i] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        result.addQuick(i + n, i, lambda * -state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        result.addQuick(i + n, i + n, lambda * state[i] * state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));

        if (i > j) {
            result.addQuick(i, j, lambda * (-gbg1b1[0] * cos - gbg1b1[1] * sin));
            result.addQuick(i + n, j + n, lambda * -state[i] * state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        } else {
            result.addQuick(j, i, lambda * (-gbg1b1[0] * cos - gbg1b1[1] * sin));
            result.addQuick(j + n, i + n, lambda * -state[i] * state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        }
    }

    public static void getHessianLineToQ(int num, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result) {
        getHessianLineToQ(num, y, state, result, 1.0);
    }

    public static void getHessianLineToQ(int num, YMatrixGetter y, double[] state, MySparseDoubleMatrix2D result, double lambda) {
        int n = y.getAdmittance()[0].getM();
        int[] ij = y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = y.getLineAdmittance(num, YMatrixGetter.LINE_TO);
        double thetaIJ = state[j + n] - state[i + n];
        double sin = Math.sin(thetaIJ);
        double cos = Math.cos(thetaIJ);
        result.addQuick(j, j, lambda * -2 * (gbg1b1[1] + gbg1b1[3]));
        result.addQuick(j + n, j, lambda * -state[i] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        result.addQuick(j + n, i, lambda * -state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        result.addQuick(j + n, j + n, lambda * state[i] * state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        result.addQuick(i + n, j, lambda * state[i] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        result.addQuick(i + n, i, lambda * state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        result.addQuick(i + n, i + n, lambda * state[i] * state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        if (i > j) {
            result.addQuick(i, j, lambda * (-gbg1b1[0] * sin + gbg1b1[1] * cos));
            result.addQuick(i + n, j + n, lambda * -state[i] * state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        } else {
            result.addQuick(j, i, lambda * (-gbg1b1[0] * sin + gbg1b1[1] * cos));
            result.addQuick(j + n, i + n, lambda * -state[i] * state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        }
    }

    public static void getHessianStruc(int[] zeroPBuses, int[] zeroQBuses, YMatrixGetter y, MySparseDoubleMatrix2D result) {
        for (int bus : zeroPBuses)
            getStrucBusPQ(bus, y, result);
        for (int bus : zeroQBuses)
            getStrucBusPQ(bus, y, result);
    }

    public static void getHessian(int[] zeroPBuses, int[] zeroQBuses, YMatrixGetter y,
                                  double[] state, MySparseDoubleMatrix2D result, double[] lambda, int index) {
        for (int i = 0; i < zeroPBuses.length; i++, index++)
            getHessianBusP(zeroPBuses[i], y, state, result, lambda[index]);
        for (int i = 0; i < zeroQBuses.length; i++, index++)
            getHessianBusQ(zeroQBuses[i], y, state, result, lambda[index]);
    }
}
