package zju.util;

import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;

import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-6-6
 */
public class HessianMakerPCSlow implements MeasTypeCons {

    public static ASparseMatrixLink[] getHessianOfFullState(MeasVector meas, YMatrixGetter admittanceGetter, AVector state) {
        ASparseMatrixLink[] result = new ASparseMatrixLink[meas.getZ().getN()];
        int index = 0;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++) {
                        result[index] = new ASparseMatrixLink(0, 0);
                    }
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        result[index] = new ASparseMatrixLink(0, 0);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        result[index] = new ASparseMatrixLink(0, 0);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        result[index] = new ASparseMatrixLink(0, 0);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        result[index] = getHessianLineFromP(num, admittanceGetter, state);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        result[index] = getHessianLineFromQ(num, admittanceGetter, state);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        result[index] = getHessianLineToP(num, admittanceGetter, state);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        result[index] = getHessianLineToQ(num, admittanceGetter, state);
                    }
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
        return result;
    }

    public static ASparseMatrixLink[] getHessian(MeasVector meas, YMatrixGetter admittanceGetter, ASparseMatrixLink jacobian, AVector state) {
        ASparseMatrixLink[] result = new ASparseMatrixLink[jacobian.getM()];
        int index = 0;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++) {
                        result[index] = new ASparseMatrixLink(state.getN(), state.getN());
                    }
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        result[index] = new ASparseMatrixLink(state.getN(), state.getN());
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        result[index] = getHessianBusP(num, index, admittanceGetter, jacobian, state);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int num = meas.getBus_q_pos()[i];//num starts from 1
                        result[index] = getHessianBusQ(num, index, admittanceGetter, jacobian, state);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        result[index] = getHessianLineFromP(num, admittanceGetter, state);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        result[index] = getHessianLineFromQ(num, admittanceGetter, state);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        result[index] = getHessianLineToP(num, admittanceGetter, state);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        result[index] = getHessianLineToQ(num, admittanceGetter, state);
                    }
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
        return result;
    }

    public static ASparseMatrixLink[] getHessian(MeasVector meas, YMatrixGetter addmittanceGetter, AVector state, int[] transformer, double[] ratios) {
        ASparseMatrixLink[] result = new ASparseMatrixLink[meas.getZ().getN()];
        int index = 0;
        int n = state.getN() / 2;
        int size = state.getN() + transformer.length;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++) {
                        result[index] = new ASparseMatrixLink(size, size);
                    }
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        result[index] = new ASparseMatrixLink(size, size);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        result[index] = new ASparseMatrixLink(size, size);
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        for (int j = state.getN(); j < size; j++) {
                            int branchId = transformer[j - state.getN()];
                            double t = ratios[j - state.getN()];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (ij[0] == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                result[index].setValue(j, ij[0] - 1, 4 * ui * gbg1b1[0] - uj * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[1] - 1, -ui * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[0] - 1 + n, ui * uj * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[1] - 1 + n, -ui * uj * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t);
                                result[index].setValue(j, j, 2 * ui * ui * gbg1b1[0] / t);
                            } else if (ij[1] == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_TO);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                result[index].setValue(j, ij[0] - 1, -uj * (gbg1b1[0] * cos - gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[1] - 1, -ui * (gbg1b1[0] * cos - gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[0] - 1 + n, ui * uj * (gbg1b1[0] * sin + gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[1] - 1 + n, -ui * uj * (gbg1b1[0] * sin + gbg1b1[1] * cos) / t);
                            }
                        }
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        result[index] = new ASparseMatrixLink(size, size);
                        int num = meas.getBus_q_pos()[i];//num starts from 1
                        for (int j = state.getN(); j < size; j++) {
                            int branchId = transformer[j - state.getN()];
                            double t = ratios[j - state.getN()];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (ij[0] == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                result[index].setValue(j, ij[0] - 1, -4 * ui * gbg1b1[1] - uj * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[1] - 1, -ui * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[0] - 1 + n, -ui * uj * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[1] - 1 + n, ui * uj * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t);
                                result[index].setValue(j, j, -2 * ui * ui * gbg1b1[1] / t);
                            } else if (ij[1] == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_TO);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                result[index].setValue(j, ij[0] - 1, uj * (gbg1b1[0] * sin + gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[1] - 1, ui * (gbg1b1[0] * sin + gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[0] - 1 + n, ui * uj * (gbg1b1[0] * cos - gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[1] - 1 + n, -ui * uj * (gbg1b1[0] * cos - gbg1b1[1] * sin) / t);
                            }
                        }
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        result[index] = new ASparseMatrixLink(size, size);
                        for (int j = state.getN(); j < size; j++) {
                            int branchId = transformer[j - state.getN()];
                            double t = ratios[j - state.getN()];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (num == branchId) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                result[index].setValue(j, ij[0] - 1, 4 * ui * gbg1b1[0] - uj * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[1] - 1, -ui * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[0] - 1 + n, ui * uj * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[1] - 1 + n, -ui * uj * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t);
                                result[index].setValue(j, j, 2 * ui * ui * gbg1b1[0] / t);
                                break;
                            }
                        }
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        result[index] = new ASparseMatrixLink(size, size);
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        for (int j = state.getN(); j < size; j++) {
                            int branchId = transformer[j - state.getN()];
                            double t = ratios[j - state.getN()];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (branchId == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                result[index].setValue(j, ij[0] - 1, -4 * ui * gbg1b1[1] - uj * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[1] - 1, -ui * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[0] - 1 + n, -ui * uj * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[1] - 1 + n, ui * uj * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t);
                                result[index].setValue(j, j, -2 * ui * ui * gbg1b1[1] / t);
                                break;
                            }
                        }
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        result[index] = new ASparseMatrixLink(size, size);
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        for (int j = state.getN(); j < size; j++) {
                            int branchId = transformer[j - state.getN()];
                            double t = ratios[j - state.getN()];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (num == branchId) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_TO);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                result[index].setValue(j, ij[0] - 1, -uj * (gbg1b1[0] * cos - gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[1] - 1, -ui * (gbg1b1[0] * cos - gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[0] - 1 + n, ui * uj * (gbg1b1[0] * sin + gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[1] - 1 + n, -ui * uj * (gbg1b1[0] * sin + gbg1b1[1] * cos) / t);
                                break;
                            }
                        }
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        result[index] = new ASparseMatrixLink2D(size, size);
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        for (int j = state.getN(); j < size; j++) {
                            int branchId = transformer[j - state.getN()];
                            double t = ratios[j - state.getN()];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (branchId == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_TO);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                result[index].setValue(j, ij[0] - 1, uj * (gbg1b1[0] * sin + gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[1] - 1, ui * (gbg1b1[0] * sin + gbg1b1[1] * cos) / t);
                                result[index].setValue(j, ij[0] - 1 + n, ui * uj * (gbg1b1[0] * cos - gbg1b1[1] * sin) / t);
                                result[index].setValue(j, ij[1] - 1 + n, -ui * uj * (gbg1b1[0] * cos - gbg1b1[1] * sin) / t);
                                break;
                            }
                        }
                    }
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
        return result;
    }

    public static ASparseMatrixLink getHessianBusP(int pos, int index, YMatrixGetter admittanceGetter, ASparseMatrixLink jacobian, AVector state) {
        ASparseMatrixLink result = new ASparseMatrixLink(state.getN(), state.getN());
        getHessianBusP(pos, index, admittanceGetter, jacobian, state, result);
        return result;
    }

    public static ASparseMatrixLink[] getHessianOfAllBusP(YMatrixGetter admittanceGetter, AVector state) {
        IEEEDataIsland island = admittanceGetter.getIsland();
        ASparseMatrixLink result = new ASparseMatrixLink(state.getN(), state.getN());
        for (BusData bus : island.getBuses()) {
            if (bus.getType() == 2 || bus.getType() == 3) continue;//todo: why?
            ASparseMatrixLink jac = JacobianMakerPCSlow.getJacobianOfBusP(bus.getBusNumber(), admittanceGetter, state);
            getHessianBusP(bus.getBusNumber(), 0, admittanceGetter, jac, state, result);
            //result.printOnScreen();
        }
        return new ASparseMatrixLink[]{result};
    }

    public static void getHessianBusP(int pos, int index, YMatrixGetter admittanceGetter, ASparseMatrixLink jacobian, AVector state, ASparseMatrixLink result) {
        int n = state.getN() / 2;
        pos = pos - 1;
        ASparseMatrixLink[] admittance = admittanceGetter.getAdmittance();
        int k = jacobian.getIA()[index];
        while (k != -1) {
            int j = jacobian.getJA().get(k);
            double v = jacobian.getVA().get(k);
            k = jacobian.getLINK().get(k);
            if (j == pos) {
                result.setValue(pos, pos, 2 * admittance[0].getValue(pos, pos), true);
            } else if (j < n) {
                double v1 = v / state.getValue(pos);
                result.setValue(j, pos, v1, true);
                result.setValue(pos, j, v1, true);
            } else {
                double v1 = v / state.getValue(j - n);
                result.setValue(j, j - n, v1, true);
                result.setValue(pos + n, j - n, -v1, true);
                result.setValue(j, pos, v / state.getValue(pos), true);
                double gij = admittance[0].getValue(pos, j - n);
                double bij = admittance[1].getValue(pos, j - n);
                double cos = Math.cos(state.getValue(pos + n) - state.getValue(j));
                double sin = Math.sin(state.getValue(pos + n) - state.getValue(j));
                double mij = -state.getValue(pos) * state.getValue(j - n) * (gij * cos + bij * sin);
                if (pos == j - n)
                    mij += StateCalByPolar.calBusP(pos + 1, admittanceGetter, state);
                result.setValue(j, j, mij, true);
                result.setValue(pos + n, j, -mij, true);
                result.setValue(j, pos + n, -mij, true);
            }
        }
    }

    public static ASparseMatrixLink getHessianBusQ(int num, int index, YMatrixGetter admittanceGetter, ASparseMatrixLink jacobian, AVector state) {
        ASparseMatrixLink result = new ASparseMatrixLink(state.getN(), state.getN());
        getHessianBusQ(num, index, admittanceGetter, jacobian, state, result);
        return result;
    }

    public static void getHessianBusQ(int num, int index, YMatrixGetter admittanceGetter, ASparseMatrixLink jacobian, AVector state, ASparseMatrixLink result) {
        int n = state.getN() / 2;
        num = num - 1;
        ASparseMatrixLink[] admittance = admittanceGetter.getAdmittance();
        int k = jacobian.getIA()[index];
        while (k != -1) {
            int j = jacobian.getJA().get(k);
            double v = jacobian.getVA().get(k);
            k = jacobian.getLINK().get(k);
            if (j == num) {
                result.setValue(num, num, -2 * admittance[1].getValue(num, num), true);
            } else if (j < n) {
                double v1 = v / state.getValue(num);
                result.setValue(j, num, v1, true);
                result.setValue(num, j, v1, true);
            } else {
                double v1 = v / state.getValue(j - n);
                result.setValue(j, j - n, v1, true);
                result.setValue(num + n, j - n, -v1, true);
                result.setValue(j, num, v / state.getValue(num), true);
                double gij = admittance[0].getValue(num, j - n);
                double bij = admittance[1].getValue(num, j - n);
                double cos = Math.cos(state.getValue(num + n) - state.getValue(j));
                double sin = Math.sin(state.getValue(num + n) - state.getValue(j));
                double hij = state.getValue(num) * state.getValue(j - n) * (gij * sin - bij * cos);
                if (num == j - n)
                    hij -= StateCalByPolar.calBusQ(num + 1, admittanceGetter, state);
                result.setValue(j, j, -hij, true);
                result.setValue(num + n, j, hij, true);
                result.setValue(j, num + n, hij, true);
            }
        }
    }

    public static ASparseMatrixLink getHessianLineFromP(int num, YMatrixGetter admittanceGetter, AVector state) {
        ASparseMatrixLink result = new ASparseMatrixLink(state.getN(), state.getN());
        getHessianLineFromP(num, admittanceGetter, state, result);
        return result;
    }

    public static void getHessianLineFromP(int num, YMatrixGetter admittanceGetter, AVector state, ASparseMatrixLink result) {
        int n = state.getN() / 2;
        int[] ij = admittanceGetter.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = admittanceGetter.getLineAdmittance(num, YMatrixGetter.LINE_FROM);
        double thetaIJ = state.getValue(i + n) - state.getValue(j + n);
        double cos = Math.cos(thetaIJ);
        double sin = Math.sin(thetaIJ);
        result.setValue(i, i, 2 * (gbg1b1[0] + gbg1b1[2]), true);
        result.setValue(i, j, -gbg1b1[0] * cos - gbg1b1[1] * sin, true);
        result.setValue(i, i + n, state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(i, j + n, -state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);

        result.setValue(j, i, -gbg1b1[0] * cos - gbg1b1[1] * sin, true);
        result.setValue(j, i + n, state.getValue(i) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(j, j + n, -state.getValue(i) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);

        result.setValue(i + n, i, state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(i + n, j, state.getValue(i) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(i + n, i + n, state.getValue(i) * state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(i + n, j + n, -state.getValue(i) * state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);

        result.setValue(j + n, i, -state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(j + n, j, -state.getValue(i) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(j + n, i + n, -state.getValue(i) * state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(j + n, j + n, state.getValue(i) * state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
    }

    public static ASparseMatrixLink getHessianLineFromQ(int num, YMatrixGetter admittanceGetter, AVector state) {
        ASparseMatrixLink result = new ASparseMatrixLink(state.getN(), state.getN());
        getHessianLineFromQ(num, admittanceGetter, state, result);
        return result;
    }

    public static void getHessianLineFromQ(int num, YMatrixGetter admittanceGetter, AVector state, ASparseMatrixLink result) {
        int n = state.getN() / 2;
        int[] ij = admittanceGetter.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = admittanceGetter.getLineAdmittance(num, YMatrixGetter.LINE_FROM);
        double thetaIJ = state.getValue(i + n) - state.getValue(j + n);
        double sin = Math.sin(thetaIJ);
        double cos = Math.cos(thetaIJ);
        result.setValue(i, i, -2 * (gbg1b1[1] + gbg1b1[3]), true);
        result.setValue(i, j, -gbg1b1[0] * sin + gbg1b1[1] * cos, true);
        result.setValue(i, i + n, -state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(i, j + n, state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);

        result.setValue(j, i, -gbg1b1[0] * sin + gbg1b1[1] * cos, true);
        result.setValue(j, i + n, -state.getValue(i) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(j, j + n, state.getValue(i) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);

        result.setValue(i + n, i, -state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(i + n, j, -state.getValue(i) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(i + n, i + n, state.getValue(i) * state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(i + n, j + n, -state.getValue(i) * state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);

        result.setValue(j + n, i, state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(j + n, j, state.getValue(i) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(j + n, i + n, -state.getValue(i) * state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(j + n, j + n, state.getValue(i) * state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
    }

    public static ASparseMatrixLink getHessianLineToP(int num, YMatrixGetter admittanceGetter, AVector state) {
        ASparseMatrixLink result = new ASparseMatrixLink(state.getN(), state.getN());
        getHessianLineToP(num, admittanceGetter, state, result);
        return result;
    }

    public static void getHessianLineToP(int num, YMatrixGetter admittanceGetter, AVector state, ASparseMatrixLink result) {
        int n = state.getN() / 2;
        int[] ij = admittanceGetter.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = admittanceGetter.getLineAdmittance(num, YMatrixGetter.LINE_TO);
        double thetaIJ = state.getValue(j + n) - state.getValue(i + n);
        double cos = Math.cos(thetaIJ);
        double sin = Math.sin(thetaIJ);
        result.setValue(j, j, 2 * (gbg1b1[0] + gbg1b1[2]), true);
        result.setValue(j, i, -gbg1b1[0] * cos - gbg1b1[1] * sin, true);
        result.setValue(j, j + n, state.getValue(i) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(j, i + n, -state.getValue(i) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);

        result.setValue(i, j, -gbg1b1[0] * cos - gbg1b1[1] * sin, true);
        result.setValue(i, j + n, state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(i, i + n, -state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);

        result.setValue(j + n, j, state.getValue(i) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(j + n, i, state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(j + n, j + n, state.getValue(i) * state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(j + n, i + n, -state.getValue(i) * state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);

        result.setValue(i + n, j, -state.getValue(i) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(i + n, i, -state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(i + n, j + n, -state.getValue(i) * state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(i + n, i + n, state.getValue(i) * state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
    }

    public static ASparseMatrixLink getHessianLineToQ(int num, YMatrixGetter admittanceGetter, AVector state) {
        ASparseMatrixLink result = new ASparseMatrixLink(state.getN(), state.getN());
        getHessianLineToQ(num, admittanceGetter, state, result);
        return result;
    }

    public static void getHessianLineToQ(int num, YMatrixGetter admittanceGetter, AVector state, ASparseMatrixLink result) {
        int n = state.getN() / 2;
        int[] ij = admittanceGetter.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = admittanceGetter.getLineAdmittance(num, YMatrixGetter.LINE_TO);
        double thetaIJ = state.getValue(j + n) - state.getValue(i + n);
        double sin = Math.sin(thetaIJ);
        double cos = Math.cos(thetaIJ);
        result.setValue(j, j, -2 * (gbg1b1[1] + gbg1b1[3]), true);
        result.setValue(j, i, -gbg1b1[0] * sin + gbg1b1[1] * cos, true);
        result.setValue(j, j + n, -state.getValue(i) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(j, i + n, state.getValue(i) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);

        result.setValue(i, j, -gbg1b1[0] * sin + gbg1b1[1] * cos, true);
        result.setValue(i, j + n, -state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(i, i + n, state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);

        result.setValue(j + n, j, -state.getValue(i) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(j + n, i, -state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(j + n, j + n, state.getValue(i) * state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(j + n, i + n, -state.getValue(i) * state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);

        result.setValue(i + n, j, state.getValue(i) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(i + n, i, state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(i + n, j + n, -state.getValue(i) * state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(i + n, i + n, state.getValue(i) * state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
    }

    public static ASparseMatrixLink[] getHessian(List<Integer> linkBuses, YMatrixGetter admittanceGetter, ASparseMatrixLink jacobian, AVector state) {
        int index = 0;
        ASparseMatrixLink[] result = new ASparseMatrixLink[2 * linkBuses.size()];
        for (int i = 0; i < linkBuses.size(); i++, index++)
            result[index] = getHessianBusP(linkBuses.get(i), index, admittanceGetter, jacobian, state);
        for (int i = 0; i < linkBuses.size(); i++, index++)
            result[index] = getHessianBusQ(linkBuses.get(i), index, admittanceGetter, jacobian, state);
        return result;
    }
}
