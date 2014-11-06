package zju.util;

import org.apache.log4j.Logger;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink;
import zju.matrix.AVector;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-19
 */
public class StateCalByRC implements MeasTypeCons {

    private static Logger log = Logger.getLogger(StateCalByRC.class);

    public static double[] calLineCurrent(int branchId, YMatrixGetter admittanceGetter, AVector state, int fromOrTo) {
        return calLineCurrent(branchId, admittanceGetter, state.getValues(), fromOrTo);
    }

    /*
   * calculate current in cartesian system
   * */
    public static double[] calLineCurrent(int branchId, YMatrixGetter admittanceGetter, double[] state, int fromOrTo) {
        IEEEDataIsland island = admittanceGetter.getIsland();
        int n = island.getBuses().size();
        BranchData branch = island.getId2branch().get(branchId);
        int fromBus = branch.getTapBusNumber() - 1;
        int toBus = branch.getZBusNumber() - 1;
        double[] eifiejfj;
        double[] gbg1b1;
        if (fromOrTo == YMatrixGetter.LINE_FROM) {
            eifiejfj = new double[]{state[fromBus], state[fromBus + n], state[toBus], state[toBus + n]};
            gbg1b1 = admittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
        } else {
            eifiejfj = new double[]{state[toBus], state[toBus + n], state[fromBus], state[fromBus + n]};
            gbg1b1 = admittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_TO);
        }
        return calLineCurrent(gbg1b1, eifiejfj);
    }

    public static double[] calLineCurrent(double[] gbg1b1, double[] eifiejfj) {
        double ei = eifiejfj[0];
        double fi = eifiejfj[1];
        double ej = eifiejfj[2];
        double fj = eifiejfj[3];
        double real = (ei - ej) * gbg1b1[0] - (fi - fj) * gbg1b1[1] + ei * gbg1b1[2] - fi * gbg1b1[3];
        double imaginary = (ei - ej) * gbg1b1[1] + (fi - fj) * gbg1b1[0] + ei * gbg1b1[3] + fi * gbg1b1[2];
        return new double[]{real, imaginary};
    }

    public static double[] calAnotherVoltage(double[] gbg1b1, double[] ui) {
        double realUi = ui[0];
        double imageUi = ui[1];
        double realI = ui[3];
        double imagI = ui[4];
        double g = gbg1b1[0];
        double b = gbg1b1[1];
        double yc = gbg1b1[3] / 2.0;
        double ej = (-g * realI - b * imagI + (g * g + b * b) * realUi - yc * (g * imageUi - b * realUi)) / (g * g + b * b);
        double fj = (b * realI - g * imagI + (g * g + b * b) * imageUi + yc * (b * imageUi + g * realUi)) / (g * g + b * b);
        return new double[]{ej, fj};
    }

    public static void getEstimatedZ_UI(MeasVector meas, YMatrixGetter y, AVector state) {
        getEstimatedZ_UI(meas, y, state.getValues());
    }

    public static void getEstimatedZ_UI(MeasVector meas, YMatrixGetter y, double[] state) {
        AVector result = meas.getZ_estimate();
        int index = 0;
        int n = y.getAdmittance()[0].getM();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    //todo: not finished...
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        int num = meas.getBus_v_pos()[i] - 1;//num starts from 0
                        double e = state[num];
                        double f = state[num + n];
                        result.setValue(index, e * e + f * f);//todo:
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        double value = calBusP_UI(num, n, state);
                        result.setValue(index, value);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int num = meas.getBus_q_pos()[i];//num starts from 1
                        double value = calBusQ_UI(num, n, state);
                        result.setValue(index, value);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        double v = calLinePFrom(num, y, state);
                        result.setValue(index, v);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        double v = calLineQFrom(num, y, state);
                        result.setValue(index, v);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        double v = calLinePTo(num, y, state);
                        result.setValue(index, v);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        double v = calLineQTo(num, y, state);
                        result.setValue(index, v);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    public static void getEstimatedZ_U(MeasVector meas, YMatrixGetter y, AVector state) {
        getEstimatedZ_U(meas, y, state.getValues());
    }

    public static void getEstimatedZ_U(MeasVector meas, YMatrixGetter y, double[] state) {
        AVector result = meas.getZ_estimate();
        int index = 0;
        int n = y.getAdmittance()[0].getM();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    //todo: not finished...
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        int num = meas.getBus_v_pos()[i] - 1;//num starts from 0
                        double e = state[num];
                        double f = state[num + n];
                        result.setValue(index, e * e + f * f);//todo:
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        double value = calBusP_U(num, y, state);
                        result.setValue(index, value);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int num = meas.getBus_q_pos()[i];//num starts from 1
                        double value = calBusQ_U(num, y, state);
                        result.setValue(index, value);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        double v = calLinePFrom(num, y, state);
                        result.setValue(index, v);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        double v = calLineQFrom(num, y, state);
                        result.setValue(index, v);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        double v = calLinePTo(num, y, state);
                        result.setValue(index, v);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        double v = calLineQTo(num, y, state);
                        result.setValue(index, v);
                    }
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
    }

    public static double calBusP_U(int num, YMatrixGetter y, AVector state) {
        return calBusP_U(num, y, state.getValues());
    }

    public static double calBusP_U(int num, YMatrixGetter y, double[] state) {
        num -= 1;
        ASparseMatrixLink[] admittance = y.getAdmittance();
        int n = admittance[0].getM();
        double value = 0;
        int k = admittance[0].getIA()[num];
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double ei = state[num];
            double fi = state[num + n];
            double ej = state[j];
            double fj = state[j + n];
            double gij = admittance[0].getVA().get(k);
            double bij = admittance[1].getVA().get(k);
            value += gij * ei * ej - bij * ei * fj + bij * ej * fi + gij * fi * fj;
            k = admittance[0].getLINK().get(k);
        }
        return value;
    }

    public static double calBusQ_U(int num, YMatrixGetter y, AVector state) {
        return calBusQ_U(num, y, state.getValues());
    }

    public static double calBusQ_U(int num, YMatrixGetter y, double[] state) {
        num -= 1;
        ASparseMatrixLink[] admittance = y.getAdmittance();
        int n = admittance[0].getM();
        double value = 0;
        int k = admittance[0].getIA()[num];
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double ei = state[num];
            double fi = state[num + n];
            double ej = state[j];
            double fj = state[j + n];
            double gij = admittance[0].getVA().get(k);
            double bij = admittance[1].getVA().get(k);
            value += -bij * ei * ej - gij * ei * fj + gij * ej * fi - bij * fi * fj;
            k = admittance[0].getLINK().get(k);
        }
        return value;
    }

    public static double calBusP_UI(int num, int n, AVector state) {
        return calBusP_UI(num, n, state.getValues());
    }

    /**
     * @param num   bus's num in admittance matrix and the least is 1
     * @param state all buses' Ux, Uy, Ix, Iy
     * @return bus's injection active power
     */
    public static double calBusP_UI(int num, int n, double[] state) {
        num -= 1;
        return state[num] * state[num + 2 * n] + state[num + n] * state[num + 3 * n];
    }

    public static double calBusQ_UI(int num, int n, AVector state) {
        return calBusQ_UI(num, n, state.getValues());
    }

    /**
     * @param num   bus's num in admittance matrix and the least is 1
     * @param state all buses' Ux, Uy, Ix, Iy
     * @return bus's injection reactive power
     */
    public static double calBusQ_UI(int num, int n, double[] state) {
        num -= 1;
        return -state[num] * state[num + 3 * n] + state[num + n] * state[num + 2 * n];

    }

    public static double calLinePFrom(int branchId, YMatrixGetter Y, AVector state) {
        return calLinePFrom(branchId, Y, state.getValues());
    }

    /**
     * @param branchId branch's branchId ieee common format data and the least is 1
     * @param Y        admittance matrix getter
     * @param state    all buses' Ux, Uy, Ix, Iy
     * @return active power of head of the branch
     */
    public static double calLinePFrom(int branchId, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(branchId);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
        double g = gbg1b1[0];
        double b = gbg1b1[1];
        double g1 = gbg1b1[2];
        double ei = state[i];
        double ej = state[j];
        double fi = state[i + n];
        double fj = state[j + n];
        return (g + g1) * (ei * ei + fi * fi) + b * fj * ei - g * ej * ei - b * ej * fi - g * fj * fi;
    }

    public static double calLinePFrom2(int branchId, YMatrixGetter Y, AVector state) {
        return calLinePFrom2(branchId, Y, state.getValues());
    }

    /**
     * @param branchId branch's branchId ieee common format data and the least is 1
     * @param Y        admittance matrix getter
     * @param state    all buses' Ux, Uy, Ix, Iy
     * @return active power of head of the branch
     */
    public static double calLinePFrom2(int branchId, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(branchId);
        int i = ij[0] - 1;
        double ei = state[i];
        double fi = state[i + n];
        double[] c = calLineCurrent(branchId, Y, state, YMatrixGetter.LINE_FROM);
        return ei * c[0] + fi * c[1];
    }

    public static double calLineQFrom(int num, YMatrixGetter Y, AVector state) {
        return calLineQFrom(num, Y, state.getValues());
    }

    /**
     * @param num   branch's num ieee common format data and the least is 1
     * @param Y     admittance matrix getter
     * @param state all buses' Ux, Uy, Ix, Iy
     * @return inactive power of head of the branch
     */
    public static double calLineQFrom(int num, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(num, YMatrixGetter.LINE_FROM);
        double ei = state[i];
        double fi = state[i + n];
        double ej = state[j];
        double fj = state[j + n];
        double g = gbg1b1[0];
        double b = gbg1b1[1];
        double b1 = gbg1b1[3];
        return -(b + b1) * (ei * ei + fi * fi) + b * ej * ei + g * fj * ei - g * ej * fi + b * fj * fi;
    }

    public static double calLineQFrom2(int branchId, YMatrixGetter Y, AVector state) {
        return calLineQFrom2(branchId, Y, state.getValues());
    }

    /**
     * @param branchId branch's branchId ieee common format data and the least is 1
     * @param Y        admittance matrix getter
     * @param state    all buses' Ux, Uy, Ix, Iy
     * @return active power of head of the branch
     */
    public static double calLineQFrom2(int branchId, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(branchId);
        int i = ij[0] - 1;
        double ei = state[i];
        double fi = state[i + n];
        double[] c = calLineCurrent(branchId, Y, state, YMatrixGetter.LINE_FROM);
        return -ei * c[1] + fi * c[0];
    }

    public static double calLinePTo(int branchId, YMatrixGetter Y, AVector state) {
        return calLinePTo(branchId, Y, state.getValues());
    }

    /**
     * @param branchId branch's num ieee common format data and the least is 1
     * @param Y        admittance matrix getter
     * @param state    all buses' Ux, Uy, Ix, Iy
     * @return active power of tail of the branch
     */
    public static double calLinePTo(int branchId, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(branchId);
        int j = ij[0] - 1;
        int i = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(branchId, YMatrixGetter.LINE_TO);
        double g = gbg1b1[0];
        double b = gbg1b1[1];
        double g1 = gbg1b1[2];
        double ei = state[i];
        double ej = state[j];
        double fi = state[i + n];
        double fj = state[j + n];
        return (g + g1) * (ei * ei + fi * fi) + b * fj * ei - g * ej * ei - b * ej * fi - g * fj * fi;
    }

    public static double calLinePTo2(int branchId, YMatrixGetter Y, AVector state) {
        return calLinePTo2(branchId, Y, state.getValues());
    }

    /**
     * @param branchId branch's branchId ieee common format data and the least is 1
     * @param Y        admittance matrix getter
     * @param state    all buses' Ux, Uy, Ix, Iy
     * @return active power of head of the branch
     */
    public static double calLinePTo2(int branchId, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(branchId);
        int i = ij[1] - 1;
        double ei = state[i];
        double fi = state[i + n];
        double[] c = calLineCurrent(branchId, Y, state, YMatrixGetter.LINE_TO);
        return ei * c[0] + fi * c[1];
    }

    public static double calLineQTo(int num, YMatrixGetter Y, AVector state) {
        return calLineQTo(num, Y, state.getValues());
    }

    /**
     * @param num   branch's num ieee common format data and the least is 1
     * @param Y     admittance matrix getter
     * @param state all buses' Ux, Uy, Ix, Iy
     * @return inactive power of tail of the branch
     */
    public static double calLineQTo(int num, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(num);
        int j = ij[0] - 1;
        int i = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(num, YMatrixGetter.LINE_TO);
        double ei = state[i];
        double fi = state[i + n];
        double ej = state[j];
        double fj = state[j + n];
        double g = gbg1b1[0];
        double b = gbg1b1[1];
        double b1 = gbg1b1[3];
        return -(b + b1) * (ei * ei + fi * fi) + b * ej * ei + g * fj * ei - g * ej * fi + b * fj * fi;
    }

    public static double calLineQTo2(int branchId, YMatrixGetter Y, AVector state) {
        return calLineQTo2(branchId, Y, state.getValues());
    }

    /**
     * @param branchId branch's branchId ieee common format data and the least is 1
     * @param Y        admittance matrix getter
     * @param state    all buses' Ux, Uy, Ix, Iy
     * @return active power of head of the branch
     */
    public static double calLineQTo2(int branchId, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(branchId);
        int i = ij[1] - 1;
        double ei = state[i];
        double fi = state[i + n];
        double[] c = calLineCurrent(branchId, Y, state, YMatrixGetter.LINE_TO);
        return -ei * c[1] + fi * c[0];
    }

    public static double calEstValue(int type, int pos, YMatrixGetter Y, AVector state) {
        return calEstValue(type, pos, Y, state.getValues());
    }

    public static double calEstValue(int type, int pos, YMatrixGetter admittanceGetter, double[] state) {
        int n = admittanceGetter.getAdmittance()[0].getM();
        switch (type) {
            case TYPE_BUS_ANGLE:
                return state[pos + n - 1];
            case TYPE_BUS_VOLOTAGE:
                return state[pos - 1];
            case TYPE_BUS_ACTIVE_POWER:
                return calBusP_UI(pos, n, state);
            case TYPE_BUS_REACTIVE_POWER:
                return calBusQ_UI(pos, n, state);
            case TYPE_LINE_FROM_ACTIVE:
                return calLinePFrom(pos, admittanceGetter, state);
            case TYPE_LINE_FROM_REACTIVE:
                return calLineQFrom(pos, admittanceGetter, state);
            case TYPE_LINE_TO_ACTIVE:
                return calLinePTo(pos, admittanceGetter, state);
            case TYPE_LINE_TO_REACTIVE:
                return calLineQTo(pos, admittanceGetter, state);
            default:
                log.warn("Unsupported measure type: " + type);
                return 0;
        }
    }

    public static void calI(YMatrixGetter y, double[] u, double[] c) {
        calI(y, u, c, 0);
    }

    public static void calI(YMatrixGetter y, double[] u, double[] c, int start) {
        int n = y.getAdmittance()[0].getN();
        for(int i = 0; i < 2 * n; i++)
            c[i + start] = 0.0;
        for (int i = 0; i < y.getAdmittance()[0].getM(); i++) {
            int k = y.getAdmittance()[0].getIA()[i];
            while (k != -1) {
                int j = y.getAdmittance()[0].getJA().get(k);
                double e = u[j];
                double f = u[j + n];
                double g = y.getAdmittance()[0].getVA().get(k);
                double b = y.getAdmittance()[1].getVA().get(k);
                c[i + start] += e * g - f * b;
                c[i + start + n] += e * b + f * g;
                k = y.getAdmittance()[0].getLINK().get(k);
            }
        }
    }
}
