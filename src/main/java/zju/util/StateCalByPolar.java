package zju.util;

import org.apache.log4j.Logger;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink;
import zju.matrix.AVector;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;

/**
 * this class provides method to calculate system state like line power load or bus power, and so on...
 *
 * @author Dong Shufeng
 * Date: 2007-12-12
 */
public class StateCalByPolar implements MeasTypeCons {

    private static Logger log = Logger.getLogger(StateCalByPolar.class);

    /**
     * @param meas measurement vector
     * @param Y    admittance matrix getter
     * @param x    4n variables
     * @return give estimated values relative to measurement value
     */
    public static void getEstZOfFullState(MeasVector meas, YMatrixGetter Y, double[] x) {
        AVector result = meas.getZ_estimate();
        int index = 0;
        int n = Y.getAdmittance()[0].getM();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++) {
                        int num = meas.getBus_a_pos()[i] - 1;
                        result.setValue(index, x[num + n]);
                    }
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        int num = meas.getBus_v_pos()[i] - 1;
                        result.setValue(index, x[num]);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int num = meas.getBus_p_pos()[i] - 1;
                        result.setValue(index, x[num + 2 * n]);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int num = meas.getBus_q_pos()[i] - 1;
                        result.setValue(index, x[num + 3 * n]);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int i = 0; i < meas.getLine_from_p_pos().length; i++, index++)
                        result.setValue(index, StateCalByPolar.calLinePFrom(meas.getLine_from_p_pos()[i], Y, x));
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int i = 0; i < meas.getLine_from_q_pos().length; i++, index++)
                        result.setValue(index, StateCalByPolar.calLineQFrom(meas.getLine_from_q_pos()[i], Y, x));
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int i = 0; i < meas.getLine_to_p_pos().length; i++, index++)
                        result.setValue(index, StateCalByPolar.calLinePTo(meas.getLine_to_p_pos()[i], Y, x));
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int i = 0; i < meas.getLine_to_q_pos().length; i++, index++)
                        result.setValue(index, StateCalByPolar.calLineQTo(meas.getLine_to_q_pos()[i], Y, x));
                    break;
                default:
                    //log.error("unsupported measure type: " + type);
                    break;
            }
        }
    }

    /**
     * @param meas  measurement vector
     * @param Y     admittance matrix getter
     * @param state bus's voltage and angle
     * @return give estimated values relative to measurement value
     */
    public static AVector getEstimatedZ(MeasVector meas, YMatrixGetter Y, AVector state) {
        return getEstimatedZ(meas, Y, state.getValues(), meas.getZ_estimate());
    }

    /**
     * @param meas  measurement vector
     * @param Y     admittance matrix getter
     * @param state bus's voltage and angle
     * @return give estimated values relative to measurement value
     */
    public static AVector getEstimatedZ(MeasVector meas, YMatrixGetter Y, double[] state) {
        return getEstimatedZ(meas, Y, state, meas.getZ_estimate());
    }

    public static float[] getEstimatedZ(MeasVector meas, YMatrixGetter Y, float[] state, int offset) {
        int index = 0;
        int n = Y.getAdmittance()[0].getM();
        float[] result = new float[meas.getZ_estimate().getN()];
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++) {
                        int num = meas.getBus_a_pos()[i] - 1;//num starts from 0
                        result[index] = state[offset + num + n];
                    }
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        int num = meas.getBus_v_pos()[i] - 1;//num starts from 0
                        result[index] = state[offset + num];
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        result[index] = calBusP(num, Y, state, offset);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int num = meas.getBus_q_pos()[i];//num starts from 1
                        result[index] = calBusQ(num, Y, state, offset);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        result[index] = calLinePFrom(num, Y, state, offset);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        result[index] = calLineQFrom(num, Y, state, offset);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        result[index] = calLinePTo(num, Y, state, offset);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        result[index] = calLineQTo(num, Y, state, offset);
                    }
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
        return result;
    }

    /**
     * @param meas   Measurement vector
     * @param Y      Admittance matrix getter
     * @param state  Bus's voltage and angle
     * @param result Estimated values relative to measurement value
     * @return give estimated values relative to measurement value
     */
    public static AVector getEstimatedZ(MeasVector meas, YMatrixGetter Y, double[] state, AVector result) {
        int index = 0;
        int n = Y.getAdmittance()[0].getM();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++) {
                        int num = meas.getBus_a_pos()[i] - 1;//num starts from 0
                        result.setValue(index, state[num + n]);
                    }
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        int num = meas.getBus_v_pos()[i] - 1;//num starts from 0
                        result.setValue(index, state[num]);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        double value = calBusP(num, Y, state);
                        result.setValue(index, value);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int num = meas.getBus_q_pos()[i];//num starts from 1
                        double value = calBusQ(num, Y, state);
                        result.setValue(index, value);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        double v = calLinePFrom(num, Y, state);
                        result.setValue(index, v);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        double v = calLineQFrom(num, Y, state);
                        result.setValue(index, v);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        double v = calLinePTo(num, Y, state);
                        result.setValue(index, v);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        double v = calLineQTo(num, Y, state);
                        result.setValue(index, v);
                    }
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
        meas.setZ_estimate(result);
        return result;
    }

    public static double calBusP(int num, YMatrixGetter Y, AVector state) {
        return calBusP(num, Y, state.getValues());
    }

    /**
     * @param num   bus's num in admittance matrix and the least is 1
     * @param Y     admittance matrix getter
     * @param state all buses' voltage and angle
     * @return bus's injection active power
     */

    public static double calBusP(int num, YMatrixGetter Y, double[] state) {
        num = num - 1;
        ASparseMatrixLink[] admittance = Y.getAdmittance();
        int n = admittance[0].getM();
        double value = 0;
        int k = admittance[0].getIA()[num];
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double thetaIJ = state[num + n] - state[j + n];
            double gij = admittance[0].getVA().get(k);
            double bij = admittance[1].getVA().get(k);
            value += (state[num] * state[j] * (gij * Math.cos(thetaIJ) + bij * Math.sin(thetaIJ)));
            k = admittance[0].getLINK().get(k);
        }
        return value;
    }

    public static float calBusP(int num, YMatrixGetter Y, float[] state, int offset) {
        num = num - 1;
        ASparseMatrixLink[] admittance = Y.getAdmittance();
        int n = admittance[0].getM();
        double value = 0;
        int k = admittance[0].getIA()[num];
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double thetaIJ = state[offset + num + n] - state[offset + j + n];
            double gij = admittance[0].getVA().get(k);
            double bij = admittance[1].getVA().get(k);
            value += (state[offset + num] * state[offset + j] * (gij * Math.cos(thetaIJ) + bij * Math.sin(thetaIJ)));
            k = admittance[0].getLINK().get(k);
        }
        return (float) value;
    }

    public static double calBusP(double[][] y, double[] state) {
        double value = 0;
        for (int i = 0; i < y[0].length; i++) {
            double thetaIJ = i == 0 ? 0 : state[y[0].length + i - 1];
            value = value + state[0] * state[i] * (y[0][i] * Math.cos(thetaIJ) + y[1][i] * Math.sin(thetaIJ));
        }
        return value;
    }

    public static double calBusQ(int num, YMatrixGetter Y, AVector state) {
        return calBusQ(num, Y, state.getValues());
    }

    /**
     * @param num   bus's num in admittance matrix and the least is 1
     * @param Y     admittance matrix getter
     * @param state all buses' voltage and angle
     * @return bus's injection reactive power
     */
    public static double calBusQ(int num, YMatrixGetter Y, double[] state) {
        num = num - 1;
        ASparseMatrixLink[] admittance = Y.getAdmittance();
        double value = 0;
        int n = Y.getAdmittance()[0].getM();
        int k = admittance[0].getIA()[num];
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double thetaIJ = state[num + n] - state[j + n];
            value += (state[num] * state[j] * (admittance[0].getVA().get(k) * Math.sin(thetaIJ)
                    - admittance[1].getVA().get(k) * Math.cos(thetaIJ)));
            k = admittance[0].getLINK().get(k);
        }
        return value;
    }

    public static float calBusQ(int num, YMatrixGetter Y, float[] state, int offset) {
        num = num - 1;
        ASparseMatrixLink[] admittance = Y.getAdmittance();
        double value = 0;
        int n = Y.getAdmittance()[0].getM();
        int k = admittance[0].getIA()[num];
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double thetaIJ = state[offset + num + n] - state[offset + j + n];
            value += (state[offset + num] * state[offset + j] * (admittance[0].getVA().get(k) * Math.sin(thetaIJ)
                    - admittance[1].getVA().get(k) * Math.cos(thetaIJ)));
            k = admittance[0].getLINK().get(k);
        }
        return (float) value;
    }

    public static double calBusQ(double[][] y, double[] state) {
        double value = 0;
        for (int i = 0; i < y[0].length; i++) {
            double thetaIJ = i == 0 ? 0 : state[y[0].length + i - 1];
            value += state[0] * state[i] * (y[0][i] * Math.sin(thetaIJ) - y[1][i] * Math.cos(thetaIJ));
        }
        return value;
    }

    public static double calLinePFrom(int branchId, YMatrixGetter Y, AVector state) {
        return calLinePFrom(branchId, Y, state.getValues());
    }

    /**
     * @param branchId branch's branchId ieee common format data and the least is 1
     * @param Y        admittance matrix getter
     * @param state    all buses' voltage and angle
     * @return active power of head of the branch
     */
    public static double calLinePFrom(int branchId, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(branchId);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
        double thetaIJ = state[i + n] - state[j + n];
        return state[i] * state[i] * (gbg1b1[0] + gbg1b1[2]) - state[i] * state[j] * (gbg1b1[0] * Math.cos(thetaIJ)
                + gbg1b1[1] * Math.sin(thetaIJ));
    }

    public static float calLinePFrom(int branchId, YMatrixGetter Y, float[] state, int offset) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(branchId);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
        double thetaIJ = state[offset + i + n] - state[offset + j + n];
        return (float) (state[offset + i] * state[offset + i] * (gbg1b1[0] + gbg1b1[2]) - state[offset + i] * state[offset + j] * (gbg1b1[0] * Math.cos(thetaIJ)
                + gbg1b1[1] * Math.sin(thetaIJ)));
    }

    public static double calLinePFrom(BranchData branch, double[] state) {
        double[] gbg1b1 = YMatrixGetter.getBranchAdmittance(branch)[0];
        double thetaIJ = state[2];
        return state[0] * state[0] * (gbg1b1[0] + gbg1b1[2]) - state[0] * state[1] * (gbg1b1[0] * Math.cos(thetaIJ)
                + gbg1b1[1] * Math.sin(thetaIJ));
    }

    public static double calLinePTo(int num, YMatrixGetter Y, AVector state) {
        return calLinePTo(num, Y, state.getValues());
    }

    /**
     * @param num   branch's num ieee common format data and the least is 1
     * @param Y     admittance matrix getter
     * @param state all buses' voltage and angle
     * @return active power of tail of the branch
     */

    public static double calLinePTo(int num, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(num, YMatrixGetter.LINE_TO);
        double thetaJI = state[j + n] - state[i + n];
        return state[j] * state[j] * (gbg1b1[0] + gbg1b1[2]) - state[i] * state[j] * (gbg1b1[0] * Math.cos(thetaJI)
                + gbg1b1[1] * Math.sin(thetaJI));
    }

    public static float calLinePTo(int num, YMatrixGetter Y, float[] state, int offset) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(num, YMatrixGetter.LINE_TO);
        double thetaJI = state[offset + j + n] - state[offset + i + n];
        return (float) (state[offset + j] * state[offset + j] * (gbg1b1[0] + gbg1b1[2]) - state[offset + i] * state[offset + j] * (gbg1b1[0] * Math.cos(thetaJI)
                + gbg1b1[1] * Math.sin(thetaJI)));
    }

    public static double calLinePTo(BranchData branch, double[] state) {
        double[] gbg1b1 = YMatrixGetter.getBranchAdmittance(branch)[1];
        double thetaJI = -state[2];
        return state[1] * state[1] * (gbg1b1[0] + gbg1b1[2]) - state[0] * state[1] * (gbg1b1[0] * Math.cos(thetaJI)
                + gbg1b1[1] * Math.sin(thetaJI));
    }

    public static double calLineQFrom(int num, YMatrixGetter Y, AVector state) {
        return calLineQFrom(num, Y, state.getValues());
    }

    /**
     * @param num   branch's num ieee common format data and the least is 1
     * @param Y     admittance matrix getter
     * @param state all buses' voltage and angle
     * @return inactive power of head of the branch
     */

    public static double calLineQFrom(int num, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(num, YMatrixGetter.LINE_FROM);
        double thetaIJ = state[i + n] - state[j + n];
        return -state[i] * state[i] * (gbg1b1[1] + gbg1b1[3]) - state[i] * state[j] * (gbg1b1[0] * Math.sin(thetaIJ)
                - gbg1b1[1] * Math.cos(thetaIJ));
    }

    public static float calLineQFrom(int num, YMatrixGetter Y, float[] state, int offset) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(num, YMatrixGetter.LINE_FROM);
        double thetaIJ = state[offset + i + n] - state[offset + j + n];
        return (float) (-state[offset + i] * state[offset + i] * (gbg1b1[1] + gbg1b1[3]) - state[offset + i] * state[offset + j] * (gbg1b1[0] * Math.sin(thetaIJ)
                - gbg1b1[1] * Math.cos(thetaIJ)));
    }

    public static double calLineQFrom(BranchData branch, double[] state) {
        double[] gbg1b1 = YMatrixGetter.getBranchAdmittance(branch)[0];
        double thetaIJ = state[2];
        return -state[0] * state[0] * (gbg1b1[1] + gbg1b1[3]) - state[0] * state[1] * (gbg1b1[0] * Math.sin(thetaIJ)
                - gbg1b1[1] * Math.cos(thetaIJ));
    }

    public static double calLineQTo(int num, YMatrixGetter Y, AVector state) {
        return calLineQTo(num, Y, state.getValues());
    }

    /**
     * @param num   branch's num ieee common format data and the least is 1
     * @param Y     admittance matrix getter
     * @param state all buses' voltage and angle
     * @return inactive power of tail of the branch
     */
    public static double calLineQTo(int num, YMatrixGetter Y, double[] state) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(num, YMatrixGetter.LINE_TO);
        double thetaJI = state[j + n] - state[i + n];
        return -state[j] * state[j] * (gbg1b1[1] + gbg1b1[3]) - state[i] * state[j] * (gbg1b1[0] * Math.sin(thetaJI)
                - gbg1b1[1] * Math.cos(thetaJI));
    }

    public static float calLineQTo(int num, YMatrixGetter Y, float[] state, int offset) {
        int n = Y.getAdmittance()[0].getM();
        int[] ij = Y.getFromTo(num);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(num, YMatrixGetter.LINE_TO);
        double thetaJI = state[offset + j + n] - state[offset + i + n];
        return (float) (-state[offset + j] * state[offset + j] * (gbg1b1[1] + gbg1b1[3]) - state[offset + i] * state[offset + j] * (gbg1b1[0] * Math.sin(thetaJI)
                - gbg1b1[1] * Math.cos(thetaJI)));
    }

    public static double calLineQTo(BranchData branch, double[] state) {
        double[] gbg1b1 = YMatrixGetter.getBranchAdmittance(branch)[1];
        double thetaJI = -state[2];
        return -state[1] * state[1] * (gbg1b1[1] + gbg1b1[3]) - state[0] * state[1] * (gbg1b1[0] * Math.sin(thetaJI)
                - gbg1b1[1] * Math.cos(thetaJI));
    }

    public static double calEstValue(int type, int pos, YMatrixGetter Y, AVector state) {
        int n = Y.getAdmittance()[0].getM();
        switch (type) {
            case TYPE_BUS_ANGLE:
                return state.getValue(pos + n - 1);
            case TYPE_BUS_VOLOTAGE:
                return state.getValue(pos - 1);
            case TYPE_BUS_ACTIVE_POWER:
                return calBusP(pos, Y, state);
            case TYPE_BUS_REACTIVE_POWER:
                return calBusQ(pos, Y, state);
            case TYPE_LINE_FROM_ACTIVE:
                return calLinePFrom(pos, Y, state);
            case TYPE_LINE_FROM_REACTIVE:
                return calLineQFrom(pos, Y, state);
            case TYPE_LINE_TO_ACTIVE:
                return calLinePTo(pos, Y, state);
            case TYPE_LINE_TO_REACTIVE:
                return calLineQTo(pos, Y, state);
            default:
                log.warn("Unsupported measure type: " + type);
                return 0;
        }
    }

    /*
     * calculate current amplitude in polar system
     * */
    public static double calLineCurrentAmp(int branchId, YMatrixGetter Y, AVector state, int fromOrTo) {
        return calLineCurrent(branchId, Y, state, fromOrTo)[0];
    }

    /*
     * calculate current angle in polar system
     * */
    public static double calLineCurrentAngle(int branchId, YMatrixGetter Y, AVector state, int fromOrTo) {
        return calLineCurrent(branchId, Y, state, fromOrTo)[1];
    }

    /*
     * calculate current in polar system
     * */
    public static double[] calLineCurrent(int branchId, YMatrixGetter Y, AVector state, int fromOrTo) {
        IEEEDataIsland island = Y.getIsland();
        int n = island.getBuses().size();
        BranchData branch = island.getId2branch().get(branchId);
        int fromBus = branch.getTapBusNumber() - 1;
        int toBus = branch.getZBusNumber() - 1;
        double[] uiaiujaj;
        double[] gbg1b1;
        if (fromOrTo == YMatrixGetter.LINE_FROM) {
            uiaiujaj = new double[]{state.getValue(fromBus), state.getValue(fromBus + n),
                    state.getValue(toBus), state.getValue(toBus + n)};
            gbg1b1 = Y.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
        } else {
            uiaiujaj = new double[]{state.getValue(toBus), state.getValue(toBus + n),
                    state.getValue(fromBus), state.getValue(fromBus + n)};
            gbg1b1 = Y.getLineAdmittance(branchId, YMatrixGetter.LINE_TO);
        }
        return calLineCurrent(gbg1b1, uiaiujaj);
    }

    //todo: phase shift is not considered 
    public static double[] calLineCurrent(double[] gbg1b1, double[] uiaiujaj) {
        double ei = uiaiujaj[0] * Math.cos(uiaiujaj[1]);
        double fi = uiaiujaj[0] * Math.sin(uiaiujaj[1]);
        double ej = uiaiujaj[2] * Math.cos(uiaiujaj[3]);
        double fj = uiaiujaj[2] * Math.sin(uiaiujaj[3]);
        double[] c = StateCalByRC.calLineCurrent(gbg1b1, new double[]{ei, fi, ej, fj});
        double current = Math.sqrt(c[0] * c[0] + c[1] * c[1]);
        double angle = Math.atan2(c[1], c[0]);
        return new double[]{current, angle};
    }

//    public static double[] calAnotherVoltage(double[] gbg1b1, double[] uiaiIiai) {
//        double ei = uiaiIiai[0] * Math.cos(uiaiIiai[1]);
//        double fi = uiaiIiai[0] * Math.sin(uiaiIiai[1]);
//        double rei = uiaiIiai[2] * Math.cos(uiaiIiai[3]);
//        double imi = uiaiIiai[2] * Math.sin(uiaiIiai[3]);
//        double[] uj = StateCalByRC.calAnotherVoltage(gbg1b1, new double[]{ei,fi,rei,imi});
//
//        double voltage = Math.sqrt(uj[0] * uj[0] + uj[1] * uj[1]);
//        double angle = Math.atan2(uj[1], uj[0]);
//        return new double[]{voltage, angle};
//    }
}
