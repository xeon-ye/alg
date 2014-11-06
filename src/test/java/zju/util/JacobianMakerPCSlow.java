package zju.util;

import org.apache.log4j.Logger;
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
 *         Date: 2007-12-5
 */
public class JacobianMakerPCSlow implements MeasTypeCons {

    private static Logger log = Logger.getLogger(JacobianMakerPCSlow.class);

    /**
     * @param meas  num of every measure type
     * @param y     admittance matrix getter
     * @param state state vector
     * @return jocobian matrix
     */
    public static ASparseMatrixLink getJacobianOfFullState(MeasVector meas, YMatrixGetter y, AVector state) {
        ASparseMatrixLink2D result = new ASparseMatrixLink2D(meas.getZ().getN(), state.getN() * 2);
        int index = 0;
        int n = y.getAdmittance()[0].getM();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++)
                        fillJacobian_bus_a(meas.getBus_a_pos()[i], n, result, index);
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++)
                        fillJacobian_bus_v(meas.getBus_v_pos()[i], result, index);
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++)
                        result.setValue(index, ((2 * n) + meas.getBus_p_pos()[i]) - 1, 1);
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++)
                        result.setValue(index, ((3 * n) + meas.getBus_q_pos()[i]) - 1, 1);
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++)
                        fillJacobian_line_from_p(meas.getLine_from_p_pos()[k], n, y, state, result, index);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++)
                        fillJacobian_line_from_q(meas.getLine_from_q_pos()[k], n, y, state, result, index);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++)
                        fillJacobian_line_to_p(meas.getLine_to_p_pos()[k], n, y, state, result, index);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++)
                        fillJacobian_line_to_q(meas.getLine_to_q_pos()[k], n, y, state, result, index);
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
        return result;
    }

    /**
     * @param meas              num of every measure type
     * @param addmittanceGetter admittance matrix getter
     * @param state             state vector
     * @return jocobian matrix
     */
    public static ASparseMatrixLink2D getJacobianOfVTheta(MeasVector meas, YMatrixGetter addmittanceGetter, AVector state) {
        ASparseMatrixLink2D result = new ASparseMatrixLink2D(meas.getZ().getN(), state.getN());
        int index = 0;
        int n = state.getN() / 2;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++)
                        fillJacobian_bus_a(meas.getBus_a_pos()[i], n, result, index);
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++)
                        fillJacobian_bus_v(meas.getBus_v_pos()[i], result, index);
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++)
                        fillJacobian_bus_p(meas.getBus_p_pos()[i], n, addmittanceGetter, state, result, index);
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++)
                        fillJacobian_bus_q(meas.getBus_q_pos()[i], n, addmittanceGetter, state, result, index);
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++)
                        fillJacobian_line_from_p(meas.getLine_from_p_pos()[k], n, addmittanceGetter, state, result, index);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++)
                        fillJacobian_line_from_q(meas.getLine_from_q_pos()[k], n, addmittanceGetter, state, result, index);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++)
                        fillJacobian_line_to_p(meas.getLine_to_p_pos()[k], n, addmittanceGetter, state, result, index);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++)
                        fillJacobian_line_to_q(meas.getLine_to_q_pos()[k], n, addmittanceGetter, state, result, index);
                    break;
                default:
                    //log.error("unsupported measure type: " + type);//todo: not good
                    break;
            }
        }
        return result;
    }

    public static ASparseMatrixLink getJacobianOfRatio(MeasVector meas, YMatrixGetter addmittanceGetter, AVector state, int[] transformerBranches, double[] ratios) {
        ASparseMatrixLink result = new ASparseMatrixLink(meas.getZ().getN(), transformerBranches.length);
        int index = 0;
        int n = state.getN() / 2;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    index += meas.getBus_a_pos().length;
                    break;
                case TYPE_BUS_VOLOTAGE:
                    index += meas.getBus_v_pos().length;
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int num = meas.getBus_p_pos()[i];//num starts from 1
                        for (int j = 0; j < transformerBranches.length; j++) {
                            int branchId = transformerBranches[j];
                            double t = ratios[j];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (ij[0] == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                double v = 2 * ui * ui * gbg1b1[0] - ui * uj * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t;
                                result.setValue(index, j, v);
                            } else if (ij[1] == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_TO);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaJI = state.getValue(ij[1] - 1 + n) - state.getValue(ij[0] - 1 + n);
                                double cos = Math.cos(thetaJI);
                                double sin = Math.sin(thetaJI);
                                double v = -ui * uj * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t;
                                result.setValue(index, j, v);
                            }
                        }
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int num = meas.getBus_q_pos()[i];//num starts from 0
                        for (int j = 0; j < transformerBranches.length; j++) {
                            int branchId = transformerBranches[j];
                            double t = ratios[j];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (ij[0] == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                double v = -2 * ui * ui * gbg1b1[1] - ui * uj * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t;
                                result.setValue(index, j, v);
                            } else if (ij[1] == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_TO);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaJI = state.getValue(ij[1] - 1 + n) - state.getValue(ij[0] - 1 + n);
                                double cos = Math.cos(thetaJI);
                                double sin = Math.sin(thetaJI);
                                double v = -ui * uj * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t;
                                result.setValue(index, j, v);
                            }
                        }
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int num = meas.getLine_from_p_pos()[k];//num starts from 1
                        for (int j = 0; j < transformerBranches.length; j++) {
                            int branchId = transformerBranches[j];
                            double t = ratios[j];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (branchId == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                double v = 2 * ui * ui * gbg1b1[0] - ui * uj * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t;
                                result.setValue(index, j, v);
                            }
                        }
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int num = meas.getLine_from_q_pos()[k];//num starts from 1
                        for (int j = 0; j < transformerBranches.length; j++) {
                            int branchId = transformerBranches[j];
                            double t = ratios[j];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (branchId == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_FROM);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaIJ = state.getValue(ij[0] - 1 + n) - state.getValue(ij[1] - 1 + n);
                                double cos = Math.cos(thetaIJ);
                                double sin = Math.sin(thetaIJ);
                                double v = -2 * ui * ui * gbg1b1[1] - ui * uj * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t;
                                result.setValue(index, j, v);
                            }
                        }
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int num = meas.getLine_to_p_pos()[k];//num starts from 1
                        for (int j = 0; j < transformerBranches.length; j++) {
                            int branchId = transformerBranches[j];
                            double t = ratios[j];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (branchId == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_TO);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaJI = state.getValue(ij[1] - 1 + n) - state.getValue(ij[0] - 1 + n);
                                double cos = Math.cos(thetaJI);
                                double sin = Math.sin(thetaJI);
                                double v = -ui * uj * (gbg1b1[0] * cos + gbg1b1[1] * sin) / t;
                                result.setValue(index, j, v);
                            }
                        }
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int num = meas.getLine_to_q_pos()[k];//num starts from 1
                        for (int j = 0; j < transformerBranches.length; j++) {
                            int branchId = transformerBranches[j];
                            double t = ratios[j];
                            int[] ij = addmittanceGetter.getFromTo(branchId);
                            if (branchId == num) {
                                double[] gbg1b1 = addmittanceGetter.getLineAdmittance(branchId, YMatrixGetter.LINE_TO);
                                double ui = state.getValue(ij[0] - 1);
                                double uj = state.getValue(ij[1] - 1);
                                double thetaJI = state.getValue(ij[1] - 1 + n) - state.getValue(ij[0] - 1 + n);
                                double cos = Math.cos(thetaJI);
                                double sin = Math.sin(thetaJI);
                                double v = -ui * uj * (gbg1b1[0] * sin - gbg1b1[1] * cos) / t;
                                result.setValue(index, j, v);
                            }
                        }
                    }
                    break;
                default:
                    log.warn("Unsupported measure type: " + type);
                    break;
            }
        }
        return result;
    }

    public static ASparseMatrixLink getJacobianOfVTheta(List<Integer> linkBuses, YMatrixGetter admittanceGetter, AVector state) {
        ASparseMatrixLink result = new ASparseMatrixLink2D(2 * linkBuses.size(), state.getN());
        int index = 0;
        int n = state.getN() / 2;
        for (int i = 0; i < linkBuses.size(); i++, index++) {
            fillJacobian_bus_p(linkBuses.get(i), n, admittanceGetter, state, result, index);
        }
        for (int i = 0; i < linkBuses.size(); i++, index++) {
            fillJacobian_bus_q(linkBuses.get(i), n, admittanceGetter, state, result, index);
        }
        return result;
    }

    public static ASparseMatrixLink getJacobianOfBusP(int num, YMatrixGetter admittanceGetter, AVector state) {
        ASparseMatrixLink result = new ASparseMatrixLink(1, state.getN());
        fillJacobian_bus_p(num, state.getN() / 2, admittanceGetter, state, result, 0);
        return result;
    }

    public static ASparseMatrixLink getJacobianOfAllBusP(YMatrixGetter admittanceGetter, AVector state) {
        ASparseMatrixLink result = new ASparseMatrixLink(1, state.getN());
        int n = state.getN() / 2;
        IEEEDataIsland island = admittanceGetter.getIsland();
        for (BusData bus : island.getBuses()) {
            if (bus.getType() == 2 || bus.getType() == 3) continue;
            //System.out.println("bus " + bus.getBusNo());
            //getJacobianOfBusP(bus.getBusNo(),Y,state).printOnScreen();
            ASparseMatrixLink[] admittance = admittanceGetter.getAdmittance();
            int num = bus.getBusNumber() - 1;//num starts from 0
            int k = admittance[0].getIA()[num];
            double qi = 0;
            while (k != -1) {
                int j = admittance[0].getJA().get(k);
                double thetaIJ = state.getValue(num + n) - state.getValue(j + n);
                double sin = Math.sin(thetaIJ);
                double cos = Math.cos(thetaIJ);
                qi = qi + state.getValue(num) * state.getValue(j) * (admittance[0].getVA().get(k) * sin - admittance[1].getVA().get(k) * cos);
                if (num == j) {
                    result.increase(0, num, (admittance[0].getVA().get(k) * state.getValue(num) * state.getValue(num)
                            + StateCalByPolar.calBusP(bus.getBusNumber(), admittanceGetter, state)) / state.getValue(num));
                    result.increase(0, num + n, -admittance[1].getVA().get(k) * state.getValue(num) * state.getValue(num));
                    k = admittance[0].getLINK().get(k);
                    continue;
                }
                result.increase(0, j, state.getValue(num) * (admittance[0].getVA().get(k) * cos + admittance[1].getVA().get(k) * sin));
                result.increase(0, j + n, state.getValue(num) * state.getValue(j) * (admittance[0].getVA().get(k) * sin - admittance[1].getVA().get(k) * cos));
                k = admittance[0].getLINK().get(k);
            }
            result.increase(0, num + n, -qi);
        }
        //result.printOnScreen();
        return result;
    }

    public static ASparseMatrixLink getJacobianOfBusQ(int num, YMatrixGetter admittanceGetter, AVector state) {
        ASparseMatrixLink result = new ASparseMatrixLink(1, state.getN());
        fillJacobian_bus_q(num, state.getN(), admittanceGetter, state, result, 0);
        return result;
    }

    public static void fillJacobian_bus_a(int pos, int n, ASparseMatrixLink result, int currentRow) {
        result.setValue(currentRow, pos + n - 1, 1);
    }

    public static void fillJacobian_bus_v(int pos, ASparseMatrixLink result, int currentRow) {
        result.setValue(currentRow, pos - 1, 1);
    }

    public static void fillJacobian_bus_p(int pos, int n, YMatrixGetter addmittanceGetter, AVector state, ASparseMatrixLink result, int index) {
        ASparseMatrixLink[] admittance = addmittanceGetter.getAdmittance();
        int num = pos - 1;//num starts from 0
        int k = admittance[0].getIA()[num];
        double qi = 0;
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double thetaIJ = state.getValue(num + n) - state.getValue(j + n);
            double sin = Math.sin(thetaIJ);
            double cos = Math.cos(thetaIJ);
            qi = qi + state.getValue(num) * state.getValue(j) * (admittance[0].getVA().get(k) * sin - admittance[1].getVA().get(k) * cos);
            if (num == j) {
                result.setValue(index, num, (admittance[0].getVA().get(k) * state.getValue(num) * state.getValue(num)
                        + StateCalByPolar.calBusP(pos, addmittanceGetter, state)) / state.getValue(num));
                result.setValue(index, num + n, -admittance[1].getVA().get(k) * state.getValue(num) * state.getValue(num));
                k = admittance[0].getLINK().get(k);
                continue;
            }
            result.setValue(index, j, state.getValue(num) * (admittance[0].getVA().get(k) * cos + admittance[1].getVA().get(k) * sin));
            result.setValue(index, j + n, state.getValue(num) * state.getValue(j) * (admittance[0].getVA().get(k) * sin - admittance[1].getVA().get(k) * cos));
            k = admittance[0].getLINK().get(k);
        }
        result.increase(index, num + n, -qi);
    }

    public static void fillJacobian_bus_q(int pos, int n, YMatrixGetter addmittanceGetter, AVector state, ASparseMatrixLink result, int index) {
        ASparseMatrixLink[] admittance = addmittanceGetter.getAdmittance();
        int num = pos - 1;//num starts from 0
        int k = admittance[0].getIA()[num];
        double pi = 0.0;
        while (k != -1) {
            int j = admittance[0].getJA().get(k);
            double thetaIJ = state.getValue(num + n) - state.getValue(j + n);
            double cos = Math.cos(thetaIJ);
            double sin = Math.sin(thetaIJ);
            pi = pi + state.getValue(num) * state.getValue(j) * (admittance[0].getVA().get(k) * cos + admittance[1].getVA().get(k) * sin);
            if (num == j) {
                result.setValue(index, num, (-admittance[1].getVA().get(k) * state.getValue(num) * state.getValue(num)
                        + StateCalByPolar.calBusQ(pos, addmittanceGetter, state)) / state.getValue(num));
                result.setValue(index, num + n, -admittance[0].getVA().get(k) * state.getValue(num) * state.getValue(num));
                k = admittance[0].getLINK().get(k);
                continue;
            }
            result.setValue(index, j, state.getValue(num) * (admittance[0].getVA().get(k) * sin - admittance[1].getVA().get(k) * cos));
            result.setValue(index, j + n, -state.getValue(num) * state.getValue(j) * (admittance[0].getVA().get(k) * cos + admittance[1].getVA().get(k) * sin));
            k = admittance[0].getLINK().get(k);
        }
        result.increase(index, num + n, pi);
    }

    public static void fillJacobian_line_from_p(int pos, int n, YMatrixGetter addmittanceGetter, AVector state, ASparseMatrixLink result, int currentRow) {
        int[] ij = addmittanceGetter.getFromTo(pos);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = addmittanceGetter.getLineAdmittance(pos, YMatrixGetter.LINE_FROM);
        double thetaIJ = state.getValue(i + n) - state.getValue(j + n);
        double cos = Math.cos(thetaIJ);
        double sin = Math.sin(thetaIJ);
        result.setValue(currentRow, i, 2 * state.getValue(i) * (gbg1b1[0] + gbg1b1[2]) - state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        result.setValue(currentRow, j, -state.getValue(i) * (gbg1b1[0] * cos + gbg1b1[1] * sin), true);
        double tmp = state.getValue(i) * state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos);
        result.setValue(currentRow, i + n, tmp, true);
        result.setValue(currentRow, j + n, -tmp, true);
    }

    public static void fillJacobian_line_from_q(int pos, int n, YMatrixGetter addmittanceGetter, AVector state, ASparseMatrixLink result, int currentRow) {
        int[] ij = addmittanceGetter.getFromTo(pos);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = addmittanceGetter.getLineAdmittance(pos, YMatrixGetter.LINE_FROM);
        double thetaIJ = state.getValue(i + n) - state.getValue(j + n);
        double sin = Math.sin(thetaIJ);
        double cos = Math.cos(thetaIJ);
        result.setValue(currentRow, i, -2 * state.getValue(i) * (gbg1b1[1] + gbg1b1[3]) - state.getValue(j) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        result.setValue(currentRow, j, -state.getValue(i) * (gbg1b1[0] * sin - gbg1b1[1] * cos), true);
        double tmp = -state.getValue(i) * state.getValue(j) * (gbg1b1[0] * cos + gbg1b1[1] * sin);
        result.setValue(currentRow, i + n, tmp, true);
        result.setValue(currentRow, j + n, -tmp, true);
    }

    public static void fillJacobian_line_to_p(int pos, int n, YMatrixGetter addmittanceGetter, AVector state, ASparseMatrixLink result, int currentRow) {
        int[] ij = addmittanceGetter.getFromTo(pos);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = addmittanceGetter.getLineAdmittance(pos, YMatrixGetter.LINE_TO);
        double thetaIJ = state.getValue(i + n) - state.getValue(j + n);
        double cos = Math.cos(thetaIJ);
        double sin = Math.sin(thetaIJ);
        result.setValue(currentRow, j, 2 * state.getValue(j) * (gbg1b1[0] + gbg1b1[2]) - state.getValue(i) * (gbg1b1[0] * cos - gbg1b1[1] * sin), true);
        result.setValue(currentRow, i, -state.getValue(j) * (gbg1b1[0] * cos - gbg1b1[1] * sin), true);
        double tmp = -state.getValue(i) * state.getValue(j) * (gbg1b1[0] * sin + gbg1b1[1] * cos);
        result.setValue(currentRow, j + n, tmp, true);
        result.setValue(currentRow, i + n, -tmp, true);
    }

    public static void fillJacobian_line_to_q(int pos, int n, YMatrixGetter addmittanceGetter, AVector state, ASparseMatrixLink result, int currentRow) {
        int[] ij = addmittanceGetter.getFromTo(pos);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = addmittanceGetter.getLineAdmittance(pos, YMatrixGetter.LINE_TO);
        double thetaIJ = state.getValue(i + n) - state.getValue(j + n);
        double sin = Math.sin(thetaIJ);
        double cos = Math.cos(thetaIJ);
        result.setValue(currentRow, j, -2 * state.getValue(j) * (gbg1b1[1] + gbg1b1[3]) + state.getValue(i) * (gbg1b1[0] * sin + gbg1b1[1] * cos), true);
        result.setValue(currentRow, i, state.getValue(j) * (gbg1b1[0] * sin + gbg1b1[1] * cos), true);
        double tmp = -state.getValue(i) * state.getValue(j) * (gbg1b1[0] * cos - gbg1b1[1] * sin);
        result.setValue(currentRow, j + n, tmp, true);
        result.setValue(currentRow, i + n, -tmp, true);
    }

    public static ASparseMatrixLink getJacobianOfVTheta(int type, int pos, YMatrixGetter addmittanceGetter, AVector state) {
        ASparseMatrixLink2D result = new ASparseMatrixLink2D(1, state.getN());
        int index = 0;
        int n = state.getN() / 2;
        switch (type) {
            case TYPE_BUS_ANGLE:
                fillJacobian_bus_a(pos, n, result, index);
                break;
            case TYPE_BUS_VOLOTAGE:
                fillJacobian_bus_v(pos, result, index);
                break;
            case TYPE_BUS_ACTIVE_POWER:
                fillJacobian_bus_p(pos, n, addmittanceGetter, state, result, index);
                break;
            case TYPE_BUS_REACTIVE_POWER:
                fillJacobian_bus_q(pos, n, addmittanceGetter, state, result, index);
                break;
            case TYPE_LINE_FROM_ACTIVE:
                fillJacobian_line_from_p(pos, n, addmittanceGetter, state, result, index);
                break;
            case TYPE_LINE_FROM_REACTIVE:
                fillJacobian_line_from_q(pos, n, addmittanceGetter, state, result, index);
                break;
            case TYPE_LINE_TO_ACTIVE:
                fillJacobian_line_to_p(pos, n, addmittanceGetter, state, result, index);
                break;
            case TYPE_LINE_TO_REACTIVE:
                fillJacobian_line_to_q(pos, n, addmittanceGetter, state, result, index);
                break;
            default:
                log.warn("Unsupported measure type: " + type);
                break;
        }
        return result;
    }
}
