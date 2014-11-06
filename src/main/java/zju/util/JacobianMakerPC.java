package zju.util;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import zju.matrix.ASparseMatrixLink;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;

import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-27
 */
public class JacobianMakerPC implements MeasTypeCons {

    /**
     * @param meas num of every measure type
     * @param Y    admittance matrix getter
     * @return jocobian matrix
     */
    public static ASparseMatrixLink2D getJacStrucOfVTheta(MeasVector meas, YMatrixGetter Y, int n) {
        int capacity = getNonZeroNum(meas, Y);
        ASparseMatrixLink2D result = new ASparseMatrixLink2D(meas.getZ().getN(), 2 * n, capacity);
        int index = 0;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++)
                        result.setValue(index, meas.getBus_a_pos()[i] + n - 1, 1);
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++)
                        result.setValue(index, meas.getBus_v_pos()[i] - 1, 1);
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        ASparseMatrixLink[] admittance = Y.getAdmittance();
                        int num = meas.getBus_p_pos()[i] - 1;//num starts from 0
                        jacStruOfBusPower(result, admittance[0], num, index);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        ASparseMatrixLink[] admittance = Y.getAdmittance();
                        int num = meas.getBus_q_pos()[i] - 1;//num starts from 0
                        jacStruOfBusPower(result, admittance[0], num, index);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        int[] ij = Y.getFromTo(meas.getLine_from_p_pos()[k]);
                        int i = ij[0] - 1;
                        int j = ij[1] - 1;
                        result.setValue(index, i, 1);
                        result.setValue(index, j, 1);
                        result.setValue(index, i + n, 1);
                        result.setValue(index, j + n, 1);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        int[] ij = Y.getFromTo(meas.getLine_from_q_pos()[k]);
                        int i = ij[0] - 1;
                        int j = ij[1] - 1;
                        result.setValue(index, i, 1);
                        result.setValue(index, j, 1);
                        result.setValue(index, i + n, 1);
                        result.setValue(index, j + n, 1);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        int[] ij = Y.getFromTo(meas.getLine_to_p_pos()[k]);
                        int i = ij[0] - 1;
                        int j = ij[1] - 1;
                        result.setValue(index, j, 1);
                        result.setValue(index, i, 1);
                        result.setValue(index, j + n, 1);
                        result.setValue(index, i + n, 1);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        int[] ij = Y.getFromTo(meas.getLine_to_q_pos()[k]);
                        int i = ij[0] - 1;
                        int j = ij[1] - 1;
                        result.setValue(index, j, 1);
                        result.setValue(index, i, 1);
                        result.setValue(index, j + n, 1);
                        result.setValue(index, i + n, 1);
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
     * 得到Jacobian矩阵中非零元个数，包括了潮流约束对应的非零元
     *
     * @param meas 量测向量
     * @param Y    导纳矩阵
     * @return Jacobian矩阵中非零元个数
     */
    public static int getNonZeroNumOfFullState(MeasVector meas, YMatrixGetter Y) {
        int nele_jac = meas.getBus_a_pos().length + meas.getBus_v_pos().length
                + meas.getBus_p_pos().length + meas.getBus_q_pos().length
                + 4 * (meas.getLine_from_p_pos().length + meas.getLine_from_q_pos().length
                + meas.getLine_to_p_pos().length + meas.getLine_to_q_pos().length) + getNonZeroNumOfFullState(Y);
        return nele_jac;
    }

    /**
     * 得到Jacobian矩阵中潮流约束对应的非零元个数
     *
     * @param Y 导纳矩阵
     * @return Jacobian矩阵中非零元个数
     */
    public static int getNonZeroNumOfFullState(YMatrixGetter Y) {
        int busNumber = Y.getAdmittance()[0].getN();
        int nele_jac = 0;
        if (Y.getConnectedBusCount() == null) {
            nele_jac += 2 * 3 * (busNumber + busNumber);
        } else {
            for (int busNum = 0; busNum < busNumber; busNum++)
                nele_jac += 4 * Y.getConnectedBusCount()[busNum] + 4;
        }
        nele_jac += 2 * busNumber;
        return nele_jac;
    }

    public static int getNonZeroNum(MeasVector meas, YMatrixGetter Y) {
        int p_from_size = meas.getLine_from_p_pos().length;
        int q_from_size = meas.getLine_from_q_pos().length;
        int p_to_size = meas.getLine_to_p_pos().length;
        int q_to_size = meas.getLine_to_q_pos().length;
        int angle_size = meas.getBus_a_pos().length;
        int v_size = meas.getBus_v_pos().length;
        int capacity = angle_size + v_size + 4 * (p_from_size + q_from_size + p_to_size + q_to_size);
        if (Y.getConnectedBusCount() == null) {
            capacity += 2 * 3 * (meas.getBus_p_pos().length + meas.getBus_q_pos().length);
        } else {
            for (int busNum : meas.getBus_p_pos())
                capacity += 2 * Y.getConnectedBusCount()[busNum - 1] + 2;
            for (int busNum : meas.getBus_q_pos())
                capacity += 2 * Y.getConnectedBusCount()[busNum - 1] + 2;
        }
        return capacity;
    }

    private static void jacStruOfBusPower(ASparseMatrixLink2D result, ASparseMatrixLink y, int num, int row) {
        int k = y.getIA()[num];
        while (k != -1) {
            int j = y.getJA().get(k);
            if (num == j) {
                result.setValue(row, num, 1);
                result.setValue(row, num + y.getN(), 1);
                k = y.getLINK().get(k);
                continue;
            }
            result.setValue(row, j, 1);
            result.setValue(row, j + y.getN(), 1);
            k = y.getLINK().get(k);
        }
    }

    /**
     * @param meas   潮流计算用的量测，只包含P,Q量测
     * @param Y      导纳阵
     * @param pqSize PQ节点个数
     * @param pvSize PV节点个数
     * @return Jacobian矩阵结构
     */
    public static ASparseMatrixLink2D getJacStrucOfVTheta(MeasVector meas, YMatrixGetter Y, int pqSize, int pvSize) {
        ASparseMatrixLink2D result = new ASparseMatrixLink2D(meas.getZ().getN(), pvSize + 2 * pqSize);
        int index = 0;
        ASparseMatrixLink[] admittance = Y.getAdmittance();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int pos = meas.getBus_p_pos()[i];
                        int num = pos - 1;//num starts from 0
                        int k = admittance[0].getIA()[num];
                        while (k != -1) {
                            int j = admittance[0].getJA().get(k);
                            if (num == j) {
                                if (num < pqSize)
                                    result.setValue(index, num, 1.0);
                                result.setValue(index, num + pqSize, 1.0);
                                k = admittance[0].getLINK().get(k);
                                continue;
                            }
                            if (j < pqSize)
                                result.setValue(index, j, 1.0);
                            if (j < pvSize + pqSize)
                                result.setValue(index, j + pqSize, 1.0);
                            k = admittance[0].getLINK().get(k);
                        }
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int pos = meas.getBus_q_pos()[i];
                        int num = pos - 1;//num starts from 0
                        int k = admittance[0].getIA()[num];
                        while (k != -1) {
                            int j = admittance[0].getJA().get(k);
                            if (num == j) {
                                if (num < pqSize)
                                    result.setValue(index, num, 1.0);
                                result.setValue(index, num + pqSize, 1.0);
                                k = admittance[0].getLINK().get(k);
                                continue;
                            }
                            if (j < pqSize)
                                result.setValue(index, j, 1.0);
                            if (j < pvSize + pqSize)
                                result.setValue(index, j + pqSize, 1.0);
                            k = admittance[0].getLINK().get(k);
                        }
                    }
                    break;
                default:
                    break;
            }
        }
        return result;
    }

    public static void getJacobianOfVTheta(MeasVector meas, YMatrixGetter y, AVector state,
                                           int pqSize, int pvSize,
                                           DoubleMatrix2D result) {
        getJacobianOfVTheta(meas, y, state.getValues(), pqSize, pvSize, result);
    }

    /**
     * @param meas   潮流计算用的量测，只包含P,Q量测
     * @param y      导纳阵
     * @param state  状态量，包含所有节点的电压幅值和相角
     * @param pqSize PQ节点个数
     * @param pvSize PV节点个数
     * @param result Jacobian矩阵
     */
    public static void getJacobianOfVTheta(MeasVector meas, YMatrixGetter y, double[] state,
                                           int pqSize, int pvSize,
                                           DoubleMatrix2D result) {
        int index = 0;
        int n = y.getAdmittance()[0].getN();
        ASparseMatrixLink[] gb = y.getAdmittance();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int pos = meas.getBus_p_pos()[i];
                        int num = pos - 1;//num starts from 0
                        int k = gb[0].getIA()[num];
                        double qi = 0;
                        while (k != -1) {
                            int j = gb[0].getJA().get(k);
                            double thetaIJ = state[num + n] - state[j + n];
                            double sin = Math.sin(thetaIJ);
                            double cos = Math.cos(thetaIJ);
                            double vi = state[num];
                            double vj = state[j];
                            qi = qi + vi * vj * (gb[0].getVA().get(k) * sin - gb[1].getVA().get(k) * cos);
                            if (num == j) {
                                if (num < pqSize)
                                    result.setQuick(index, num, (gb[0].getVA().get(k) * vi * vi
                                            + StateCalByPolar.calBusP(pos, y, state)) / vi);
                                result.setQuick(index, num + pqSize, -gb[1].getVA().get(k) * vi * vi);
                                k = gb[0].getLINK().get(k);
                                continue;
                            }
                            if (j < pqSize)
                                result.setQuick(index, j, vi * (gb[0].getVA().get(k) * cos + gb[1].getVA().get(k) * sin));
                            if (j != pvSize + pqSize)
                                result.setQuick(index, j + pqSize, vi * vj * (gb[0].getVA().get(k) * sin - gb[1].getVA().get(k) * cos));
                            k = gb[0].getLINK().get(k);
                        }
                        result.setQuick(index, num + pqSize, result.getQuick(index, num + pqSize) - qi);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int pos = meas.getBus_q_pos()[i];
                        int num = pos - 1;//num starts from 0
                        int k = gb[0].getIA()[num];
                        double pi = 0.0;
                        while (k != -1) {
                            int j = gb[0].getJA().get(k);
                            double thetaIJ = state[num + n] - state[j + n];
                            double cos = Math.cos(thetaIJ);
                            double sin = Math.sin(thetaIJ);
                            double vi = state[num];
                            double vj = state[j];
                            pi = pi + vi * vj * (gb[0].getVA().get(k) * cos + gb[1].getVA().get(k) * sin);
                            if (num == j) {
                                if (num < pqSize)
                                    result.setQuick(index, num, (-gb[1].getVA().get(k) * vi * vi
                                            + StateCalByPolar.calBusQ(pos, y, state)) / vi);
                                result.setQuick(index, num + pqSize, -gb[0].getVA().get(k) * vi * vi);
                                k = gb[0].getLINK().get(k);
                                continue;
                            }
                            if (j < pqSize)
                                result.setQuick(index, j, vi * (gb[0].getVA().get(k) * sin - gb[1].getVA().get(k) * cos));
                            if (j < pvSize + pqSize)
                                result.setQuick(index, j + pqSize, -vi * vj * (gb[0].getVA().get(k) * cos + gb[1].getVA().get(k) * sin));
                            k = gb[0].getLINK().get(k);
                        }
                        result.setQuick(index, num + pqSize, result.getQuick(index, num + pqSize) + pi);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    public static SparseDoubleMatrix2D getJacobianOfVTheta(MeasVector meas, YMatrixGetter y, AVector state) {
        return getJacobianOfVTheta(meas, y, state.getValues());
    }

    public static void getJacobianOfVTheta(MeasVector meas, YMatrixGetter y, double[] state, DoubleMatrix2D result) {
        int index = 0, n = y.getAdmittance()[0].getN();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++)
                        result.setQuick(index, meas.getBus_a_pos()[i] + n - 1, 1.0);
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++)
                        result.setQuick(index, meas.getBus_v_pos()[i] - 1, 1.0);
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++)
                        fillJacobian_bus_p(meas.getBus_p_pos()[i], y, state, result, index);
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++)
                        fillJacobian_bus_q(meas.getBus_q_pos()[i], y, state, result, index);
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
                    break;
            }
        }
    }

    /**
     * @param meas  measurements which ara
     * @param y     admittance matrix
     * @param state bus's v and angle in radian
     * @return jacobian matrix
     */
    public static SparseDoubleMatrix2D getJacobianOfVTheta(MeasVector meas, YMatrixGetter y, double[] state) {
        int n = y.getAdmittance()[0].getN();
        int capacity = getNonZeroNum(meas, y);
        SparseDoubleMatrix2D result = new MySparseDoubleMatrix2D(meas.getZ().getN(), 2 * n, capacity, 0.9, 0.95);
        getJacobianOfVTheta(meas, y, state, result);
        return result;
    }

    public static void getConstantPartOfFullState(MeasVector meas, YMatrixGetter y, DoubleMatrix2D H, int index) {
        int n = y.getAdmittance()[0].getN();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++)
                        H.setQuick(index, meas.getBus_a_pos()[i] + n - 1, 1);
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++)
                        H.setQuick(index, meas.getBus_v_pos()[i] - 1, 1.0);
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++)
                        H.setQuick(index, 2 * n + meas.getBus_p_pos()[i] - 1, 1.0);
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++)
                        H.setQuick(index, 3 * n + meas.getBus_q_pos()[i] - 1, 1.0);
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++)
                        index++;
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++)
                        index++;
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++)
                        index++;
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++)
                        index++;
                    break;
                default:
                    break;
            }
        }
    }

    public static void getVariablePartOfFullState(MeasVector meas, YMatrixGetter y, double[] state, DoubleMatrix2D H, int index) {
        int n = y.getAdmittance()[0].getN();
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
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++)
                        fillJacobian_line_from_p(meas.getLine_from_p_pos()[k], n, y, state, H, index);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++)
                        fillJacobian_line_from_q(meas.getLine_from_q_pos()[k], n, y, state, H, index);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++)
                        fillJacobian_line_to_p(meas.getLine_to_p_pos()[k], n, y, state, H, index);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++)
                        fillJacobian_line_to_q(meas.getLine_to_q_pos()[k], n, y, state, H, index);
                    break;
                default:
                    break;
            }
        }
    }

    public static int getNoneZeroNum(int[] zeroPBuses, int[] zeroQBues, YMatrixGetter y) {
        int capacity = 0;
        if (y.getConnectedBusCount() == null) {
            capacity = 2 * 3 * 2 * (zeroPBuses.length + zeroQBues.length);
        } else {
            for (Integer busNum : zeroPBuses)
                capacity += 2 * y.getConnectedBusCount()[busNum - 1] + 2;
            for (Integer busNum : zeroQBues)
                capacity += 2 * y.getConnectedBusCount()[busNum - 1] + 2;
        }
        return capacity;
    }

    public static void getJacobianOfVTheta(int[] zeroPBuses, int[] zeroQBues, YMatrixGetter y,
                                           double[] state, DoubleMatrix2D jac, int index) {
        for (int i = 0; i < zeroPBuses.length; i++, index++)
            fillJacobian_bus_p(zeroPBuses[i], y, state, jac, index);
        for (int i = 0; i < zeroQBues.length; i++, index++)
            fillJacobian_bus_q(zeroQBues[i], y, state, jac, index);
    }

    public static DoubleMatrix2D getPJacobianOfVTheta(List<Integer> linkPBuses, YMatrixGetter y, AVector state, int[] connectedBusCount) {
        int capacity = 0;
        if (connectedBusCount == null) {
            capacity = 2 * 3 * linkPBuses.size();
        } else {
            for (Integer busNum : linkPBuses)
                capacity += 2 * connectedBusCount[busNum - 1] + 2;
        }
        DoubleMatrix2D result = new MySparseDoubleMatrix2D(linkPBuses.size(), state.getN(), capacity, 0.2, 0.5);
        int index = 0;
        int n = y.getAdmittance()[0].getM();
        for (int i = 0; i < linkPBuses.size(); i++, index++) {
            fillJacobian_bus_p(linkPBuses.get(i), y, state, result, index);
        }
        return result;
    }

    public static void fillJacobian_bus_p(int pos, YMatrixGetter Y, AVector state, DoubleMatrix2D result, int index) {
        fillJacobian_bus_p(pos, Y, state.getValues(), result, index);
    }

    public static void fillJacobian_bus_p(int pos, YMatrixGetter Y, double[] state, DoubleMatrix2D result, int index) {
        ASparseMatrixLink[] GB = Y.getAdmittance();
        int n = GB[0].getN();
        int num = pos - 1;//num starts from 0
        int k = GB[0].getIA()[num];
        double qi = 0;
        while (k != -1) {
            int j = GB[0].getJA().get(k);
            double thetaIJ = state[num + n] - state[j + n];
            double sin = Math.sin(thetaIJ);
            double cos = Math.cos(thetaIJ);
            double vi = state[num];
            double vj = state[j];
            qi = qi + vi * vj * (GB[0].getVA().get(k) * sin - GB[1].getVA().get(k) * cos);
            if (num == j) {
                result.setQuick(index, num, (GB[0].getVA().get(k) * vi * vi
                        + StateCalByPolar.calBusP(pos, Y, state)) / vi);
                result.setQuick(index, num + n, -GB[1].getVA().get(k) * vi * vi);
                k = GB[0].getLINK().get(k);
                continue;
            }
            result.setQuick(index, j, vi * (GB[0].getVA().get(k) * cos + GB[1].getVA().get(k) * sin));
            result.setQuick(index, j + n, vi * vj * (GB[0].getVA().get(k) * sin - GB[1].getVA().get(k) * cos));
            k = GB[0].getLINK().get(k);
        }
        result.setQuick(index, num + n, result.getQuick(index, num + n) - qi);
    }

    public static void fillJacobian_bus_q(int pos, YMatrixGetter Y, AVector state, DoubleMatrix2D result, int index) {
        fillJacobian_bus_q(pos, Y, state.getValues(), result, index);
    }

    public static void fillJacobian_bus_q(int pos, YMatrixGetter Y, double[] state, DoubleMatrix2D result, int index) {
        ASparseMatrixLink[] GB = Y.getAdmittance();
        int n = GB[0].getN();
        int num = pos - 1;//num starts from 0
        int k = GB[0].getIA()[num];
        double pi = 0.0;
        while (k != -1) {
            int j = GB[0].getJA().get(k);
            double thetaIJ = state[num + n] - state[j + n];
            double cos = Math.cos(thetaIJ);
            double sin = Math.sin(thetaIJ);
            double vi = state[num];
            double vj = state[j];
            pi = pi + vi * vj * (GB[0].getVA().get(k) * cos + GB[1].getVA().get(k) * sin);
            if (num == j) {
                result.setQuick(index, num, (-GB[1].getVA().get(k) * vi * vi
                        + StateCalByPolar.calBusQ(pos, Y, state)) / vi);
                result.setQuick(index, num + n, -GB[0].getVA().get(k) * vi * vi);
                k = GB[0].getLINK().get(k);
                continue;
            }
            result.setQuick(index, j, vi * (GB[0].getVA().get(k) * sin - GB[1].getVA().get(k) * cos));
            result.setQuick(index, j + n, -vi * vj * (GB[0].getVA().get(k) * cos + GB[1].getVA().get(k) * sin));
            k = GB[0].getLINK().get(k);
        }
        result.setQuick(index, num + n, result.getQuick(index, num + n) + pi);
    }

    public static void fillJacobian_line_from_p(int pos, int n, YMatrixGetter Y, AVector state, DoubleMatrix2D result, int currentRow) {
        fillJacobian_line_from_p(pos, n, Y, state.getValues(), result, currentRow);
    }

    public static void fillJacobian_line_from_p(int pos, int n, YMatrixGetter Y, double[] state, DoubleMatrix2D result, int currentRow) {
        int[] ij = Y.getFromTo(pos);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(pos, YMatrixGetter.LINE_FROM);
        double thetaIJ = state[i + n] - state[j + n];
        double cos = Math.cos(thetaIJ);
        double sin = Math.sin(thetaIJ);
        result.setQuick(currentRow, i, 2 * state[i] * (gbg1b1[0] + gbg1b1[2]) - state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        result.setQuick(currentRow, j, -state[i] * (gbg1b1[0] * cos + gbg1b1[1] * sin));
        double tmp = state[i] * state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos);
        result.setQuick(currentRow, i + n, tmp);
        result.setQuick(currentRow, j + n, -tmp);
    }

    public static void fillJacobian_line_from_q(int pos, int n, YMatrixGetter Y, AVector state, DoubleMatrix2D result, int currentRow) {
        fillJacobian_line_from_q(pos, n, Y, state.getValues(), result, currentRow);
    }

    public static void fillJacobian_line_from_q(int pos, int n, YMatrixGetter Y, double[] state, DoubleMatrix2D result, int currentRow) {
        int[] ij = Y.getFromTo(pos);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(pos, YMatrixGetter.LINE_FROM);
        double thetaIJ = state[i + n] - state[j + n];
        double sin = Math.sin(thetaIJ);
        double cos = Math.cos(thetaIJ);
        result.setQuick(currentRow, i, -2 * state[i] * (gbg1b1[1] + gbg1b1[3]) - state[j] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        result.setQuick(currentRow, j, -state[i] * (gbg1b1[0] * sin - gbg1b1[1] * cos));
        double tmp = -state[i] * state[j] * (gbg1b1[0] * cos + gbg1b1[1] * sin);
        result.setQuick(currentRow, i + n, tmp);
        result.setQuick(currentRow, j + n, -tmp);
    }

    public static void fillJacobian_line_to_p(int pos, int n, YMatrixGetter Y, AVector state, DoubleMatrix2D result, int currentRow) {
        fillJacobian_line_to_p(pos, n, Y, state.getValues(), result, currentRow);
    }

    public static void fillJacobian_line_to_p(int pos, int n, YMatrixGetter Y, double[] state, DoubleMatrix2D result, int currentRow) {
        int[] ij = Y.getFromTo(pos);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(pos, YMatrixGetter.LINE_TO);
        double thetaIJ = state[i + n] - state[j + n];
        double cos = Math.cos(thetaIJ);
        double sin = Math.sin(thetaIJ);
        result.setQuick(currentRow, j, 2 * state[j] * (gbg1b1[0] + gbg1b1[2]) - state[i] * (gbg1b1[0] * cos - gbg1b1[1] * sin));
        result.setQuick(currentRow, i, -state[j] * (gbg1b1[0] * cos - gbg1b1[1] * sin));
        double tmp = -state[i] * state[j] * (gbg1b1[0] * sin + gbg1b1[1] * cos);
        result.setQuick(currentRow, j + n, tmp);
        result.setQuick(currentRow, i + n, -tmp);
    }

    public static void fillJacobian_line_to_q(int pos, int n, YMatrixGetter Y, AVector state, DoubleMatrix2D result, int currentRow) {
        fillJacobian_line_to_q(pos, n, Y, state.getValues(), result, currentRow);
    }

    public static void fillJacobian_line_to_q(int pos, int n, YMatrixGetter Y, double[] state, DoubleMatrix2D result, int currentRow) {
        int[] ij = Y.getFromTo(pos);
        int i = ij[0] - 1;
        int j = ij[1] - 1;
        double[] gbg1b1 = Y.getLineAdmittance(pos, YMatrixGetter.LINE_TO);
        double thetaIJ = state[i + n] - state[j + n];
        double sin = Math.sin(thetaIJ);
        double cos = Math.cos(thetaIJ);
        result.setQuick(currentRow, j, -2 * state[j] * (gbg1b1[1] + gbg1b1[3]) + state[i] * (gbg1b1[0] * sin + gbg1b1[1] * cos));
        result.setQuick(currentRow, i, state[j] * (gbg1b1[0] * sin + gbg1b1[1] * cos));
        double tmp = -state[i] * state[j] * (gbg1b1[0] * cos - gbg1b1[1] * sin);
        result.setQuick(currentRow, j + n, tmp);
        result.setQuick(currentRow, i + n, -tmp);
    }
}
