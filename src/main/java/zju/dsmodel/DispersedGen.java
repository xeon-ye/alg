package zju.dsmodel;

import zju.dsntp.LcbPfModel;
import zju.matrix.ASparseMatrixLink2D;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/3/28
 */
public class DispersedGen implements DsModelCons, Serializable {
    //PV节点
    public static final int MODE_PV = 1;
    //异步电动机模型
    public static final int MODE_IM = 2;

    //模式
    private int mode;
    //用于PV节点
    private int[] phases;
    //PV节点模型的P和V
    private double[] vAmpl, pOutput;
    //下面的参数用于异步发电机模型
    private InductionMachine motor;
    //DG所连接的节点
    private DsTopoNode tn;
    //用于异步电机模型
    private double[][] jacOfMotorEdgeI;
    //变量在数组中的起始位置
    private int stateIndex, slipIndex;

    public DispersedGen(int model) {
        this.mode = model;
        if (mode == MODE_IM)
            jacOfMotorEdgeI = new double[5][6];
    }

    public int getMode() {
        return mode;
    }

    public void setMode(int mode) {
        this.mode = mode;
    }

    public InductionMachine getMotor() {
        return motor;
    }

    public void setMotor(InductionMachine motor) {
        this.motor = motor;
    }

    /**
     * 根据节点电压计算电流
     *
     * @param v        三相电压
     * @param tmpLoadI 用于存储结算结果
     */
    public void calI(double[][] v, double[][] tmpLoadI) {
        switch (mode) {
            case MODE_IM:
                motor.calI(v, tmpLoadI);
                break;
            case MODE_PV:
                break;
            default:
                break;
        }
    }

    /**
     * 计算支路电压跌落
     *
     * @param edge 支路
     * @param v    用于存储计算结果
     */
    public void calVDrop(LcbPfModel m, DetailedEdge edge, double[] v) {
        switch (mode) {
            case MODE_IM:
                getMotor().calVLN(v, edge.getPhase());
                break;
            case MODE_PV:
                int phaseIndex = getPhaseIndex(edge.getPhase());
                v[0] = m.getState().getValue(stateIndex + phaseIndex);
                v[1] = m.getState().getValue(stateIndex + phaseIndex + m.getDimension());
                break;
            default:
                break;
        }
    }

    public void fillJacStrucOfVDrop(LcbPfModel m, DetailedEdge e) {
        fillJacStrucOfVDrop(m.getJacobianStruc(), m, e, 0);
    }

    public void fillJacStrucOfVDrop(ASparseMatrixLink2D jacStruc, LcbPfModel m, DetailedEdge e, int index) {
        int k = m.getB().getJA2()[m.getEdgeToNo().get(e)], i;
        double v;
        int dim = m.getDimension();
        switch (mode) {
            case MODE_IM:
                while (k != -1) {
                    i = m.getB().getIA2().get(k);
                    v = m.getB().getVA().get(k);
                    k = m.getB().getLINK2().get(k);

                    jacStruc.setValue(i + index, stateIndex, v * InductionMachine.A_real[e.getPhase()][1], true);
                    jacStruc.setValue(i + index, stateIndex + 1, v * InductionMachine.A_real[e.getPhase()][2], true);
                    jacStruc.setValue(i + index, stateIndex + dim, -v * InductionMachine.A_imag[e.getPhase()][1], true);
                    jacStruc.setValue(i + index, stateIndex + dim + 1, -v * InductionMachine.A_imag[e.getPhase()][2], true);
                    jacStruc.setValue(i + index + m.getLoopSize(), stateIndex, v * InductionMachine.A_imag[e.getPhase()][1], true);
                    jacStruc.setValue(i + index + m.getLoopSize(), stateIndex + 1, v * InductionMachine.A_imag[e.getPhase()][2], true);
                    jacStruc.setValue(i + index + m.getLoopSize(), stateIndex + dim, v * InductionMachine.A_real[e.getPhase()][1], true);
                    jacStruc.setValue(i + index + m.getLoopSize(), stateIndex + dim + 1, v * InductionMachine.A_real[e.getPhase()][2], true);
                }
                break;
            case MODE_PV:
                while (k != -1) {
                    i = m.getB().getIA2().get(k);
                    v = m.getB().getVA().get(k);
                    k = m.getB().getLINK2().get(k);
                    int phaseIndex = getPhaseIndex(e.getPhase());
                    jacStruc.setValue(i + index, stateIndex + phaseIndex, v, true);
                    jacStruc.setValue(i + index + m.getLoopSize(), stateIndex + +phaseIndex + dim, v, true);
                }
                break;
            default:
                break;
        }
    }

    public int calZ(LcbPfModel m, int index) {
        switch (mode) {
            case MODE_IM:
                motor.calZ(false);
                System.arraycopy(motor.getZ_est().getValues(), 0, m.getZ_est().getValues(), index, motor.getStateSize());
                break;
            case MODE_PV:
                DetailedEdge e;
                double vx, vy;
                double[] c = new double[2];
                double p;
                for (int i = 0; i < phases.length; i++) {
                    e = tn.getIsland().getDetailedG().getEdge(tn.getTnNo() + "-" + phases[i], DsTopoIsland.EARTH_NODE_ID);
                    m.calCurrent(m.getEdgeToNo().get(e), m.getState(), c);
                    vx = m.getState().getValue(stateIndex + i);
                    vy = m.getState().getValue(stateIndex + i + m.getDimension());
                    p = pOutput[i];
                    if (e.getS_real() > ZERO_LIMIT)
                        p -= e.getS_real();
                    if (e.getI_ampl() > ZERO_LIMIT)
                        p -= vAmpl[i] * e.getI_ampl() * Math.cos(e.getI_angle());
                    if (e.getZ_real() > ZERO_LIMIT) {
                        double zAmple = Math.sqrt(e.getZ_real() * e.getZ_real() + e.getZ_image() * e.getZ_image());
                        p -= vAmpl[i] * vAmpl[i] * e.getZ_real() / zAmple;
                    }
                    m.getZ_est().setValue(index + i, vx * c[0] + vy * c[1] + p);
                    m.getZ_est().setValue(index + i + phases.length, vx * vx + vy * vy - vAmpl[i] * vAmpl[i]);
                }
                break;
            default:
                break;
        }
        return getStateSize();
    }

    public int fillJacStruc(LcbPfModel m, int index) {
        int dim = m.getDimension();
        DetailedEdge e;
        switch (mode) {
            case MODE_IM:
                motor.fillJacStrucOfU12(m.getJacobianStruc(), stateIndex, dim, index, slipIndex);
                //tn = motors.get(motor);
                for (int j = 0; j < motor.getStateSize(); j++, index++) {
                    for (int i = 0; i < 3; i++) {
                        e = tn.getIsland().getDetailedG().getEdge(tn.getTnNo() + "-" + i, tn.getTnNo() + "-motor-3");
                        m.fillJacStruc(m.getEdgeToNo().get(e), index, 1.0, 0);
                        m.fillJacStruc(m.getEdgeToNo().get(e), index, 1.0, dim);
                    }
                }
                break;
            case MODE_PV:
                for (int i = 0; i < phases.length; i++) {
                    e = tn.getIsland().getDetailedG().getEdge(tn.getTnNo() + "-" + phases[i], DsTopoIsland.EARTH_NODE_ID);
                    m.fillJacStruc(m.getEdgeToNo().get(e), index + i, 1.0, 0);
                    m.fillJacStruc(m.getEdgeToNo().get(e), index + i, 1.0, dim);
                    m.getJacobianStruc().setValue(index + i, stateIndex + i, 1.0);
                    m.getJacobianStruc().setValue(index + i, stateIndex + i + dim, 1.0);
                    m.getJacobianStruc().setValue(index + i + phases.length, stateIndex + i, 1.0);
                    m.getJacobianStruc().setValue(index + i + phases.length, stateIndex + i + dim, 1.0);
                }
                break;
            default:
                break;
        }
        return getStateSize();
    }

    public int fillJac(LcbPfModel m, int index) {
        int dim = m.getDimension();
        DetailedEdge e;
        switch (mode) {
            case MODE_IM:
                motor.fillJacOfU12(m.getJacobian(), stateIndex, dim, index, slipIndex);
                motor.fillJacOfI(jacOfMotorEdgeI, 0);
                for (int j = 0; j < motor.getStateSize(); j++, index++) {
                    for (int i = 0; i < 3; i++) {
                        e = tn.getIsland().getDetailedG().getEdge(tn.getTnNo() + "-" + i, tn.getTnNo() + "-motor-3");
                        m.cleanJac(m.getEdgeToNo().get(e), index, 0);
                        m.cleanJac(m.getEdgeToNo().get(e), index, dim);
                    }
                    for (int i = 0; i < 3; i++) {
                        e = tn.getIsland().getDetailedG().getEdge(tn.getTnNo() + "-" + i, tn.getTnNo() + "-motor-3");
                        m.fillJac2(m.getEdgeToNo().get(e), index, jacOfMotorEdgeI[j][i], 0);
                        m.fillJac2(m.getEdgeToNo().get(e), index, jacOfMotorEdgeI[j][i + 3], dim);
                    }
                }
                break;
            case MODE_PV:
                double vx, vy;
                double[] c = new double[2];
                for (int i = 0; i < phases.length; i++) {
                    e = tn.getIsland().getDetailedG().getEdge(tn.getTnNo() + "-" + phases[i], DsTopoIsland.EARTH_NODE_ID);
                    m.calCurrent(m.getEdgeToNo().get(e), m.getState(), c);
                    vx = m.getState().getValue(stateIndex + i);
                    vy = m.getState().getValue(stateIndex + i + m.getDimension());
                    m.fillJac(m.getEdgeToNo().get(e), index + i, vx, 0);
                    m.fillJac(m.getEdgeToNo().get(e), index + i, vy, dim);
                    m.getJacobian().setQuick(index + i, stateIndex + i, c[0]);
                    m.getJacobian().setQuick(index + i, stateIndex + i + dim, c[1]);
                    m.getJacobian().setQuick(index + i + phases.length, stateIndex + i, 2 * vx);
                    m.getJacobian().setQuick(index + i + phases.length, stateIndex + i + dim, 2 * vy);
                }
                break;
            default:
                break;
        }
        return getStateSize();
    }

    public int getStateSize() {
        switch (mode) {
            case MODE_IM:
                return motor.getStateSize();
            case MODE_PV:
                return 2 * phases.length;
            default:
                break;
        }
        return 0;
    }

    public void updateState(LcbPfModel m) {
        DetailedEdge edge;
        switch (mode) {
            case MODE_IM:
                motor.setSlip(m.getState().getValue(slipIndex));
                motor.getVLN_12()[0][0] = m.getState().getValue(stateIndex);
                motor.getVLN_12()[0][1] = m.getState().getValue(stateIndex + m.getDimension());
                motor.getVLN_12()[1][0] = m.getState().getValue(stateIndex + 1);
                motor.getVLN_12()[1][1] = m.getState().getValue(stateIndex + m.getDimension() + 1);
                double[] values = motor.getState().getValues();
                for (int i = 0; i < values.length; i++)
                    values[i] = 0.0;
                values[4] = m.getState().getValue(slipIndex);
                double[] tempI = new double[2];
                for (int i = 0; i < 3; i++) {
                    edge = tn.getIsland().getDetailedG().getEdge(tn.getTnNo() + "-" + i, tn.getTnNo() + "-motor-3");
                    m.calCurrent(m.getEdgeToNo().get(edge), m.getState(), tempI);
                    values[0] += InductionMachine.A_inv_real[1][i] * tempI[0] - InductionMachine.A_inv_imag[1][i] * tempI[1];
                    values[2] += InductionMachine.A_inv_real[1][i] * tempI[1] + InductionMachine.A_inv_imag[1][i] * tempI[0];
                    values[1] += InductionMachine.A_inv_real[2][i] * tempI[0] - InductionMachine.A_inv_imag[2][i] * tempI[1];
                    values[3] += InductionMachine.A_inv_real[2][i] * tempI[1] + InductionMachine.A_inv_imag[2][i] * tempI[0];
                }
                break;
            default:
                break;
        }
    }

    public void initialState(LcbPfModel m) {
        double[][] VLN_abc = tn.getIsland().getBusV().get(tn);
        switch (mode) {
            case MODE_IM:
                double[][] VLL_abc = motor.getVLL_abc();
                for (int j = 0; j < 3; j++) {
                    VLL_abc[j][0] = VLN_abc[j][0] - VLN_abc[(j + 1) % 3][0];
                    VLL_abc[j][1] = VLN_abc[j][1] - VLN_abc[(j + 1) % 3][1];
                }
                motor.setVLL_abc(VLL_abc);
                break;
            case MODE_PV:
                for(int i = 0; i < phases.length; i++) {
                    m.getState().setValue(stateIndex + i, VLN_abc[phases[i]][0]);
                    m.getState().setValue(stateIndex + i + m.getDimension(), VLN_abc[phases[i]][1]);
                }
        }
    }

    public void updateOuterState(LcbPfModel m) {
        switch (mode) {
            case MODE_IM:
                m.getState().setValue(stateIndex, motor.getVLN_12()[0][0]);
                m.getState().setValue(stateIndex + 1, motor.getVLN_12()[1][0]);
                m.getState().setValue(stateIndex + m.getDimension(), motor.getVLN_12()[0][1]);
                m.getState().setValue(stateIndex + m.getDimension() + 1, motor.getVLN_12()[1][1]);
                break;
            default:
                break;
        }
    }

    public DsTopoNode getTn() {
        return tn;
    }

    public void setTn(DsTopoNode tn) {
        this.tn = tn;
    }

    public int getStateIndex() {
        return stateIndex;
    }

    public void setStateIndex(int stateIndex) {
        this.stateIndex = stateIndex;
    }

    public int getSlipIndex() {
        return slipIndex;
    }

    public void setSlipIndex(int slipIndex) {
        this.slipIndex = slipIndex;
    }

    public int[] getPhases() {
        return phases;
    }

    public void setPhases(int[] phases) {
        this.phases = phases;
    }

    public double[] getvAmpl() {
        return vAmpl;
    }

    public void setvAmpl(double[] vAmpl) {
        this.vAmpl = vAmpl;
    }

    public double[] getpOutput() {
        return pOutput;
    }

    public void setpOutput(double[] pOutput) {
        this.pOutput = pOutput;
    }

    public int getPhaseIndex(int phase) {
        for (int i = 0; i < phases.length; i++)
            if (phases[i] == phase)
                return i;
        return -1;
        //return phase;
    }
}
