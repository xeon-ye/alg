package zju.se;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.matrix.AVector;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;
import zju.pso.*;
import zju.util.StateCalByPolar;
import zju.util.StateCalByRC;

/**
 * @Description: 状态估计的粒子群算法
 * @Author: Fang Rui
 * @Date: 2018/5/29
 * @Time: 11:40
 */
public class PsoSeAlg extends AbstractSeAlg implements OptModel, ParallelOptModel, MeasTypeCons {

    protected SeObjective objFunc = new SeObjective();
    private MeasVector pqMeasure;
    private int slackBusCol;
    private int busNumber;
    private double[] initVariableState;
    private boolean isWarmStart;
    private double objective;

    private void initial() {
        if (getSlackBusNum() > 0)
            slackBusCol = getSlackBusNum() - 1; // 得到松弛节点的列号
        busNumber = Y.getAdmittance()[0].getM(); // 节点数量
        isSlackBusVoltageFixed = false;
    }

    @Override
    public AVector getFinalVTheta() {
        switch (variable_type) {
            case VARIABLE_VTHETA:
                return new AVector(variableState);
            case VARIABLE_VTHETA_PQ:
                double[] vTheta = new double[2 * busNumber];
                System.arraycopy(variableState, 0, vTheta, 0, vTheta.length);
                return new AVector(vTheta);
            case VARIABLE_U:
            case VARIABLE_UI:
                double[] state = new double[2 * busNumber];
                for (int i = 0; i < busNumber; i++) {
                    double a = variableState[i];
                    double b = variableState[i + busNumber];
                    state[i] = Math.sqrt(a * a + b * b);
                    state[i + busNumber] = Math.atan2(b, a);
                }
                return new AVector(state);
            default:
                return null;
        }
    }

    @Override
    public void doSeAnalyse() {
        long start = System.currentTimeMillis();
        initial();
//        HybridPso solver;
//        if (isWarmStart) {
//            solver = new HybridPso(this, initVariableState);
//        } else {
//            solver = new HybridPso(this);
//        }
        ParallelPso solver;
        if (isWarmStart) {
            solver = new ParallelPso(this, initVariableState);
        } else {
            solver = new ParallelPso(this);
        }
        solver.execute();
        variableState = solver.getgBestLocation().getLoc();
        if (variable_type == IpoptSeAlg.VARIABLE_VTHETA)
            StateCalByPolar.getEstimatedZ(meas, Y, variableState); //获得了测点的估计值
        else if (variable_type == IpoptSeAlg.VARIABLE_U)
            StateCalByRC.getEstimatedZ_U(meas, Y, variableState); //获得了测点的估计值

        if (solver.isGBestfeasible())
            objective = solver.getgBest();

        setTimeUsed(System.currentTimeMillis() - start);
    }

    @Override
    public double evalObj(Location location) {
        double[] variableState = location.getLoc();
        double[] z_est = new double[0];

        if (variable_type == IpoptSeAlg.VARIABLE_VTHETA)
            z_est = StateCalByPolar.getEstimatedZ(meas, Y, variableState).getValues(); //获得了测点的估计值
        else if (variable_type == IpoptSeAlg.VARIABLE_U)
            z_est = StateCalByRC.getEstimatedZ_U(meas, Y, variableState).getValues(); //获得了测点的估计值
        double[] z = meas.getZ().getValues(); // 获得了测点的量测值
        double obj = 0;

        double[] threshold = objFunc.getThresholds();
        for (int i = 0; i < z_est.length; i++) {
            double d = (z_est[i] - z[i]) / threshold[i]; // 测点相对偏移
            if (Math.abs(d) > 1)
                obj++;
        }
        return obj;
    }

    @Override
    public double paraEvalObj(float[] location, int offset) {

        float[] z_est = StateCalByPolar.getEstimatedZ(meas, Y, location, offset); //获得了测点的估计值
        double[] z = meas.getZ().getValues(); // 获得了测点的量测值
        double obj = 0;

        double[] threshold = objFunc.getThresholds();
        for (int i = 0; i < z_est.length; i++) {
            double d = (z_est[i] - z[i]) / threshold[i]; // 测点相对偏移
            if (Math.abs(d) > 1)
                obj++;
        }
        return obj;
    }

    @Override
    public double evalConstr(Location location) {
        // 处理等式约束
        double violation = 0;
        double[] variableState = location.getLoc();
        int[] zeroPBuses = this.zeroPBuses;
        int[] zeroQBuses = this.zeroQBuses;
        double[] constr = new double[zeroPBuses.length + zeroQBuses.length];
        for (int zeroPBuse : zeroPBuses) {
            double p = StateCalByPolar.calBusP(zeroPBuse, Y, variableState);
            double deviation = Math.abs(p) - tol_p;
            if (deviation > 0)
                violation += deviation;
        }
        for (int zeroQBus : zeroQBuses) {
            double q = StateCalByPolar.calBusQ(zeroQBus, Y, variableState);
            double deviation = Math.abs(q) - tol_q;
            if (deviation > 0)
                violation += deviation;
        }
        return violation;
    }

    @Override
    public float paraEvalConstr(float[] location, int offset) {
        // 处理等式约束
        float violation = 0;
        int[] zeroPBuses = this.zeroPBuses;
        int[] zeroQBuses = this.zeroQBuses;
        for (int zeroPBuse : zeroPBuses) {
            float p = StateCalByPolar.calBusP(zeroPBuse, Y, location, offset);
            float deviation = Math.abs(p) - (float) tol_p;
            if (deviation > 0)
                violation += deviation;
        }
        for (int zeroQBus : zeroQBuses) {
            float q = StateCalByPolar.calBusQ(zeroQBus, Y, location, offset);
            float deviation = Math.abs(q) - (float) tol_q;
            if (deviation > 0)
                violation += deviation;
        }
        return violation;
    }

    @Override
    public double[] getMinLoc() {
        double[] minLoc = new double[getDimentions()];
        if (variable_type == VARIABLE_VTHETA
                || variable_type == VARIABLE_VTHETA_PQ) {
            for (int i = 0; i < busNumber; i++) {
                minLoc[i] = 0.9;
                minLoc[i + busNumber] = -Math.PI / 2;
            }
            if (slackBusCol >= 0) {
                minLoc[slackBusCol + busNumber] = getSlackBusAngle();
                if (isSlackBusVoltageFixed()) {
                    minLoc[slackBusCol] = getSlackBusVoltage();
                }
            }
        } else {
            for (int i = 0; i < 2 * busNumber; i++) {
                minLoc[i] = -2.0;
            }
        }
        return minLoc;
    }

    @Override
    public float[] paraGetMinLoc() {
        float[] minLoc = new float[getDimentions()];
        if (variable_type == VARIABLE_VTHETA
                || variable_type == VARIABLE_VTHETA_PQ) {
            for (int i = 0; i < busNumber; i++) {
                minLoc[i] = 0.9f;
                minLoc[i + busNumber] = (float) (-Math.PI / 2);
            }
            if (slackBusCol >= 0) {
                minLoc[slackBusCol + busNumber] = (float) getSlackBusAngle();
                if (isSlackBusVoltageFixed()) {
                    minLoc[slackBusCol] = (float) getSlackBusVoltage();
                }
            }
        } else {
            for (int i = 0; i < 2 * busNumber; i++) {
                minLoc[i] = -2.0f;
            }
        }
        return minLoc;
    }

    @Override
    public double[] getMaxLoc() {
        double[] maxLoc = new double[getDimentions()];
        if (variable_type == VARIABLE_VTHETA
                || variable_type == VARIABLE_VTHETA_PQ) {
            for (int i = 0; i < busNumber; i++) {
                maxLoc[i] = 1.1;
                maxLoc[i + busNumber] = Math.PI / 2;
            }
            if (slackBusCol >= 0) {
                maxLoc[slackBusCol + busNumber] = getSlackBusAngle();
                if (isSlackBusVoltageFixed()) {
                    maxLoc[slackBusCol] = getSlackBusVoltage();
                }
            }
        } else {
            for (int i = 0; i < 2 * busNumber; i++) {
                maxLoc[i] = 2;
            }
        }
        return maxLoc;
    }

    @Override
    public float[] paraGetMaxLoc() {
        float[] maxLoc = new float[getDimentions()];
        if (variable_type == VARIABLE_VTHETA
                || variable_type == VARIABLE_VTHETA_PQ) {
            for (int i = 0; i < busNumber; i++) {
                maxLoc[i] = 1.1f;
                maxLoc[i + busNumber] = (float) (Math.PI / 2);
            }
            if (slackBusCol >= 0) {
                maxLoc[slackBusCol + busNumber] = (float) getSlackBusAngle();
                if (isSlackBusVoltageFixed()) {
                    maxLoc[slackBusCol] = (float) getSlackBusVoltage();
                }
            }
        } else {
            for (int i = 0; i < 2 * busNumber; i++) {
                maxLoc[i] = 2f;
            }
        }
        return maxLoc;
    }

    @Override
    public double[] getMinVel() {
        double[] minLoc = getMinLoc();
        double[] maxLoc = getMaxLoc();
        double[] minVel = new double[getDimentions()];
        for (int i = 0; i < getDimentions(); i++) {
            minVel[i] = (minLoc[i] - maxLoc[i]) * 0.2;
        }
        return minVel;
    }

    @Override
    public float[] paraGetMinVel() {
        float[] minLoc = paraGetMinLoc();
        float[] maxLoc = paraGetMaxLoc();
        float[] minVel = new float[getDimentions()];
        for (int i = 0; i < getDimentions(); i++) {
            minVel[i] = (minLoc[i] - maxLoc[i]) * 0.2f;
        }
        return minVel;
    }

    @Override
    public double[] getMaxVel() {
        double[] minLoc = getMinLoc();
        double[] maxLoc = getMaxLoc();
        double[] maxVel = new double[getDimentions()];
        for (int i = 0; i < getDimentions(); i++) {
            maxVel[i] = (maxLoc[i] - minLoc[i]) * 0.2;
        }
        return maxVel;
    }

    @Override
    public float[] paraGetMaxVel() {
        float[] minLoc = paraGetMinLoc();
        float[] maxLoc = paraGetMaxLoc();
        float[] maxVel = new float[getDimentions()];
        for (int i = 0; i < getDimentions(); i++) {
            maxVel[i] = (maxLoc[i] - minLoc[i]) * 0.2f;
        }
        return maxVel;
    }

    @Override
    public int getDimentions() {
        return variableState.length;
    }

    @Override
    public int getMaxIter() {
        return 10000;
    }

    @Override
    public double getTolFitness() {
        return -99999;
    }

    public MeasVector getPqMeasure() {
        return pqMeasure;
    }

    public void setPqMeasure(MeasVector pqMeasure) {
        this.pqMeasure = pqMeasure;
    }

    public SeObjective getObjFunc() {
        return objFunc;
    }

    public void setWarmStart(boolean warmStart) {
        isWarmStart = warmStart;
    }

    public double[] getInitVariableState() {
        return initVariableState;
    }

    public void setInitVariableState(double[] initVariableState) {
        this.initVariableState = initVariableState;
    }
}
