package zju.se;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.matrix.AVector;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;
import zju.pso.HybridPso;
import zju.pso.Location;
import zju.pso.OptModel;
import zju.util.StateCalByPolar;
import zju.util.StateCalByRC;

/**
 * @Description: 状态估计的粒子群算法
 * @Author: Fang Rui
 * @Date: 2018/5/29
 * @Time: 11:40
 */
public class PsoSeAlg extends AbstractSeAlg implements OptModel, MeasTypeCons {

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
        HybridPso solver;
        if (isWarmStart) {
            solver = new HybridPso(this, 1000, initVariableState);
        } else {
            solver = new HybridPso(this, 1000);
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

//        double[] a = objFunc.getA();
//        double[] b = objFunc.getB();
//        for (int i = 0; i < z_est.length; i++) {
//            double v = z_est[i] - z[i];
//            double y1 = 1.0 / (1.0 + Math.exp(-b[i] * (v - a[i])));
//            double y2 = 1.0 / (1.0 + Math.exp(b[i] * (v + a[i])));
//            obj += (y1 + y2);
//        }
        double[] threshold = objFunc.getThresholds();
        for (int i = 0; i < z_est.length; i++) {
            double d = (z_est[i] - z[i]) / threshold[i]; // 测点相对偏移
            if (Math.abs(d) > 1)
                obj++;
        }
        return obj;
    }

    @Override
    public double[] evalConstr(Location location) {
        // 处理等式约束
        double[] variableState = location.getLoc();
        int[] zeroPBuses = this.zeroPBuses;
        int[] zeroQBuses = this.zeroQBuses;
        double[] constr = new double[zeroPBuses.length + zeroQBuses.length];
        int index = 0;
        for (int zeroPBuse : zeroPBuses) {
            double p = StateCalByPolar.calBusP(zeroPBuse, Y, variableState);
            constr[index++] = Math.abs(p) - tol_p;
        }
        for (int zeroQBus : zeroQBuses) {
            double q = StateCalByPolar.calBusQ(zeroQBus, Y, variableState);
            constr[index++] = Math.abs(q) - tol_q;
        }

        return constr;
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
    public int getDimentions() {
        return variableState.length;
    }

    @Override
    public int getMaxIter() {
        return 100000;
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
