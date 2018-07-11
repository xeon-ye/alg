package zju.se;

import zju.matrix.AVector;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;
import zju.pso.HybridPso;
import zju.pso.Location;
import zju.pso.OptModel;
import zju.pso.PsoProcess;
import zju.util.PfUtil;
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
    private double objective;

    private void initial() {
        if (getSlackBusNum() > 0)
            slackBusCol = getSlackBusNum() - 1; // 得到松弛节点的列号
        busNumber = Y.getAdmittance()[0].getM(); // 节点数量

    }

    @Override
    public AVector getFinalVTheta() {
        return null;
    }

    @Override
    public void doSeAnalyse() {
        long start = System.currentTimeMillis();
        initial();
        HybridPso solver = new HybridPso(this, 1000);
        solver.execute();

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

        assert z_est.length == z.length;
        double[] threshold = objFunc.getThresholds();
        double obj = 0;
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
                minLoc[i] = 1.;
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
            minVel[i] = (minLoc[i] - maxLoc[i]) * 0.5;
        }
        return minVel;
    }

    @Override
    public double[] getMaxVel() {
        double[] minLoc = getMinLoc();
        double[] maxLoc = getMaxLoc();
        double[] maxVel = new double[getDimentions()];
        for (int i = 0; i < getDimentions(); i++) {
            maxVel[i] = (maxLoc[i] - minLoc[i]) * 0.5;
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

    public MeasVector getPqMeasure() {
        return pqMeasure;
    }

    public void setPqMeasure(MeasVector pqMeasure) {
        this.pqMeasure = pqMeasure;
    }

    public SeObjective getObjFunc() {
        return objFunc;
    }
}
