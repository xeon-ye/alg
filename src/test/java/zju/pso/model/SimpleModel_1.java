package zju.pso.model;

import zju.pso.Location;
import zju.pso.OptModel;
import zju.pso.ParallelOptModel;

/**
 * 使用粒子群算法求解非线性规划
 * f(x, y) = (2.8125 - x + x * y^4)^2 + (2.25 - x + x * y^2)^2 + (1.5 - x + x*y)^2
 * where 1 <= x <= 4, and -1 <= y <= 1
 *
 * @Author: Fang Rui
 * @Date: 2018/6/19
 * @Time: 14:53
 */
public class SimpleModel_1 implements OptModel, ParallelOptModel {

    @Override
    public double evalObj(Location location) {
        double x = location.getLoc()[0];
        double y = location.getLoc()[1];

        return Math.pow(2.8125 - x + x * Math.pow(y, 4), 2) +
                Math.pow(2.25 - x + x * Math.pow(y, 2), 2) +
                Math.pow(1.5 - x + x * y, 2);
    }

    @Override
    public int getDimentions() {
        return 2;
    }

    @Override
    public double[] getMinLoc() {
        return new double[]{1, -1};
    }

    @Override
    public double[] getMaxLoc() {
        return new double[]{4, 1};
    }

    @Override
    public double[] getMinVel() {
        return new double[]{-1, -1};
    }

    @Override
    public double[] getMaxVel() {
        return new double[]{1, 1};
    }

    @Override
    public double paraEvalObj(float[] location, int offset) {
        double x = location[offset];
        double y = location[offset + 1];

        return Math.pow(2.8125 - x + x * Math.pow(y, 4), 2) +
                Math.pow(2.25 - x + x * Math.pow(y, 2), 2) +
                Math.pow(1.5 - x + x * y, 2);
    }

    @Override
    public float[] paraGetMinLoc() {
        return new float[]{1, -1};
    }

    @Override
    public float[] paraGetMaxLoc() {
        return new float[]{4, 1};
    }

    @Override
    public float[] paraGetMinVel() {
        return new float[]{-1, -1};
    }

    @Override
    public float[] paraGetMaxVel() {
        return new float[]{1, 1};
    }

    @Override
    public int getMaxIter() {
        return 1000;
    }

    @Override
    public double getTolFitness() {
        return -99999;
    }
}
