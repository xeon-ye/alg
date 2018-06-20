package zju.pso;

import org.junit.Test;

/**
 * @Description: 测试pso的非线性规划求解
 * @Author: Fang Rui
 * @Date: 2018/6/19
 * @Time: 14:53
 */
public class PsoTest implements OptModel {

    @Test
    public void test() {
        new PsoProcess(this, 30).execute();
    }

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
}
