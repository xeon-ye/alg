package zju.pso.model;

import zju.pso.Location;
import zju.pso.OptModel;

/**
 * @Author: Fang Rui
 * @Date: 2018/6/21
 * @Time: 15:41
 */
public class DifficultModel_2 implements OptModel {

    @Override
    public double evalObj(Location location) {
        double[] x = location.getLoc();
        return Math.exp(x[0] * x[1] * x[2] * x[3] * x[4]);
    }

    @Override
    public double[] evalConstr(Location location) {
        double[] x = location.getLoc();
        double[] constr = new double[6];

        constr[0] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] + x[4] * x[4] - 10 - 0.001;
        constr[1] = x[1] * x[2] - 5 * x[3] * x[4] - 0.001;
        constr[2] = x[0] * x[0] * x[0] + x[1] * x[1] * x[1] + 1 - 0.001;

        constr[3] = -(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3] + x[4] * x[4] - 10) - 0.001;
        constr[4] = -(x[1] * x[2] - 5 * x[3] * x[4]) - 0.001;
        constr[5] = -(x[0] * x[0] * x[0] + x[1] * x[1] * x[1] + 1) - 0.001;
        return constr;
    }

    @Override
    public double[] getMinLoc() {
        return new double[]{-2.3, -2.3, -3.2, -3.2, -3.2};
    }

    @Override
    public double[] getMaxLoc() {
        return new double[]{2.3, 2.3, 3.2, 3.2, 3.2};
    }

    @Override
    public double[] getMinVel() {
        return new double[]{-0.2, -0.2, -0.3, -0.3, -0.3};
    }

    @Override
    public double[] getMaxVel() {
        return new double[]{0.2, 0.2, 0.3, 0.3, 0.3};
    }

    @Override
    public int getDimentions() {
        return 5;
    }

    @Override
    public int getMaxIter() {
        return 500;
    }
}
