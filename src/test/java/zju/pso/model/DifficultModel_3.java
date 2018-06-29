package zju.pso.model;

import zju.pso.Location;
import zju.pso.OptModel;

/**
 * @Author: Fang Rui
 * @Date: 2018/6/22
 * @Time: 10:03
 */
public class DifficultModel_3 implements OptModel {

    @Override
    public double evalObj(Location location) {
        double[] x = location.getLoc();
        return x[0] * x[0] + (x[1] - 1) * (x[1] - 1);
    }

    @Override
    public double[] evalConstr(Location location) {
        double[] x = location.getLoc();
        double[] constr = new double[1];
        constr[0] = Math.abs(x[1] - x[0] * x[0]) - 1e-3;
        return constr;
    }

    @Override
    public double[] getMinLoc() {
        return new double[]{-1, -1};
    }

    @Override
    public double[] getMaxLoc() {
        return new double[]{1, 1};
    }

    @Override
    public double[] getMinVel() {
        return new double[]{-0.5, -0.5};
    }

    @Override
    public double[] getMaxVel() {
        return new double[]{0.5, 0.5};
    }

    @Override
    public int getDimentions() {
        return 2;
    }

}
