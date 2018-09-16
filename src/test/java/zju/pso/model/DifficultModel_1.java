package zju.pso.model;

import zju.pso.Location;
import zju.pso.OptModel;
import zju.pso.ParallelOptModel;

import java.util.Arrays;

/**
 * @Author: Fang Rui
 * @Date: 2018/6/20
 * @Time: 21:16
 */
public class DifficultModel_1 implements OptModel, ParallelOptModel {

    @Override
    public double evalObj(Location location) {
        double[] x = location.getLoc();
        double obj = 0;
        for (int i = 0; i < 4; i++) {
            obj += 5 * x[i];
        }
        for (int i = 0; i < 4; i++) {
            obj -= 5 * x[i] * x[i];
        }
        for (int i = 4; i < 13; i++) {
            obj -= x[i];
        }
        return obj;
    }

    @Override
    public double evalConstr(Location location) {
        double[] x = location.getLoc();
        double constr = 0;

        if (2 * x[0] + 2 * x[1] + x[9] + x[10] - 10 > 0)
            constr += 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10;
        if (2 * x[0] + 2 * x[2] + x[9] + x[11] - 10 > 0)
            constr += 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10;
        if (2 * x[1] + 2 * x[2] + x[10] + x[11] - 10 > 0)
            constr += 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10;
        if (-8 * x[0] + x[9] > 0)
            constr += -8 * x[0] + x[9];
        if (-8 * x[1] + x[10] > 0)
            constr += -8 * x[1] + x[10];
        if (-8 * x[2] + x[11] > 0)
            constr += -8 * x[2] + x[11];
        if (-2 * x[3] - x[4] + x[9] > 0)
            constr += -2 * x[3] - x[4] + x[9];
        if (-2 * x[5] - x[6] + x[10] > 0)
            constr += -2 * x[5] - x[6] + x[10];
        if (-2 * x[7] - x[8] + x[11] > 0)
            constr += -2 * x[7] - x[8] + x[11];

        return constr;
    }

    @Override
    public double[] getMinLoc() {
        double[] minLoc = new double[getDimentions()];
        Arrays.fill(minLoc, 0, 9, 0);
        Arrays.fill(minLoc, 9, 12, 0);
        minLoc[12] = 0;
        return minLoc;
    }

    @Override
    public double[] getMaxLoc() {
        double[] maxLoc = new double[getDimentions()];
        Arrays.fill(maxLoc, 0, 9, 1);
        Arrays.fill(maxLoc, 9, 12, 100);
        maxLoc[12] = 1;
        return maxLoc;
    }

    @Override
    public double[] getMinVel() {
        double[] minVel = new double[getDimentions()];
        Arrays.fill(minVel, 0, 9, -0.5);
        Arrays.fill(minVel, 9, 12, -50);
        minVel[12] = -0.5;
        return minVel;
    }

    @Override
    public double[] getMaxVel() {
        double[] maxVel = new double[getDimentions()];
        Arrays.fill(maxVel, 0, 9, 0.5);
        Arrays.fill(maxVel, 9, 12, 50);
        maxVel[12] = 0.5;
        return maxVel;
    }

    @Override
    public double paraEvalObj(float[] location, int offset) {
        double obj = 0;
        for (int i = offset; i < offset + 4; i++) {
            obj += 5 * location[i];
        }
        for (int i = offset; i < offset + 4; i++) {
            obj -= 5 * location[i] * location[i];
        }
        for (int i = offset + 4; i < offset + 13; i++) {
            obj -= location[i];
        }
        return obj;
    }

    @Override
    public float paraEvalConstr(float[] location, int offset) {
        float constr = 0;

        if (2 * location[offset] + 2 * location[offset + 1] + location[offset + 9] + location[offset + 10] - 10 > 0)
            constr += 2 * location[offset] + 2 * location[offset + 1] + location[offset + 9] + location[offset + 10] - 10;
        if (2 * location[offset] + 2 * location[offset + 2] + location[offset + 9] + location[offset + 11] - 10 > 0)
            constr += 2 * location[offset] + 2 * location[offset + 2] + location[offset + 9] + location[offset + 11] - 10;
        if (2 * location[offset + 1] + 2 * location[offset + 2] + location[offset + 10] + location[offset + 11] - 10 > 0)
            constr += 2 * location[offset + 1] + 2 * location[offset + 2] + location[offset + 10] + location[offset + 11] - 10;
        if (-8 * location[offset] + location[offset + 9] > 0)
            constr += -8 * location[offset] + location[offset + 9];
        if (-8 * location[offset + 1] + location[offset + 10] > 0)
            constr += -8 * location[offset + 1] + location[offset + 10];
        if (-8 * location[offset + 2] + location[offset + 11] > 0)
            constr += -8 * location[offset + 2] + location[offset + 11];
        if (-2 * location[offset + 3] - location[offset + 4] + location[offset + 9] > 0)
            constr += -2 * location[offset + 3] - location[offset + 4] + location[offset + 9];
        if (-2 * location[offset + 5] - location[offset + 6] + location[offset + 10] > 0)
            constr += -2 * location[offset + 5] - location[offset + 6] + location[offset + 10];
        if (-2 * location[offset + 7] - location[offset + 8] + location[offset + 11] > 0)
            constr += -2 * location[offset + 7] - location[offset + 8] + location[offset + 11];

        return constr;
    }

    @Override
    public float[] paraGetMinLoc() {
        float[] minLoc = new float[getDimentions()];
        Arrays.fill(minLoc, 0, 9, 0);
        Arrays.fill(minLoc, 9, 12, 0);
        minLoc[12] = 0;
        return minLoc;
    }

    @Override
    public float[] paraGetMaxLoc() {
        float[] maxLoc = new float[getDimentions()];
        Arrays.fill(maxLoc, 0, 9, 1);
        Arrays.fill(maxLoc, 9, 12, 100);
        maxLoc[12] = 1;
        return maxLoc;
    }

    @Override
    public float[] paraGetMinVel() {
        float[] minVel = new float[getDimentions()];
        Arrays.fill(minVel, 0, 9, -0.5f);
        Arrays.fill(minVel, 9, 12, -50);
        minVel[12] = -0.5f;
        return minVel;
    }

    @Override
    public float[] paraGetMaxVel() {
        float[] maxVel = new float[getDimentions()];
        Arrays.fill(maxVel, 0, 9, 0.5f);
        Arrays.fill(maxVel, 9, 12, 50);
        maxVel[12] = 0.5f;
        return maxVel;
    }

    @Override
    public int getDimentions() {
        return 13;
    }

    @Override
    public int getMaxIter() {
        return 2000;
    }

    @Override
    public double getTolFitness() {
        return -99999;
    }
}
