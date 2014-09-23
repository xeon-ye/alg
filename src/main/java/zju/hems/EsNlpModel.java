package zju.hems;

import zju.common.IpoptModel;
import zju.common.IpoptSolver;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/5/23
 */
public class EsNlpModel implements IpoptModel {

    //最大迭代次数
    private int maxIter = 500;
    //收敛精度,目标函数,存储效率,储能能量初值,储能最大变化量
    private double tolerance = 1e-4, objective, iniEnergy, maxEnergyChange;
    //充电效率,放电效率，如果这两者不一样则目前程序还没有处理
    private double esChargeEff = 0.7, esDischargeEff = 0.7;
    //用于平滑的参数
    private double buffer = 0.001;
    //收敛结果
    boolean isConverged;
    //problem sizes
    protected int n, m, nele_jac, nele_hess;
    //储能能量上限,储能能量下限,负荷需求,电价,优化结果
    double[] x_L, x_U, pNeed, pricePerKwh, result;
    //平滑充电和放电曲线的参数
    double x1,x2, x_c, y_c, r_square;

    /**
     * 执行优化
     */
    public void doEsOpt() {
        initial();
        IpoptSolver solver = new IpoptSolver(this);
        solver.solve(maxIter, tolerance, true, true);
        setConverged(solver.isConverged());
        objective = solver.getObjective();
    }

    private void initial() {
        m = pNeed.length;
        n = pNeed.length - 1;
        result = new double[n];

        nele_jac = 2 * n;
        nele_hess = 2 * n - 1;

        //x1 = buffer / Math.sqrt(1 + esChargeEff * esChargeEff);
        //x2 = -buffer / Math.sqrt(1 + esDischargeEff * esDischargeEff);
        //double y1 = buffer * esChargeEff / Math.sqrt(1 + esChargeEff * esChargeEff);
        //double y2 = -buffer * esDischargeEff / Math.sqrt(1 + esDischargeEff * esDischargeEff);
        //double b1 = y1 + x1 / esChargeEff;
        //double b2 = y2 + x2 / esDischargeEff;
        //x_c = (b2 - b1) / (1 / esDischargeEff - 1 / esChargeEff);
        //y_c = -x_c / esDischargeEff + b2;
        //r_square = (x1 - x_c) * (x1 - x_c) + (y1 - y_c) * (y1 - y_c);//这里实际上半径的平方
        x1 = 0;
        x2 = 0;
    }

    /**
     * 设置初值
     * @param x 待设置的变量
     */
    public void setStartingPoint(double[] x) {
        for (int i = 0; i < x.length; i++)
            x[i] = iniEnergy;
    }

    @Override
    public boolean isPrintPath() {
        return true;
    }

    /**
     * 求解目标函数
     */
    public boolean eval_f(int n, double[] x, boolean new_x, double[] obj_value) {
        double tmp, a, b, r, deltaE;
        obj_value[0] = 0;
        for (int i = 0; i < n + 1; i++) {
            if (i == 0) {
                deltaE = x[i] - iniEnergy;
                tmp = pNeed[i] + esChargeEff * (x[i] - iniEnergy);
            } else if (i == n) {
                deltaE = iniEnergy - x[i - 1];
                tmp = pNeed[i] + esChargeEff * (iniEnergy - x[i - 1]);
            } else {
                deltaE = x[i] - x[i - 1];
                tmp = pNeed[i] + esChargeEff * (x[i] - x[i - 1]);
            }

            //计算tmp
            //if (deltaE > x1) {//充电
            //    tmp = pNeed[i] + esChargeEff * deltaE;
            //} else {
            //    if (deltaE < x2) {
            //        tmp = pNeed[i] + esDischargeEff * deltaE;
            //    } else {
            //        tmp = pNeed[i] + y_c + Math.sqrt(r_square - (deltaE - x_c) * (deltaE - x_c));
            //    }
            //}
            //计算tmp结束

            a = buffer / Math.sqrt(1 + pricePerKwh[i] * pricePerKwh[i]);
            if (tmp >= a) {
                obj_value[0] += pricePerKwh[i] * tmp;
            } else if (tmp > -buffer) {
                b = buffer * pricePerKwh[i] / Math.sqrt(1 + pricePerKwh[i] * pricePerKwh[i]);
                r = buffer / pricePerKwh[i] + b + a / pricePerKwh[i];
                obj_value[0] += r - Math.sqrt(r * r - (tmp + buffer) * (tmp + buffer));
            }
        }
        return true;
    }

    /**
     * 求解目标函数的导数
     */
    public boolean eval_grad_f(int n, double[] x, boolean new_x, double[] grad_f) {
        for (int i = 0; i < grad_f.length; i++)
            grad_f[i] = 0.0;
        double tmp, tmp2, a, b, r, deltaE;
        double grad0;
        for (int i = 0; i < n + 1; i++) {
            if (i == 0) {
                deltaE = x[i] - iniEnergy;
                tmp = pNeed[i] + esChargeEff * (x[i] - iniEnergy);
            } else if (i == n) {
                deltaE = iniEnergy - x[i - 1];
                tmp = pNeed[i] + esChargeEff * (iniEnergy - x[i - 1]);
            } else {
                deltaE = x[i] - x[i - 1];
                tmp = pNeed[i] + esChargeEff * (x[i] - x[i - 1]);
            }
            //计算grad0
            if (deltaE > x1) {//充电
                grad0 = esChargeEff;
            } else {
                if (deltaE < x2) {
                    grad0 = esDischargeEff;
                } else {
                    grad0 = (x_c - deltaE) / Math.sqrt(r_square - (deltaE - x_c) * (deltaE - x_c));
                }
            }
            grad0 = esChargeEff;
            //计算grad0结束

            a = buffer / Math.sqrt(1 + pricePerKwh[i] * pricePerKwh[i]);
            if (tmp >= a) {
                if (i != n)
                    grad_f[i] += pricePerKwh[i] * esChargeEff;
                if (i != 0)
                    grad_f[i - 1] -= pricePerKwh[i] * esChargeEff;
            } else if (tmp > -buffer) {
                b = buffer * pricePerKwh[i] / Math.sqrt(1 + pricePerKwh[i] * pricePerKwh[i]);
                r = buffer / pricePerKwh[i] + b + a / pricePerKwh[i];
                tmp2 = (tmp + buffer) / Math.sqrt(r * r - (tmp + buffer) * (tmp + buffer));
                if (i != n)
                    grad_f[i] += tmp2 * grad0;
                if (i != 0)
                    grad_f[i - 1] -= tmp2 * grad0;
            }
        }
        return true;
    }

    /**
     * 求解约束的值
     */
    public boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g) {
        for (int i = 0; i < m; i++) {
            if (i == 0)
                g[i] = x[i] - iniEnergy;
            else if (i == n)
                g[i] = iniEnergy - x[i - 1];
            else
                g[i] = x[i] - x[i - 1];
        }
        return true;
    }

    @Override
    public boolean eval_jac_g(int n, double[] x, boolean new_x, int m, int nele_jac, int[] iRow, int[] jCol, double[] values) {
        if (values == null) {
            int count = 0;
            for (int i = 0; i < m; i++) {
                iRow[count] = i;
                if (i == 0)
                    jCol[count] = i;
                else if (i == n)
                    jCol[count] = i - 1;
                else {
                    jCol[count++] = i;
                    iRow[count] = i;
                    jCol[count] = i - 1;
                }
                count++;
            }
        } else {
            int count = 0;
            for (int i = 0; i < m; i++) {
                if (i == 0)
                    values[count++] = 1.0;
                else if (i == n)
                    values[count++] = -1.0;
                else {
                    values[count++] = 1.0;
                    values[count++] = -1.0;
                }
            }
        }
        return true;
    }

    @Override
    public boolean eval_h(int n, double[] x, boolean new_x, double obj_factor, int m, double[] lambda, boolean new_lambda, int nele_hess, int[] iRow, int[] jCol, double[] values) {
        if (values == null) {
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (i != 0) {
                    iRow[count] = i;
                    jCol[count++] = i - 1;
                }
                iRow[count] = i;
                jCol[count++] = i;
            }
        } else {
            for (int i = 0; i < nele_hess; i++)
                values[i] = 0;
            double tmp, tmp2, a, b, r;
            int count;
            for (int i = 0; i < n; i++) {
                count = 2 * i;//对角线上元素的位置
                if (i == 0)
                    tmp = pNeed[i] + esChargeEff * (x[i] - iniEnergy);
                else
                    tmp = pNeed[i] + esChargeEff * (x[i] - x[i - 1]);
                a = buffer / Math.sqrt(1 + pricePerKwh[i] * pricePerKwh[i]);
                b = buffer * pricePerKwh[i] / Math.sqrt(1 + pricePerKwh[i] * pricePerKwh[i]);
                if (tmp > -buffer && tmp < a) {
                    r = buffer / pricePerKwh[i] + b + a / pricePerKwh[i];
                    tmp2 = Math.sqrt(r * r - (tmp + buffer) * (tmp + buffer));
                    values[count] = obj_factor * esChargeEff * esChargeEff * (1. / tmp2 +
                            (tmp + buffer) * (tmp + buffer) / (tmp2 * tmp2 * tmp2));
                    if (i != 0)
                        values[count - 1] = -values[count];
                }

                if (i == n - 1)
                    tmp = pNeed[i + 1] + esChargeEff * (iniEnergy - x[i]);
                else
                    tmp = pNeed[i + 1] + esChargeEff * (x[i + 1] - x[i]);
                a = buffer / Math.sqrt(1 + pricePerKwh[i + 1] * pricePerKwh[i + 1]);
                b = buffer * pricePerKwh[i + 1] / Math.sqrt(1 + pricePerKwh[i + 1] * pricePerKwh[i + 1]);

                if (tmp > -buffer && tmp < a) {
                    r = buffer / pricePerKwh[i + 1] + b + a / pricePerKwh[i + 1];
                    tmp2 = Math.sqrt(r * r - (tmp + buffer) * (tmp + buffer));
                    values[count] += obj_factor * esChargeEff * esChargeEff * (1. / tmp2 +
                            (tmp + buffer) * (tmp + buffer) / (tmp2 * tmp2 * tmp2));
                }
            }
        }
        return true;
    }

    /**
     * 设置状态变量的上下限
     * @param x_L 上限
     * @param x_U 下限
     */
    public void setXLimit(double[] x_L, double[] x_U) {
        System.arraycopy(this.x_L, 0, x_L, 0, x_L.length);
        System.arraycopy(this.x_U, 0, x_U, 0, x_U.length);
    }

    /**
     * 设置约束的上下限
     * @param g_L 上限
     * @param g_U 下限
     */
    public void setGLimit(double[] g_L, double[] g_U) {
        for (int i = 0; i < g_L.length; i++) {
            g_L[i] = -maxEnergyChange;
            g_U[i] = maxEnergyChange;
        }
    }

    @Override
    public int getNele_jac() {
        return nele_jac;
    }

    @Override
    public int getNele_hess() {
        return nele_hess;
    }

    @Override
    public int getM() {
        return m;
    }

    @Override
    public int getN() {
        return n;
    }

    @Override
    public void fillState(double[] x) {
        System.arraycopy(x, 0, result, 0, x.length);
    }

    public int getMaxIter() {
        return maxIter;
    }

    public void setMaxIter(int maxIter) {
        this.maxIter = maxIter;
    }

    public double getTolerance() {
        return tolerance;
    }

    public void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }

    public double getObjective() {
        return objective;
    }

    public void setObjective(double objective) {
        this.objective = objective;
    }

    public boolean isConverged() {
        return isConverged;
    }

    public void setConverged(boolean isConverged) {
        this.isConverged = isConverged;
    }

    public double getEsChargeEff() {
        return esChargeEff;
    }

    public void setEsChargeEff(double esChargeEff) {
        this.esChargeEff = esChargeEff;
    }

    public double getIniEnergy() {
        return iniEnergy;
    }

    public void setIniEnergy(double iniEnergy) {
        this.iniEnergy = iniEnergy;
    }

    public double getMaxEnergyChange() {
        return maxEnergyChange;
    }

    public void setMaxEnergyChange(double maxEnergyChange) {
        this.maxEnergyChange = maxEnergyChange;
    }

    public double[] getX_L() {
        return x_L;
    }

    public void setX_L(double[] x_L) {
        this.x_L = x_L;
    }

    public double[] getX_U() {
        return x_U;
    }

    public void setX_U(double[] x_U) {
        this.x_U = x_U;
    }

    public double[] getpNeed() {
        return pNeed;
    }

    public void setpNeed(double[] pNeed) {
        this.pNeed = pNeed;
    }

    public double[] getPricePerKwh() {
        return pricePerKwh;
    }

    public void setPricePerKwh(double[] pricePerKwh) {
        this.pricePerKwh = pricePerKwh;
    }

    public double[] getResult() {
        return result;
    }

    public double getBuffer() {
        return buffer;
    }

    public void setBuffer(double buffer) {
        this.buffer = buffer;
    }

    public double getEsDischargeEff() {
        return esDischargeEff;
    }

    public void setEsDischargeEff(double esDischargeEff) {
        this.esDischargeEff = esDischargeEff;
    }
}
