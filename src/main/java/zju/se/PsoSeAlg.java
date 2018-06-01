package zju.se;

import zju.matrix.AVector;
import zju.measure.MeasTypeCons;

/**
 * @Description: 状态估计的粒子群算法
 * @Author: Fang Rui
 * @Date: 2018/5/29
 * @Time: 11:40
 */
public class PsoSeAlg extends AbstractSeAlg implements MeasTypeCons {

    private double tol_p = 0.005, tol_q = 0.005;

    @Override
    public AVector getFinalVTheta() {
        return null;
    }

    @Override
    public void doSeAnalyse() {

    }

    public double getTol_p() {
        return tol_p;
    }

    public void setTol_p(double tol_p) {
        this.tol_p = tol_p;
    }

    public double getTol_q() {
        return tol_q;
    }

    public void setTol_q(double tol_q) {
        this.tol_q = tol_q;
    }
}
