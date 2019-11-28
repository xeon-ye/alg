package zju.hems;

/**
 * 风电
 * @author Xu Chengsi
 * @date 2019/11/28
 */
public class WindPower {

    double coper;   // 单位输出功率的运维成本
    double[] power; // 风电出力预测

    public WindPower(double coper, double[] power) {
        this.coper = coper;
        this.power = power;
    }

    public double getCoper() {
        return coper;
    }

    public void setCoper(double coper) {
        this.coper = coper;
    }

    public double[] getPower() {
        return power;
    }

    public void setPower(double[] power) {
        this.power = power;
    }
}
