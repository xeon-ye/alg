package zju.hems;

/**
 * 光伏
 */
public class Photovoltaic {

    double coper;   // 单位输出功率的运维成本
    double[] power; // 光伏出力预测

    public Photovoltaic(double coper) {
        this.coper = coper;
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
