package zju.hems;

/**
 * 光伏
 * @author Xu Chengsi
 * @date 2019/7/25
 */
public class Photovoltaic {

    double coper;   // 单位输出功率的运维成本
    double[] power; // 光伏出力预测
    double[] heatPower; // 光伏产热功率预测

    public Photovoltaic(double coper, double[] power) {
        this.coper = coper;
        this.power = power;
    }

    public Photovoltaic(double coper, double[] power, double[] heatPower) {
        this(coper, power);
        this.heatPower = heatPower;
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

    public double[] getHeatPower() {
        return heatPower;
    }

    public void setHeatPower(double[] heatPower) {
        this.heatPower = heatPower;
    }
}
