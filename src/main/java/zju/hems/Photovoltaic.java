package zju.hems;

/**
 * 光伏
 */
public class Photovoltaic {

    double coper;   // 单位输出功率的运维成本

    public Photovoltaic(double coper) {
        this.coper = coper;
    }

    public double getCoper() {
        return coper;
    }

    public void setCoper(double coper) {
        this.coper = coper;
    }
}
