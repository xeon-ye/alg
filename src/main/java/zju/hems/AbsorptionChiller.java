package zju.hems;

/**
 * 吸收式制冷机
 */
public class AbsorptionChiller {

    double coper;   // 单位输出功率的运维成本
    double calorificValue;  // 蒸汽热值
    double minH;    // 最小吸热功率
    double maxH;    // 最大吸热功率
    double Ic;  // 制冷能效比
    double Ih;  // 制热能效比

    public AbsorptionChiller(double coper, double calorificValue, double minH, double maxH, double Ic, double Ih) {
        this.coper = coper;
        this.calorificValue = calorificValue;
        this.minH = minH;
        this.maxH = maxH;
        this.Ic = Ic;
        this.Ih = Ih;
    }

    public double getCoper() {
        return coper;
    }

    public void setCoper(double coper) {
        this.coper = coper;
    }

    public double getCalorificValue() {
        return calorificValue;
    }

    public void setCalorificValue(double calorificValue) {
        this.calorificValue = calorificValue;
    }

    public double getMinH() {
        return minH;
    }

    public void setMinH(double minH) {
        this.minH = minH;
    }

    public double getMaxH() {
        return maxH;
    }

    public void setMaxH(double maxH) {
        this.maxH = maxH;
    }

    public double getIc() {
        return Ic;
    }

    public void setIc(double ic) {
        Ic = ic;
    }

    public double getIh() {
        return Ih;
    }

    public void setIh(double ih) {
        Ih = ih;
    }
}
