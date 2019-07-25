package zju.hems;

/**
 * 燃气锅炉
 */
public class GasBoiler {

    double coper;   // 单位输出功率的运维成本
    double calorificValue;  // 燃气热值
    double css; // 启停成本
    double Effgb;   // 产热效率
    double minH;    // 最小产热功率
    double maxH;    // 最大产热功率
    double rampRate;    // 爬坡率约束

    public GasBoiler(double coper, double calorificValue, double css, double Effgb, double minH, double maxH, double rampRate) {
        this.coper = coper;
        this.calorificValue = calorificValue;
        this.css = css;
        this.Effgb = Effgb;
        this.maxH = maxH;
        this.minH = minH;
        this.rampRate = rampRate;
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

    public double getCss() {
        return css;
    }

    public void setCss(double css) {
        this.css = css;
    }

    public double getEffgb() {
        return Effgb;
    }

    public void setEffgb(double effgb) {
        Effgb = effgb;
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

    public double getRampRate() {
        return rampRate;
    }

    public void setRampRate(double rampRate) {
        this.rampRate = rampRate;
    }
}
