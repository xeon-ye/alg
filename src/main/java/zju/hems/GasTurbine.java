package zju.hems;

/**
 * 燃气轮机和余热锅炉
 * @author Xu Chengsi
 * @date 2019/7/25
 */
public class GasTurbine {

    double coper;   // 单位输出功率的运维成本
    double calorificValue;  // 天然气热值
    double Effe;   // 燃气轮机发电效率
    double Effh;   // 中品味热回收效率
    double css; // 启停成本
    double minP;    // 最小产电功率
    double maxP;    // 最大产电功率
    double minRampRate;    // 爬坡率下限
    double maxRampRate;    // 爬坡率上限
    int initState;   // 初始启停状态

    public GasTurbine(double coper, double Effe, double Effh, double css, double minP,
                      double maxP, double minRampRate, double maxRampRate, int initState) {
        this.coper = coper;
        this.Effe = Effe;
        this.Effh = Effh;
        this.css = css;
        this.maxP = maxP;
        this.minP = minP;
        this.minRampRate = minRampRate;
        this.maxRampRate = maxRampRate;
        this.initState = initState;
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

    public double getEffe() {
        return Effe;
    }

    public void setEffe(double effe) {
        this.Effe = effe;
    }

    public double getEffh() {
        return Effh;
    }

    public void setEffh(double effh) {
        this.Effh = effh;
    }

    public double getCss() {
        return css;
    }

    public void setCss(double css) {
        this.css = css;
    }

    public double getMinP() {
        return minP;
    }

    public void setMinP(double minP) {
        this.minP = minP;
    }

    public double getMaxP() {
        return maxP;
    }

    public void setMaxP(double maxP) {
        this.maxP = maxP;
    }

    public double getMinRampRate() {
        return minRampRate;
    }

    public void setMinRampRate(double minRampRate) {
        this.minRampRate = minRampRate;
    }

    public double getMaxRampRate() {
        return maxRampRate;
    }

    public void setMaxRampRate(double maxRampRate) {
        this.maxRampRate = maxRampRate;
    }

    public int getInitState() {
        return initState;
    }

    public void setInitState(int initState) {
        this.initState = initState;
    }
}
