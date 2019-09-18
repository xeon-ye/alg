package zju.hems;

/**
 * 电制冷/热中央空调
 * @author Xu Chengsi
 * @date 2019/7/25
 */
public class AirCon {

    double coper;   // 单位输出功率的运维成本
    double Cwater;    // 冷冻水的比热容
    double density; // 冷冻水的密度
    double dTac;  // 回水和供水温度差
    double Effac;  // 供冷效率
    double consumCoef;  // 能耗比例系数
    double minP;    // 最小耗电功率
    double maxP;    // 最大耗电功率
    double EERc;    // 制冷能效比
    double EERh;    // 制热能效比

    public AirCon(double coper, double Effac, double consumCoef, double minP, double maxP, double EERc) {
        this.coper = coper;
        this.Effac = Effac;
        this.consumCoef = consumCoef;
        this.minP = minP;
        this.maxP = maxP;
        this.EERc = EERc;
    }

    public double getCoper() {
        return coper;
    }

    public void setCoper(double coper) {
        this.coper = coper;
    }

    public double getCwater() {
        return Cwater;
    }

    public void setCwater(double cwater) {
        Cwater = cwater;
    }

    public double getDensity() {
        return density;
    }

    public void setDensity(double density) {
        this.density = density;
    }

    public double getdTac() {
        return dTac;
    }

    public void setdTac(double dTac) {
        this.dTac = dTac;
    }

    public double getEffac() {
        return Effac;
    }

    public void setEffac(double effac) {
        Effac = effac;
    }

    public double getConsumCoef() {
        return consumCoef;
    }

    public void setConsumCoef(double consumCoef) {
        this.consumCoef = consumCoef;
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

    public double getEERc() {
        return EERc;
    }

    public void setEERc(double EERc) {
        this.EERc = EERc;
    }

    public double getEERh() {
        return EERh;
    }

    public void setEERh(double EERh) {
        this.EERh = EERh;
    }
}
