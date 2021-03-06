package zju.hems;

/**
 * 电池储能
 * @author Xu Chengsi
 * @date 2019/6/20
 */
public class Storage {

    // 原始参数
    double coper;   // 单位输出功率的运维成本
    double crep;    // 蓄电池更换成本
    double Qlife = 9615;   // 电池单体全寿命输出总电量（kWh）
    double maxPIn;  // 最大充电功率
    double maxPOut; // 最大放电功率
    double R;   // 电池容量
    double minSOC;  // 最小SOC
    double maxSOC;  // 最大SOC
    double initSOC;  // 初始SOC
    double endSOC;  // 控制结束时的SOC
    double yin = 1; // 充电爬坡率约束系数
    double yout = 1; // 放电爬坡率约束系数
    double lossCoef;    // 自损耗系数
    double effIn;   // 充电效率
    double effOut;   // 放电效率

    // 计算参数
    double cbw; // 蓄电池累计充电1kWh的折旧成本
    double minS;    // 电池最低电量
    double maxS;    // 电池最高电量
    double initS;   // 电池初始电量
    double endS;   // 电池终止电量

    public Storage(double coper, double cbw, double maxPIn, double maxPOut, double R, double minSOC, double maxSOC,
                   double initSOC, double endSOC, double yin, double yout, double lossCoef, double effIn, double effOut) {
        this.coper = coper;
        this.cbw = cbw;
        this.maxPIn = maxPIn;
        this.maxPOut = maxPOut;
        this.R = R;
        this.minSOC = minSOC;
        this.maxSOC = maxSOC;
        this.initSOC = initSOC;
        this.endSOC = endSOC;
        this.yin = yin;
        this.yout = yout;
        this.lossCoef = lossCoef;
        this.effIn = effIn;
        this.effOut = effOut;

        minS = minSOC * R;
        maxS = maxSOC * R;
        initS = initSOC * R;
        endS = endSOC * R;
    }

    public double getCoper() {
        return coper;
    }

    public void setCoper(double coper) {
        this.coper = coper;
    }

    public double getCrep() {
        return crep;
    }

    public void setCrep(double crep) {
        this.crep = crep;
    }

    public double getQlife() {
        return Qlife;
    }

    public void setQlife(double qlife) {
        Qlife = qlife;
    }

    public double getMaxPIn() {
        return maxPIn;
    }

    public void setMaxPIn(double maxPIn) {
        this.maxPIn = maxPIn;
    }

    public double getMaxPOut() {
        return maxPOut;
    }

    public void setMaxPOut(double maxPOut) {
        this.maxPOut = maxPOut;
    }

    public double getR() {
        return R;
    }

    public void setR(double r) {
        R = r;
    }

    public double getMinSOC() {
        return minSOC;
    }

    public void setMinSOC(double minSOC) {
        this.minSOC = minSOC;
    }

    public double getMaxSOC() {
        return maxSOC;
    }

    public void setMaxSOC(double maxSOC) {
        this.maxSOC = maxSOC;
    }

    public double getInitSOC() {
        return initSOC;
    }

    public void setInitSOC(double initSOC) {
        this.initSOC = initSOC;
    }

    public double getEndSOC() {
        return endSOC;
    }

    public void setEndSOC(double endSOC) {
        this.endSOC = endSOC;
    }

    public double getYin() {
        return yin;
    }

    public void setYin(double yin) {
        this.yin = yin;
    }

    public double getYout() {
        return yout;
    }

    public void setYout(double yout) {
        this.yout = yout;
    }

    public double getLossCoef() {
        return lossCoef;
    }

    public void setLossCoef(double lossCoef) {
        this.lossCoef = lossCoef;
    }

    public double getEffIn() {
        return effIn;
    }

    public void setEffIn(double effIn) {
        this.effIn = effIn;
    }

    public double getEffOut() {
        return effOut;
    }

    public void setEffOut(double effOut) {
        this.effOut = effOut;
    }

    public double getCbw() {
        return cbw;
    }

    public double getMinS() {
        return minS;
    }

    public double getMaxS() {
        return maxS;
    }

    public double getInitS() {
        return initS;
    }

    public double getEndS() {
        return endS;
    }
}
