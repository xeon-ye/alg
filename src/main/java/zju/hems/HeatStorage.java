package zju.hems;

/**
 * 储热罐
 * @author Xu Chengsi
 * @date 2020/12/24
 */
public class HeatStorage {

    // 原始参数
    double coper;   // 单位输出功率的运维成本
    double Effin;  // 储热效率
    double Effout; // 放热效率
    double lossCoef;    // 自损耗系数
    double coeffPower;    // 保热耗电系数
    double S;   // 储热罐容量
    double minT;  // 最小储热状态
    double maxT;  // 最大储热状态
    double initT;  // 初始储热状态
    double maxPin;    // 最大储热功率
    double maxPout;    // 最大放热功率

    // 计算参数
    double minS;    // 最小储热量
    double maxS;    // 最大储热量
    double initS;   // 初始储热量

    public HeatStorage(double coper, double Effin, double Effout, double lossCoef, double coeffPower, double S,
                       double minT, double maxT, double initT, double maxPin, double maxPout) {
        this.coper = coper;
        this.Effin = Effin;
        this.Effout = Effout;
        this.lossCoef = lossCoef;
        this.coeffPower = coeffPower;
        this.S = S;
        this.minT = minT;
        this.maxT = maxT;
        this.initT = initT;
        this.maxPin = maxPin;
        this.maxPout = maxPout;

        minS = minT * S;
        maxS = maxT * S;
        initS = initT * S;
    }

    public double getCoper() {
        return coper;
    }

    public void setCoper(double coper) {
        this.coper = coper;
    }

    public double getEffin() {
        return Effin;
    }

    public void setEffin(double effin) {
        Effin = effin;
    }

    public double getEffout() {
        return Effout;
    }

    public void setEffout(double effout) {
        Effout = effout;
    }

    public double getLossCoef() {
        return lossCoef;
    }

    public void setLossCoef(double lossCoef) {
        this.lossCoef = lossCoef;
    }

    public double getCoeffPower() {
        return coeffPower;
    }

    public void setCoeffPower(double coeffPower) {
        this.coeffPower = coeffPower;
    }

    public double getS() {
        return S;
    }

    public void setS(double s) {
        S = s;
    }

    public double getMinT() {
        return minT;
    }

    public void setMinT(double minT) {
        this.minT = minT;
    }

    public double getMaxT() {
        return maxT;
    }

    public void setMaxT(double maxT) {
        this.maxT = maxT;
    }

    public double getInitT() {
        return initT;
    }

    public void setInitT(double intT) {
        this.initT = initT;
    }

    public double getMinS() {
        return minS;
    }

    public void setMinS(double minS) {
        this.minS = minS;
    }

    public double getMaxS() {
        return maxS;
    }

    public void setMaxS(double maxS) {
        this.maxS = maxS;
    }

    public double getInitS() {
        return initS;
    }

    public void setInitS(double initS) {
        this.initS = initS;
    }

    public double getMaxPin() {
        return maxPin;
    }

    public void setMaxPin(double maxPin) {
        this.maxPin = maxPin;
    }

    public double getMaxPout() {
        return maxPout;
    }

    public void setMaxPout(double maxPout) {
        this.maxPout = maxPout;
    }
}
