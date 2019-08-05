package zju.hems;

/**
 * 冰蓄冷空调
 */
public class IceStorageAc {

    // 原始参数
    double coper;   // 单位输出功率的运维成本
    double Cgly;    // 乙二醇溶液的比热容
    double density; // 乙二醇溶液的密度
    double dTice;  // 蓄冰工况时经过蓄冰槽的乙二醇溶液回水和供水温度差
    double Effice;  // 制冰效率
    double EERc; // 制冷能效比
    double EERice; // 制冰能效比
    double dTmelt;  // 融冰工况时经过蓄冰槽的乙二醇溶液回水和供水温度差
    double Effmelt; // 融冰效率
    double dTref;  // 制冷工况时经过制冷机的乙二醇溶液回水和供水温度差
    double Effref;  // 制冷效率
    double lossCoef;    // 自损耗系数
    double maxP;    // 最大耗电功率
    double S;   // 蓄冰槽容量
    double minT;  // 最小蓄冰状态
    double maxT;  // 最大蓄冰状态
    double initT;  // 初始蓄冰状态
    double consumCoef;  // 能耗比例系数
    double maxPice;    // 最大制冰功率
    double maxPmelt;    // 最大融冰功率

    // 计算参数
    double minS;    // 蓄冰槽最小蓄冰量
    double maxS;    // 蓄冰槽最大蓄冰量
    double initS;   // 蓄冰槽初始蓄冰量

    public IceStorageAc(double coper, double Effice, double EERc,
                        double EERice, double Effmelt, double Effref, double lossCoef,
                        double maxP, double S, double minT, double maxT, double initT, double consumCoef, double maxPice,
                        double maxPmelt) {
        this.coper = coper;
        this.Effice = Effice;
        this.EERc = EERc;
        this.EERice = EERice;
        this.Effmelt = Effmelt;
        this.Effref = Effref;
        this.lossCoef = lossCoef;
        this.maxP = maxP;
        this.S = S;
        this.minT = minT;
        this.maxT = maxT;
        this.initT = initT;
        this.consumCoef = consumCoef;
        this.maxPice = maxPice;
        this.maxPmelt = maxPmelt;

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

    public double getCgly() {
        return Cgly;
    }

    public void setCgly(double cgly) {
        Cgly = cgly;
    }

    public double getDensity() {
        return density;
    }

    public void setDensity(double density) {
        this.density = density;
    }

    public double getdTice() {
        return dTice;
    }

    public void setdTice(double dTice) {
        this.dTice = dTice;
    }

    public double getEffice() {
        return Effice;
    }

    public void setEffice(double effice) {
        Effice = effice;
    }

    public double getEERc() {
        return EERc;
    }

    public void setEERc(double EERc) {
        this.EERc = EERc;
    }

    public double getEERice() {
        return EERice;
    }

    public void setEERice(double EERice) {
        this.EERice = EERice;
    }

    public double getdTmelt() {
        return dTmelt;
    }

    public void setdTmelt(double dTmelt) {
        this.dTmelt = dTmelt;
    }

    public double getEffmelt() {
        return Effmelt;
    }

    public void setEffmelt(double effmelt) {
        Effmelt = effmelt;
    }

    public double getdTref() {
        return dTref;
    }

    public void setdTref(double dTref) {
        this.dTref = dTref;
    }

    public double getEffref() {
        return Effref;
    }

    public void setEffref(double effref) {
        Effref = effref;
    }

    public double getLossCoef() {
        return lossCoef;
    }

    public void setLossCoef(double lossCoef) {
        this.lossCoef = lossCoef;
    }

    public double getMaxP() {
        return maxP;
    }

    public void setMaxP(double maxP) {
        this.maxP = maxP;
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

    public double getConsumCoef() {
        return consumCoef;
    }

    public void setConsumCoef(double consumCoef) {
        this.consumCoef = consumCoef;
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

    public double getMaxPice() {
        return maxPice;
    }

    public void setMaxPice(double maxPice) {
        this.maxPice = maxPice;
    }

    public double getMaxPmelt() {
        return maxPmelt;
    }

    public void setMaxPmelt(double maxPmelt) {
        this.maxPmelt = maxPmelt;
    }
}
