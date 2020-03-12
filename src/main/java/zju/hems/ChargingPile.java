package zju.hems;

public class ChargingPile {
    double minP;    // 最小充电功率
    double maxP;    // 额定功率
    int mode;   // 充电模式。1：自动充满模式，2：按时间充电模式，3：按金额充电模式，4：按电量充电模式
    double Sm;  // 自动充满模式下，剩余需要的充电量（kWh）
    double Te;    // 按时间充电模式下，设定的停止充电时刻（时段数）
    double M;   // 按金额充电模式下，设置的金额（元）
    double S;   // 按电量充电模式下，剩余需要的充电量（kWh）

    public ChargingPile(double minP, double maxP, int mode, double Sm, double Te, double M, double S) {
        this.minP = minP;
        this.maxP = maxP;
        this.mode = mode;
        this.Sm = Sm;
        this.Te = Te;
        this.M = M;
        this.S = S;
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

    public int getMode() {
        return mode;
    }

    public void setMode(int mode) {
        this.mode = mode;
    }

    public double getSm() {
        return Sm;
    }

    public void setSm(double sm) {
        Sm = sm;
    }

    public double getTe() {
        return Te;
    }

    public void setTe(double te) {
        Te = te;
    }

    public double getM() {
        return M;
    }

    public void setM(double m) {
        M = m;
    }

    public double getS() {
        return S;
    }

    public void setS(double s) {
        S = s;
    }
}
