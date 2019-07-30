package zju.hems;

public class SteamLoad {

    double[] demand;  // 负荷需求
    double EER; // 热回收效率

    public SteamLoad(double[] demand, double EER) {
        this.demand = demand;
        this.EER = EER;
    }

    public double[] getDemand() {
        return demand;
    }

    public void setDemand(double[] demand) {
        this.demand = demand;
    }

    public double getEER() {
        return EER;
    }

    public void setEER(double EER) {
        this.EER = EER;
    }
}
