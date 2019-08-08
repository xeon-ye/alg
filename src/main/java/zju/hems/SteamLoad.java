package zju.hems;

/**
 * 蒸汽驱动负荷
 * @author Xu Chengsi
 * @date 2019/7/25
 */
public class SteamLoad {

    double[] demand;  // 负荷需求
    double Effh; // 热回收效率

    public SteamLoad(double[] demand, double Effh) {
        this.demand = demand;
        this.Effh = Effh;
    }

    public double[] getDemand() {
        return demand;
    }

    public void setDemand(double[] demand) {
        this.demand = demand;
    }

    public double getEffh() {
        return Effh;
    }

    public void setEffh(double effh) {
        Effh = effh;
    }
}
