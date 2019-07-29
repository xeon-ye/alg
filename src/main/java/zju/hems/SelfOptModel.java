package zju.hems;

public class SelfOptModel {

    Microgrid microgrid;
    int periodNum = 96; // 一天的时段数
    double t;   // 单位时段长度
    double[] elecPrices;    // 电价
    double[] gasPrices;    // 天然气价格
    double[] steamPrices;    // 园区CHP蒸汽价格

    public SelfOptModel(Microgrid microgrid) {
        this.microgrid = microgrid;
    }

    public void doSelfOpt() {

    }
}
