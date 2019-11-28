package zju.hems;

import java.util.Map;

public interface DispatchOpt {

    /**
     *
     * @param microgrid 微网模型
     * @param dispatchTime 调度总时长
     * @param periodNum 时段数
     * @param elecPrices 电价
     * @param gasPrices 天然气价格
     * @param steamPrices 热蒸汽价格
     * @return
     */
    Map<String, UserResult> doDispatchOpt(Microgrid microgrid, double dispatchTime, int periodNum, double[] elecPrices,
                                          double[] gasPrices, double[] steamPrices);
}
