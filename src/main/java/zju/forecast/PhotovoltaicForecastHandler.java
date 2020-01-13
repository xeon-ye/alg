package zju.forecast;

import java.util.List;

public interface PhotovoltaicForecastHandler extends ForecastHandler{

    /**
     * 光伏预测
     *
     * @param weathers 天气特征
     * @return 需要预测的时间点的预测值
     */
    double[] predictPhotovoltaic(List<Weather> weathers);

    /**
     * 重新训练模型
     *
     * @param weathers     天气特征
     * @param measurements 量测特征
     */
    void fitPhotovoltaic(List<Weather> weathers,
                         List<Measurement> measurements);
}
