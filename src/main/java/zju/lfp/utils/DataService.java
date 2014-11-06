package zju.lfp.utils;

import java.util.Calendar;
import java.util.Map;

/**
 * get useful data according forecasting time
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-9-24
 * Time: 17:28:34
 */
public abstract class DataService {
    protected long numSamples;

    /**
     * 设定训练样本数量
     *
     * @param numSamples 训练样本数量
     */
    public void setSamplesNum(long numSamples) {
        this.numSamples = numSamples;
    }

    /**
     * 得到待预测时段所需决策属性的数据集
     * @param beginCal 待预测时段开始时刻
     * @param endCal 待预测时段结束时刻
     * @param gapMinutes 采样时间间隔
     * @return dataMap
     */
    public abstract Map<Calendar, Double> getForecastedDataMap(Calendar beginCal, Calendar endCal, int gapMinutes);
}
