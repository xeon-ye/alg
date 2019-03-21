package zju.lfp.forecasters.analogyExtrapolation;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.lfp.MultiTimeSeriesPredictor;
import zju.lfp.forecasters.singleDimensionalForecaster.extrapolation.ExtrapolationPredictor;
import zju.lfp.utils.MultiTimeSeries;

/**
 * 类比外推预测：节假日预测方案
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-12-29
 * Time: 11:06:03
 */
public class AnalogyExtrapolationPredictor implements MultiTimeSeriesPredictor {
    private final static Logger log = LogManager.getLogger(AnalogyExtrapolationPredictor.class);
    private ExtrapolationPredictor extrapolationPredictor;

    public AnalogyExtrapolationPredictor(int longPeriod, int shortPeriod, int usefulPointsNum) {
        extrapolationPredictor = new ExtrapolationPredictor(longPeriod, shortPeriod, usefulPointsNum);
    }

    public void predict(MultiTimeSeries multiTimeSeries) {
        int numReffers = multiTimeSeries.getNumAttributes() - 1;
        if (numReffers == 0) {
            log.error("Can't find enough data for analogy!");            
            extrapolationPredictor.predict(multiTimeSeries);
            return;
        }
        int beginIndex = multiTimeSeries.getMissValueIndex();
        int lenForecasting = multiTimeSeries.getNLen() - beginIndex;
        MultiTimeSeries des = multiTimeSeries.getChildMultiTimeSeries(numReffers, beginIndex);
        extrapolationPredictor.predict(des);
        double[] originalResult = des.getTimeSeries(0, beginIndex, lenForecasting);
        double[] result = new double[lenForecasting];
        for (int i = 0; i < numReffers; i++) {
            MultiTimeSeries ref = multiTimeSeries.getChildMultiTimeSeries(i, beginIndex);
            extrapolationPredictor.predict(ref);
            double[] refForecastResult = ref.getTimeSeries(0, beginIndex, lenForecasting);
            double[] refHistoryResult = multiTimeSeries.getTimeSeries(i, beginIndex, lenForecasting);
            for (int j = 0; j < lenForecasting; j++) {
                result[j] += refHistoryResult[j] * originalResult[j] / refForecastResult[j];
            }
        }
        for (int j = 0; j < lenForecasting; j++) {
            result[j] /= numReffers;
        }
        multiTimeSeries.addOrUpdateTimeSeries(numReffers, beginIndex, result);
    }
}
