package zju.lfp.forecasters.singleDimensionalForecaster.extrapolation;

import zju.lfp.MultiTimeSeriesPredictor;
import zju.lfp.utils.MultiTimeSeries;

/**
 * 双周期时间序列的简单多点外推算法：
 * 序列在时间轴上的是连续的
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-12-21
 * Time: 9:32:02
 */
public class ExtrapolationPredictor implements MultiTimeSeriesPredictor {
    private static final double extrapolationWeight = 0.75;
    private static final double longPeriodForecastWeight = 0.25;
    private static final double useLongPeriodCriticaProportion = 0.12;

    // 时间序列的长周期
    private static final double longPeriodWeight = -0.5;
    private int longPeriod;
    // 时间序列的短周期
    private static final double shortPeriodWeight = -1;
    private int shortPeriod;

    private int usefulPointsNum;

    public ExtrapolationPredictor(int longPeriod, int shortPeriod, int usefulPointsNum) {
        this.longPeriod = longPeriod;
        this.shortPeriod = shortPeriod;
        this.usefulPointsNum = usefulPointsNum;
    }

    public void predict(MultiTimeSeries multiTimeSeries) {
        int forecastedSeriesIndex = multiTimeSeries.getNumAttributes() - 1;
        int lenPredict = multiTimeSeries.getNLen() - multiTimeSeries.getMissValueIndex();
        int circleNum = longPeriod / shortPeriod;

        // 1. 由远小近大确定各点权值
        double[] pointsWeight = new double[usefulPointsNum];
        double sumPointsWeight = 0;
        for (int i = 0; i < usefulPointsNum; i++) {
            pointsWeight[i] = (i + 1.0) / (i + 2.0);
            sumPointsWeight += pointsWeight[i];
        }
        for (int i = 0; i < usefulPointsNum; i++) {
            pointsWeight[i] /= sumPointsWeight;
        }

        // 2. 样本预测及其权重确定
        // 参考点的位置
        int beginRefOriIndex = multiTimeSeries.getMissValueIndex() - usefulPointsNum;
        //
        double[] refOri, refDes, ori, tempDes, des1;
        // 原始参考序列（多点）
        ori = multiTimeSeries.getTimeSeries(forecastedSeriesIndex, beginRefOriIndex, usefulPointsNum);
        des1 = new double[lenPredict];
        int count = 0;
        int longPeriodNum;
        int shortPeriodNum;
        double sampleWeight;
        double sampleWeightSum = 0;
        while (true) {
            beginRefOriIndex -= (lenPredict / (shortPeriod + 1) + 1) * shortPeriod;
            if (beginRefOriIndex < 0) break;
            count++;
            longPeriodNum = count / circleNum;
            shortPeriodNum = count % circleNum;
            //
            refOri = multiTimeSeries.getTimeSeries(forecastedSeriesIndex, beginRefOriIndex,
                    usefulPointsNum);
            refDes = multiTimeSeries.getTimeSeries(forecastedSeriesIndex,
                    beginRefOriIndex + usefulPointsNum, lenPredict);
            tempDes = samplePredict(refOri, refDes, ori, pointsWeight);
            // 样本权重
            sampleWeight = Math.exp(longPeriodWeight * longPeriodNum)
                    * Math.exp(shortPeriodWeight * shortPeriodNum);
            sampleWeightSum += sampleWeight;
            for (int i = 0; i < lenPredict; i++) {
                tempDes[i] *= sampleWeight;
                des1[i] += tempDes[i];
            }
        }
        for (int i = 0; i < lenPredict; i++) {
            des1[i] /= sampleWeightSum;
        }

        // 3 长周期预测
        int beginIndex = multiTimeSeries.getMissValueIndex() - longPeriod;
        if (beginIndex < 0) {
            multiTimeSeries.addOrUpdateTimeSeries(forecastedSeriesIndex,
                    multiTimeSeries.getMissValueIndex(), des1);
            return;
        }
        double[] des2 = multiTimeSeries.getTimeSeries(forecastedSeriesIndex, beginIndex, lenPredict);
        int c = canBeRefered(des2, des1);

        if (c != 0) {
            if (c == -1) {
                for (int i = 0; i < lenPredict; i++) {
                    des1[i] = 0.98 * des1[i]; // 参数调整
                }
            } else if (c == 1) {
                for (int i = 0; i < lenPredict; i++) {
                    des1[i] = 1.02 * des1[i]; // 参数调整
                }
            }
            multiTimeSeries.addOrUpdateTimeSeries(forecastedSeriesIndex,
                    multiTimeSeries.getMissValueIndex(), des1);
            return;
        }

        double[] des = new double[lenPredict];
        for (int i = 0; i < lenPredict; i++) {
            des[i] = extrapolationWeight * des1[i] + longPeriodForecastWeight * des2[i];
        }

        multiTimeSeries.addOrUpdateTimeSeries(forecastedSeriesIndex,
                multiTimeSeries.getMissValueIndex(), des);
    }

    private double[] samplePredict(double[] refOri, double[] refDes, double[] ori, double[] pointsWeight) {
        int predictNum = refDes.length;
        double[] des = new double[predictNum];
        // 外推计算 des
        for (int j = 0; j < predictNum; j++) {
            for (int i = 0; i < usefulPointsNum; i++) {
                des[j] += pointsWeight[i] * (refDes[j] - refOri[i] + ori[i]);
            }
        }
        return des;
    }

    private int canBeRefered(double[] refArray, double[] desArray) {
        double ref = calAvg(refArray);
        double des = calAvg(desArray);
        double d = (ref - des) / ref;
        if (d > useLongPeriodCriticaProportion) return -1;
        if (Math.abs(d) > useLongPeriodCriticaProportion) return 1;
        else return 0;
    }

    private double calAvg(double[] array) {
        if (array == null || array.length == 0) return Double.NaN;
        double sum = 0;
        for (double a : array) {
            sum += a;
        }
        return sum / array.length;
    }
}
