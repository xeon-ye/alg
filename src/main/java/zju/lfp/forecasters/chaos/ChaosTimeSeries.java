package zju.lfp.forecasters.chaos;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-11-23
 * Time: 15:10:13
 */
public class ChaosTimeSeries {
    private static Logger log = LogManager.getLogger(ChaosTimeSeries.class);

    private double[] series;

    public ChaosTimeSeries(double[] series) {
        this.series = series;
    }

    // ======================== public ===================================
    public int beginIndexReformed(int delayPoints, int reformDimensions) {
        return delayPoints * (reformDimensions - 1);
    }

    // 重构相空间中的点
    public double[] getReformedSpacePoint(int index, int delayPoints, int reformDimensions) {
        int beginIndex = index - beginIndexReformed(delayPoints, reformDimensions);
        if (beginIndex < 0) {
            log.error("Point can't be reformed " +
                    "because the length of timeSeries is too short!");
            return null;
        }
        return getSubSeries(beginIndex, delayPoints, reformDimensions);
    }

    public double distanceBetweenSpacePoints(int index1, int index2,
                                             int delayPoints, int reformDimensions) {
        double[] point1 = getReformedSpacePoint(index1, delayPoints, reformDimensions);
        double[] point2 = getReformedSpacePoint(index2, delayPoints, reformDimensions);
        return calDistance(point1, point2);
    }

    // 得到子序列
    public double[] getSubSeries(int beginIndex, int gap, int len) {
        double[] subSeries = new double[len];
        for (int n = 0; n < len; n++) {
            int k = beginIndex + n * gap;
            if (k >= series.length)
                break;
            subSeries[n] = series[k];
        }
        return subSeries;
    }

    // 计算序列平均值
    public double mean() {
        double fMean = 0.0f;
        for (double sery : series) fMean += sery;
        fMean /= series.length;
        return fMean;
    }

    // 计算线性自相关因素
    public double[] calLinearAutoRelationFactor(int len) {
        double[] larf = new double[len];
        double m = mean();
        double s0 = 0.0f;
        for (double sery : series) s0 += (sery - m) * (sery - m);
        ///
        larf[0] = 1;
        for (int k = 1; k < len; k++) {
            double s = 0.0f;
            for (int t = 0; t < series.length - k; t++)
                s += (series[t] - m) *( series[t + k] - m);
            larf[k] = s / s0;
        }
        return larf;
    }

    // ===================== getter and setter ===========================
    public int getLength() {
        return series.length;
    }

    public double[] getSeries() {
        return series;
    }

    // 超空间中两点的欧式距离
    public static double calDistance(double[] array1, double[] array2) {
        int len1 = array1.length;
        int len2 = array2.length;
        if(len1 != len2) {
            return Double.NaN;
        }
        double sum = 0;
        for(int i = 0; i < len1; i++)
            sum += Math.pow((array1[i] - array2[i]), 2);
        return Math.sqrt(sum);
    }
}
