package zju.lfp.evaluation;

/**
 * Created by IntelliJ IDEA.
 * User: Administrator
 * Date: 2006-7-24
 * Time: 11:47:49
 */
public class SeriesEvaluation {
    private final static double avgPrecisionWeight = 0.6; //todo 在config中配置
    private final static double extremPrecisionWeight = 0.4;

    public final static double DOUBLE_MISS_VALUE = Double.NaN;  //空值
    public final static double ACCEPTE_VALUE = -9999;
    public final static double SPECIAL_VERACITY = -100;  //特殊精度（免考核）

    private double[] historyArray;
    private double[] forecastArray;
    private boolean[] usefulMarker;
    private int seriesLength;

    private double[] veracityArray;
    // 日平均准确率
    private double avgPrecision;
    // 最大值准确率
    private double maxPrecision;
    // 最小值准确率
    private double minPrecision;
    // 极值准确率
    private double extPrecision;
    // 综合准确率
    private double intPrecision;

    // ========================= constructor ============================
    public SeriesEvaluation() {

    }

    public SeriesEvaluation(double[] historyArray, double[] forecastArray) {
        this.historyArray = historyArray;
        this.forecastArray = forecastArray;
        if (historyArray != null)
            seriesLength = historyArray.length;
        usefulMarker = new boolean[seriesLength];
        for (int i = 0; i < seriesLength; i++) usefulMarker[i] = true;
        initialVeracity();
    }

    public SeriesEvaluation(double[] historyArray, double[] forecastArray, boolean[] usefulMarker) {
        this.historyArray = historyArray;
        this.forecastArray = forecastArray;
        this.usefulMarker = usefulMarker;
        if (historyArray != null)
            seriesLength = historyArray.length;
        initialVeracity();
    }

    private void initialVeracity() {
        calVeracityArray();
        boolean b = true;
        for (int i = 0; i < seriesLength; i++) {
            if (!Double.isNaN(veracityArray[i])) {
                b = false;
            }
        }
        if (b) {
            avgPrecision = DOUBLE_MISS_VALUE;
            maxPrecision = DOUBLE_MISS_VALUE;
            minPrecision = DOUBLE_MISS_VALUE;
            extPrecision = DOUBLE_MISS_VALUE;
            return;
        }
        calAverageVeracity();
        calExtremumVeracity();
        calIntegrationVeracity();
    }


    // =================================== getter and setter =================================
    public ComplexVeracity getComplexVeracity() {
        return new ComplexVeracity(avgPrecision, maxPrecision, minPrecision, extPrecision, intPrecision);
    }

    public double[] getHistoryArray() {
        return historyArray;
    }

    public void setHistoryArray(double[] historyArray) {
        this.historyArray = historyArray;
    }

    public double[] getForecastArray() {
        return forecastArray;
    }

    public void setForecastArray(double[] forecastArray) {
        this.forecastArray = forecastArray;
    }

    public boolean[] getUsefulMarker() {
        return usefulMarker;
    }

    public void setUsefulMarker(boolean[] usefulMarker) {
        this.usefulMarker = usefulMarker;
    }

    public double[] getVeracityArray() {
        return veracityArray;
    }

    public void setVeracityArray(double[] veracityArray) {
        this.veracityArray = veracityArray;
    }

    public double getAvgPrecision() {
        return avgPrecision;
    }

    public void setAvgPrecision(double avgPrecision) {
        this.avgPrecision = avgPrecision;
    }

    public double getMaxPrecision() {
        return maxPrecision;
    }

    public void setMaxPrecision(double maxPrecision) {
        this.maxPrecision = maxPrecision;
    }

    public double getMinPrecision() {
        return minPrecision;
    }

    public void setMinPrecision(double minPrecision) {
        this.minPrecision = minPrecision;
    }

    public double getExtPrecision() {
        return extPrecision;
    }

    public void setExtPrecision(double extPrecision) {
        this.extPrecision = extPrecision;
    }

    public double getIntPrecision() {
        return intPrecision;
    }

    public void setIntPrecision(double intPrecision) {
        this.intPrecision = intPrecision;
    }

    // ============================= public ============================
    public void calVeracityArray() {
        if (seriesLength == 0) {
            veracityArray = null;
            return;
        }
        veracityArray = new double[seriesLength];
        for (int i = 0; i < seriesLength; i++)
            if (isAcceptValue(historyArray[i]))
                veracityArray[i] = 1 - Math.abs((forecastArray[i] - historyArray[i]) / historyArray[i]);
            else
                veracityArray[i] = DOUBLE_MISS_VALUE;
    }

//    /**
//     * 根据 usefulMarker 得到 series 的子序列
//     * eg:
//     * series = [1 2 3 4 5 6 7]  usefulMarker = [1 1 0 1 0 0 1]
//     * return [1 2 4 7]
//     *
//     */
//    private double[] getUsefulSeries(double[] series, boolean[] usefulMarker) {
//        if (series == null || usefulMarker == null || series.length == 0 || usefulMarker.length == 0)
//            return null;
//        assert (series.length == usefulMarker.length);
//        int count = 0;
//        for (boolean anUsefulMarker : usefulMarker)
//            if (anUsefulMarker)
//                count++;
//        if (count == 0) return null;
//        if (count == series.length) return series;
//        double[] usefulSeries = new double[count];
//        count = 0;
//        for (int i = 0; i < series.length; i++)
//            if (usefulMarker[i])
//                usefulSeries[count++] = series[i];
//        return usefulSeries;
//    }

    public void calAverageVeracity() {
        double sum = 0;
        int count = 0;
        for (int i = 0; i < seriesLength; i++) {
            if (usefulMarker[i] && !Double.isNaN(veracityArray[i])) {
                sum += StrictMath.pow(1 - veracityArray[i], 2);
                count++;
            }
        }
        if (count == 0) avgPrecision = SPECIAL_VERACITY;
        else avgPrecision = 1 - StrictMath.pow(sum / count, 0.5);
    }

    public void calExtremumVeracity() {
        double maxHistory = maxValue(historyArray, usefulMarker);
        double maxForecast = maxValue(forecastArray, usefulMarker);
        double minHistory = minValue(historyArray, usefulMarker);
        double minForecast = minValue(forecastArray, usefulMarker);
        if (maxHistory == Double.MIN_VALUE || minHistory == Double.MAX_VALUE) {
            maxPrecision = SPECIAL_VERACITY;
            minPrecision = SPECIAL_VERACITY;
            extPrecision = SPECIAL_VERACITY;
            return;
        }
        maxPrecision = 1 - StrictMath.abs(maxHistory - maxForecast) / maxHistory;
        minPrecision = 1 - StrictMath.abs(minHistory - minForecast) / minHistory;
        extPrecision = (maxPrecision + minPrecision) / 2;
    }

    public void calIntegrationVeracity() {
        if (avgPrecision == SPECIAL_VERACITY) intPrecision = SPECIAL_VERACITY;
        else intPrecision = avgPrecisionWeight * avgPrecision + extremPrecisionWeight * extPrecision;
    }

    // ================================= private =================================
    private double maxValue(double[] valueArray, boolean[] usefulMarker) {
        double max = Double.MIN_VALUE;
        if (valueArray != null)
            for (int i = 0; i < valueArray.length; i++)
                if (usefulMarker[i] && valueArray[i] > max && isAcceptValue(valueArray[i]))
                    max = valueArray[i];
        return max;
    }

    private double minValue(double[] valueArray, boolean[] usefulMarker) {
        double min = Double.MAX_VALUE;
        if (valueArray != null)
            for (int i = 0; i < valueArray.length; i++)
                if (usefulMarker[i] && valueArray[i] < min && isAcceptValue(valueArray[i]))
                    min = valueArray[i];
        return min;
    }

    private boolean isAcceptValue(double d) {
        return !(Double.isNaN(d) || d == 0 || d < ACCEPTE_VALUE);
    }
}
