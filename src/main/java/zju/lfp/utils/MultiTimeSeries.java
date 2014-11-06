package zju.lfp.utils;

import org.apache.log4j.Logger;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-12-6
 * Time: 16:57:34
 */
public class MultiTimeSeries {
    private final static Logger log = Logger.getLogger(MultiTimeSeries.class);

    private final static double MISS_VALUE = Double.NaN;

    private double[][] dataSet;

    private int numAttributes;
    private int nLen;

    private int missValueIndex;

    public MultiTimeSeries(int numAttributes, int nLen) {
        this.numAttributes = numAttributes;
        this.nLen = nLen;
        dataSet = new double[numAttributes][nLen];
        missValueIndex = 0;
        for (int i = 0; i < numAttributes; i++)
            for (int j = 0; j < nLen; j++)
                dataSet[i][j] = MISS_VALUE;
    }

    // ==================== setter =============================
    public void addOrUpdateTimeSeries(int index, MultiTimeSeries multiTimeSeries) {
        for (int i = 0; i < multiTimeSeries.getNumAttributes(); i++)
            addOrUpdateTimeSeries(index + i, multiTimeSeries.getTimeSeries(i));
        missValueIndex = multiTimeSeries.getMissValueIndex();
    }

    public void addOrUpdateTimeSeries(int index, double[] series) {
        addOrUpdateTimeSeries(index, 0, series);
    }

    public void addOrUpdateTimeSeries(int index, int nBegin, double[] series) {
        int lengthSeries = series.length;
        missValueIndex = nBegin + lengthSeries;
        if (missValueIndex > nLen) {
            log.error("series is too long to add: " + (nBegin + lengthSeries) + " > "
                    + nLen);
            return;
        }
        if (missValueIndex == nLen) {
            missValueIndex = 0;
        }
        System.arraycopy(series, 0, dataSet[index], nBegin, lengthSeries);
    }

    public void addOrUpdateTimeSeries(int index, int n, double value) {
        dataSet[index][n] = value;
        missValueIndex++;
        if (missValueIndex == nLen)
            missValueIndex = 0;
    }

    // ============================= getter ===============================
    public MultiTimeSeries getChildMultiTimeSeries(int index) {
        return getChildMultiTimeSeries(index, nLen);
    }

    public MultiTimeSeries getChildMultiTimeSeries(int index, int len) {
        if (index < 0 || index >= numAttributes) {
            log.info("Child MultiTimeSeries's index out of boundary!");
            return null;
        }
        MultiTimeSeries childMultiTimeSeries = new MultiTimeSeries(1, nLen);
        double[] data = getTimeSeries(index, 0, len);
        childMultiTimeSeries.addOrUpdateTimeSeries(0, data);
        return childMultiTimeSeries;
    }

    public double getValue(int index, int n) {
        return dataSet[index][n];
    }

    public double[] getTimeStampValues(int n) {
        double[] timeStampValue = new double[numAttributes];
        for (int i = 0; i < numAttributes; i++) {
            timeStampValue[i] = dataSet[i][n];
        }
        return timeStampValue;
    }

    public double[] getTimeSeries(int index) {
        return dataSet[index];
    }

    public double[] getTimeSeries(int index, int nBegin, int len) {
        double[] series = new double[len];
        System.arraycopy(dataSet[index], nBegin, series, 0, len);
        return series;
    }

    public int getNumAttributes() {
        return numAttributes;
    }

    public int getNLen() {
        return nLen;
    }

    public int getMissValueIndex() {
        return missValueIndex;
    }

    // ========================== MultiTimeSeries Instances (for app test) =======================
    public static MultiTimeSeries periodInstance(int period, int numPeriod, int lenUnknow) {
        MultiTimeSeries multiTimeSeries = new MultiTimeSeries(1, period * numPeriod + lenUnknow);
        for (int i = 0; i < period * numPeriod; i++) {
            multiTimeSeries.addOrUpdateTimeSeries(0, i, i % period);
        }
        return multiTimeSeries;
    }

    public static MultiTimeSeries doublePeriodInstance(int longPeriod, int shortPeriod,
                                                       int numShortPeriod, int lenUnknown) {
        MultiTimeSeries multiTimeSeries = new MultiTimeSeries(1,
                shortPeriod * numShortPeriod + lenUnknown);
        for (int i = 0; i < shortPeriod * numShortPeriod; i++) {
            if (i % longPeriod < shortPeriod) {
                multiTimeSeries.addOrUpdateTimeSeries(0, i, 2 * (i % shortPeriod));
            } else {
                multiTimeSeries.addOrUpdateTimeSeries(0, i, (i % shortPeriod));
            }
        }
        return multiTimeSeries;
    }

    public static MultiTimeSeries multiDoublePeriodInstance(int numAttributes, int longPeriod, int shortPeriod,
                                                            int numShortPeriod, int lenUnknown) {
        if (numAttributes <= 1) {
            log.error("Parameter error: numAttributes > 1 but was " + numAttributes);
            return null;
        }
        MultiTimeSeries multiTimeSeries = new MultiTimeSeries(numAttributes,
                shortPeriod * numShortPeriod + lenUnknown);
        for (int j = 0; j < numAttributes - 1; j++)
            for (int i = 0; i < multiTimeSeries.getNLen(); i++) {
                if (i % longPeriod < shortPeriod) {
                    multiTimeSeries.addOrUpdateTimeSeries(j, i, 2 * (i % shortPeriod));
                } else {
                    multiTimeSeries.addOrUpdateTimeSeries(j, i, (i % shortPeriod));
                }
            }
        for (int i = 0; i < shortPeriod * numShortPeriod; i++) {
            if (i % longPeriod < shortPeriod) {
                multiTimeSeries.addOrUpdateTimeSeries(numAttributes - 1, i, 2 * (i % shortPeriod));
            } else {
                multiTimeSeries.addOrUpdateTimeSeries(numAttributes - 1, i, (i % shortPeriod));
            }
        }
        return multiTimeSeries;
    }
}
