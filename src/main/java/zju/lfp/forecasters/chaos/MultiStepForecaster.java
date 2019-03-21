package zju.lfp.forecasters.chaos;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.lfp.utils.MultiTimeSeries;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-12-10
 * Time: 16:26:43
 */
public abstract class MultiStepForecaster {
    private final static Logger log = LogManager.getLogger(MultiStepForecaster.class);

    protected MultiTimeSeries multiTimeSeries;
    protected int period;
    protected int nBegin;
    protected int nEnd;

    protected MultiStepForecaster(MultiTimeSeries multiTimeSeries, int period, int nBegin, int nEnd) {
        this.multiTimeSeries = multiTimeSeries;
        this.period = period;
        this.nBegin = nBegin;
        this.nEnd = nEnd;
    }

    protected abstract void buildModel();

    protected abstract double forecast(int n);

    public void forecast() {
        buildModel();
        for(int i = nBegin; i < nEnd; i++) {
            double result = forecast(i);
            multiTimeSeries.addOrUpdateTimeSeries(multiTimeSeries.getNumAttributes() - 1,
                    i, result);
        }
    }
}