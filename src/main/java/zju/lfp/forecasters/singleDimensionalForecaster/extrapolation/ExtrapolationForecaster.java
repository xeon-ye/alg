package zju.lfp.forecasters.singleDimensionalForecaster.extrapolation;

import org.apache.log4j.Logger;
import zju.lfp.Forecaster;
import zju.lfp.utils.DataServiceFactory;
import zju.lfp.utils.MultiTimeSeries;

import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import java.util.Calendar;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-9-26
 * Time: 10:05:21
 */
public class ExtrapolationForecaster extends UnicastRemoteObject implements Forecaster {
    private final static Logger log = Logger.getLogger(ExtrapolationForecaster.class);

    private final static String forecasterName = "Extrapolation";

    private ExtrapolationPredictor predictor;
    private IExtrapolationInput dataService;

    public ExtrapolationForecaster() throws RemoteException {
        try {
            dataService = DataServiceFactory.getExtrapolationService();
        } catch (Exception e) {
            log.error(e);
        }
        predictor = new ExtrapolationPredictor(dataService.getLongPeriod(),
                dataService.getShortPeriod(), dataService.getUsefulPointsNum());
    }

    public String toString() {
        return forecasterName;
    }

    public double[] forecast(Calendar beginCal, Calendar endCal, int gapMinutes) throws RemoteException {
        MultiTimeSeries multiTimeSeries = dataService.formMultiTimeSeries(beginCal, endCal, gapMinutes);
        int beginIndex = multiTimeSeries.getMissValueIndex();
        predictor.predict(multiTimeSeries);
        int seriesIndexForecasting = multiTimeSeries.getNumAttributes() - 1;
        int lenForecasting = multiTimeSeries.getNLen() - beginIndex;
        return multiTimeSeries.getTimeSeries(seriesIndexForecasting, beginIndex, lenForecasting);
    }

    public double forecast(Calendar cal, int gapMinutes) throws RemoteException {
        Calendar endCal = (Calendar) cal.clone();
        endCal.add(Calendar.MINUTE, gapMinutes);
        return forecast(cal, endCal, gapMinutes)[0];
    }

    public void setOptions(Map<String, Object> options) {

    }

    public Map<String, Object> getOptions() {
        return null;
    }
}
