package zju.lfp.forecasters.analogyExtrapolation;

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
 * Date: 2007-12-29
 * Time: 14:13:37
 */
public class AnalogyExtrapolationForecaster extends UnicastRemoteObject implements Forecaster {
    private final static Logger log = Logger.getLogger(AnalogyExtrapolationForecaster.class);

    private final static String forecasterName = "AnalogyExtrapolation";

    private AnalogyExtrapolationPredictor predictor;
    private IAnalogyExtrapolationInput dataService;

    public AnalogyExtrapolationForecaster() throws RemoteException {
        try {
            dataService = DataServiceFactory.getAnalogyExtrapolationService();
        } catch (Exception e) {
            log.error(e);
        }
        predictor = new AnalogyExtrapolationPredictor(dataService.getLongPeriod(),
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

