package zju.lfp.forecasters.singleDimensionalForecaster.arma;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
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
 * Date: 2007-9-25
 * Time: 16:06:29
 */
public class ArmaForecaster extends UnicastRemoteObject implements Forecaster {
    private final static Logger log = LogManager.getLogger(ArmaForecaster.class);

    private final static String forecasterName = "Arma";

    private ArmaPredictor predictor;
    private IArmaInput dataService;

    public ArmaForecaster() throws RemoteException {
        try {
            dataService = DataServiceFactory.getArmaService();
        } catch (Exception e) {
            log.error(e);
        }
        predictor = new ArmaPredictor(dataService.getLongPeriod(),
                dataService.getShortPeriod());
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
