package zju.lfp.forecasters.singleDimensionalForecaster.arma;

import zju.lfp.utils.MultiTimeSeries;

import java.util.Calendar;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-12-22
 * Time: 2:23:04
 */
public interface IArmaInput {
    MultiTimeSeries formMultiTimeSeries(Calendar beginCal, Calendar endCal, int gapMinutes);

    int getLongPeriod();

    int getShortPeriod();
}
