package zju.lfp.forecasters.singleDimensionalForecaster.extrapolation;

import zju.lfp.utils.MultiTimeSeries;

import java.io.Serializable;
import java.util.Calendar;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-12-21
 * Time: 15:33:46
 */
public interface IExtrapolationInput extends Serializable {
    MultiTimeSeries formMultiTimeSeries(Calendar beginCal, Calendar endCal, int gapMinutes);

    int getLongPeriod();

    int getShortPeriod();

    int getUsefulPointsNum();
}
