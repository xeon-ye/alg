package zju.lfp.utils;

import java.util.Calendar;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-10-22
 * Time: 14:37:16
 */
public interface DataInfoService {
    String getForecastingContext();

    void setForecastingContext(String contextName);

    boolean shouldForecast(String contextName);

    int getSampleBeginMinutesInDay();

    int getSamplePeriodMinutes();

    boolean isSampleStage(Calendar cal);

    boolean saveForecastResult(double[] forecastResult, Calendar beginCal, int gapMinutes, String savedType);
}
