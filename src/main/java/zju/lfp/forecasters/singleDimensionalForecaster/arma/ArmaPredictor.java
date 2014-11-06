package zju.lfp.forecasters.singleDimensionalForecaster.arma;

import zju.lfp.MultiTimeSeriesPredictor;
import zju.lfp.utils.MultiTimeSeries;
import zju.lfp.utils.TimeSeries;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-12-22
 * Time: 1:55:47
 */
public class ArmaPredictor implements MultiTimeSeriesPredictor {
    // 时间序列的长周期
    private int longPeriod;
    // 时间序列的短周期
    private int shortPeriod;

    public ArmaPredictor(int longPeriod, int shortPeriod) {
        this.longPeriod = longPeriod;
        this.shortPeriod = shortPeriod;
    }

    public void predict(MultiTimeSeries multiTimeSeries) {
        int forecastedSeriesIndex = multiTimeSeries.getNumAttributes() - 1;
        int lenPredict = multiTimeSeries.getNLen() - multiTimeSeries.getMissValueIndex();

        TimeSeries pData = new TimeSeries();
        pData.SetSeries(multiTimeSeries.getMissValueIndex(),
                multiTimeSeries.getTimeSeries(forecastedSeriesIndex, 0, multiTimeSeries.getMissValueIndex()));

        TimeSeries pControl = null;
        TimeSeries pControlFore = null;

        TimeSeries pFore = new TimeSeries();
        pFore.SetSeries(lenPredict);

        boolean bSuccess = predictWithDifferent(pData, pControl, pControlFore, pFore);

        if (bSuccess)
            multiTimeSeries.addOrUpdateTimeSeries(forecastedSeriesIndex, multiTimeSeries.getMissValueIndex(),
                    pFore.getData());
    }

    public boolean predictWithDifferent(TimeSeries s1, TimeSeries c1, TimeSeries c2, TimeSeries r1) {

        s1.Different(longPeriod,1);
        s1.Different(shortPeriod, 1);

        //
        TimeSeries s11 = new TimeSeries();
        TimeSeries c11 = new TimeSeries();
        
        int len = s1.m_nLen - longPeriod - shortPeriod;

        s1.GetSubSeries(longPeriod + shortPeriod, 1, s11, len);
        if (c1 != null && c1.m_nLen > 0) c1.GetSubSeries(longPeriod + shortPeriod, 1, c11, len);
        boolean bSuccess = predictIt(s11, c11, c2, r1);
        //
        TimeSeries y = new TimeSeries();
        y.SetSeries(s1.m_nLen + r1.m_nLen);

        for (int i = 0; i < s1.m_nLen; i++)
            y.m_pSeries[i] = s1.m_pSeries[i];
        for (int i = 0; i < r1.m_nLen; i++)
            y.m_pSeries[i + s1.m_nLen] = r1.m_pSeries[i];
        //y.Intergral(1,1);
        y.Intergral(shortPeriod,1);
        y.Intergral(longPeriod, 1);

        for (int i = 0; i < s1.m_nLen; i++)
            s1.m_pSeries[i] = y.m_pSeries[i];
        for (int i = 0; i < r1.m_nLen; i++)
            r1.m_pSeries[i] = y.m_pSeries[s1.m_nLen + i];

        return bSuccess;
    }

    boolean predictIt(TimeSeries pS1, TimeSeries pC1, TimeSeries pC2, TimeSeries pR1) {
        if (pC1 != null && pC1.m_nLen == 0) pC1 = null;
        if (pC2 != null && pC2.m_nLen == 0) pC2 = null;
        //
        int pMax = 1000;//pS1.m_nLen/10;
        String methodName = "AR";
        Arma arma = new Arma();
        if (methodName.equals("AR")) arma.SetModel(AForecastModel._Model_Ar);
        else if (methodName.equals("ARLONG")) arma.SetModel(AForecastModel._Model_LongAr);
        else if (methodName.equals("ARX")) arma.SetModel(AForecastModel._Model_Arx);
        else if (methodName.equals("ARMA")) arma.SetModel(AForecastModel._Model_Arma);

        if (!arma.Forecast(pS1, pC1, pC2, pR1, pMax)) return false;
        return true;
    }
}
