package zju.lfp;

import zju.lfp.utils.MultiTimeSeries;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-12-6
 * Time: 23:13:03
 */
public interface MultiTimeSeriesPredictor extends Serializable {
    void predict(MultiTimeSeries multiTimeSeries);
}
