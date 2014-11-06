package zju.lfp;

import java.rmi.Remote;
import java.rmi.RemoteException;
import java.util.Calendar;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-9-25
 * Time: 14:33:03
 */
public interface Forecaster extends Remote {
    /**
     * 连续预测器
     * @param beginCal 预测开始时刻
     * @param endCal 预测结束时刻
     * @param gapMinutes 采样时间间隔（分钟）
     * @return doubleArray
     * @throws RemoteException e
     */
    double[] forecast(Calendar beginCal, Calendar endCal, int gapMinutes) throws RemoteException;

    /**
     * 单点预测器
     * @param cal 预测时刻
     * @param gapMinutes 采样时间间隔（分钟）
     * @return forecast result
     * @throws RemoteException e
     */
    double forecast(Calendar cal, int gapMinutes) throws RemoteException;

    /**
     * 预测器参数设置
     * @param options map
     * @throws RemoteException e
     */
    void setOptions(Map<String, Object> options) throws RemoteException;

    /**
     * 得到预测器参数
     * @return map
     * @throws RemoteException e
     */
    Map<String, Object> getOptions()throws RemoteException;
}
