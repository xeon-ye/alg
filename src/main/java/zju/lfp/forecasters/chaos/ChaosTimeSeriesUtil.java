package zju.lfp.forecasters.chaos;

import org.apache.log4j.Logger;
import zju.lfp.utils.MultiTimeSeries;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-10-23
 * Time: 9:36:51
 */
public class ChaosTimeSeriesUtil {
    private static Logger log = Logger.getLogger(ChaosTimeSeriesUtil.class);

    public static int[] getTimeDelayPoints(ChaosTimeSeries[] chaosTimeSerieses){
        return null;
    }

    public static int[] getReformDimensions(ChaosTimeSeries[] chaosTimeSerieses) {
        return null;
    }

    /**
     * 多维时间序列的重构
     * @param chaosTimeSeries 多维时间序列
     * @param pointIndex 原时间序列中的点位置
     *@param delayPoints 多变量的延迟点数
     * @param minReformDims 多变量的重构维数
     * @return 超空间中的点
     */
    public static double[] getReformedSpacePoint(ChaosTimeSeries[] chaosTimeSeries, int pointIndex,int[] delayPoints, int[] minReformDims) {
        int sumLength = 0;
        List<double[]> list = new ArrayList<double[]>();
        double[] point;
        for(int i = 0; i< chaosTimeSeries.length; i++) {
            point = chaosTimeSeries[i].getReformedSpacePoint(pointIndex, delayPoints[i], minReformDims[i]);
            list.add(point);
            sumLength += minReformDims[i];
        }
        double[] phaseSpacePoint = new double[sumLength];
        int index = 0;
        for(double[] p : list) {
            System.arraycopy(p, 0, phaseSpacePoint, index, p.length);
            index += p.length;
        }
        return phaseSpacePoint;
    }

    public static double distanceBetweenSpacePoints(ChaosTimeSeries[] chaosTimeSeries, int index1,int index2, int[] delayPoints, int[] minReformDims) {
        double[] point1 = getReformedSpacePoint(chaosTimeSeries, index1, delayPoints, minReformDims);
        double[] point2 = getReformedSpacePoint(chaosTimeSeries, index2, delayPoints, minReformDims);
        return ChaosTimeSeries.calDistance(point1, point2);
    }

    public static int beginUsefulPointIndex(int[] delayPoints, int[] minReformDims) {
        int max = Integer.MIN_VALUE;
        for(int i = 0; i < delayPoints.length; i++) {
            if(delayPoints[i] * minReformDims[i] > max) {
                max = delayPoints[i] * minReformDims[i];
            }
        }
        return max;
    }

    public static ChaosTimeSeries[] covertToChaosTimeSeries(MultiTimeSeries multiTimeSeries) {
        ChaosTimeSeries[] chaosTimeSerieses = new ChaosTimeSeries[multiTimeSeries.getNumAttributes()];
        for(int i = 0; i < multiTimeSeries.getNumAttributes(); i++) {
            double[] series = multiTimeSeries.getTimeSeries(i, 0, multiTimeSeries.getMissValueIndex());
            chaosTimeSerieses[i] = new ChaosTimeSeries(series);
        }
        return chaosTimeSerieses;
    }
}
