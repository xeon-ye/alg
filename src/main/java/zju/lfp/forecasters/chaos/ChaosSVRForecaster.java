package zju.lfp.forecasters.chaos;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import weka.classifiers.functions.SMOreg;
import weka.core.Instance;
import weka.core.Instances;
import zju.lfp.utils.MultiTimeSeries;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-12-10
 * Time: 17:02:06
 */
public class ChaosSVRForecaster extends MultiStepForecaster{
    private final static Logger log = LogManager.getLogger(ChaosSVRForecaster.class);

    private int nExtend;
    private int[] delayPoints;
    private int[] reformDims;

    private ChaosInstances chaosInstances;
    private SMOreg smoReg;

    public ChaosSVRForecaster(MultiTimeSeries multiTimeSeries, int period,
                              int nBegin, int nEnd,
                              int[] delayPoints, int[] reformDims) {
        super(multiTimeSeries, period, nBegin, nEnd);
        nExtend = (nEnd - nBegin) / 3;
        this.delayPoints = delayPoints;
        this.reformDims = reformDims;

        chaosInstances = new ChaosInstances(multiTimeSeries, delayPoints, reformDims);
    }

    private int[] getNecessarySamplesId() {
        List<Integer> list = new ArrayList<Integer>();
        int beginUsefulPointIndex = ChaosTimeSeriesUtil.beginUsefulPointIndex(delayPoints, reformDims);
        // set begin point and end point
        int nBeginTemp = nBegin % period;
        nBeginTemp -= nExtend;
        int nEndTemp = nBeginTemp + 2 * nExtend + (nEnd - nBegin);
        // loop
        while(true) {
            if(nBeginTemp > beginUsefulPointIndex)
                break;
            nBeginTemp += period;
            nEndTemp += period;
        }
        while(nEndTemp < nEnd) {
            for(int i = nBeginTemp; i < nEndTemp; i++) {
                list.add(i);
            }
            nBeginTemp += period;
            nEndTemp += period;
        }
        for(int i = nBegin - nExtend; i < nBegin - 1; i++) {
            list.add(i);
        }
        //
        int[] array = new int[list.size()];
        int j = 0;
        for(Integer i : list) {
            array[j++] = i;
        }
        return array;
    }

    protected void buildModel() {
        int[] necessarySamplesId = getNecessarySamplesId();
        Instances instances = chaosInstances.getChaosInstances(necessarySamplesId);
        smoReg = new SMOreg();
        try {
            smoReg.buildClassifier(instances);
        } catch (Exception e) {
            log.error("weka error: Can't build classifier!");
        }
    }

    protected double forecast(int n) {
        Instance instance = chaosInstances.getWorkingInstance(n);
        try {
            return smoReg.classifyInstance(instance);
        } catch (Exception e) {
            log.error("weka error: Can't forecast instance!");
            return Double.NaN;
        }
    }
}
