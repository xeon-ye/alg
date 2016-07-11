package zju.lfp.forecasters.chaos;

import org.apache.log4j.Logger;
import weka.core.*;
import zju.lfp.utils.MultiTimeSeries;

import java.util.ArrayList;

/**
 * Created by IntelliJ IDEA.
 * User: zhangsiyuan
 * Date: 2007-12-7
 * Time: 13:12:25
 */
public class ChaosInstances {
    private final static Logger log = Logger.getLogger(ChaosInstances.class);
    private final static String ATTRIBUTE_NAME = "attribute_";

    private MultiTimeSeries multiTimeSeries;
    private ChaosTimeSeries[] chaosTimeSerieses;
    private int[] delayPoints;
    private int[] reformDims;

    private int totalAttributeDims;

    public ChaosInstances(MultiTimeSeries multiTimeSeries, int[] delayPoints, int[] reformDims) {
        if(delayPoints.length != multiTimeSeries.getNumAttributes()){
            log.error("Inequality Dimension: was " + multiTimeSeries.getNumAttributes()
                    + " but " + delayPoints.length);
            return;
        }
        this.multiTimeSeries = multiTimeSeries;
        this.delayPoints = delayPoints;
        this.reformDims = reformDims;
        totalAttributeDims = 0;
        for (int reformDim : reformDims)
            totalAttributeDims += reformDim;
    }

    private Instances formInstancesHeader() {
        String instancesName = toString();
        ArrayList<Attribute> attrInfo = new ArrayList<Attribute>();
        for (int i = 0; i < totalAttributeDims + 1; i++)
            attrInfo.add(new Attribute(ATTRIBUTE_NAME + i));
        Instances instances = new Instances(instancesName, attrInfo, 0);
        instances.setClass(instances.attribute(totalAttributeDims));
        return instances;
    }

    private void addInstance(Instances instances, int n) {
        double[] reformedSpacePoint = ChaosTimeSeriesUtil.getReformedSpacePoint(chaosTimeSerieses,
                n, delayPoints, reformDims);
        Instance instance = new DenseInstance(totalAttributeDims + 1);
        for(int i = 0; i < totalAttributeDims; i++) {
            instance.setValue(i, reformedSpacePoint[i]);
        }
        instance.setValue(totalAttributeDims, multiTimeSeries.getValue(
                multiTimeSeries.getNumAttributes() - 1, n+1)); // to test
        instances.add(instance);
    }

    public Instances getChaosInstances(int[] pointArray) {
        Instances instances = formInstancesHeader();
        chaosTimeSerieses = ChaosTimeSeriesUtil.covertToChaosTimeSeries(multiTimeSeries);
        for (int aPointArray : pointArray) {
            addInstance(instances, aPointArray);
        }
        return instances;
    }

    public Instance getWorkingInstance(int n) {
        Instance instance = new DenseInstance(totalAttributeDims + 1);
        chaosTimeSerieses = ChaosTimeSeriesUtil.covertToChaosTimeSeries(multiTimeSeries);
        double[] reformedSpacePoint = ChaosTimeSeriesUtil.getReformedSpacePoint(chaosTimeSerieses,
                n-1, delayPoints, reformDims);
        for(int i = 0; i < totalAttributeDims; i++) {
            instance.setValue(i, reformedSpacePoint[i]);
        }
        return instance;
    }
}
