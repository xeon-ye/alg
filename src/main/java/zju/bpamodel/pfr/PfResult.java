package zju.bpamodel.pfr;

import java.io.Serializable;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-14
 */
public class PfResult implements Serializable {
    private boolean isConverged;
    Map<String, BusPfResult> busData;
    //Map<String, BusPfResult> busData;

    public boolean isConverged() {
        return isConverged;
    }

    public void setConverged(boolean converged) {
        isConverged = converged;
    }

    public Map<String, BusPfResult> getBusData() {
        return busData;
    }

    public void setBusData(Map<String, BusPfResult> busData) {
        this.busData = busData;
    }
}
