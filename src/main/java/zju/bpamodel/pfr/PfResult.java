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
    Map<String, BranchPfResult> branchData;
    Map<String, TransformerPfResult> transformerData;

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

    public Map<String, BranchPfResult> getBranchData() {
        return branchData;
    }

    public void setBranchData(Map<String, BranchPfResult> branchData) {
        this.branchData = branchData;
    }

    public Map<String, TransformerPfResult> getTransformerData() {
        return transformerData;
    }

    public void setTransformerData(Map<String, TransformerPfResult> transformerData) {
        this.transformerData = transformerData;
    }
}
