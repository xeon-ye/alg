package zju.dsntp;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by XuChengsi on 2017/1/19.
 */
public class LoadTransferOptResult {
    String[] supplyId;
    String[] feederId;
    int[] minSwitch;
    List<String[]> switchChanged;

    String[] loadId;
    double[] maxLoad;

    public LoadTransferOptResult(int feederNum, int loadNum) {
        this.feederId = new String[feederNum];
        this.minSwitch = new int[feederNum];
        this.switchChanged = new ArrayList<>(feederNum);
        this.loadId = new String[loadNum];
        this.maxLoad = new double[loadNum];
    }

    public void setSupplyId(int index, String supplyId) {
        this.supplyId[index] = supplyId;
    }

    public void setFeederId(int index, String feederId) {
        this.feederId[index] = feederId;
    }

    public void setMinSwitch(int index, int minSwitch) {
        this.minSwitch[index] = minSwitch;
    }

    public void setSwitchChanged(String[] switchChanged) {
        this.switchChanged.add(switchChanged);
    }

    public void setLoadId(String[] loadId) {
        this.loadId = loadId;
    }

    public void setMaxLoad(double[] maxLoad) {
        this.maxLoad = maxLoad;
    }

    public String[] getSupplyId() {
        return supplyId;
    }

    public String[] getFeederId() {
        return feederId;
    }

    public int[] getMinSwitch() {
        return minSwitch;
    }

    public List<String[]> getSwitchChanged() {
        return switchChanged;
    }

    public String[] getLoadId() {
        return loadId;
    }

    public double[] getMaxLoad() {
        return maxLoad;
    }
}
