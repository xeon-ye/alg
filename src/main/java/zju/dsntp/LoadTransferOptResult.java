package zju.dsntp;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by XuChengsi on 2017/1/19.
 */
public class LoadTransferOptResult {
    String[] supplyId;
    int[] minSwitch;
    List<String[]> switchChanged;

    String[] loadId;
    double[] maxLoad;

    public LoadTransferOptResult(int supplyNum, int loadNum) {
        this.supplyId = new String[supplyNum];
        this.minSwitch = new int[supplyNum];
        this.switchChanged = new ArrayList<>(supplyNum);
        this.loadId = new String[loadNum];
        this.maxLoad = new double[loadNum];
    }

    public void setSupplyId(int index, String supplyId) {
        this.supplyId[index] = supplyId;
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
