package zju.bpamodel.swir;

import java.util.List;

public class BusData {
    private String busName = "";
    private double baseKv;
    private List<BusOneStepData> busOneStepDataList;

    public String getBusName() {
        return busName;
    }

    public void setBusName(String busName) {
        this.busName = busName;
    }

    public double getBaseKv() {
        return baseKv;
    }

    public void setBaseKv(double baseKv) {
        this.baseKv = baseKv;
    }

    public List<BusOneStepData> getBusOneStepDataList() {
        return busOneStepDataList;
    }

    public void setBusOneStepDataList(List<BusOneStepData> busOneStepDataList) {
        this.busOneStepDataList = busOneStepDataList;
    }
}
