package zju.bpamodel.swir;

import java.util.List;

public class GeneratorData {
    private String busName1 = "";
    private double baseKv1;
    private String busName2 = "";
    private double baseKv2;
    private List<GenOneStepData> genOneStepDataList;

    public String getBusName1() {
        return busName1;
    }

    public void setBusName1(String busName1) {
        this.busName1 = busName1;
    }

    public double getBaseKv1() {
        return baseKv1;
    }

    public void setBaseKv1(double baseKv1) {
        this.baseKv1 = baseKv1;
    }

    public String getBusName2() {
        return busName2;
    }

    public void setBusName2(String busName2) {
        this.busName2 = busName2;
    }

    public double getBaseKv2() {
        return baseKv2;
    }

    public void setBaseKv2(double baseKv2) {
        this.baseKv2 = baseKv2;
    }

    public List<GenOneStepData> getGenOneStepDataList() {
        return genOneStepDataList;
    }

    public void setGenOneStepDataList(List<GenOneStepData> genOneStepDataList) {
        this.genOneStepDataList = genOneStepDataList;
    }
}
