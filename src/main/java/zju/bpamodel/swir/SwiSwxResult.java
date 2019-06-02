package zju.bpamodel.swir;

import java.util.List;

public class SwiSwxResult {
    private List<GeneratorData> generatorDataList;
    private List<BusData> busDataList;
    private List<LineData> lineDataList;

    public List<GeneratorData> getGeneratorDataList() {
        return generatorDataList;
    }

    public void setGeneratorDataList(List<GeneratorData> generatorDataList) {
        this.generatorDataList = generatorDataList;
    }

    public List<BusData> getBusDataList() {
        return busDataList;
    }

    public void setBusDataList(List<BusData> busDataList) {
        this.busDataList = busDataList;
    }

    public List<LineData> getLineDataList() {
        return lineDataList;
    }

    public void setLineDataList(List<LineData> lineDataList) {
        this.lineDataList = lineDataList;
    }
}
