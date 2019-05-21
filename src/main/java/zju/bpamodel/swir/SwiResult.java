package zju.bpamodel.swir;

import java.io.Serializable;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-5-21
 */
public class SwiResult implements Serializable {
    List<MonitorData> monitorDataList;

    public List<MonitorData> getMonitorDataList() {
        return monitorDataList;
    }

    public void setMonitorDataList(List<MonitorData> monitorDataList) {
        this.monitorDataList = monitorDataList;
    }
}
