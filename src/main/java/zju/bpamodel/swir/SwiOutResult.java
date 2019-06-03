package zju.bpamodel.swir;

import java.io.Serializable;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-5-21
 */
public class SwiOutResult implements Serializable {
    private List<MonitorData> monitorDataList;
    private List<Damping> dampings;

    public List<MonitorData> getMonitorDataList() {
        return monitorDataList;
    }

    public void setMonitorDataList(List<MonitorData> monitorDataList) {
        this.monitorDataList = monitorDataList;
    }

    public List<Damping> getDampings() {
        return dampings;
    }

    public void setDampings(List<Damping> dampings) {
        this.dampings = dampings;
    }
}
