package zju.bpamodel.pf;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-25
 */

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-25
 */
public class ElectricIsland implements Serializable {
    PfCtrlInfo pfCtrlInfo = new PfCtrlInfo();
    private List<Bus> buses;
    private List<Transformer> transformers;
    private List<AcLine> aclines;
    private Map<String, Bus> nameToBus;
    private Map<String, List<Transformer>> busToTransformers;
    private Map<String, List<AcLine>> busToAclines;

    public void modifyData(HashMap<String, DataModifyInfo> zone2dataModifyInfoMap) {
        String zone;
        DataModifyInfo dataModifyInfo;
        for (Bus bus : buses) {
            zone = bus.getZoneName();
            if (zone2dataModifyInfoMap.containsKey(zone)) {
                dataModifyInfo = zone2dataModifyInfoMap.get(zone);
                bus.setLoadMw(bus.getLoadMw() * dataModifyInfo.getLoadPFactor());
                bus.setLoadMvar(bus.getLoadMvar() * dataModifyInfo.getLoadQFactor());
                bus.setGenMw(bus.getGenMw() * dataModifyInfo.getGenPFactor());
                bus.setGenMvar(bus.getGenMvar() * dataModifyInfo.getGenQFactor());
            }
        }
    }

    public void buildTopo() {
        busToTransformers = new HashMap<String, List<Transformer>>();
        for (Transformer t : transformers) {
            if (!busToTransformers.containsKey(t.getBusName1() + (int) (t.getBaseKv1())))
                busToTransformers.put(t.getBusName1() + (int) (t.getBaseKv1()), new ArrayList<Transformer>());
            busToTransformers.get(t.getBusName1() + (int) (t.getBaseKv1())).add(t);
            if (!busToTransformers.containsKey(t.getBusName2() + (int) (t.getBaseKv2())))
                busToTransformers.put(t.getBusName2() + (int) (t.getBaseKv2()), new ArrayList<Transformer>());
            busToTransformers.get(t.getBusName2() + (int) (t.getBaseKv2())).add(t);
        }
        busToAclines = new HashMap<String, List<AcLine>>();
        for (AcLine acline : aclines) {
            if (!busToAclines.containsKey(acline.getBusName1() + (int) (acline.getBaseKv1())))
                busToAclines.put(acline.getBusName1() + (int) (acline.getBaseKv1()), new ArrayList<AcLine>());
            busToAclines.get(acline.getBusName1() + (int) (acline.getBaseKv1())).add(acline);
            if (!busToAclines.containsKey(acline.getBusName2() + (int) (acline.getBaseKv2())))
                busToAclines.put(acline.getBusName2() + (int) (acline.getBaseKv2()), new ArrayList<AcLine>());
            busToAclines.get(acline.getBusName2() + (int) (acline.getBaseKv2())).add(acline);
        }
    }

    public Map<String, Bus> getNameToBus() {
        if (nameToBus == null || nameToBus.size() != buses.size()) {
            nameToBus = new HashMap<String, Bus>(buses.size());
            for (Bus b : buses)
                nameToBus.put(b.getName() + (int) (b.getBaseKv()), b);
        }
        return nameToBus;
    }

    public PfCtrlInfo getPfCtrlInfo() {
        return pfCtrlInfo;
    }

    public void setPfCtrlInfo(PfCtrlInfo pfCtrlInfo) {
        this.pfCtrlInfo = pfCtrlInfo;
    }

    public List<Bus> getBuses() {
        return buses;
    }

    public void setBuses(List<Bus> buses) {
        this.buses = buses;
    }

    public List<Transformer> getTransformers() {
        return transformers;
    }

    public void setTransformers(List<Transformer> transformers) {
        this.transformers = transformers;
    }

    public Map<String, List<Transformer>> getBusToTransformers() {
        return busToTransformers;
    }

    public void setBusToTransformers(Map<String, List<Transformer>> busToTransformers) {
        this.busToTransformers = busToTransformers;
    }

    public List<AcLine> getAclines() {
        return aclines;
    }

    public void setAclines(List<AcLine> aclines) {
        this.aclines = aclines;
    }

    public Map<String, List<AcLine>> getBusToAclines() {
        return busToAclines;
    }

    public void setBusToAclines(Map<String, List<AcLine>> busToAclines) {
        this.busToAclines = busToAclines;
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        for (Bus b : buses)
            str.append(b.toString()).append("\n");
        for (AcLine line : aclines)
            str.append(line.toString()).append("\n");
        for (Transformer t : transformers)
            str.append(t.toString()).append("\n");
        return str.toString();
    }
}

