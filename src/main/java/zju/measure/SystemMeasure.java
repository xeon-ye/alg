package zju.measure;

import org.apache.log4j.Logger;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * This class storing system measurement information.
 * <br>All string key of may is position id except physical measurement.</br>
 *
 * @author Dong Shufeng
 *         Date: 2007-11-23
 */
public class SystemMeasure implements Serializable, MeasTypeCons, Cloneable {

    public static Logger log = Logger.getLogger(SystemMeasure.class);

    //physical measurements
    private Map<String, List<MeasureInfo>> id2Measure = new HashMap<>();

    //the following measurement is efficient measurements
    Map<String, MeasureInfo> bus_v = new HashMap<>();

    Map<String, MeasureInfo> bus_a = new HashMap<>();

    Map<String, MeasureInfo> bus_p = new HashMap<>();

    Map<String, MeasureInfo> bus_q = new HashMap<>();

    Map<String, MeasureInfo> line_from_p = new HashMap<>();

    Map<String, MeasureInfo> line_to_p = new HashMap<>();

    Map<String, MeasureInfo> line_from_q = new HashMap<>();

    Map<String, MeasureInfo> line_to_q = new HashMap<>();

    Map<String, MeasureInfo> line_i_amp = new HashMap<>();

    Map<String, MeasureInfo> line_i_a = new HashMap<>();

    Map<String, MeasureInfo> line_from_i_amp = new HashMap<>();

    Map<String, MeasureInfo> line_from_i_a = new HashMap<>();

    Map<String, MeasureInfo> line_to_i_amp = new HashMap<>();

    Map<String, MeasureInfo> line_to_i_a = new HashMap<>();

    public SystemMeasure() {
    }

    public int getMeasureNum() {
        int m = 0;
        for (int type : DEFAULT_TYPES)
            m += getContainer(type).size();
        return m;
    }

    public void addMeasure(MeasureInfo info) {
        id2Measure.putIfAbsent(info.getPositionId(), new ArrayList<>());
        id2Measure.get(info.getPositionId()).add(info);
    }

    public Map<String, MeasureInfo> getBus_v() {
        return bus_v;
    }

    public void setBus_v(Map<String, MeasureInfo> bus_v) {
        this.bus_v = bus_v;
    }

    public Map<String, MeasureInfo> getBus_a() {
        return bus_a;
    }

    public void setBus_a(Map<String, MeasureInfo> bus_a) {
        this.bus_a = bus_a;
    }

    public Map<String, MeasureInfo> getBus_p() {
        return bus_p;
    }

    public void setBus_p(Map<String, MeasureInfo> bus_p) {
        this.bus_p = bus_p;
    }

    public Map<String, MeasureInfo> getBus_q() {
        return bus_q;
    }

    public void setBus_q(Map<String, MeasureInfo> bus_q) {
        this.bus_q = bus_q;
    }

    public Map<String, MeasureInfo> getLine_from_p() {
        return line_from_p;
    }

    public void setLine_from_p(Map<String, MeasureInfo> line_from_p) {
        this.line_from_p = line_from_p;
    }

    public Map<String, MeasureInfo> getLine_to_p() {
        return line_to_p;
    }

    public void setLine_to_p(Map<String, MeasureInfo> line_to_p) {
        this.line_to_p = line_to_p;
    }

    public Map<String, MeasureInfo> getLine_from_q() {
        return line_from_q;
    }

    public void setLine_from_q(Map<String, MeasureInfo> line_from_q) {
        this.line_from_q = line_from_q;
    }

    public Map<String, MeasureInfo> getLine_to_q() {
        return line_to_q;
    }

    public void setLine_to_q(Map<String, MeasureInfo> line_to_q) {
        this.line_to_q = line_to_q;
    }

    public Map<String, MeasureInfo> getLine_i_amp() {
        return line_i_amp;
    }

    public void setLine_i_amp(Map<String, MeasureInfo> line_i_amp) {
        this.line_i_amp = line_i_amp;
    }

    public Map<String, MeasureInfo> getLine_i_a() {
        return line_i_a;
    }

    public void setLine_i_a(Map<String, MeasureInfo> line_i_a) {
        this.line_i_a = line_i_a;
    }

    public Map<String, MeasureInfo> getLine_from_i_amp() {
        return line_from_i_amp;
    }

    public void setLine_from_i_amp(Map<String, MeasureInfo> line_from_i_amp) {
        this.line_from_i_amp = line_from_i_amp;
    }

    public Map<String, MeasureInfo> getLine_from_i_a() {
        return line_from_i_a;
    }

    public void setLine_from_i_a(Map<String, MeasureInfo> line_from_i_a) {
        this.line_from_i_a = line_from_i_a;
    }

    public Map<String, MeasureInfo> getLine_to_i_amp() {
        return line_to_i_amp;
    }

    public void setLine_to_i_amp(Map<String, MeasureInfo> line_to_i_amp) {
        this.line_to_i_amp = line_to_i_amp;
    }

    public Map<String, MeasureInfo> getLine_to_i_a() {
        return line_to_i_a;
    }

    public void setLine_to_i_a(Map<String, MeasureInfo> line_to_i_a) {
        this.line_to_i_a = line_to_i_a;
    }

    /**
     * Get physical measurements on the position.
     *
     * @param id position id: bus number, branch id and so on.
     * @return physical measurements related to the position
     */
    public MeasureInfo[] getMeasure(String id) {
        return id2Measure.get(id) == null ? null : id2Measure.get(id).toArray(new MeasureInfo[]{});
    }

    public Map<String, List<MeasureInfo>> getId2Measure() {
        return id2Measure;
    }

    public void setId2Measure(Map<String, List<MeasureInfo>> id2Measure) {
        this.id2Measure = id2Measure;
    }

    public void removeEfficientMeasure(MeasureInfo info) {
        getContainer(info.getMeasureType()).remove(info.getPositionId());
    }

    public void addEfficientMeasure(MeasureInfo info) {
        getContainer(info.getMeasureType()).put(info.getPositionId(), info);
    }

    public MeasureInfo getEfficientMeasure(int type, String positionId) {
        return getContainer(type).get(positionId);
    }

    public Map<String, MeasureInfo> getContainer(int type) {
        switch (type) {
            case TYPE_BUS_ANGLE:
                return this.getBus_a();
            case TYPE_BUS_VOLOTAGE:
                return this.getBus_v();
            case TYPE_BUS_ACTIVE_POWER:
                return this.getBus_p();
            case TYPE_BUS_REACTIVE_POWER:
                return this.getBus_q();
            case TYPE_LINE_FROM_ACTIVE:
                return this.getLine_from_p();
            case TYPE_LINE_FROM_REACTIVE:
                return this.getLine_from_q();
            case TYPE_LINE_TO_ACTIVE:
                return this.getLine_to_p();
            case TYPE_LINE_TO_REACTIVE:
                return this.getLine_to_q();
            case TYPE_LINE_CURRENT:
                return this.getLine_i_amp();
            case TYPE_LINE_CURRENT_ANGLE:
                return this.getLine_i_a();
            case TYPE_LINE_FROM_CURRENT:
                return this.getLine_from_i_amp();
            case TYPE_LINE_FROM_CURRENT_ANGLE:
                return this.getLine_from_i_a();
            case TYPE_LINE_TO_CURRENT:
                return this.getLine_to_i_amp();
            case TYPE_LINE_TO_CURRENT_ANGLE:
                return this.getLine_to_i_a();
            default:
                log.warn("unsupported measure type: " + type);
                return null;
        }
    }

    public Object clone() {
        Object obj = null;
        try {
            obj = super.clone();
        } catch (CloneNotSupportedException exception) {
            System.out.println("SystemMeasure is not Cloneable");
        }

        return obj;
    }

    public boolean contains(MeasureInfo meas) {
        return getContainer(meas.getMeasureType()).containsKey(meas.getPositionId());
    }
}

