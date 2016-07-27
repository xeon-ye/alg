package zju.ieeeformat;

import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AbstractMatrix;
import zju.util.JOFileUtil;

import java.io.Serializable;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Class IEEEDataIsland
 * <p> isolate island of bus and branch </P>
 *
 * @author Dong Shufeng
 * @version 1.0
 * Date: 2006-8-29
 */
public class IEEEDataIsland implements Serializable, Cloneable {

    private static final long serialVersionUID = 7877631297451034684l;

    private TitleData title = null;

    private List<BusData> buses = null;

    private List<BranchData> branches = null;

    private List<LossZoneData> lossZones = null;

    private List<InterchangeData> interchanges = null;

    private List<TieLineData> tieLines = null;

    private Map<Integer, BranchData> id2branch = null;

    private int pvBusSize = 0;

    private int pqBusSize = 0;

    private int slackBusSize = 0;

    public IEEEDataIsland() {
    }

    public IEEEDataIsland(TitleData title, List<BusData> buses, List<BranchData> branches,
                          List<LossZoneData> lossZones, List<InterchangeData> interchanges, List<TieLineData> tieLines) {
        this.title = title;
        this.buses = buses;
        this.branches = branches;
        this.lossZones = lossZones;
        this.interchanges = interchanges;
        this.tieLines = tieLines;

        buildBranchIndex();
        buildBusSummary();
    }

    public void buildBusSummary() {
        pvBusSize = 0;
        pqBusSize = 0;
        slackBusSize = 0;
        for (BusData bus : getBuses()) {
            int type = bus.getType();
            switch (type) {
                case BusData.BUS_TYPE_LOAD_PQ://
                case BusData.BUS_TYPE_GEN_PQ://
                    pqBusSize++;
                    break;
                case BusData.BUS_TYPE_GEN_PV://
                    pvBusSize++;
                    break;
                case BusData.BUS_TYPE_SLACK://
                    slackBusSize++;
                    break;
                default://
                    break;
            }
        }
    }

    public Map<Integer, BusData> getBusMap() {
        Map<Integer, BusData> busMap = new HashMap<Integer, BusData>(getBuses().size());
        for (BusData bus : getBuses()) {
            busMap.put(bus.getBusNumber(), bus);
        }
        return busMap;
    }

    public void buildBranchIndex() {
        id2branch = new HashMap<Integer, BranchData>(this.getBranches().size());
        for (int i = 0; i < getBranches().size(); i++) {
            BranchData branch = getBranches().get(i);
            branch.setId(i + 1);
            id2branch.put(branch.getId(), branch);
        }
    }

    public TitleData getTitle() {
        return title;
    }

    public List<BusData> getBuses() {
        return buses;
    }

    public List<BranchData> getBranches() {
        return branches;
    }

    public List<LossZoneData> getLossZones() {
        return lossZones;
    }

    public List<InterchangeData> getInterchanges() {
        return interchanges;
    }

    public List<TieLineData> getTieLines() {
        return tieLines;
    }

    public void setTitle(TitleData title) {
        this.title = title;
    }

    public void setBuses(List<BusData> buses) {
        this.buses = buses;
        buildBusSummary();
    }

    public void setBranches(List<BranchData> branches) {
        this.branches = branches;
        buildBranchIndex();
    }

    public void setLossZones(List<LossZoneData> lossZones) {
        this.lossZones = lossZones;
    }

    public void setInterchanges(List<InterchangeData> interchanges) {
        this.interchanges = interchanges;
    }

    public void setTieLines(List<TieLineData> tieLines) {
        this.tieLines = tieLines;
    }

    public BusData getBus(int num) {
        if (buses != null) {
            for (BusData b : buses)
                if (b.getBusNumber() == num)
                    return b;
        }
        return null;
    }

    public BusData getBus(String name) {
        if (buses != null) {
            for (BusData b : buses)
                if (b.getName().equals(name))
                    return b;
        }
        return null;
    }

    public int getSlackBusNum() {
        if (buses == null)
            return -1;
        for (BusData b : buses) {
            if (b.getType() == BusData.BUS_TYPE_SLACK)
                return b.getBusNumber();
        }
        return -1;
    }

    public int getSlackBusSize() {
        return slackBusSize;
    }

    public Map<Integer, BranchData> getId2branch() {
        return id2branch;
    }

    public void setId2branch(Map<Integer, BranchData> id2branch) {
        this.id2branch = id2branch;
    }

    public int getPvBusSize() {
        return pvBusSize;
    }

    public int getPqBusSize() {
        return pqBusSize;
    }

    public String toString() {
        StringWriter w = new StringWriter();
        new IcfWriter(this).write(w);
        return w.toString();
    }

    @SuppressWarnings({"CloneDoesntDeclareCloneNotSupportedException", "CloneDoesntCallSuperClone"})
    public IEEEDataIsland clone() {
        return (IEEEDataIsland) JOFileUtil.cloneObj(this);
    }
}