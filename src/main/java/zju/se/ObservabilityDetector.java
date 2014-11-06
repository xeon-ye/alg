package zju.se;

import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink;
import zju.measure.MeasureInfo;
import zju.measure.SystemMeasure;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-15
 *         Time: 21:33:58
 */
public class ObservabilityDetector {
    public static final int OBSERVABLE = 0;
    public static final int PTHEATA_OBSERVABLE = 1;
    public static final int UNOBSERVABLE = 2;

    private int observability = OBSERVABLE;
    private boolean hasVoltageMeasure = false;
    private List<Integer> busNeedPMeasure = new ArrayList<Integer>();
    private List<Integer> busNeedQMeasure = new ArrayList<Integer>();
    private IEEEDataIsland island;
    private SystemMeasure meas;
    private ASparseMatrixLink weight;

    public ObservabilityDetector() {
    }

    public ObservabilityDetector(IEEEDataIsland island, SystemMeasure meas) {
        this.setIsland(island);
        this.setMeas(meas);
    }

    public IEEEDataIsland getIsland() {
        return island;
    }

    public void setIsland(IEEEDataIsland island) {
        this.island = island;
        weight = new ASparseMatrixLink(island.getBuses().size());
        for (BranchData branch : island.getBranches()) {
            int index1 = branch.getTapBusNumber() - 1;
            int index2 = branch.getZBusNumber() - 1;
            weight.increase(Math.min(index1, index2), Math.max(index1, index2), 1);
            weight.increase(Math.max(index1, index2), Math.min(index1, index2), 1);
        }
    }

    public SystemMeasure getMeas() {
        return meas;
    }

    public void setMeas(SystemMeasure meas) {
        this.meas = meas;
    }

    public int analysisObservability() {
        busNeedPMeasure.clear();
        busNeedQMeasure.clear();
        observability = OBSERVABLE;
        if (meas.getBus_v() == null || meas.getBus_v().size() < 1) {
            hasVoltageMeasure = false;
            observability = UNOBSERVABLE;
        } else
            hasVoltageMeasure = true;
        List<Integer> pCoveredBus = new ArrayList<Integer>();
        int np;
        for (MeasureInfo d : meas.getBus_p().values()) {
            int bus = Integer.parseInt(d.getPositionId());
            if (!pCoveredBus.contains(bus))
                pCoveredBus.add(bus);
            int k = weight.getIA()[bus - 1];
            while (k != -1) {
                int j = weight.getJA().get(k);
                if (!pCoveredBus.contains(j + 1))
                    pCoveredBus.add(j + 1);
                k = weight.getLINK().get(k);
            }
        }
        np = meas.getBus_p().size();
        for (MeasureInfo d : meas.getLine_from_p().values()) {
            int line = Integer.parseInt(d.getPositionId());
            int head = island.getId2branch().get(line).getTapBusNumber();
            int tail = island.getId2branch().get(line).getZBusNumber();
            if (!pCoveredBus.contains(head))
                pCoveredBus.add(head);
            if (!pCoveredBus.contains(tail))
                pCoveredBus.add(tail);
        }
        np += meas.getLine_from_p().size();

        for (MeasureInfo d : meas.getLine_to_p().values()) {
            int line = Integer.parseInt(d.getPositionId());
            int head = island.getId2branch().get(line).getTapBusNumber();
            int tail = island.getId2branch().get(line).getZBusNumber();
            if (!pCoveredBus.contains(head))
                pCoveredBus.add(head);
            if (!pCoveredBus.contains(tail))
                pCoveredBus.add(tail);
        }
        np += meas.getLine_to_p().size();

        if (pCoveredBus.size() < island.getBuses().size() || np < island.getBuses().size() - 1) {
            observability = UNOBSERVABLE;
            for (BusData bus : island.getBuses())
                if (!pCoveredBus.contains(bus.getBusNumber()))
                    busNeedPMeasure.add(bus.getBusNumber());
        }

        List<Integer> qCoveredBus = new ArrayList<Integer>();
        int nq;
        int nv = 0;

        for (MeasureInfo d : meas.getBus_v().values()) {
            int bus = Integer.parseInt(d.getPositionId());
            if (!qCoveredBus.contains(bus))
                qCoveredBus.add(bus);
        }
        nv += meas.getBus_v().size();
        for (MeasureInfo d : meas.getBus_q().values()) {
            int bus = Integer.parseInt(d.getPositionId());
            if (!qCoveredBus.contains(bus))
                qCoveredBus.add(bus);
            int k = weight.getIA()[bus - 1];
            while (k != -1) {
                int j = weight.getJA().get(k);
                if (!qCoveredBus.contains(j + 1))
                    qCoveredBus.add(j + 1);
                k = weight.getLINK().get(k);
            }
        }
        nq = meas.getBus_q().size();
        for (MeasureInfo d : meas.getLine_from_q().values()) {
            int line = Integer.parseInt(d.getPositionId());
            int head = island.getId2branch().get(line).getTapBusNumber();
            int tail = island.getId2branch().get(line).getZBusNumber();
            if (!qCoveredBus.contains(head))
                qCoveredBus.add(head);
            if (!qCoveredBus.contains(tail))
                qCoveredBus.add(tail);
        }
        nq += meas.getLine_from_q().size();

        for (MeasureInfo d : meas.getLine_to_q().values()) {
            int line = Integer.parseInt(d.getPositionId());
            int head = island.getId2branch().get(line).getTapBusNumber();
            int tail = island.getId2branch().get(line).getZBusNumber();
            if (!qCoveredBus.contains(head))
                qCoveredBus.add(head);
            if (!qCoveredBus.contains(tail))
                qCoveredBus.add(tail);
        }
        nq += meas.getLine_to_q().size();
        if (qCoveredBus.size() < island.getBuses().size() || (nq + nv) < island.getBuses().size() - 1) {
            if (observability != UNOBSERVABLE)
                observability = PTHEATA_OBSERVABLE;
            for (BusData bus : island.getBuses())
                if (!qCoveredBus.contains(bus.getBusNumber()))
                    busNeedQMeasure.add(bus.getBusNumber());
        }
        return observability;
    }

    public void dealUnobservable() {
        while (true) {
            if (analysisObservability() == OBSERVABLE)
                return;
            List<Integer> toDeal = new ArrayList<Integer>(busNeedPMeasure);
            for (int bus : busNeedQMeasure) {
                if (toDeal.contains(new Integer(bus)))
                    toDeal.remove(new Integer(bus));
                else
                    toDeal.add(bus);
            }
            if (toDeal.size() == 0)
                return;
            for (int bus : toDeal) {
                String key = String.valueOf(bus);
                meas.getBus_a().remove(key);
                meas.getBus_p().remove(key);
                meas.getBus_q().remove(key);
                meas.getBus_v().remove(key);
                for (BranchData branch : getBranches(bus)) {
                    key = String.valueOf(branch.getId());
                    meas.getLine_from_p().remove(key);
                    meas.getLine_from_q().remove(key);
                    meas.getLine_to_p().remove(key);
                    meas.getLine_to_q().remove(key);
                    if (branch.getTapBusNumber() == bus)
                        key = String.valueOf(branch.getZBusNumber());
                    else
                        key = String.valueOf(branch.getTapBusNumber());
                    meas.getBus_p().remove(key);
                    meas.getBus_q().remove(key);
                }
            }
        }
    }

    public List<BranchData> getBranches(int bus) {
        List<BranchData> result = new ArrayList<BranchData>();
        for (BranchData branch : island.getBranches()) {
            if (branch.getTapBusNumber() == bus || branch.getZBusNumber() == bus)
                result.add(branch);
        }
        return result;
    }

    public List<Integer> getUnobservableBuses() {
        List<Integer> result = new ArrayList<Integer>(this.getBusNeedPMeasure());
        for (int i : this.getBusNeedQMeasure())
            if (!result.contains(i))
                result.add(i);
        return result;
    }

    public int getObservability() {
        return observability;
    }

    public void setObservability(int observability) {
        this.observability = observability;
    }

    public boolean isHasVoltageMeasure() {
        return hasVoltageMeasure;
    }

    public void setHasVoltageMeasure(boolean hasVoltageMeasure) {
        this.hasVoltageMeasure = hasVoltageMeasure;
    }

    public List<Integer> getBusNeedPMeasure() {
        return busNeedPMeasure;
    }

    public void setBusNeedPMeasure(List<Integer> busNeedPMeasure) {
        this.busNeedPMeasure = busNeedPMeasure;
    }

    public List<Integer> getBusNeedQMeasure() {
        return busNeedQMeasure;
    }

    public void setBusNeedQMeasure(List<Integer> busNeedQMeasure) {
        this.busNeedQMeasure = busNeedQMeasure;
    }
}
