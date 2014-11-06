package zju.se;

import org.apache.log4j.Logger;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.AVector;
import zju.measure.MeasTypeCons;
import zju.measure.MeasureInfo;
import zju.measure.SystemMeasure;
import zju.util.StateCalByPolar;
import zju.util.YMatrixGetter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * author: wangbin
 * date: 2010-8-16
 */
public class SeResultFiller implements MeasTypeCons {
    private static Logger log = Logger.getLogger(SeResultFiller.class);

    private double baseMVA;
    private IEEEDataIsland island;
    private YMatrixGetter Y;
    private SystemMeasure sm;
    private int primaryMeasureNum;

    public SeResultFiller(IEEEDataIsland island, YMatrixGetter y, SystemMeasure sm) {
        this.island = island;
        this.Y = y;
        this.sm = sm;
        baseMVA = island.getTitle().getMvaBase();
    }

    /**
     * Fill island and primary measurements.
     * @param vTheta voltage amplitude and angle of every bus
     * @param seResult state estimation result
     */
    public void fillSeResult(AVector vTheta, SeResultInfo seResult) {
        fillSeIsland(vTheta);
        fillPrimaryMeasureValue();
        seResult.setAnalogNum(primaryMeasureNum);
        seResult.setEligibleRate(SeStatistics.calEligibleRate(island, sm)[0]);
    }

    public void fillSeIsland(AVector vTheta) {
        int n = island.getBuses().size();
        for (BusData bus : island.getBuses()) {
            bus.setLoadMW(0);
            bus.setLoadMVAR(0);
            bus.setGenerationMW(0);
            bus.setGenerationMVAR(0);
            //bus.setFinalVoltage(vTheta.getValue(bus.getBusNumber() - 1));
            //bus.setFinalAngle((vTheta.getValue(bus.getBusNumber() - 1 + n)) * 180 / Math.PI);
            boolean hasPMeasure = false;
            if (sm.getBus_p().containsKey(String.valueOf(bus.getBusNumber()))) {
                MeasureInfo busP = sm.getBus_p().get(String.valueOf(bus.getBusNumber()));
                String id = "bus" + busP.getPositionId();
                if (!sm.getId2Measure().containsKey(id))
                    continue;
                for (MeasureInfo o : sm.getId2Measure().get(id)) {
                    if (o.getPowerType() == MeasTypeCons.POWER_LOAD) {
                        bus.setLoadMW(bus.getLoadMW() + o.getValue_est());
                    } else if (o.getPowerType() == MeasTypeCons.POWER_GEN)
                        bus.setGenerationMW(bus.getGenerationMW() + o.getValue_est());
                }
                hasPMeasure = true;
            }
            if (!hasPMeasure)
                bus.setLoadMW(-StateCalByPolar.calBusP(bus.getBusNumber(), Y, vTheta) * baseMVA);
            boolean hasQMeasure = false;
            if (sm.getBus_q().containsKey(String.valueOf(bus.getBusNumber()))) {
                MeasureInfo busQ = sm.getBus_q().get(String.valueOf(bus.getBusNumber()));
                String id = "bus" + busQ.getPositionId();
                if (!sm.getId2Measure().containsKey(id))
                    continue;
                for (MeasureInfo o : sm.getId2Measure().get(id)) {
                    if (o.getPowerType() == MeasTypeCons.POWER_LOAD) {
                        bus.setLoadMVAR(bus.getLoadMVAR() + o.getValue_est());
                    } else if (o.getPowerType() == MeasTypeCons.POWER_GEN)
                        bus.setGenerationMVAR(bus.getGenerationMVAR() + o.getValue_est());
                }
                hasQMeasure = true;
            }
            if (!hasQMeasure)
                bus.setLoadMVAR(-StateCalByPolar.calBusQ(bus.getBusNumber(), Y, vTheta) * baseMVA);
        }

        //compensator
        for (BusData bus : island.getBuses()) {
            String id = "bus" + bus.getBusNumber();
            if (!sm.getId2Measure().containsKey(id))
                continue;
            for (MeasureInfo info : sm.getId2Measure().get(id)) {
                if (info.getPowerType() != POWER_COMPENSATOR)
                    continue;
                double v = bus.getFinalVoltage();
                info.setValue_est(-v * v * info.getGenMVA()); // as load
            }
        }
    }

    public void fillPrimaryMeasureValue() {
        primaryMeasureNum = 0;
        List<MeasureInfo> bus_p_measure = new ArrayList<MeasureInfo>();
        List<MeasureInfo> bus_q_measure = new ArrayList<MeasureInfo>();
        for (BusData bus : island.getBuses()) {
            String id = "bus" + bus.getBusNumber();
            if (!sm.getId2Measure().containsKey(id))
                continue;

            String id2 = String.valueOf(bus.getBusNumber());
            double baseV = bus.getBaseVoltage();
            bus_p_measure.clear();
            bus_q_measure.clear();
            for (MeasureInfo obj : sm.getId2Measure().get(id)) {
                if(obj.getMeasureType() == -1)
                    continue;
                switch (obj.getMeasureType()) {
                    case TYPE_BUS_ACTIVE_POWER:
                        bus_p_measure.add(obj);
                        break;
                    case TYPE_BUS_REACTIVE_POWER:
                        bus_q_measure.add(obj);
                        break;
                    case TYPE_BUS_VOLOTAGE:
                        if (sm.getBus_v().containsKey(id2)) {
                            obj.setValue_est(sm.getBus_v().get(id2).getValue_est() * baseV); //todo:phase voltage
                            primaryMeasureNum++;
                        }
                        break;
                    case TYPE_BUS_ANGLE:
                        if (sm.getBus_a().containsKey(id2)) {
                            obj.setValue_est(sm.getBus_a().get(id2).getValue_est() * 180 / Math.PI);
                            primaryMeasureNum++;
                        }
                        break;
                    default:
                        log.warn("This type has not been supported yet, type = " + obj.getMeasureType());
                        break;
                }
            }
            if (sm.getBus_p().containsKey(id2)) {
                setPrimaryOfBusPQ(bus_p_measure, sm.getBus_p().get(id2), baseV);
                primaryMeasureNum += bus_p_measure.size();
            }
            if (sm.getBus_q().containsKey(id2)) {
                setPrimaryOfBusPQ(bus_q_measure, sm.getBus_q().get(id2), baseV);
                primaryMeasureNum += bus_q_measure.size();
            }
        }


        Map<Integer, BusData> busMap = island.getBusMap();
        for (BranchData branch : island.getBranches()) {
            String id = "branch" + branch.getId();
            if (!sm.getId2Measure().containsKey(id))
                continue;
            String id2 = String.valueOf(branch.getId());
            BusData fromBus = busMap.get(branch.getTapBusNumber());
            BusData toBus = busMap.get(branch.getTapBusNumber());
            double base1 = baseMVA / Math.sqrt(3) * 1000.0 / fromBus.getBaseVoltage(); //todo:hard coding: 1000.0
            double base2 = baseMVA / Math.sqrt(3) * 1000.0 / toBus.getBaseVoltage();
            for (MeasureInfo obj : sm.getId2Measure().get(id)) {
                if(obj.getMeasureType() == -1)
                    continue;
                switch (obj.getMeasureType()) {
                    case TYPE_LINE_FROM_ACTIVE:
                    case TYPE_LINE_FROM_REACTIVE:
                    case TYPE_LINE_TO_ACTIVE:
                    case TYPE_LINE_TO_REACTIVE:
                        if (sm.getContainer(obj.getMeasureType()).containsKey(id2)) {
                            obj.setValue_est(sm.getContainer(obj.getMeasureType()).get(id2).getValue_est() * baseMVA);
                            primaryMeasureNum ++;
                        }
                        break;
                    case TYPE_LINE_FROM_CURRENT:
                        if (sm.getContainer(obj.getMeasureType()).containsKey(id2)) {
                            obj.setValue_est(sm.getContainer(obj.getMeasureType()).get(id2).getValue_est() * base1);
                            primaryMeasureNum ++;
                        }
                        break;
                    case TYPE_LINE_TO_CURRENT:
                        if (sm.getContainer(obj.getMeasureType()).containsKey(id2)) {
                            obj.setValue_est(sm.getContainer(obj.getMeasureType()).get(id2).getValue_est() * base2);
                            primaryMeasureNum ++;
                        }
                        break;
                    case TYPE_LINE_FROM_CURRENT_ANGLE:
                    case TYPE_LINE_TO_CURRENT_ANGLE:
                        if (sm.getContainer(obj.getMeasureType()).containsKey(id2)) {
                            obj.setValue_est(sm.getContainer(obj.getMeasureType()).get(id2).getValue_est() * 180 / Math.PI);
                            primaryMeasureNum ++;
                        }
                        break;
                    default:
                        log.warn("This type has not been supported yet, type = " + obj.getMeasureType());
                        break;
                }
            }
        }
    }

    private void setPrimaryOfBusPQ(List<MeasureInfo> physicalMeasurements, MeasureInfo efficientMeasure, double baseV) {
        double value_est = efficientMeasure.getValue_est() * baseMVA;
        double tmp;
        double min_p = 1e-2;
        if (efficientMeasure.getPowerType() == POWER_GEN && baseV < 60) {//todo: hard coding: 60
            if (physicalMeasurements.size() > 1)
                log.info("There are more than one real points of gen power measure");
        }
        if (physicalMeasurements.size() == 1) {
            MeasureInfo o = physicalMeasurements.get(0);
            if (o.getPowerType() == POWER_LOAD)
                tmp = -1.0;
            else
                tmp = 1.0;
            o.setValue_est(tmp * value_est);
            return;
        }
        double value_orig = 0;
        boolean isAllZero = true;
        for (MeasureInfo o : physicalMeasurements) {
            if (o.getPowerType() == POWER_LOAD)
                tmp = -1.0;
            else
                tmp = 1.0;
            value_orig += tmp * o.getValue();
            if (Math.abs(o.getValue()) > min_p)
                isAllZero = false;
        }
        int num = physicalMeasurements.size();
        if (isAllZero) {
            for (MeasureInfo o : physicalMeasurements) {
                if (o.getPowerType() == POWER_LOAD)
                    tmp = -1.0;
                else
                    tmp = 1.0;
                o.setValue_est(tmp * value_est / num);
            }
        } else {
            for (MeasureInfo o : physicalMeasurements) {
                if (Math.abs(value_orig) < min_p && Math.abs(value_est) < num * min_p) {
                    o.setValue_est(0);
                } else if (o.getPowerType() == POWER_GEN) {
                    o.setValue_est(value_est * o.getValue() / value_orig);
                } else if (o.getPowerType() == POWER_LOAD)
                    o.setValue_est(-value_est * o.getValue() / value_orig);
            }
        }
    }
}
