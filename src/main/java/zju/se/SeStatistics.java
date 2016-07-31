package zju.se;

import org.apache.log4j.Logger;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;
import zju.measure.MeasureInfo;
import zju.measure.SystemMeasure;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * This class provides static method to cal statistics of se result.
 * <br>Static variables used in calculation used be set before the class is used.</br>
 * <br>Voltage level less than 60 is not considered now.</br>
 *
 * @author Dong Shufeng
 *         Date: 2008-1-2
 */
public class SeStatistics implements MeasTypeCons {
    private static Logger log = Logger.getLogger(SeStatistics.class);

    public static double tol_voltage = 0.02;
    public static double tol_active = 0.02;
    public static double tol_reactive = 0.03;
    public static double tol_vol_angle = 0.5;
    public static double tol_current_angle = 0.5;
    public static double tol_current = 0.02;

    public static double base_mva_500 = 1082;
    public static double base_mva_330 = 686;
    public static double base_mva_220 = 305;
    public static double base_mva_110 = 114;
    public static double base_mva_66 = 69.7;
    public static double base_voltage_500 = 600;
    public static double base_voltage_330 = 396;
    public static double base_voltage_220 = 264;
    public static double base_voltage_110 = 132;
    public static double base_voltage_66 = 79.2;

    /**
     * This method calculate eligible rate of efficient measurements
     *
     * @param vector measurement vector
     * @return eligible rate of efficient measurements
     */
    public static double[] calEligibleRate(MeasVector vector) {
        int eligibleNum = 0;
        int index = 0;
        for (int type : vector.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < vector.getBus_p_pos().length; i++, index++) {
                        if (!vector.getBadMeasId().contains(index)) {
                            double v1 = vector.getZ().getValue(index);
                            double v2 = vector.getZ_estimate().getValue(index);
                            if (Math.abs(v1) < 1e-3 && Math.abs(v2) < 1e-3)
                                eligibleNum++;
                            else if (Math.abs((v1 - v2) / v1) <= tol_active)
                                eligibleNum++;
                            else
                                log.debug("bus mw " + vector.getBus_p_pos()[i] + " measure value:" + v1 + " estimate value:" + v2 + " error:" + (v1 - v2));
                        }
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int i = 0; i < vector.getLine_from_p_pos().length; i++, index++) {
                        if (!vector.getBadMeasId().contains(index)) {
                            double v1 = vector.getZ().getValue(index);
                            double v2 = vector.getZ_estimate().getValue(index);
                            if (Math.abs(v1) < 1e-3 && Math.abs(v2) < 1e-3)
                                eligibleNum++;
                            else if (Math.abs((v1 - v2) / v1) <= tol_active)
                                eligibleNum++;
                            else
                                log.debug("line from mw " + vector.getLine_from_p_pos()[i] + " measure value:" + v1 + " estimate value:" + v2 + " error:" + (v1 - v2));
                        }
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int i = 0; i < vector.getLine_to_p_pos().length; i++, index++) {
                        if (!vector.getBadMeasId().contains(index)) {
                            double v1 = vector.getZ().getValue(index);
                            double v2 = vector.getZ_estimate().getValue(index);
                            if (Math.abs(v1) < 1e-3 && Math.abs(v2) < 1e-3)
                                eligibleNum++;
                            else if (Math.abs((v1 - v2) / v1) <= tol_active)
                                eligibleNum++;
                            else
                                log.debug("line to mw " + vector.getLine_to_p_pos()[i] + " measure value:" + v1 + " estimate value:" + v2 + " error:" + (v1 - v2));
                        }
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < vector.getBus_q_pos().length; i++, index++) {
                        if (!vector.getBadMeasId().contains(index)) {
                            double v1 = vector.getZ().getValue(index);
                            double v2 = vector.getZ_estimate().getValue(index);
                            if (Math.abs(v1) < 1e-3 && Math.abs(v2) < 1e-3)
                                eligibleNum++;
                            else if (Math.abs((v1 - v2) / v1) <= tol_reactive)
                                eligibleNum++;
                            else
                                log.debug("bus mvar " + vector.getBus_q_pos()[i] + " measure value:" + v1 + " estimate value:" + v2 + " error:" + (v1 - v2));
                        }
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int i = 0; i < vector.getLine_from_q_pos().length; i++, index++) {
                        if (!vector.getBadMeasId().contains(index)) {
                            double v1 = vector.getZ().getValue(index);
                            double v2 = vector.getZ_estimate().getValue(index);
                            if (Math.abs(v1) < 1e-3 && Math.abs(v2) < 1e-3)
                                eligibleNum++;
                            else if (Math.abs((v1 - v2) / v1) <= tol_reactive)
                                eligibleNum++;
                            else
                                log.debug("line from mvar " + vector.getLine_from_q_pos()[i] + " measure value:" + v1 + " estimate value:" + v2 + " error:" + (v1 - v2));
                        }
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int i = 0; i < vector.getLine_to_q_pos().length; i++, index++) {
                        if (!vector.getBadMeasId().contains(index)) {
                            double v1 = vector.getZ().getValue(index);
                            double v2 = vector.getZ_estimate().getValue(index);
                            if (Math.abs(v1) < 1e-3 && Math.abs(v2) < 1e-3)
                                eligibleNum++;
                            else if (Math.abs((v1 - v2) / v1) <= tol_reactive)
                                eligibleNum++;
                            else
                                log.debug("line to mvar " + vector.getLine_to_q_pos()[i] + " measure value:" + v1 + " estimate value:" + v2 + " error:" + (v1 - v2));
                        }
                    }
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < vector.getBus_v_pos().length; i++, index++) {
                        if (!vector.getBadMeasId().contains(index)) {
                            double v1 = vector.getZ().getValue(index);
                            double v2 = vector.getZ_estimate().getValue(index);
                            if (Math.abs(v1) < 1e-3 && Math.abs(v2) < 1e-3)
                                eligibleNum++;
                            else if (Math.abs((v1 - v2) / v1) <= tol_voltage)
                                eligibleNum++;
                            else
                                log.debug("bus v " + vector.getBus_v_pos()[i] + " measure value:" + v1 + " estimate value:" + v2 + " error:" + (v1 - v2));
                        }
                    }
                    break;
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < vector.getBus_a_pos().length; i++, index++) {
                        if (!vector.getBadMeasId().contains(index)) {
                            double v1 = vector.getZ().getValue(index);
                            double v2 = vector.getZ_estimate().getValue(index);
                            if (Math.abs(v1) < 1e-3 && Math.abs(v2) < 1e-3)
                                eligibleNum++;
                            else if (Math.abs((v1 - v2) / Math.PI * 180) <= tol_vol_angle)
                                eligibleNum++;
                            else
                                log.debug("bus v " + vector.getBus_a_pos()[i] + " measure value:" + v1 + " estimate value:" + v2 + "error:" + (v1 - v2));
                        }
                    }
                    break;
            }
        }
        int totalNum = vector.getZ().getN();
        double rate = ((double) eligibleNum) / ((double) (totalNum));
        log.info("Total measurement num: " + totalNum + " eligible measurement num: " + eligibleNum + "eligible rate " + rate);
        return new double[]{rate, eligibleNum};
    }


    /**
     * This method calculate eligible rate of efficient measurements witch no pmu measurement type.
     *
     * @param island ieee format data
     * @param sm     system measurements
     * @return eligible rate and eligible number
     */
    public static double[] calEligibleRate(IEEEDataIsland island, SystemMeasure sm) {
        return calEligibleRate(island, sm, false);
    }

    /**
     * This method calculate eligible rate of efficient measurements
     *
     * @param island        ieee format data
     * @param sm            system measurements
     * @param isPmuInvolved whether pmu measurements is involved
     * @return eligible rate and eligible number
     */
    public static double[] calEligibleRate(IEEEDataIsland island, SystemMeasure sm, boolean isPmuInvolved) {
        int eligibleNum = 0;
        int totalNum;
        double baseMVA = island.getTitle().getMvaBase();
        Map<Integer, BusData> buses = island.getBusMap();
        int num = 0;
        for (MeasureInfo info : sm.getBus_v().values()) {
            double baseV = buses.get(info.getPositionIdOfIntFormat()).getBaseVoltage();
            double base = getBaseVol4Cal(baseV);
            boolean isEligible = isEligible(info, base, tol_voltage);
            info.setEligible(isEligible);
            if (isEligible)
                eligibleNum++;
            else
                log.debug(buses.get(info.getPositionIdOfIntFormat()).getName() + " " + info.getValue_est() + " " + info.getValue());
        }
        log.info("电压量测, 合格个数: " + (eligibleNum - num) + ",总个数: " + sm.getBus_v().size());
        num = eligibleNum;
        for (MeasureInfo info : sm.getBus_p().values()) {
            BusData bus = buses.get(info.getPositionIdOfIntFormat());
            double base = getBusPBase(bus, sm, info, baseMVA);
            boolean isEligible = isEligible(info, base, tol_active);
            info.setEligible(isEligible);
            if (isEligible)
                eligibleNum++;
            else
                log.debug(buses.get(info.getPositionIdOfIntFormat()).getName() + " " + info.getValue_est() + " " + info.getValue());
        }
        log.info("节点有功, 合格个数: " + (eligibleNum - num) + ",总个数: " + sm.getBus_p().size());
        num = eligibleNum;
        for (MeasureInfo info : sm.getBus_q().values()) {
            BusData bus = buses.get(info.getPositionIdOfIntFormat());
            double base = getBusQBase(bus, sm, info, baseMVA);
            boolean isEligible = isEligible(info, base, tol_reactive);
            info.setEligible(isEligible);
            if (isEligible)
                eligibleNum++;
            else
                log.debug(buses.get(info.getPositionIdOfIntFormat()).getName() + " " + info.getValue_est() + " " + info.getValue());
        }
        log.info("节点无功, 合格个数: " + (eligibleNum - num) + ",总个数: " + sm.getBus_q().size());
        num = eligibleNum;
        eligibleNum += getEligibleNum(sm.getLine_from_p().values(), island.getId2branch(), buses, baseMVA, tol_active);
        eligibleNum += getEligibleNum(sm.getLine_to_p().values(), island.getId2branch(), buses, baseMVA, tol_active);
        int size = sm.getLine_from_p().size() + sm.getLine_to_p().size();
        log.info("线路有功, 合格个数: " + (eligibleNum - num) + ",总个数: " + size);
        num = eligibleNum;
        eligibleNum += getEligibleNum(sm.getLine_from_q().values(), island.getId2branch(), buses, baseMVA, tol_reactive);
        eligibleNum += getEligibleNum(sm.getLine_to_q().values(), island.getId2branch(), buses, baseMVA, tol_reactive);
        size = sm.getLine_from_q().size() + sm.getLine_to_q().size();
        log.info("线路无功, 合格个数: " + (eligibleNum - num) + ",总个数: " + size);
        totalNum = sm.getBus_v().size() + sm.getBus_p().size() + sm.getBus_q().size()
                + sm.getLine_from_p().size() + sm.getLine_from_q().size()
                + sm.getLine_to_p().size() + sm.getLine_to_q().size();
        if (isPmuInvolved) {
            num = eligibleNum;
            eligibleNum += getEligibleNumOfAngle(sm.getBus_a().values(), tol_vol_angle);
            log.info("节点相角, 合格个数: " + (eligibleNum - num) + ",总个数: " + sm.getBus_a().size());
            num = eligibleNum;
            eligibleNum += getEligibleNumOfCurrent(sm.getLine_from_i_amp().values(), island.getId2branch(), buses, baseMVA, tol_current);
            eligibleNum += getEligibleNumOfCurrent(sm.getLine_to_i_amp().values(), island.getId2branch(), buses, baseMVA, tol_current);
            size = sm.getLine_from_i_amp().size() + sm.getLine_to_i_amp().size();
            log.info("线路电流, 合格个数: " + (eligibleNum - num) + ",总个数: " + size);
            num = eligibleNum;
            eligibleNum += getEligibleNumOfAngle(sm.getLine_from_i_a().values(), tol_current_angle);
            eligibleNum += getEligibleNumOfAngle(sm.getLine_to_i_a().values(), tol_current_angle);
            size = sm.getLine_from_i_a().size() + sm.getLine_to_i_a().size();
            log.info("线路电流相角, 合格个数: " + (eligibleNum - num) + ",总个数: " + size);
            totalNum += sm.getBus_a().size() + sm.getLine_from_i_a().size() +
                    sm.getLine_from_i_amp().size() + sm.getLine_to_i_a().size() + sm.getLine_to_i_amp().size();
        }
        double rate = ((double) eligibleNum) / ((double) (totalNum));
        log.info("量测总数: " + totalNum + ",合格个数: " + eligibleNum + ",合格率: " + rate);
        return new double[]{((double) eligibleNum) / ((double) (totalNum)), eligibleNum};
    }

    public static double getBusPBase(BusData bus, SystemMeasure sm, MeasureInfo info, double baseMVA) {
        double baseV = bus.getBaseVoltage();
        double base;
        if (bus.getType() == 0 || baseV > 60)//todo:
            base = getBaseMVA4Cal(baseV, baseMVA);
        else
            base = getGenPBase(sm, info);
        return base;
    }

    public static double getGenPBase(SystemMeasure sm, MeasureInfo info) {
        double base;
        if (info.getGenMVA() > 0)
            return info.getGenMVA();
        MeasureInfo qMeas = sm.getEfficientMeasure(TYPE_BUS_REACTIVE_POWER, info.getPositionId());
        if (qMeas != null) //todo:
            base = Math.sqrt(qMeas.getValue() * qMeas.getValue() + info.getValue() * info.getValue());
        else {
            log.info("There is no reactive power measurement at gen");
            base = Math.abs(info.getValue() * 0.05);
        }
        return base;
    }

    public static double getBusQBase(BusData bus, SystemMeasure sm, MeasureInfo info, double baseMVA) {
        double baseV = bus.getBaseVoltage();
        double base;
        if (bus.getType() == 0 || baseV > 60)//todo:
            base = getBaseMVA4Cal(baseV, baseMVA);
        else {
            base = getGenQBase(sm, info);
        }
        return base;
    }

    public static double getGenQBase(SystemMeasure sm, MeasureInfo info) {
        double base;
        if (info.getGenMVA() > 0)
            return info.getGenMVA();
        MeasureInfo pMeas = sm.getEfficientMeasure(TYPE_BUS_ACTIVE_POWER, info.getPositionId());
        if (pMeas != null)//todo:
            base = Math.sqrt(pMeas.getValue() * pMeas.getValue() + info.getValue() * info.getValue());
        else {
            log.info("There is no active power measurement at gen");
            base = Math.abs(info.getValue() * 0.1);
        }
        return base;
    }

    public static double getBaseVol4Cal(double baseV) {
        double base;
        if (baseV > 600) {
            log.info("base voltage is higher than 600");
            base = 1.1;
        } else if (baseV >= 490)
            base = base_voltage_500 / baseV;
        else if (baseV >= 300)
            base = base_voltage_330 / baseV;
        else if (baseV >= 200)
            base = base_voltage_220 / baseV;
        else if (baseV >= 100)
            base = base_voltage_110 / baseV;
        else if (baseV >= 60)
            base = base_voltage_66 / baseV;
        else
            base = 1.1;
        return base;
    }

    public static double getBaseMVA4Cal(double baseV, double baseMVA) {
        double base;
        if (baseV >= 490)
            base = base_mva_500 / baseMVA;
        else if (baseV >= 300)
            base = base_mva_330 / baseMVA;
        else if (baseV >= 200)
            base = base_mva_220 / baseMVA;
        else if (baseV >= 100)
            base = base_mva_110 / baseMVA;
        else
            base = base_mva_66 / baseMVA;//todo: this may not right
        //if (baseV < 60)
        //    log.warn("line base voltage is lower than 60kv");
        return base;
    }

    public static int getEligibleNum(Collection<MeasureInfo> meases, Map<Integer, BranchData> branches, Map<Integer, BusData> buses, double baseMVA, double tolerance) {
        int eligibleNum = 0;
        for (MeasureInfo info : meases) {
            BranchData branch = branches.get(info.getPositionIdOfIntFormat());
            double baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getTapBusNumber()).getBaseVoltage());
            double base = getBaseMVA4Cal(baseV, baseMVA);
            boolean isEligible = isEligible(info, base, tolerance);
            info.setEligible(isEligible);
            if (isEligible)
                eligibleNum++;
        }
        return eligibleNum;
    }

    public static int getEligibleNumOfCurrent(Collection<MeasureInfo> meases, Map<Integer, BranchData> branches, Map<Integer, BusData> buses, double baseMVA, double tolerance) {
        int eligibleNum = 0;
        for (MeasureInfo info : meases) {
            BranchData branch = branches.get(info.getPositionIdOfIntFormat());
            double baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getTapBusNumber()).getBaseVoltage());
            double baseV4Cal = getBaseVol4Cal(baseV);
            double baseMVA4Cal = getBaseMVA4Cal(baseV, baseMVA);
            boolean isEligible = isEligible(info, baseMVA4Cal / baseV4Cal, tolerance); //todo:this may be not right
            //boolean isEligible = isEligible(info, tolerance);
            info.setEligible(isEligible);
            if (isEligible)
                eligibleNum++;
        }
        return eligibleNum;
    }

    private static int getEligibleNum(Collection<MeasureInfo> meases, double tolerance) {
        int eligibleNum = 0;
        for (MeasureInfo info : meases) {
            boolean isEligible = isEligible(info, tolerance);
            info.setEligible(isEligible);
            if (isEligible)
                eligibleNum++;
        }
        return eligibleNum;
    }

    private static int getEligibleNumOfAngle(Collection<MeasureInfo> meases, double tolerance) {
        int eligibleNum = 0;
        for (MeasureInfo info : meases) {
            boolean isEligible = isEligible(info, Math.PI / 180, tolerance);
            info.setEligible(isEligible);
            if (isEligible)
                eligibleNum++;
        }
        return eligibleNum;
    }

    public static boolean isEligible(MeasureInfo info, double base, double tolerance) {
        double v1 = info.getValue();
        double v2 = info.getValue_est();
        return Math.abs((v1 - v2) / base) <= tolerance;
    }

    /**
     * this method cal eligible rate when the every measurement is checked.
     *
     * @param sm measurements
     * @return eligible rate and eligible number
     */
    public static double[] calEligibleRate(SystemMeasure sm) {
        int eligibleNum = 0;
        int totalNum = 0;
        eligibleNum += getEligibleNumOfAngle(sm.getBus_a().values(), tol_vol_angle);
        totalNum += sm.getBus_a().size();
        eligibleNum += getEligibleNum(sm.getBus_v().values(), tol_voltage);
        totalNum += sm.getBus_v().size();
        eligibleNum += getEligibleNum(sm.getBus_p().values(), tol_active);
        totalNum += sm.getBus_p().size();
        eligibleNum += getEligibleNum(sm.getBus_q().values(), tol_reactive);
        totalNum += sm.getBus_q().size();
        eligibleNum += getEligibleNum(sm.getLine_from_p().values(), tol_active);
        totalNum += sm.getLine_from_p().size();
        eligibleNum += getEligibleNum(sm.getLine_from_q().values(), tol_reactive);
        totalNum += sm.getLine_from_q().size();
        eligibleNum += getEligibleNum(sm.getLine_to_p().values(), tol_active);
        totalNum += sm.getLine_to_p().size();
        eligibleNum += getEligibleNum(sm.getLine_to_q().values(), tol_reactive);
        totalNum += sm.getLine_to_q().size();
        log.info("Total measurement num: " + totalNum);
        log.info("Eligible measurement num: " + eligibleNum);
        return new double[]{((double) eligibleNum) / ((double) (totalNum)), eligibleNum};
    }

    private static boolean isEligible(MeasureInfo info, double tolerance) {
        double v1 = info.getValue();
        double v2 = info.getValue_est();
        return (Math.abs(v1) < 10e-3 && Math.abs(v2) < 10e-3) || Math.abs((v1 - v2) / v1) <= tolerance;
    }

    public static double[] getMaxAllowResidual(IEEEDataIsland island, SystemMeasure sm, MeasVector meas) {
        double[] a = new double[meas.getZ().getN()];
        int index = 0;
        Map<Integer, BusData> buses = new HashMap<Integer, BusData>();
        for (BusData bus : island.getBuses())
            buses.put(bus.getBusNumber(), bus);
        double baseMVA = island.getTitle().getMvaBase();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++) {
                        a[index] = tol_vol_angle / 180 * Math.PI;
                    }
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        int busNum = meas.getBus_v_pos()[i];
                        MeasureInfo info = sm.getEfficientMeasure(TYPE_BUS_VOLOTAGE, String.valueOf(busNum));
                        double baseV;
                        if (info == null) {
                            baseV = buses.get(busNum).getBaseVoltage();
                        } else {
                            baseV = buses.get(Integer.parseInt(info.getPositionId())).getBaseVoltage();
                        }
                        double base = getBaseVol4Cal(baseV);
                        a[index] = base * tol_voltage;
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        int busNum = meas.getBus_p_pos()[i];
                        MeasureInfo info = sm.getEfficientMeasure(TYPE_BUS_ACTIVE_POWER, String.valueOf(busNum));
                        double base = getBusPBase(buses.get(busNum), sm, info, baseMVA);
                        a[index] = base * tol_active;
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        int busNum = meas.getBus_q_pos()[i];
                        MeasureInfo info = sm.getEfficientMeasure(TYPE_BUS_REACTIVE_POWER, String.valueOf(busNum));
                        double base = getBusQBase(buses.get(busNum), sm, info, baseMVA);
                        a[index] = base * tol_reactive;
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int i = 0; i < meas.getLine_from_p_pos().length; i++, index++) {
                        int branchId = meas.getLine_from_p_pos()[i];
                        MeasureInfo info = sm.getEfficientMeasure(TYPE_LINE_FROM_ACTIVE, String.valueOf(branchId));
                        BranchData branch = island.getId2branch().get(Integer.parseInt(info.getPositionId()));
                        double baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getTapBusNumber()).getBaseVoltage());
                        double base = getBaseMVA4Cal(baseV, baseMVA);
                        a[index] = base * tol_active;
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int i = 0; i < meas.getLine_to_p_pos().length; i++, index++) {
                        int branchId = meas.getLine_to_p_pos()[i];
                        MeasureInfo info = sm.getEfficientMeasure(TYPE_LINE_TO_ACTIVE, String.valueOf(branchId));
                        BranchData branch = island.getId2branch().get(Integer.parseInt(info.getPositionId()));
                        double baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getTapBusNumber()).getBaseVoltage());
                        double base = getBaseMVA4Cal(baseV, baseMVA);
                        a[index] = base * tol_active;
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int i = 0; i < meas.getLine_from_q_pos().length; i++, index++) {
                        int branchId = meas.getLine_from_q_pos()[i];
                        MeasureInfo info = sm.getEfficientMeasure(TYPE_LINE_FROM_REACTIVE, String.valueOf(branchId));
                        BranchData branch = island.getId2branch().get(Integer.parseInt(info.getPositionId()));
                        double baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getTapBusNumber()).getBaseVoltage());
                        double base = getBaseMVA4Cal(baseV, baseMVA);
                        a[index] = base * tol_reactive;
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int i = 0; i < meas.getLine_to_q_pos().length; i++, index++) {
                        int branchId = meas.getLine_to_q_pos()[i];
                        MeasureInfo info = sm.getEfficientMeasure(TYPE_LINE_TO_REACTIVE, String.valueOf(branchId));
                        BranchData branch = island.getId2branch().get(Integer.parseInt(info.getPositionId()));
                        double baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getTapBusNumber()).getBaseVoltage());
                        double base = getBaseMVA4Cal(baseV, baseMVA);
                        a[index] = base * tol_reactive;
                    }
                    break;
                case TYPE_LINE_FROM_CURRENT:
                    for (int i = 0; i < meas.getLine_from_i_amp_pos().length; i++, index++) {
                        int branchId = meas.getLine_from_i_amp_pos()[i];
                        MeasureInfo info = sm.getEfficientMeasure(TYPE_LINE_FROM_CURRENT, String.valueOf(branchId));
                        BranchData branch = island.getId2branch().get(Integer.parseInt(info.getPositionId()));
                        double baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getTapBusNumber()).getBaseVoltage());
                        double baseV4Cal = getBaseVol4Cal(baseV);
                        double baseMVA4Cal = getBaseMVA4Cal(baseV, baseMVA);
                        double base = baseMVA4Cal / baseV4Cal;
                        a[index] = base * tol_current;
                    }
                    break;
                case TYPE_LINE_FROM_CURRENT_ANGLE:
                    for (int i = 0; i < meas.line_from_i_a_pos.length; i++, index++) {
                        a[index] = tol_vol_angle / 180 * Math.PI;
                    }
                    break;
                case TYPE_LINE_TO_CURRENT:
                    for (int i = 0; i < meas.getLine_to_i_amp_pos().length; i++, index++) {
                        int branchId = meas.getLine_to_i_amp_pos()[i];
                        MeasureInfo info = sm.getEfficientMeasure(TYPE_LINE_TO_CURRENT, String.valueOf(branchId));
                        BranchData branch = island.getId2branch().get(Integer.parseInt(info.getPositionId()));
                        double baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getTapBusNumber()).getBaseVoltage());
                        double baseV4Cal = getBaseVol4Cal(baseV);
                        double baseMVA4Cal = getBaseMVA4Cal(baseV, baseMVA);
                        double base = baseMVA4Cal / baseV4Cal;
                        a[index] = base * tol_current;
                    }
                    break;
                case TYPE_LINE_TO_CURRENT_ANGLE:
                    for (int i = 0; i < meas.getLine_to_i_a_pos().length; i++, index++) {
                        a[index] = tol_vol_angle / 180 * Math.PI;
                    }
                    break;
                default:
                    break;
            }
        }
        for (int i = 0; i < a.length; i++) {
            if (a[i] < 10e-4)
                a[i] = 0.01;
        }
        return a;
    }

    public static double[] getMaxAllowResidualOfBranch(MeasureInfo[] meas, double baseV, double baseMVA) {
        double[] a = new double[meas.length];
        int index = 0;
        double base;
        for (MeasureInfo info : meas) {
            int type = info.getMeasureType();
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    base = getBaseVol4Cal(baseV);
                    a[index++] = base * tol_voltage;
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    base = getBaseMVA4Cal(baseV, baseMVA);
                    a[index++] = base * tol_active;
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    base = getBaseMVA4Cal(baseV, baseMVA);
                    a[index++] = base * tol_active;
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    base = getBaseMVA4Cal(baseV, baseMVA);
                    a[index++] = base * tol_reactive;
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    base = getBaseMVA4Cal(baseV, baseMVA);
                    a[index++] = base * tol_reactive;
                    break;
                default:
                    return null;
            }
        }
        return a;
    }

    public static double[] getMaxAllowResidual(IEEEDataIsland island, MeasureInfo[] meas) {
        double[] a = new double[meas.length];
        int index = 0;
        Map<Integer, BusData> buses = new HashMap<Integer, BusData>();
        for (BusData bus : island.getBuses())
            buses.put(bus.getBusNumber(), bus);
        double baseMVA = island.getTitle().getMvaBase();
        double base;
        for (MeasureInfo info : meas) {
            int type = info.getMeasureType();
            switch (type) {
                case TYPE_BUS_ANGLE:
                    log.warn("Angle measurement has not been dealt yet.");
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    a[index++] = Math.abs(info.getValue()) * 0.05; //todo:
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    a[index++] = Math.abs(info.getValue()) * 0.1; //todo:
                    break;
                case TYPE_BUS_VOLOTAGE:
                    double baseV = buses.get(Integer.parseInt(info.getPositionId())).getBaseVoltage();
                    base = getBaseVol4Cal(baseV);
                    a[index++] = base * tol_voltage;
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    BranchData branch = island.getId2branch().get(Integer.parseInt(info.getPositionId()));
                    baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getZBusNumber()).getBaseVoltage());
                    base = getBaseMVA4Cal(baseV, baseMVA);
                    a[index++] = base * tol_active;
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    branch = island.getId2branch().get(Integer.parseInt(info.getPositionId()));
                    baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getZBusNumber()).getBaseVoltage());
                    base = getBaseMVA4Cal(baseV, baseMVA);
                    a[index++] = base * tol_active;
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    branch = island.getId2branch().get(Integer.parseInt(info.getPositionId()));
                    baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getZBusNumber()).getBaseVoltage());
                    base = getBaseMVA4Cal(baseV, baseMVA);
                    a[index++] = base * tol_reactive;
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    branch = island.getId2branch().get(Integer.parseInt(info.getPositionId()));
                    baseV = Math.max(buses.get(branch.getTapBusNumber()).getBaseVoltage(), buses.get(branch.getZBusNumber()).getBaseVoltage());
                    base = getBaseMVA4Cal(baseV, baseMVA);
                    a[index++] = base * tol_reactive;
                    break;
                default:
                    return null;
            }
        }
        for (int i = 0; i < a.length; i++) {
            if (a[i] < 10e-4)
                a[i] = 0.01;
        }
        return a;
    }
}
