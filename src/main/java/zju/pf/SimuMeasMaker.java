package zju.pf;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.AVector;
import zju.measure.*;
import zju.util.NumberOptHelper;
import zju.util.RandomMaker;
import zju.util.StateCalByPolar;
import zju.util.YMatrixGetter;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 * Date: 2007-12-12
 */
public class SimuMeasMaker implements MeasTypeCons {

    private static Logger log = LogManager.getLogger(SimuMeasMaker.class);

    private static double min_p_value = 0.0001;

    private static double min_q_value = 0.0001;

    /**
     * @param meas             vector to store result
     * @param admittanceGetter admittance matrix getter
     * @param vtheta           all bus's voltage and angle
     * @param sigmas           parameter of normal distribution
     */
    public static void createNormalMeasure(MeasVector meas, YMatrixGetter admittanceGetter, AVector vtheta, AVector sigmas) {
        //compute measure
        AVector measure = StateCalByPolar.getEstimatedZ(meas, admittanceGetter, vtheta);
        AVector trueZ = new AVector(measure);
        meas.setZ_true(trueZ);
        for (int i = 0; i < measure.getN(); i++)
            measure.setValue(i, measure.getValue(i) + RandomMaker.randomNorm(0, sigmas.getValue(i)));
        meas.setZ(measure);
    }

    public static SystemMeasure createFullMeasure_withBadData(IEEEDataIsland island, int errorDistribution, double ratio, double badRate) {
        SystemMeasure sm = createFullMeasure(island, errorDistribution, ratio);
        addBadData(sm, badRate);
        return sm;
    }

    /**
     * this method using powerflow result as true value and add random noise as measurement value
     *
     * @param island            ieee common format data
     * @param errorDistribution 0: equality 1:Gauss
     * @param ratio             error limit
     * @return every bus has injection p, q  and thresholds1 measurement, every branch has power measurement
     */
    public static SystemMeasure createFullMeasure(IEEEDataIsland island, int errorDistribution, double ratio) {
        return createMeasureOfTypes(island, new int[]{
                TYPE_BUS_VOLOTAGE,
                TYPE_BUS_ACTIVE_POWER,
                TYPE_BUS_REACTIVE_POWER,

                TYPE_LINE_FROM_ACTIVE,
                TYPE_LINE_FROM_REACTIVE,
                TYPE_LINE_TO_ACTIVE,
                TYPE_LINE_TO_REACTIVE
        }, errorDistribution, ratio);
    }

    /**
     * this method using powerflow result as true value and add random noise as measurement value
     *
     * @param island            ieee common format data
     * @param types             measure types you want to return
     * @param errorDistribution 0: equality 1:Gauss
     * @param ratio             error limit
     * @return return system measure of the types you set
     */
    public static SystemMeasure createMeasureOfTypes(IEEEDataIsland island, int[] types, int errorDistribution, double ratio) {
        NumberOptHelper helper = new NumberOptHelper();
        helper.simple(island);
        Map<Integer, Integer> new2old = helper.getNew2old();
        helper.trans(island);

        YMatrixGetter admittanceGetter = new YMatrixGetter(island);
        admittanceGetter.formYMatrix();

        List<Integer> measureTypes = new ArrayList<Integer>(types.length);
        for (int type : types)
            measureTypes.add(type);
        int n = island.getBuses().size();
        AVector vTheta = new AVector(2 * n);
        for (int i = 0; i < n; i++) {
            BusData bus = island.getBuses().get(i);
            vTheta.setValue(bus.getBusNumber() - 1, bus.getFinalVoltage());
            vTheta.setValue(bus.getBusNumber() + n - 1, bus.getFinalAngle() * Math.PI / 180.0);
        }
        SystemMeasure sm = new SystemMeasure();
        for (BusData bus : island.getBuses()) {
            if (measureTypes.contains(TYPE_BUS_ACTIVE_POWER)) {
                double trueValue = StateCalByPolar.calBusP(bus.getBusNumber(), admittanceGetter, vTheta);
                if (Math.abs(trueValue) > min_p_value) {
                    MeasureInfo injectionP = new MeasureInfo(String.valueOf(bus.getBusNumber()), TYPE_BUS_ACTIVE_POWER, 0);
                    formMeasure(trueValue, errorDistribution, ratio, injectionP);
                    sm.addEfficientMeasure(injectionP);
                }
            }
            if (measureTypes.contains(TYPE_BUS_REACTIVE_POWER)) {
                double trueValue = StateCalByPolar.calBusQ(bus.getBusNumber(), admittanceGetter, vTheta);
                if (Math.abs(trueValue) > min_q_value) {
                    MeasureInfo injectionQ = new MeasureInfo(String.valueOf(bus.getBusNumber()), TYPE_BUS_REACTIVE_POWER, 0);
                    formMeasure(trueValue, errorDistribution, ratio, injectionQ);
                    sm.addEfficientMeasure(injectionQ);
                }
            }
            if (measureTypes.contains(TYPE_BUS_VOLOTAGE)) {
                double trueValue = bus.getFinalVoltage();
                MeasureInfo voltage = new MeasureInfo(String.valueOf(bus.getBusNumber()), TYPE_BUS_VOLOTAGE, 0);
                formMeasure(trueValue, errorDistribution, ratio, voltage);
                sm.addEfficientMeasure(voltage);
            }
            if (measureTypes.contains(TYPE_BUS_ANGLE)) {
                double trueValue = vTheta.getValue(bus.getBusNumber() - 1 + n);
                if (bus.getType() == 2) {
                    MeasureInfo angleInfo = new MeasureInfo(String.valueOf(bus.getBusNumber()), TYPE_BUS_ANGLE, 0);
                    //if (Math.abs(trueValue) < MIN_MEASURE_VALUE)
                    formMeasure(trueValue, errorDistribution, 1.0, angleInfo); //todo:
                    //else
                    //    formMeasure(trueValue, errorDistribution, 0.5 / 3.0 / trueValue / 180 * Math.PI, angleInfo); //todo:
                    sm.addEfficientMeasure(angleInfo);
                }
            }
        }

        for (int i = 0; i < island.getBranches().size(); i++) {
            if (island.getBranches().get(i).getType() != 0)//transformer
                continue;
            int branchId = island.getBranches().get(i).getId();
            if (measureTypes.contains(TYPE_LINE_FROM_ACTIVE)) {
                double trueValue = StateCalByPolar.calLinePFrom(branchId, admittanceGetter, vTheta);
                if (Math.abs(trueValue) > min_p_value) {
                    MeasureInfo fromP = new MeasureInfo(String.valueOf(branchId), TYPE_LINE_FROM_ACTIVE, 0);
                    formMeasure(trueValue, errorDistribution, ratio, fromP);
                    sm.addEfficientMeasure(fromP);
                }
            }
            if (measureTypes.contains(TYPE_LINE_FROM_REACTIVE)) {
                double trueValue = StateCalByPolar.calLineQFrom(branchId, admittanceGetter, vTheta);
                if (Math.abs(trueValue) > min_q_value) {
                    MeasureInfo fromQ = new MeasureInfo(String.valueOf(branchId), TYPE_LINE_FROM_REACTIVE, 0);
                    formMeasure(trueValue, errorDistribution, ratio, fromQ);
                    sm.addEfficientMeasure(fromQ);
                }
            }
            if (measureTypes.contains(TYPE_LINE_TO_ACTIVE)) {
                double trueValue = StateCalByPolar.calLinePTo(branchId, admittanceGetter, vTheta);
                if (Math.abs(trueValue) > min_p_value) {
                    MeasureInfo toP = new MeasureInfo(String.valueOf(branchId), TYPE_LINE_TO_ACTIVE, 0);
                    formMeasure(trueValue, errorDistribution, ratio, toP);
                    sm.addEfficientMeasure(toP);
                }
            }
            if (measureTypes.contains(TYPE_LINE_TO_REACTIVE)) {
                double trueValue = StateCalByPolar.calLineQTo(branchId, admittanceGetter, vTheta);
                if (Math.abs(trueValue) > min_q_value) {
                    MeasureInfo toQ = new MeasureInfo(String.valueOf(branchId), TYPE_LINE_TO_REACTIVE, 0);
                    formMeasure(trueValue, errorDistribution, ratio, toQ);
                    sm.addEfficientMeasure(toQ);
                }
            }
            double[] current = StateCalByPolar.calLineCurrent(branchId, admittanceGetter, vTheta, YMatrixGetter.LINE_FROM);
            if (Math.abs(current[0]) <= 1e-3 || Math.abs(current[1]) <= 1e-3)
                continue;
            if (measureTypes.contains(TYPE_LINE_FROM_CURRENT)) {
                double currentAmp = current[0];
                //if (Math.abs(currentAmp) > MIN_MEASURE_VALUE) {
                MeasureInfo currentInfo = new MeasureInfo(String.valueOf(branchId), TYPE_LINE_FROM_CURRENT, 0);
                formMeasure(currentAmp, errorDistribution, ratio, currentInfo);
                sm.addEfficientMeasure(currentInfo);
                //}
            }
            if (measureTypes.contains(TYPE_LINE_FROM_CURRENT_ANGLE)) {
                double angle = current[1];
                //if (Math.abs(angle) > MIN_MEASURE_VALUE) {
                MeasureInfo angleInfo = new MeasureInfo(String.valueOf(branchId), TYPE_LINE_FROM_CURRENT_ANGLE, 0);
                //if (Math.abs(angle) < MIN_MEASURE_VALUE)
                formMeasure(angle, errorDistribution, 1.0, angleInfo);
                //else
                //    formMeasure(angle, errorDistribution, 0.5 / 3.0 / angle / 180 * Math.PI, angleInfo);
                sm.addEfficientMeasure(angleInfo);
                //}
            }

            current = StateCalByPolar.calLineCurrent(branchId, admittanceGetter, vTheta, YMatrixGetter.LINE_TO);
            if (Math.abs(current[0]) <= 1e-3 || Math.abs(current[1]) <= 1e-3) continue;
            if (measureTypes.contains(TYPE_LINE_TO_CURRENT)) {
                double currentAmp = current[0];
                //if (Math.abs(currentAmp) > 1e-3) {
                MeasureInfo currentInfo = new MeasureInfo(String.valueOf(branchId), TYPE_LINE_TO_CURRENT, 0);
                formMeasure(currentAmp, errorDistribution, ratio, currentInfo);
                sm.addEfficientMeasure(currentInfo);
                //}
            }
            if (measureTypes.contains(TYPE_LINE_TO_CURRENT_ANGLE)) {
                double angle = current[1];
                //if (Math.abs(angle) > 1e-3) {
                MeasureInfo angleInfo = new MeasureInfo(String.valueOf(branchId), TYPE_LINE_TO_CURRENT_ANGLE, 0);
                if (Math.abs(angle) < 1e-3)
                    formMeasure(angle, errorDistribution, 1.0, angleInfo);
                else
                    formMeasure(angle, errorDistribution, 0.5 / 3.0 / angle / 180 * Math.PI, angleInfo);
                sm.addEfficientMeasure(angleInfo);
                //}
            }
        }
        MeasureUtil.trans(sm, new2old);
        helper.revert(island);
        return sm;
    }

    /**
     * this method using voltage & current result as true value and add random noise as measurement value
     *
     * @param island            ieee common format data
     * @param errorDistribution 0: equality 1:Gauss
     * @param ratio             error limit
     * @return every bus voltage & angle , every branch current & angle
     */

    public static SystemMeasure createFullMeasure4PMU(IEEEDataIsland island, int errorDistribution, double ratio) {
        return createMeasureOfTypes(island, new int[]{TYPE_BUS_VOLOTAGE,
                TYPE_BUS_ANGLE,
                TYPE_LINE_FROM_CURRENT,
                TYPE_LINE_FROM_CURRENT_ANGLE,
                TYPE_LINE_TO_CURRENT,
                TYPE_LINE_TO_CURRENT_ANGLE}, errorDistribution, ratio);
    }

    public static SystemMeasure createFullMeasure4PMU_withBadData(IEEEDataIsland island, int errorDistribution, double ratio, double rate) {
        SystemMeasure sm = createFullMeasure4PMU(island, errorDistribution, ratio);
        addBadData4PMU(sm, rate);
        return sm;
    }

    /**
     * this method using powerflow result as true value and add random noise as measurement value
     *
     * @param island            ieee common format data
     * @param errorDistribution 0: equality 1:Gauss
     * @param ratio             error limit
     * @return every bus has injection p, q  and thresholds1 measurement, every branch has power measurement, and current measurement
     */
    public static SystemMeasure createMixedMeasure(IEEEDataIsland island, int errorDistribution, double ratio) {
        return createMeasureOfTypes(island, new int[]{TYPE_BUS_ACTIVE_POWER,
                TYPE_BUS_REACTIVE_POWER,
                TYPE_BUS_VOLOTAGE,
                TYPE_BUS_ANGLE,
                TYPE_LINE_FROM_ACTIVE,
                TYPE_LINE_FROM_REACTIVE,
                TYPE_LINE_TO_ACTIVE,
                TYPE_LINE_TO_REACTIVE,
                TYPE_LINE_FROM_CURRENT,
                TYPE_LINE_FROM_CURRENT_ANGLE,
                TYPE_LINE_TO_CURRENT,
                TYPE_LINE_TO_CURRENT_ANGLE}, errorDistribution, ratio);
    }

    public static SystemMeasure createMixedMeasure(IEEEDataIsland island, int errorDistribution, double ratio_SCAdA, double ratio_PMU) {
        SystemMeasure sm1 = createMeasureOfTypes(island, new int[]{TYPE_BUS_ACTIVE_POWER,
                TYPE_BUS_REACTIVE_POWER,
                TYPE_LINE_FROM_ACTIVE,
                TYPE_LINE_FROM_REACTIVE,
                TYPE_LINE_TO_ACTIVE,
                TYPE_LINE_TO_REACTIVE}, errorDistribution, ratio_SCAdA);
        SystemMeasure sm2 = createMeasureOfTypes(island, new int[]{TYPE_BUS_VOLOTAGE,
                TYPE_BUS_ANGLE,
                TYPE_LINE_FROM_CURRENT,
                TYPE_LINE_FROM_CURRENT_ANGLE,
                TYPE_LINE_TO_CURRENT,
                TYPE_LINE_TO_CURRENT_ANGLE}, errorDistribution, ratio_PMU);
        sm1.setBus_a(sm2.getBus_a());
        sm1.setBus_v(sm2.getBus_v());
        sm1.setLine_from_i_amp(sm2.getLine_from_i_amp());
        sm1.setLine_from_i_a(sm2.getLine_from_i_a());
        sm1.setLine_to_i_amp(sm2.getLine_to_i_amp());
        sm1.setLine_to_i_a(sm2.getLine_to_i_a());
        return sm1;
    }

    public static SystemMeasure createMixedMeasure_withBadData(IEEEDataIsland island, int errorDistribution, double ratio_SCAdA, double ratio_PMU, double ratio_bad) {
        SystemMeasure sm1 = createMeasureOfTypes(island, new int[]{TYPE_BUS_ACTIVE_POWER,
                TYPE_BUS_REACTIVE_POWER,
                TYPE_LINE_FROM_ACTIVE,
                TYPE_LINE_FROM_REACTIVE,
                TYPE_LINE_TO_ACTIVE,
                TYPE_LINE_TO_REACTIVE}, errorDistribution, ratio_SCAdA);
        addBadData(sm1, ratio_bad);
        SystemMeasure sm2 = createMeasureOfTypes(island, new int[]{TYPE_BUS_VOLOTAGE,
                TYPE_BUS_ANGLE,
                TYPE_LINE_FROM_CURRENT,
                TYPE_LINE_FROM_CURRENT_ANGLE,
                TYPE_LINE_TO_CURRENT,
                TYPE_LINE_TO_CURRENT_ANGLE}, errorDistribution, ratio_PMU);
        sm1.setBus_a(sm2.getBus_a());
        sm1.setBus_v(sm2.getBus_v());
        sm1.setLine_from_i_amp(sm2.getLine_from_i_amp());
        sm1.setLine_from_i_a(sm2.getLine_from_i_a());
        sm1.setLine_to_i_amp(sm2.getLine_to_i_amp());
        sm1.setLine_to_i_a(sm2.getLine_to_i_a());
        return sm1;
    }

    public static void formMeasure(double trueValue, int errorDistribution, double ratio, MeasureInfo m) {
        double p;
        double error = Math.abs(ratio * trueValue);
        double sigma;
        if (errorDistribution == 0) {
            p = trueValue + (2.0 * Math.random() - 1.0) * error;
            sigma = Math.sqrt(4.0 * error * error / 12.0);
        } else {
            sigma = error;
            p = trueValue + RandomMaker.randomNorm(0, error);
        }
        m.setValue(p);
        if (error < 1e-6) {
            m.setWeight(1.0);
            m.setSigma(1.0);
        } else {
            m.setSigma(sigma);
            double weiht = 1.0 / (sigma * sigma) / 1e6;//todo: wrong!
            m.setWeight(weiht);
        }
        m.setValue_true(trueValue);
    }

    public static void addBadData(SystemMeasure sm, double rate) {
        int total = sm.getBus_a().size() + sm.getBus_p().size() + sm.getBus_q().size() + sm.getBus_v().size() +
                sm.getLine_from_p().size() + sm.getLine_from_q().size() + sm.getLine_to_p().size() + sm.getLine_to_q().size();
        //int n = (int) Math.ceil(sm.getLine_from_p().size() * rate);
        int n = (int) Math.floor(total * rate);
        log.info("Add bad data number:" + n + "\ttotal measurement number:" + total);
        int[] j = new int[7];
        MeasureInfo[][] infos = new MeasureInfo[7][];
        infos[0] = sm.getLine_from_p().values().toArray(new MeasureInfo[]{});
        infos[1] = sm.getLine_from_q().values().toArray(new MeasureInfo[]{});
        infos[2] = sm.getLine_to_p().values().toArray(new MeasureInfo[]{});
        infos[3] = sm.getLine_to_q().values().toArray(new MeasureInfo[]{});
        infos[4] = sm.getBus_p().values().toArray(new MeasureInfo[]{});
        infos[5] = sm.getBus_q().values().toArray(new MeasureInfo[]{});
        infos[6] = sm.getBus_v().values().toArray(new MeasureInfo[]{});
        while (n > 0) {
            double d = Math.random();
            for (int i = 0; i < infos.length; i++) {
                if (j[i] >= infos[i].length)
                    continue;
                double v = infos[i][j[i]].getValue();
                if (i == 6)
                    infos[i][j[i]].setValue(0.0);
                else if (d > 0.3)
                    infos[i][j[i]].setValue(-v);
                else if (d > 0.15)
                    infos[i][j[i]].setValue(1.2 * v);
                else
                    infos[i][j[i]].setValue(0.8 * v);

                MeasureInfo info = infos[i][j[i]];
                switch (info.getMeasureType()) {
                    case TYPE_BUS_VOLOTAGE:
                    case TYPE_BUS_ACTIVE_POWER:
                    case TYPE_BUS_REACTIVE_POWER:
                        log.info("Bus " + infos[i][j[i]].getPositionId() + " " + v + " " + infos[i][j[i]].getValue());
                        break;
                    case TYPE_LINE_FROM_ACTIVE:
                    case TYPE_LINE_FROM_REACTIVE:
                    case TYPE_LINE_TO_ACTIVE:
                    case TYPE_LINE_TO_REACTIVE:
                        log.info("Branch " + infos[i][j[i]].getPositionId() + " " + v + " " + infos[i][j[i]].getValue());
                        break;
                }

                j[i]++;
                n--;
                if (n <= 0)
                    break;
            }
        }
    }

    public static void addBadData4PMU(SystemMeasure sm, double rate) {
        int total = sm.getBus_a().size() + sm.getBus_v().size() +
                sm.getLine_from_i_a().size() + sm.getLine_from_i_amp().size() +
                sm.getLine_to_i_a().size() + sm.getLine_to_i_amp().size();
        int n = (int) Math.floor(total * rate);
        System.out.println("Add bad data number:" + n + "\ttotal measurement number:" + total);
        MeasureInfo[][] infos = new MeasureInfo[6][];
        infos[0] = sm.getLine_from_i_a().values().toArray(new MeasureInfo[]{});
        infos[1] = sm.getLine_from_i_amp().values().toArray(new MeasureInfo[]{});
        infos[2] = sm.getLine_to_i_a().values().toArray(new MeasureInfo[]{});
        infos[3] = sm.getLine_to_i_amp().values().toArray(new MeasureInfo[]{});
        infos[4] = sm.getBus_a().values().toArray(new MeasureInfo[]{});
        infos[5] = sm.getBus_v().values().toArray(new MeasureInfo[]{});

        while (n > 0) {
            double d1 = Math.random();
            double d2 = Math.random();
            int i = (int) (d1 * 6);
            if (i < 6) {
                int j = (int) (d2 * infos[i].length);
                if (j < infos[i].length) {
                    double d3 = Math.random();
                    double v = infos[i][j].getValue();
                    if (d3 > 0.5) {
                        infos[i][j].setValue(0.9 * v);
                    } else if (d3 > 0.25) {
                        infos[i][j].setValue(1.1 * v);
                    } else {
                        infos[i][j].setValue(0.0);
                    }
                    n--;
                    if (n <= 0)
                        break;
                }
            }
        }
    }

    public static SystemMeasure createMeasureWithCurrent(IEEEDataIsland island, int errorDistribution, double ratio) {
        return createMeasureOfTypes(island, new int[]{TYPE_BUS_ACTIVE_POWER,
                TYPE_BUS_REACTIVE_POWER,
                TYPE_BUS_VOLOTAGE,
                TYPE_LINE_FROM_ACTIVE,
                TYPE_LINE_FROM_REACTIVE,
                TYPE_LINE_TO_ACTIVE,
                TYPE_LINE_TO_REACTIVE,
                TYPE_LINE_CURRENT,
        }, errorDistribution, ratio);
    }
}
