package zju.measure;

import zju.matrix.AVector;
import zju.util.StateCalByPolar;
import zju.util.YMatrixGetter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-6-19
 */
public class MeasureUtil implements MeasTypeCons {

    public static void setVTheta(double vTheta[], SystemMeasure dest, int n) {
        for (int i = 1; i <= n; i++) {
            if (dest.getBus_a().containsKey(String.valueOf(i)))
                dest.getBus_a().get(String.valueOf(i)).setValue_est(vTheta[i + n - 1]);
            if (dest.getBus_v().containsKey(String.valueOf(i)))
                dest.getBus_v().get(String.valueOf(i)).setValue_est(vTheta[i - 1]);
        }
    }

    public static void setVThetaPq(double[] state, SystemMeasure dest, int n) {
        for (int i = 1; i <= n; i++) {
            if (dest.getBus_a().containsKey(String.valueOf(i)))
                dest.getBus_a().get(String.valueOf(i)).setValue_est(state[i + n - 1]);
            if (dest.getBus_v().containsKey(String.valueOf(i)))
                dest.getBus_v().get(String.valueOf(i)).setValue_est(state[i - 1]);
            if (dest.getBus_p().containsKey(String.valueOf(i)))
                dest.getBus_p().get(String.valueOf(i)).setValue_est(state[i + 2 * n - 1]);
            if (dest.getBus_q().containsKey(String.valueOf(i)))
                dest.getBus_q().get(String.valueOf(i)).setValue_est(state[i + 3 * n - 1]);
        }
    }

    public static void setEstValue(MeasVector source, SystemMeasure dest) {
        int index = 0;
        for (int type : source.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ANGLE:
                    if (source.getBus_a_phase() == null)
                        for (int i = 0; i < source.getBus_a_pos().length; i++, index++) {
                            dest.getBus_a().get(String.valueOf(source.getBus_a_pos()[i])).setValue_est(source.getZ_estimate().getValue(index));
                            //for (MeasureInfo info : dest.getBus_a().values()) {
                            //    if (info.getPositionId().equals(String.valueOf(source.getBus_a_pos()[i]))
                            //            || info.getPositionId().startsWith(String.valueOf(source.getBus_a_pos()[i]) + "_")) {
                            //        info.setValue_est(source.getZ_estimate().getValue(index));
                            //    }
                            //}
                        }
                    else
                        for (int i = 0; i < source.getBus_a_pos().length; i++, index++)
                            dest.getBus_a().get(source.getBus_a_pos()[i] + "_" + source.getBus_a_phase()[i]).setValue_est(source.getZ_estimate().getValue(index));
                    break;
                case TYPE_BUS_VOLOTAGE:
                    if (source.getBus_v_phase() == null)
                        for (int i = 0; i < source.getBus_v_pos().length; i++, index++) {
                            dest.getBus_v().get(String.valueOf(source.getBus_v_pos()[i])).setValue_est(source.getZ_estimate().getValue(index));
                            //for (MeasureInfo info : dest.getBus_v().values()) {
                            //    if (info.getPositionId().equals(String.valueOf(source.getBus_v_pos()[i]))
                            //            || info.getPositionId().startsWith(String.valueOf(source.getBus_v_pos()[i]) + "_")) {
                            //        info.setValue_est(source.getZ_estimate().getValue(index));
                            //    }
                            //}
                        }
                    else
                        for (int i = 0; i < source.getBus_v_pos().length; i++, index++)
                            dest.getBus_v().get(source.getBus_v_pos()[i] + "_" + source.getBus_v_phase()[i]).setValue_est(source.getZ_estimate().getValue(index));
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    if (source.getBus_p_phase() == null)
                        for (int i = 0; i < source.getBus_p_pos().length; i++, index++) {
                            dest.getBus_p().get(String.valueOf(source.getBus_p_pos()[i])).setValue_est(source.getZ_estimate().getValue(index));
                            //for (MeasureInfo info : dest.getBus_p().values()) {
                            //    if (info.getPositionId().equals(String.valueOf(source.getBus_p_pos()[i]))
                            //            || info.getPositionId().startsWith(String.valueOf(source.getBus_p_pos()[i]) + "_")) {
                            //        info.setValue_est(source.getZ_estimate().getValue(index));
                            //    }
                            //}
                        }
                    else
                        for (int i = 0; i < source.getBus_p_pos().length; i++, index++)
                            dest.getBus_p().get(source.getBus_p_pos()[i] + "_" + source.getBus_p_phase()[i]).setValue_est(source.getZ_estimate().getValue(index));
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    if (source.getBus_q_phase() == null)
                        for (int i = 0; i < source.getBus_q_pos().length; i++, index++) {
                            dest.getBus_q().get(String.valueOf(source.getBus_q_pos()[i])).setValue_est(source.getZ_estimate().getValue(index));
                            //for (MeasureInfo info : dest.getBus_q().values()) {
                            //    if (info.getPositionId().equals(String.valueOf(source.getBus_q_pos()[i]))
                            //            || info.getPositionId().startsWith(String.valueOf(source.getBus_q_pos()[i]) + "_")) {
                            //        info.setValue_est(source.getZ_estimate().getValue(index));
                            //    }
                            //}
                        }
                    else
                        for (int i = 0; i < source.getBus_q_pos().length; i++, index++)
                            dest.getBus_q().get(source.getBus_q_pos()[i] + "_" + source.getBus_q_phase()[i]).setValue_est(source.getZ_estimate().getValue(index));
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    if (source.getLine_from_p_phase() == null)
                        for (int k = 0; k < source.getLine_from_p_pos().length; k++, index++)
                            dest.getLine_from_p().get(String.valueOf(source.getLine_from_p_pos()[k])).setValue_est(source.getZ_estimate().getValue(index));
                    else
                        for (int k = 0; k < source.getLine_from_p_pos().length; k++, index++)
                            dest.getLine_from_p().get(source.getLine_from_p_pos()[k] + "_" + source.getLine_from_p_phase()[k]).setValue_est(source.getZ_estimate().getValue(index));
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    if (source.getLine_from_q_phase() == null)
                        for (int k = 0; k < source.getLine_from_q_pos().length; k++, index++)
                            dest.getLine_from_q().get(String.valueOf(source.getLine_from_q_pos()[k])).setValue_est(source.getZ_estimate().getValue(index));
                    else
                        for (int k = 0; k < source.getLine_from_q_pos().length; k++, index++)
                            dest.getLine_from_q().get(source.getLine_from_q_pos()[k] + "_" + source.getLine_from_q_phase()[k]).setValue_est(source.getZ_estimate().getValue(index));

                    break;
                case TYPE_LINE_TO_ACTIVE:
                    if (source.getLine_to_p_phase() == null)
                        for (int k = 0; k < source.getLine_to_p_pos().length; k++, index++)
                            dest.getLine_to_p().get(String.valueOf(source.getLine_to_p_pos()[k])).setValue_est(source.getZ_estimate().getValue(index));
                    else
                        for (int k = 0; k < source.getLine_to_p_pos().length; k++, index++)
                            dest.getLine_to_p().get(source.getLine_to_p_pos()[k] + "_" + source.getLine_to_p_phase()[k]).setValue_est(source.getZ_estimate().getValue(index));
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    if (source.getLine_to_q_phase() == null)
                        for (int k = 0; k < source.getLine_to_q_pos().length; k++, index++)
                            dest.getLine_to_q().get(String.valueOf(source.getLine_to_q_pos()[k])).setValue_est(source.getZ_estimate().getValue(index));
                    else
                        for (int k = 0; k < source.getLine_to_q_pos().length; k++, index++)
                            dest.getLine_to_q().get(source.getLine_to_q_pos()[k] + "_" + source.getLine_to_q_phase()[k]).setValue_est(source.getZ_estimate().getValue(index));

                    break;
                case TYPE_LINE_CURRENT:
                    if (source.getLine_i_amp_phase() == null)
                        for (int k = 0; k < source.getLine_i_amp_pos().length; k++, index++)
                            dest.getLine_i_amp().get(String.valueOf(source.getLine_i_amp_pos()[k])).setValue_est(source.getZ_estimate().getValue(index));
                    else
                        for (int k = 0; k < source.getLine_i_amp_pos().length; k++, index++)
                            dest.getLine_i_amp().get(source.getLine_i_amp_pos()[k] + "_" + source.getLine_i_amp_phase()[k]).setValue_est(source.getZ_estimate().getValue(index));
                    break;
                case TYPE_LINE_CURRENT_ANGLE:
                    if (source.getLine_i_amp_phase() == null)
                        for (int k = 0; k < source.line_i_a_pos.length; k++, index++)
                            dest.getLine_i_a().get(String.valueOf(source.line_i_amp_pos[k])).setValue_est(source.z_estimate.getValue(index));
                    else
                        for (int k = 0; k < source.getLine_i_amp_pos().length; k++, index++)
                            dest.getLine_i_a().get(source.line_i_a_pos[k] + "_" + source.line_i_amp_phase[k]).setValue_est(source.z_estimate.getValue(index));
                    break;
                case TYPE_LINE_FROM_CURRENT:
                    if (source.getLine_from_i_amp_phase() == null)
                        for (int k = 0; k < source.getLine_from_i_amp_pos().length; k++, index++)
                            dest.getLine_from_i_amp().get(String.valueOf(source.getLine_from_i_amp_pos()[k])).setValue_est(source.getZ_estimate().getValue(index));
                    else
                        for (int k = 0; k < source.getLine_from_i_amp_pos().length; k++, index++)
                            dest.getLine_from_i_amp().get(source.getLine_from_i_amp_pos()[k] + "_" + source.getLine_from_i_amp_phase()[k]).setValue_est(source.getZ_estimate().getValue(index));
                    break;
                case TYPE_LINE_FROM_CURRENT_ANGLE:
                    if (source.getLine_from_i_a_phase() == null)
                        for (int k = 0; k < source.line_from_i_a_pos.length; k++, index++)
                            dest.getLine_from_i_a().get(String.valueOf(source.line_from_i_a_pos[k])).setValue_est(source.getZ_estimate().getValue(index));
                    else
                        for (int k = 0; k < source.line_from_i_a_pos.length; k++, index++)
                            dest.getLine_from_i_a().get(source.line_from_i_a_pos[k] + "_" + source.getLine_from_i_a_phase()[k]).setValue_est(source.getZ_estimate().getValue(index));
                    break;
                case TYPE_LINE_TO_CURRENT:
                    if (source.getLine_to_i_amp_phase() == null)
                        for (int k = 0; k < source.getLine_to_i_amp_pos().length; k++, index++)
                            dest.getLine_to_i_amp().get(String.valueOf(source.getLine_to_i_amp_pos()[k])).setValue_est(source.getZ_estimate().getValue(index));
                    else
                        for (int k = 0; k < source.getLine_to_i_amp_pos().length; k++, index++)
                            dest.getLine_to_i_amp().get(source.getLine_to_i_amp_pos()[k] + "_" + source.getLine_to_i_amp_phase()[k]).setValue_est(source.getZ_estimate().getValue(index));
                    break;
                case TYPE_LINE_TO_CURRENT_ANGLE:
                    if (source.getLine_to_i_a_phase() == null)
                        for (int k = 0; k < source.getLine_to_i_a_pos().length; k++, index++)
                            dest.getLine_to_i_a().get(String.valueOf(source.getLine_to_i_a_pos()[k])).setValue_est(source.getZ_estimate().getValue(index));
                    else
                        for (int k = 0; k < source.getLine_to_i_a_pos().length; k++, index++)
                            dest.getLine_to_i_a().get(source.getLine_to_i_a_pos()[k] + "_" + source.getLine_to_i_a_phase()[k]).setValue_est(source.getZ_estimate().getValue(index));
                    break;
                default:
                    //log.warn("Unsupported measure type: " + type);
                    break;
            }
        }
    }

    public static void setEstValue(SystemMeasure dest, AVector state, YMatrixGetter yMatrixGetter) {
        int n = state.getN() / 2;
        for (MeasureInfo info : dest.getBus_v().values()) {
            int number = info.getPositionIdOfIntFormat();   //todo:without optimization
            info.setValue_est(state.getValue(number - 1));
        }
        for (MeasureInfo info : dest.getBus_a().values()) {
            int number = info.getPositionIdOfIntFormat();
            info.setValue_est(state.getValue(number - 1 + n));
        }
        for (MeasureInfo info : dest.getBus_p().values()) {
            int number = info.getPositionIdOfIntFormat();
            info.setValue_est(StateCalByPolar.calBusP(number, yMatrixGetter, state));
        }
        for (MeasureInfo info : dest.getBus_q().values()) {
            int number = info.getPositionIdOfIntFormat();
            info.setValue_est(StateCalByPolar.calBusQ(number, yMatrixGetter, state));
        }
        for (MeasureInfo info : dest.getLine_from_p().values()) {
            int number = info.getPositionIdOfIntFormat();
            info.setValue_est(StateCalByPolar.calLinePFrom(number, yMatrixGetter, state));
        }
        for (MeasureInfo info : dest.getLine_from_q().values()) {
            int number = info.getPositionIdOfIntFormat();
            info.setValue_est(StateCalByPolar.calLineQFrom(number, yMatrixGetter, state));
        }
        for (MeasureInfo info : dest.getLine_to_p().values()) {
            int number = info.getPositionIdOfIntFormat();
            info.setValue_est(StateCalByPolar.calLinePTo(number, yMatrixGetter, state));
        }
        for (MeasureInfo info : dest.getLine_to_q().values()) {
            int number = info.getPositionIdOfIntFormat();
            info.setValue_est(StateCalByPolar.calLineQTo(number, yMatrixGetter, state));
        }
        for (MeasureInfo info : dest.getLine_from_i_a().values()) {
            int number = info.getPositionIdOfIntFormat();
            info.setValue_est(StateCalByPolar.calLineCurrentAngle(number, yMatrixGetter, state, YMatrixGetter.LINE_FROM));
        }
        for (MeasureInfo info : dest.getLine_from_i_amp().values()) {
            int number = info.getPositionIdOfIntFormat();
            info.setValue_est(StateCalByPolar.calLineCurrentAmp(number, yMatrixGetter, state, YMatrixGetter.LINE_FROM));
        }
        for (MeasureInfo info : dest.getLine_to_i_a().values()) {
            int number = info.getPositionIdOfIntFormat();
            info.setValue_est(StateCalByPolar.calLineCurrentAngle(number, yMatrixGetter, state, YMatrixGetter.LINE_TO));
        }
        for (MeasureInfo info : dest.getLine_to_i_amp().values()) {
            int number = info.getPositionIdOfIntFormat();
            info.setValue_est(StateCalByPolar.calLineCurrentAmp(number, yMatrixGetter, state, YMatrixGetter.LINE_TO));
        }
    }

    /**
     * get eligible and uneligible measurement list
     *
     * @param sm total measurements of system
     * @return list array: the first list is eligible measurement list and the second list is uneligible measurement list
     */
    public static List<MeasureInfo>[] getTwoMeasureList(SystemMeasure sm) {
        List<MeasureInfo> eligible = new ArrayList<MeasureInfo>();
        List<MeasureInfo> ineligible = new ArrayList<MeasureInfo>();
        for (int type : MeasTypeCons.DEFAULT_TYPES) {
            for (MeasureInfo info : sm.getContainer(type).values()) {
                if (info.isEligible())
                    eligible.add(info);
                else
                    ineligible.add(info);
            }
        }
        return new List[]{eligible, ineligible};
    }

    /**
     * get bad measurement list
     *
     * @param sm total measurements of system
     * @return bad measurement list
     */
    public static List<MeasureInfo> getEstBadMeasure(SystemMeasure sm) {
        List<MeasureInfo> ineligible = new ArrayList<MeasureInfo>();
        for (int type : MeasTypeCons.DEFAULT_TYPES) {
            for (MeasureInfo info : sm.getContainer(type).values()) {
                if (Math.abs(info.getValue() - info.getValue_est()) > info.getSigma() * 3)
                    ineligible.add(info);
            }
        }
        return ineligible;
    }

    public static Map<String, MeasureInfo> getBadMeasure(SystemMeasure sm) {
        Map<String, MeasureInfo> bds = new HashMap<String, MeasureInfo>();
        for (int type : MeasTypeCons.DEFAULT_TYPES) {
            for (MeasureInfo info : sm.getContainer(type).values()) {
                if (Math.abs(info.getValue() - info.getValue_true()) > info.getSigma() * 3)
                    bds.put(info.getMeasureType() + "_" + info.getPositionId(), info);
            }
        }
        return bds;
    }

    /**
     * transfer bus number in ieee common format data and measurement information is transfered also
     *
     * @param mapping key: current bus number value: new bus number
     * @param sysMeas system measure to be translated
     */
    public static void trans(SystemMeasure sysMeas, Map<Integer, Integer> mapping) {
        Map<String, MeasureInfo> m = new HashMap<String, MeasureInfo>(sysMeas.getBus_a().size());
        for (MeasureInfo info : sysMeas.getBus_a().values()) {
            String[] v = info.getPositionId().split("_");
            String newKey = String.valueOf(mapping.get(Integer.parseInt(v[0])));
            if (v.length > 1)
                newKey += "_" + v[1];
            info.setPositionId(newKey);
            m.put(newKey, info);
        }
        sysMeas.setBus_a(m);
        m = new HashMap<String, MeasureInfo>(sysMeas.getBus_v().size());
        for (MeasureInfo info : sysMeas.getBus_v().values()) {
            String[] v = info.getPositionId().split("_");
            String newKey = String.valueOf(mapping.get(Integer.parseInt(v[0])));
            if (v.length > 1)
                newKey += "_" + v[1];
            info.setPositionId(newKey);
            m.put(newKey, info);
        }
        sysMeas.setBus_v(m);
        m = new HashMap<String, MeasureInfo>(sysMeas.getBus_p().size());
        for (MeasureInfo info : sysMeas.getBus_p().values()) {
            String[] v = info.getPositionId().split("_");
            String newKey = String.valueOf(mapping.get(Integer.parseInt(v[0])));
            if (v.length > 1)
                newKey += "_" + v[1];
            info.setPositionId(newKey);
            m.put(newKey, info);
        }
        sysMeas.setBus_p(m);
        m = new HashMap<String, MeasureInfo>(sysMeas.getBus_q().size());
        for (MeasureInfo info : sysMeas.getBus_q().values()) {
            String[] v = info.getPositionId().split("_");
            String newKey = String.valueOf(mapping.get(Integer.parseInt(v[0])));
            if (v.length > 1)
                newKey += "_" + v[1];
            info.setPositionId(newKey);
            m.put(newKey, info);
        }
        sysMeas.setBus_q(m);

        if (sysMeas.getId2Measure() != null) {
            Map<String, List<MeasureInfo>> newId2Measure = new HashMap<String, List<MeasureInfo>>();
            for (String id : sysMeas.getId2Measure().keySet()) {
                if (!id.startsWith("bus")) {
                    newId2Measure.put(id, sysMeas.getId2Measure().get(id));
                    continue;
                }
                int oldBusNum = Integer.parseInt(id.substring("bus".length()));
                int newBusNum = mapping.get(oldBusNum);
                newId2Measure.put("bus" + newBusNum, sysMeas.getId2Measure().get(id));
            }
            sysMeas.setId2Measure(newId2Measure);
        }
    }

    /**
     * transfer bus number in ieee common format data and measurement information is transfered also
     *
     * @param mapping key: current bus number value: new bus number
     * @param sysMeas system measure to be translated
     */
    public static void trans_3phase(SystemMeasure sysMeas, Map<Integer, Integer> mapping) {
        Map<String, MeasureInfo> m = new HashMap<String, MeasureInfo>(sysMeas.getBus_a().size());
        for (MeasureInfo info : sysMeas.getBus_a().values())
            tranMeasure(info, mapping, m);
        sysMeas.setBus_a(m);
        m = new HashMap<String, MeasureInfo>(sysMeas.getBus_v().size());
        for (MeasureInfo info : sysMeas.getBus_v().values())
            tranMeasure(info, mapping, m);

        sysMeas.setBus_v(m);
        m = new HashMap<String, MeasureInfo>(sysMeas.getBus_p().size());
        for (MeasureInfo info : sysMeas.getBus_p().values())
            tranMeasure(info, mapping, m);
        sysMeas.setBus_p(m);
        m = new HashMap<String, MeasureInfo>(sysMeas.getBus_q().size());
        for (MeasureInfo info : sysMeas.getBus_q().values())
            tranMeasure(info, mapping, m);
        sysMeas.setBus_q(m);
    }

    private static void tranMeasure(MeasureInfo info, Map<Integer, Integer> mapping, Map<String, MeasureInfo> m) {
        String[] v = info.getPositionId().split("_");
        int key = Integer.parseInt(v[0]);
        if (!mapping.containsKey(key)) {
            System.out.println("Bus number " + key + " is not found in old2New map!");
            return;
        }
        String newKey = mapping.get(key) + "_" + v[1];
        info.setPositionId(newKey);
        m.put(newKey, info);
    }

    public static void trans(MeasVector meas, Map<Integer, Integer> mapping) {
        for (int i = 0; i < meas.getBus_a_pos().length; i++)
            meas.getBus_a_pos()[i] = mapping.get(meas.getBus_a_pos()[i]);
        for (int i = 0; i < meas.getBus_v_pos().length; i++)
            meas.getBus_v_pos()[i] = mapping.get(meas.getBus_v_pos()[i]);
        for (int i = 0; i < meas.getBus_p_pos().length; i++)
            meas.getBus_p_pos()[i] = mapping.get(meas.getBus_p_pos()[i]);
        for (int i = 0; i < meas.getBus_q_pos().length; i++)
            meas.getBus_q_pos()[i] = mapping.get(meas.getBus_q_pos()[i]);
    }
}
