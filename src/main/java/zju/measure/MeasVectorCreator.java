package zju.measure;

import org.apache.log4j.Logger;
import zju.matrix.AVector;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-6
 */
@SuppressWarnings({"unchecked"})
public class MeasVectorCreator implements MeasTypeCons {
    private static Logger log = Logger.getLogger(MeasVectorCreator.class);

    public double[] measures;
    public double[] sigmas;
    public double[] weights;
    public double[] est_values;
    public double[] true_values;
    public int[] measTypes;
    public int[] measPos;

    private int[] measureType;

    public int[] getMeasureType() {
        return measureType;
    }

    public void setMeasureType(int[] measureType) {
        this.measureType = measureType;
    }

    public MeasVector getMeasureVector(SystemMeasure sysMeas) {
        return getMeasureVector(sysMeas, false);
    }

    public MeasVector getMeasureVector(SystemMeasure sysMeas, boolean withPhase) {

        int count = sysMeas.getBus_a().size();
        count += sysMeas.getBus_v().size();
        count += sysMeas.getBus_p().size();
        count += sysMeas.getBus_q().size();
        count += sysMeas.getLine_from_p().size();
        count += sysMeas.getLine_from_q().size();
        count += sysMeas.getLine_to_p().size();
        count += sysMeas.getLine_to_q().size();
        count += sysMeas.getLine_i_amp().size();
        count += sysMeas.getLine_from_i_amp().size();
        count += sysMeas.getLine_from_i_a().size();
        count += sysMeas.getLine_to_i_amp().size();
        count += sysMeas.getLine_to_i_a().size();

        measures = new double[count];
        sigmas = new double[count];
        weights = new double[count];
        est_values = new double[count];
        true_values = new double[count];
        measPos = new int[count];
        measTypes = new int[count];

        MeasVector meas = new MeasVector();
        if (measureType == null)
            meas.setMeasureOrder(DEFAULT_TYPES);
        else
            meas.setMeasureOrder(measureType);
        try {
            int index = 0, rowStart;
            for (int type : meas.getMeasureOrder()) {
                String field = null;
                switch (type) {
                    case TYPE_BUS_ANGLE:
                        field = "bus_a";
                        meas.setBus_a_index(index);
                        for(int i = index; i < sysMeas.bus_a.size() + index; i++)
                            measTypes[i] = TYPE_BUS_ANGLE;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.bus_a.size();
                        break;
                    case TYPE_BUS_VOLOTAGE:
                        field = "bus_v";
                        meas.setBus_v_index(index);
                        for(int i = index; i < sysMeas.bus_v.size() + index; i++)
                            measTypes[i] = TYPE_BUS_VOLOTAGE;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.bus_v.size();
                        break;
                    case TYPE_BUS_ACTIVE_POWER:
                        field = "bus_p";
                        meas.setBus_p_index(index);
                        for(int i = index; i < sysMeas.bus_p.size() + index; i++)
                            measTypes[i] = TYPE_BUS_ACTIVE_POWER;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.bus_p.size();
                        break;
                    case TYPE_BUS_REACTIVE_POWER:
                        field = "bus_q";
                        meas.setBus_q_index(index);
                        for(int i = index; i < sysMeas.bus_q.size() + index; i++)
                            measTypes[i] = TYPE_BUS_REACTIVE_POWER;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.bus_q.size();
                        break;
                    case TYPE_LINE_FROM_ACTIVE:
                        field = "line_from_p";
                        meas.setLine_from_p_index(index);
                        for(int i = index; i < sysMeas.line_from_p.size() + index; i++)
                            measTypes[i] = TYPE_LINE_FROM_ACTIVE;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.line_from_p.size();
                        break;
                    case TYPE_LINE_FROM_REACTIVE:
                        field = "line_from_q";
                        meas.setLine_from_q_index(index);
                        for(int i = index; i < sysMeas.line_from_q.size() + index; i++)
                            measTypes[i] = TYPE_LINE_FROM_REACTIVE;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.line_from_q.size();
                        break;
                    case TYPE_LINE_TO_ACTIVE:
                        field = "line_to_p";
                        meas.setLine_to_p_index(index);
                        for(int i = index; i < sysMeas.line_to_p.size() + index; i++)
                            measTypes[i] = TYPE_LINE_TO_ACTIVE;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.line_to_p.size();
                        break;
                    case TYPE_LINE_TO_REACTIVE:
                        field = "line_to_q";
                        meas.setLine_to_q_index(index);
                        for(int i = index; i < sysMeas.line_to_q.size() + index; i++)
                            measTypes[i] = TYPE_LINE_TO_REACTIVE;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.line_to_q.size();
                        break;
                    case TYPE_LINE_CURRENT:
                        field = "line_i_amp";
                        meas.line_i_amp_index = index;
                        for(int i = index; i < sysMeas.line_i_amp.size() + index; i++)
                            measTypes[i] = TYPE_LINE_CURRENT;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.line_i_amp.size();
                        break;
                    case TYPE_LINE_CURRENT_ANGLE:
                        field = "line_i_a";
                        meas.line_i_a_index = index;
                        for(int i = index; i < sysMeas.line_i_a.size() + index; i++)
                            measTypes[i] = TYPE_LINE_CURRENT_ANGLE;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index+= sysMeas.line_i_a.size();
                        break;
                    case TYPE_LINE_FROM_CURRENT:
                        field = "line_from_i_amp";
                        meas.line_from_i_amp_index = index;
                        for(int i = index; i < sysMeas.line_from_i_amp.size() + index; i++)
                            measTypes[i] = TYPE_LINE_FROM_CURRENT;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.line_from_i_amp.size();
                        break;
                    case TYPE_LINE_FROM_CURRENT_ANGLE:
                        field = "line_from_i_a";
                        meas.setLine_from_i_a_index(index);
                        for(int i = index; i < sysMeas.line_from_i_a.size() + index; i++)
                            measTypes[i] = TYPE_LINE_FROM_CURRENT_ANGLE;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.line_from_i_a.size();
                        break;
                    case TYPE_LINE_TO_CURRENT:
                        field = "line_to_i_amp";
                        meas.setLine_to_i_amp_index(index);
                        for(int i = index; i < sysMeas.line_to_i_amp.size() + index; i++)
                            measTypes[i] = TYPE_LINE_TO_CURRENT;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.line_to_i_amp.size();
                        break;
                    case TYPE_LINE_TO_CURRENT_ANGLE:
                        field = "line_to_i_a";
                        meas.setLine_to_i_a_index(index);
                        for(int i = index; i < sysMeas.line_to_i_a.size() + index; i++)
                            measTypes[i] = TYPE_LINE_TO_CURRENT_ANGLE;
                        setMeasValue(field, sysMeas, meas, withPhase, index);
                        index += sysMeas.line_to_i_a.size();
                        break;
                    default:
                        log.warn("unsupported measure type: " + type);
                        break;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            log.error(e);
            return null;
        }
        meas.setZ(new AVector(measures.length));
        for (int i = 0; i < measures.length; i++)
            meas.getZ().setValue(i, measures[i]);
        if (sigmas.length > 0) {
            meas.setSigma(new AVector(sigmas.length));
            for (int i = 0; i < sigmas.length; i++)
                meas.getSigma().setValue(i, sigmas[i]);
        }
        if (weights.length > 0) {
            meas.setWeight(new AVector(weights.length));
            for (int i = 0; i < weights.length; i++)
                meas.getWeight().setValue(i, weights[i]);
        }
        if (est_values.length > 0) {
            meas.setZ_estimate(new AVector(est_values.length));
            for (int i = 0; i < est_values.length; i++)
                meas.getZ_estimate().setValue(i, est_values[i]);
        }
        if (true_values.length > 0) {
            meas.setZ_true(new AVector(true_values.length));
            for (int i = 0; i < true_values.length; i++)
                meas.getZ_true().setValue(i, true_values[i]);
        }
        return meas;
    }

    public MeasVector parse(InputStream in) {
        return parse(new InputStreamReader(in));
    }

    public MeasVector parse(Reader in) {
        return getMeasureVector(DefaultMeasParser.parse(in));
    }

    private void setMeasValue(String field, SystemMeasure sm, MeasVector meas, boolean withPhase, int index) throws IOException, IllegalAccessException, InvocationTargetException, NoSuchMethodException {
        Map<String, MeasureInfo> content = null;
        String getter = "get" + field;
        Method[] methods = sm.getClass().getMethods();
        for (Method m : methods) {
            if (m.getName().equalsIgnoreCase(getter)) {
                content = (Map<String, MeasureInfo>) m.invoke(sm);
                break;
            }
        }
        assert content != null;
        methods = meas.getClass().getMethods();
        String setter = "set" + field;
        AVector v1 = new AVector(content.size());
        int[] v2 = new int[content.size()];
        int[] v3 = null;
        if (withPhase)
            v3 = new int[content.size()];
        for (Method m : methods) {
            if (m.getName().equalsIgnoreCase(setter)) {
                m.invoke(meas, v1);
            } else if (m.getName().equalsIgnoreCase(setter + "_pos")) {
                m.invoke(meas, (Object) v2);
            } else if (m.getName().equalsIgnoreCase(setter + "_phase")) {
                m.invoke(meas, (Object) v3);
            }
        }
        int i = 0;
        for (MeasureInfo info : content.values()) {
            if (withPhase) {
                String[] v = info.getPositionId().split("_");
                v2[i] = Integer.parseInt(v[0]);
                v3[i] = Integer.parseInt(v[1]);
                measPos[index] = v2[i];
            } else {
                String[] v = info.getPositionId().split("_");
                if (v[0].equals("null"))
                    continue;
                v2[i] = Integer.parseInt(v[0]); //todo:solve the key problem
                measPos[index] = v2[i];
            }
            v1.setValue(i, info.getValue());
            measures[index] = info.getValue();
            sigmas[index] = info.getSigma();
            weights[index] = info.getWeight();
            est_values[index] = info.getValue_est();
            true_values[index] = info.getValue_true();
            i++;
            index++;
        }
    }

    public static MeasVector createVector(MeasureInfo[] infos) {
        MeasVector meas = new MeasVector();
        List<Double> measures = new ArrayList<>(infos.length);
        List<Double> sigmas = new ArrayList<>(infos.length);
        List<Double> weights = new ArrayList<>(infos.length);
        List<Double> est_values = new ArrayList<>(infos.length);
        List<Double> true_values = new ArrayList<>(infos.length);
        for (MeasureInfo info : infos) {
            measures.add(info.getValue());
            sigmas.add(info.getSigma());
            weights.add(info.getWeight());
            est_values.add(info.getValue_est());
            true_values.add(info.getValue_true());
        }
        meas.setZ(new AVector(measures.size()));
        for (int i = 0; i < measures.size(); i++)
            meas.z.setValue(i, measures.get(i));
        meas.setSigma(new AVector(sigmas.size()));
        for (int i = 0; i < sigmas.size(); i++)
            meas.sigma.setValue(i, sigmas.get(i));
        meas.setWeight(new AVector(weights.size()));
        for (int i = 0; i < weights.size(); i++)
            meas.weight.setValue(i, weights.get(i));
        meas.setZ_estimate(new AVector(est_values.size()));
        for (int i = 0; i < est_values.size(); i++)
            meas.z_estimate.setValue(i, est_values.get(i));
        meas.setZ_true(new AVector(true_values.size()));
        for (int i = 0; i < true_values.size(); i++)
            meas.z_true.setValue(i, true_values.get(i));
        return meas;
    }

    public MeasVector parse(String filePath) {
        return this.parse(new File(filePath));
    }

    public MeasVector parse(File file) {
        try {
            return this.parse(new BufferedReader(new FileReader(file)));
        } catch (FileNotFoundException e) {
            return null;
        }
    }
}
