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

    private List<Double> measures;
    private List<Double> sigmas;
    private List<Double> weights;
    private List<Double> est_values;
    private List<Double> true_values;

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

        measures = new ArrayList<Double>(count);
        sigmas = new ArrayList<Double>(count);
        weights = new ArrayList<Double>(count);
        est_values = new ArrayList<Double>(count);
        true_values = new ArrayList<Double>(count);

        MeasVector meas = new MeasVector();
        if (measureType == null)
            meas.setMeasureOrder(DEFAULT_TYPES);
        else
            meas.setMeasureOrder(measureType);
        try {
            int index = 0;
            for (int type : meas.getMeasureOrder()) {
                String field = null;
                switch (type) {
                    case TYPE_BUS_ANGLE:
                        field = "bus_a";
                        meas.setBus_a_index(index);
                        index += sysMeas.getBus_a().size();
                        break;
                    case TYPE_BUS_VOLOTAGE:
                        field = "bus_v";
                        meas.setBus_v_index(index);
                        index += sysMeas.getBus_v().size();
                        break;
                    case TYPE_BUS_ACTIVE_POWER:
                        field = "bus_p";
                        meas.setBus_p_index(index);
                        index += sysMeas.getBus_p().size();
                        break;
                    case TYPE_BUS_REACTIVE_POWER:
                        field = "bus_q";
                        meas.setBus_q_index(index);
                        index += sysMeas.getBus_q().size();
                        break;
                    case TYPE_LINE_FROM_ACTIVE:
                        field = "line_from_p";
                        meas.setLine_from_p_index(index);
                        index += sysMeas.getLine_from_p().size();
                        break;
                    case TYPE_LINE_FROM_REACTIVE:
                        field = "line_from_q";
                        meas.setLine_from_q_index(index);
                        index += sysMeas.getLine_from_q().size();
                        break;
                    case TYPE_LINE_TO_ACTIVE:
                        field = "line_to_p";
                        meas.setLine_to_p_index(index);
                        index += sysMeas.getLine_to_p().size();
                        break;
                    case TYPE_LINE_TO_REACTIVE:
                        field = "line_to_q";
                        meas.setLine_to_q_index(index);
                        index += sysMeas.getLine_to_q().size();
                        break;
                    case TYPE_LINE_CURRENT:
                        field = "line_i_amp";
                        meas.setLine_i_amp_index(index);
                        index += sysMeas.getLine_i_amp().size();
                        break;
                    case TYPE_LINE_CURRENT_ANGLE:
                        field = "line_i_a";
                        //meas.setLine_i_a_index(index);
                        //index+= sysMeas.getLine_i_a().size();
                        break;
                    case TYPE_LINE_FROM_CURRENT:
                        field = "line_from_i_amp";
                        meas.setLine_i_amp_index(index);
                        index += sysMeas.getLine_from_i_amp().size();
                        break;
                    case TYPE_LINE_FROM_CURRENT_ANGLE:
                        field = "line_from_i_a";
                        meas.setLine_from_i_a_index(index);
                        index += sysMeas.getLine_from_i_a().size();
                        break;
                    case TYPE_LINE_TO_CURRENT:
                        field = "line_to_i_amp";
                        meas.setLine_to_i_amp_index(index);
                        index += sysMeas.getLine_to_i_amp().size();
                        break;
                    case TYPE_LINE_TO_CURRENT_ANGLE:
                        field = "line_to_i_a";
                        meas.setLine_to_i_a_index(index);
                        index += sysMeas.getLine_to_i_a().size();
                        break;
                    default:
                        log.warn("unsupported measure type: " + type);
                        break;
                }
                setMeasValue(field, sysMeas, meas, withPhase);
            }
        } catch (Exception e) {
            e.printStackTrace();
            log.error(e);
            return null;
        }
        meas.setZ(new AVector(measures.size()));
        for (int i = 0; i < measures.size(); i++)
            meas.getZ().setValue(i, measures.get(i));
        if (sigmas.size() > 0) {
            meas.setSigma(new AVector(sigmas.size()));
            for (int i = 0; i < sigmas.size(); i++)
                meas.getSigma().setValue(i, sigmas.get(i));
        }
        if (weights.size() > 0) {
            meas.setWeight(new AVector(weights.size()));
            for (int i = 0; i < weights.size(); i++)
                meas.getWeight().setValue(i, weights.get(i));
        }
        if (est_values.size() > 0) {
            meas.setZ_estimate(new AVector(est_values.size()));
            for (int i = 0; i < est_values.size(); i++)
                meas.getZ_estimate().setValue(i, est_values.get(i));
        }
        if (true_values.size() > 0) {
            meas.setZ_true(new AVector(true_values.size()));
            for (int i = 0; i < true_values.size(); i++)
                meas.getZ_true().setValue(i, true_values.get(i));
        }
        return meas;
    }

    public MeasVector parse(InputStream in) {
        return parse(new InputStreamReader(in));
    }

    public MeasVector parse(Reader in) {
        return getMeasureVector(DefaultMeasParser.parse(in));
    }

    private void setMeasValue(String field, SystemMeasure sm, MeasVector meas, boolean withPhase) throws IOException, IllegalAccessException, InvocationTargetException, NoSuchMethodException {
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
                m.invoke(meas, v2);
            } else if (m.getName().equalsIgnoreCase(setter + "_phase")) {
                m.invoke(meas, v3);
            }
        }
        int i = 0;
        for (MeasureInfo info : content.values()) {
            if (withPhase) {
                String[] v = info.getPositionId().split("_");
                v2[i] = Integer.parseInt(v[0]);
                v3[i] = Integer.parseInt(v[1]);
            } else {
                String[] v = info.getPositionId().split("_");
                if (v[0].equals("null"))
                    continue;
                v2[i] = Integer.parseInt(v[0]); //todo:solve the key problem
                //v2[i] = Integer.parseInt(info.getPositionId());
            }
            v1.setValue(i, info.getValue());
            measures.add(info.getValue());
            sigmas.add(info.getSigma());
            weights.add(info.getWeight());
            est_values.add(info.getValue_est());
            true_values.add(info.getValue_true());
            i++;
        }
    }

    public static MeasVector createVector(MeasureInfo[] infos) {
        MeasVector meas = new MeasVector();
        List<Double> measures = new ArrayList<Double>(infos.length);
        List<Double> sigmas = new ArrayList<Double>(infos.length);
        List<Double> weights = new ArrayList<Double>(infos.length);
        List<Double> est_values = new ArrayList<Double>(infos.length);
        List<Double> true_values = new ArrayList<Double>(infos.length);
        for (MeasureInfo info : infos) {
            measures.add(info.getValue());
            sigmas.add(info.getSigma());
            weights.add(info.getWeight());
            est_values.add(info.getValue_est());
            true_values.add(info.getValue_true());
        }
        meas.setZ(new AVector(measures.size()));
        for (int i = 0; i < measures.size(); i++)
            meas.getZ().setValue(i, measures.get(i));
        meas.setSigma(new AVector(sigmas.size()));
        for (int i = 0; i < sigmas.size(); i++)
            meas.getSigma().setValue(i, sigmas.get(i));
        meas.setWeight(new AVector(weights.size()));
        for (int i = 0; i < weights.size(); i++)
            meas.getWeight().setValue(i, weights.get(i));
        meas.setZ_estimate(new AVector(est_values.size()));
        for (int i = 0; i < est_values.size(); i++)
            meas.getZ_estimate().setValue(i, est_values.get(i));
        meas.setZ_true(new AVector(true_values.size()));
        for (int i = 0; i < true_values.size(); i++)
            meas.getZ_true().setValue(i, true_values.get(i));
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
