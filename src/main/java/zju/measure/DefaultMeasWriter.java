package zju.measure;

import org.apache.log4j.Logger;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-11-18
 */
public class DefaultMeasWriter implements MeasTypeCons {

    private static Logger log = Logger.getLogger(DefaultMeasWriter.class);
    private static DecimalFormat format = new DecimalFormat("##.####");

    //private static String[] typeNames = new String[]{"bus voltage", "bus injection active power", "bus injection reactive power",
    //        "line active power from", "line active power to", "line reactive power from", "line reactive power to"};

    public DefaultMeasWriter() {
    }

    public static void writeSimple(SystemMeasure sysMeasure, Writer writer) {
        writeSimple(writer, sysMeasure.getBus_a().values(), sysMeasure.getBus_v().values(), sysMeasure.getBus_p().values(), sysMeasure.getBus_q().values(),
                sysMeasure.getLine_from_p().values(), sysMeasure.getLine_from_q().values(), sysMeasure.getLine_to_p().values(), sysMeasure.getLine_to_q().values());
    }

    public static void writeFully(SystemMeasure sysMeasure, Writer writer) {
        BufferedWriter w = new BufferedWriter(writer);
        try {
            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_a");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getBus_a().values()) {
                //w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma_x_estimate() + "\t" + info.getWeight());
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_v");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getBus_v().values()) {
                //w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma_x_estimate() + "\t" + info.getWeight());
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_p");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getBus_p().values()) {
                //w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma_x_estimate() + "\t" + info.getWeight());
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_q");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getBus_q().values()) {
                //w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma_x_estimate() + "\t" + info.getWeight() + "\t" + info.getGenMVA());
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_from_p");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_from_p().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_from_q");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_from_q().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_to_p");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_to_p().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_to_q");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_to_q().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_i_amp");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_i_amp().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_from_i_amp");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_from_i_amp().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_from_i_a");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_from_i_a().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_to_i_amp");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_to_i_amp().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_to_i_a");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_to_i_a().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight());
                w.newLine();
            }
            w.write("-999");

            w.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            log.error(e);
        }
    }

    public static void writeFullyWithTrueValue(SystemMeasure sysMeasure, Writer writer) {
        BufferedWriter w = new BufferedWriter(writer);
        try {
            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_a");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getBus_a().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_v");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getBus_v().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_p");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getBus_p().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true() + "\t" + info.getGenMVA());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_q");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getBus_q().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true() + "\t" + info.getGenMVA());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_from_p");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_from_p().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_from_q");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_from_q().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_to_p");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_to_p().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_to_q");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_to_q().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_i_amp");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_i_amp().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_from_i_amp");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_from_i_amp().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_from_i_a");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_from_i_a().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_to_i_amp");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_to_i_amp().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_to_i_a");
            w.newLine();
            for (MeasureInfo info : sysMeasure.getLine_to_i_a().values()) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getSigma() + "\t" + info.getWeight() + "\t" + info.getValue_true());
                w.newLine();
            }
            w.write("-999");
            w.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            log.error(e);
        }
    }

    public static void writeSimple(Writer writer, Collection<MeasureInfo> bus_a,
                                   Collection<MeasureInfo> bus_v,
                                   Collection<MeasureInfo> bus_p,
                                   Collection<MeasureInfo> bus_q,
                                   Collection<MeasureInfo> line_from_p,
                                   Collection<MeasureInfo> line_from_q,
                                   Collection<MeasureInfo> line_to_p,
                                   Collection<MeasureInfo> line_to_q
    ) {
        BufferedWriter w = new BufferedWriter(writer);
        try {
            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_a");
            w.newLine();
            for (MeasureInfo info : bus_a) {
                w.write(info.getPositionId() + "\t" + info.getValue());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_v");
            w.newLine();
            for (MeasureInfo info : bus_v) {
                w.write(info.getPositionId() + "\t" + info.getValue());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_p");
            w.newLine();
            for (MeasureInfo info : bus_p) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getGenMVA());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("bus_q");
            w.newLine();
            for (MeasureInfo info : bus_q) {
                w.write(info.getPositionId() + "\t" + info.getValue() + "\t" + info.getGenMVA());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_from_p");
            w.newLine();
            for (MeasureInfo info : line_from_p) {
                w.write(info.getPositionId() + "\t" + info.getValue());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_from_q");
            w.newLine();
            for (MeasureInfo info : line_from_q) {
                w.write(info.getPositionId() + "\t" + info.getValue());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_to_p");
            w.newLine();
            for (MeasureInfo info : line_to_p) {
                w.write(info.getPositionId() + "\t" + info.getValue());
                w.newLine();
            }
            w.write("-999");
            w.newLine();

            w.write("-------------------------------------------");
            w.newLine();
            w.write("line_to_q");
            w.newLine();
            for (MeasureInfo info : line_to_q) {
                w.write(info.getPositionId() + "\t" + info.getValue());
                w.newLine();
            }
            w.write("-999");
            w.close();
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
            log.error(e);
        }
    }

    public static void writeSimple(MeasureInfo[][] meases, Writer w) {
        List<MeasureInfo> bus_a = new ArrayList<MeasureInfo>();
        List<MeasureInfo> bus_v = new ArrayList<MeasureInfo>();
        List<MeasureInfo> bus_p = new ArrayList<MeasureInfo>();
        List<MeasureInfo> bus_q = new ArrayList<MeasureInfo>();
        List<MeasureInfo> line_from_p = new ArrayList<MeasureInfo>();
        List<MeasureInfo> line_from_q = new ArrayList<MeasureInfo>();
        List<MeasureInfo> line_to_p = new ArrayList<MeasureInfo>();
        List<MeasureInfo> line_to_q = new ArrayList<MeasureInfo>();
        for (MeasureInfo[] infos : meases) {
            for (MeasureInfo info : infos) {
                switch (info.getMeasureType()) {
                    case TYPE_BUS_ANGLE:
                        bus_a.add(info);
                        break;
                    case TYPE_BUS_VOLOTAGE:
                        bus_v.add(info);
                        break;
                    case TYPE_BUS_ACTIVE_POWER:
                        bus_p.add(info);
                        break;
                    case TYPE_BUS_REACTIVE_POWER:
                        bus_q.add(info);
                        break;
                    case TYPE_LINE_FROM_ACTIVE:
                        line_from_p.add(info);
                        break;
                    case TYPE_LINE_FROM_REACTIVE:
                        line_from_q.add(info);
                        break;
                    case TYPE_LINE_TO_ACTIVE:
                        line_to_p.add(info);
                        break;
                    case TYPE_LINE_TO_REACTIVE:
                        line_to_q.add(info);
                        break;
                    default:
                        log.warn("Not supported type: " + info.getMeasureType());
                }
            }
        }
        writeSimple(w, bus_a, bus_v, bus_p, bus_q, line_from_p, line_from_q, line_to_p, line_to_q);
    }

    public static void writeFileSimple(MeasureInfo[][] meases, String path) {
        try {
            writeSimple(meases, new FileWriter(path));
        } catch (IOException e) {
            log.warn("Can not write file of measurement information, file path:" + path);
        }
    }

    public static void writeFileSimple(SystemMeasure sysMeasure, String path) {
        try {
            writeSimple(sysMeasure, new FileWriter(path));
        } catch (IOException e) {
            log.warn("Can not write file of measurement information, file path:" + path);
        }
    }

    public static void writeFileFully(SystemMeasure sysMeasure, String path) {
        try {
            writeFully(sysMeasure, new FileWriter(path));
        } catch (IOException e) {
            log.warn("Can not write file of measurement information, file path:" + path);
        }
    }

    public static void writeFullyWithTrueValue(SystemMeasure sysMeasure, String path) {
        try {
            writeFullyWithTrueValue(sysMeasure, new FileWriter(path));
        } catch (IOException e) {
            log.warn("Can not write file of measurement information, file path:" + path);
        }
    }

    public static void writeMeasList(SystemMeasure sysMeasure, String filePath) {
        writeMeasList(sysMeasure, DEFAULT_TYPES, filePath);
    }

    public static void writeMeasList(SystemMeasure sysMeasure, int[] typeOrder, String filePath) {
        try {
            writeMeasList(sysMeasure, typeOrder, new FileWriter(filePath));
        } catch (IOException e) {
            log.warn("Failed to write measure to file: " + filePath);
            e.printStackTrace();
        }
    }

    public static void writeMeasList(SystemMeasure sysMeasure, int[] typeOrder, Writer writer) {
        for (int type : typeOrder) {
            Map<String, MeasureInfo> container = sysMeasure.getContainer(type);
            if (container == null)
                continue;
            BufferedWriter w = new BufferedWriter(writer);
            for (MeasureInfo meas : container.values()) {
                try {
                    w.write(String.format("%d", meas.getType()));//todo:not finished.
                } catch (IOException e) {
                    log.warn("Fail to write " + meas.getType() + "\t" + meas.getPositionId());
                    e.printStackTrace();
                }
            }
            try {
                writer.close();
                w.close();
            } catch (IOException e) {
                log.warn("IOException occured:" + e);
                e.printStackTrace();
            }
        }
    }

    public static void writeMeasure(MeasVector meas, Writer w) {
        BufferedWriter writer = new BufferedWriter(w);
        int index = 0;
        try {
            writer.write("================================================================");
            writer.newLine();
            writer.write("no.\tmeasure item\tmeasure value\ttrue value\terror\tsigma\tweight");
            writer.newLine();
            writer.write("================================================================");
            writer.newLine();
            for (int type : meas.getMeasureOrder()) {
                switch (type) {
                    case TYPE_BUS_ANGLE:
                        for (int i = 0; i < meas.getBus_a_pos().length; i++, index++) {
                            int num = meas.getBus_a_pos()[i];
                            double v = meas.getZ().getValue(index);
                            double v_true = meas.getZ_true().getValue(index);
                            String item = "\tbus angle(" + num + ")\t";
                            writer.write((index + 1) + item + format.format(v) + "\t" + format.format(v_true)
                                    + "\t" + format.format(v - v_true) + "\t" + format.format(meas.getSigma().getValue(index))
                                    + "\t" + format.format(meas.getWeight().getValue(index)));
                            writer.newLine();
                        }
                        break;
                    case TYPE_BUS_VOLOTAGE:
                        for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                            int num = meas.getBus_v_pos()[i];
                            double v = meas.getZ().getValue(index);
                            double v_true = meas.getZ_true().getValue(index);
                            String item = "\tbus voltage(" + num + ")\t";
                            writer.write((index + 1) + item + format.format(v) + "\t" + format.format(v_true)
                                    + "\t" + format.format(v - v_true) + "\t" + format.format(meas.getSigma().getValue(index))
                                    + "\t" + format.format(meas.getWeight().getValue(index)));
                            writer.newLine();

                        }
                        break;
                    case TYPE_BUS_ACTIVE_POWER:
                        for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                            int num = meas.getBus_p_pos()[i];
                            double v = meas.getZ().getValue(index);
                            double v_true = meas.getZ_true().getValue(index);
                            String item = "\tbus p(" + num + ")\t";
                            writer.write((index + 1) + item + format.format(v) + "\t" + format.format(v_true)
                                    + "\t" + format.format(v - v_true) + "\t" + format.format(meas.getSigma().getValue(index))
                                    + "\t" + format.format(meas.getWeight().getValue(index)));
                            writer.newLine();

                        }
                        break;
                    case TYPE_BUS_REACTIVE_POWER:
                        for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                            int num = meas.getBus_q_pos()[i];
                            double v = meas.getZ().getValue(index);
                            double v_true = meas.getZ_true().getValue(index);
                            String item = "\tbus q(" + num + ")\t";
                            writer.write((index + 1) + item + format.format(v) + "\t" + format.format(v_true)
                                    + "\t" + format.format(v - v_true) + "\t" + format.format(meas.getSigma().getValue(index))
                                    + "\t" + format.format(meas.getWeight().getValue(index)));
                            writer.newLine();
                        }
                        break;
                    case TYPE_LINE_FROM_ACTIVE:
                        for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                            int num = meas.getLine_from_p_pos()[k];
                            double v = meas.getZ().getValue(index);
                            double v_true = meas.getZ_true().getValue(index);
                            String item = "\tline from p(" + num + ")\t";
                            writer.write((index + 1) + item + format.format(v) + "\t" + format.format(v_true)
                                    + "\t" + format.format(v - v_true) + "\t" + format.format(meas.getSigma().getValue(index))
                                    + "\t" + format.format(meas.getWeight().getValue(index)));
                            writer.newLine();
                        }
                        break;
                    case TYPE_LINE_FROM_REACTIVE:
                        for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                            int num = meas.getLine_from_q_pos()[k];
                            double v = meas.getZ().getValue(index);
                            double v_true = meas.getZ_true().getValue(index);
                            String item = "\tline from q(" + num + ")\t";
                            writer.write((index + 1) + item + format.format(v) + "\t" + format.format(v_true)
                                    + "\t" + format.format(v - v_true) + "\t" + format.format(meas.getSigma().getValue(index))
                                    + "\t" + format.format(meas.getWeight().getValue(index)));
                            writer.newLine();
                        }
                        break;
                    case TYPE_LINE_TO_ACTIVE:
                        for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                            int num = meas.getLine_to_p_pos()[k];
                            double v = meas.getZ().getValue(index);
                            double v_true = meas.getZ_true().getValue(index);
                            String item = "\tline to p(" + num + ")\t";
                            writer.write((index + 1) + item + format.format(v) + "\t" + format.format(v_true)
                                    + "\t" + format.format(v - v_true) + "\t" + format.format(meas.getSigma().getValue(index))
                                    + "\t" + format.format(meas.getWeight().getValue(index)));
                            writer.newLine();
                        }
                        break;
                    case TYPE_LINE_TO_REACTIVE:
                        for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                            int num = meas.getLine_to_q_pos()[k];
                            double v = meas.getZ().getValue(index);
                            double v_true = meas.getZ_true().getValue(index);
                            String item = "\tline to q(" + num + ")\t";
                            writer.write((index + 1) + item + format.format(v) + "\t" + format.format(v_true)
                                    + "\t" + format.format(v - v_true) + "\t" + format.format(meas.getSigma().getValue(index))
                                    + "\t" + format.format(meas.getWeight().getValue(index)));
                            writer.newLine();
                        }
                        break;
                    case TYPE_LINE_CURRENT:
                        for (int k = 0; k < meas.getLine_i_amp_pos().length; k++, index++) {
                            int num = meas.getLine_i_amp_pos()[k];
                            double v = meas.getZ().getValue(index);
                            double v_true = meas.getZ_true().getValue(index);
                            String item = "\tline i(" + num + ")\t";
                            writer.write((index + 1) + item + format.format(v) + "\t" + format.format(v_true)
                                    + "\t" + format.format(v - v_true) + "\t" + format.format(meas.getSigma().getValue(index))
                                    + "\t" + format.format(meas.getWeight().getValue(index)));
                            writer.newLine();
                        }
                    default:
                        break;
                }//end switch
            }//end for
            writer.close();
            w.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
 