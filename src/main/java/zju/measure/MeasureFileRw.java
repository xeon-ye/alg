package zju.measure;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * this class provides methods to get system measurement information by parsing file
 *
 * @author Dong Shufeng
 *         Date: 2007-12-19
 */
@SuppressWarnings({"unchecked"})
public class MeasureFileRw implements MeasTypeCons {
    private static Logger log = LogManager.getLogger(MeasureFileRw.class);
    private static DecimalFormat format = new DecimalFormat("##.####");

    /**
     * @param in reader of source
     * @return system measurement information stored in class SystemMeasure
     */
    public static SystemMeasure parse(Reader in) {
        SystemMeasure meas = new SystemMeasure();
        BufferedReader reader = new BufferedReader(in);
        try {
            String str;
            while ((str = reader.readLine()) != null) {
                if (str.equals(""))
                    continue;
                String str2 = str.trim();
                if (str2.equalsIgnoreCase("bus_a")) {
                    setMeasValue(reader, meas, TYPE_BUS_ANGLE);
                } else if (str2.equalsIgnoreCase("bus_v")) {
                    setMeasValue(reader, meas, TYPE_BUS_VOLOTAGE);
                } else if (str2.equalsIgnoreCase("bus_p")) {
                    setMeasValue(reader, meas, TYPE_BUS_ACTIVE_POWER);
                } else if (str2.equalsIgnoreCase("bus_q")) {
                    setMeasValue(reader, meas, TYPE_BUS_REACTIVE_POWER);
                } else if (str2.equalsIgnoreCase("line_from_p")) {
                    setMeasValue(reader, meas, TYPE_LINE_FROM_ACTIVE);
                } else if (str2.equalsIgnoreCase("line_from_q")) {
                    setMeasValue(reader, meas, TYPE_LINE_FROM_REACTIVE);
                } else if (str2.equalsIgnoreCase("line_to_p")) {
                    setMeasValue(reader, meas, TYPE_LINE_TO_ACTIVE);
                } else if (str2.equalsIgnoreCase("line_to_q")) {
                    setMeasValue(reader, meas, TYPE_LINE_TO_REACTIVE);
                } else if (str2.equalsIgnoreCase("line_i_amp")) {
                    setMeasValue(reader, meas, TYPE_LINE_CURRENT);
                } else if (str2.equalsIgnoreCase("line_from_i_amp")) {
                    setMeasValue(reader, meas, TYPE_LINE_FROM_CURRENT);
                } else if (str2.equalsIgnoreCase("line_from_i_a")) {
                    setMeasValue(reader, meas, TYPE_LINE_FROM_CURRENT_ANGLE);
                } else if (str2.equalsIgnoreCase("line_to_i_amp")) {
                    setMeasValue(reader, meas, TYPE_LINE_TO_CURRENT);
                } else if (str2.equalsIgnoreCase("line_to_i_a")) {
                    setMeasValue(reader, meas, TYPE_LINE_TO_CURRENT_ANGLE);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
        return meas;
    }

    private static void setMeasValue(BufferedReader reader, SystemMeasure meas, int measType) throws IOException, IllegalAccessException, InvocationTargetException {
        Map<String, MeasureInfo> container = meas.getContainer(measType);
        ArrayList<MeasureInfo> list = new ArrayList<>();
        setMeasValue(reader, list, measType);
        for (MeasureInfo info : list)
            container.put(info.getPositionId(), info);
    }

    private static void setMeasValue(BufferedReader reader, List<MeasureInfo> container, int measType) throws IOException, IllegalAccessException, InvocationTargetException {
        List<String> content = new ArrayList<String>();
        String str;
        while (true) {
            str = reader.readLine();
            if (str.trim().equalsIgnoreCase("-999"))
                break;
            if (str.trim().equals(""))
                continue;
            content.add(str);
        }
        for (String aContent : content) {
            String[] s = aContent.split("\t");
            MeasureInfo measure = new MeasureInfo();
            measure.setPositionId(s[0]);
            measure.setValue(Double.parseDouble(s[1]));
            container.add(measure);
            if (s.length == 3 || s.length > 4)
                measure.setGenMVA(Double.parseDouble(s[s.length - 1]));
            if (s.length > 3) {
                measure.setSigma(Double.parseDouble(s[2]));
                measure.setWeight(Double.parseDouble(s[3]));
                if (s.length > 4)
                    measure.setValue_true(Double.parseDouble(s[4]));
            }
            measure.setMeasureType(measType);
        }
    }

    /**
     * parse a file and return system measurement infomation
     *
     * @param filePath absolute path of file
     * @return system measurement information
     */
    public static SystemMeasure parse(String filePath) {
        return parse(new File(filePath));
    }

    public static SystemMeasure parse(String filePath, String charset) {
        return parse(new File(filePath), charset);
    }

    /**
     * parse a file and return system measurement infomation
     *
     * @param file file to parse
     * @return system measurement information
     */
    public static SystemMeasure parse(File file) {
        try {
            return parse(new FileReader(file));
        } catch (FileNotFoundException e) {
            log.warn("解析量测数据时发生[找不到文件]错误:" + e.getMessage());
            return null;
        }
    }

    public static SystemMeasure parse(File file, String charset) {
        try {
            return parse(new FileInputStream(file), charset);
        } catch (FileNotFoundException e) {
            log.warn("解析量测数据时发生[找不到文件]错误:" + e.getMessage());
            return null;
        }
    }

    public static SystemMeasure parse(InputStream s, String charset) {
        try {
            return parse(new InputStreamReader(s, charset));
        } catch (UnsupportedEncodingException e) {
            log.warn("解析量测数据时发生[不支持的编码]错误:" + e.getMessage());
            return null;
        }
    }

    public static SystemMeasure parse(InputStream s) {
        return parse(new InputStreamReader(s));
    }

    public static List<MeasureInfo>[] parseToList(String path) {
        return parseToList(new File(path));
    }

    public static List<MeasureInfo>[] parseToList(String path, String charset) {
        return parseToList(new File(path), charset);
    }

    public static List<MeasureInfo>[] parseToList(File in) {
        try {
            return parseToList(new FileReader(in));
        } catch (FileNotFoundException e) {
            log.warn("解析量测数据时发生[找不到文件]错误:" + e.getMessage());
        }
        return null;
    }

    public static List<MeasureInfo>[] parseToList(File in, String charset) {
        try {
            return parseToList(new InputStreamReader(new FileInputStream(in), charset));
        } catch (FileNotFoundException e) {
            log.warn("解析量测数据时发生[找不到文件]错误:" + e.getMessage());
        } catch (UnsupportedEncodingException e) {
            log.warn("解析量测数据时发生[不支持的编码]错误:" + e.getMessage());
        }
        return null;
    }

    public static List<MeasureInfo>[] parseToList(Reader in) {
        BufferedReader reader = new BufferedReader(in);
        List<MeasureInfo>[] lists = new List[8];
        for (int i = 0; i < lists.length; i++)
            lists[i] = new ArrayList<MeasureInfo>();
        try {
            String str;
            while ((str = reader.readLine()) != null) {
                if (str.equals(""))
                    continue;
                String str2 = str.trim();
                if (str2.equalsIgnoreCase("bus_a")) {
                    setMeasValue(reader, lists[0], TYPE_BUS_ANGLE);
                } else if (str2.equalsIgnoreCase("bus_v")) {
                    setMeasValue(reader, lists[1], TYPE_BUS_VOLOTAGE);
                } else if (str2.equalsIgnoreCase("bus_p")) {
                    setMeasValue(reader, lists[2], TYPE_BUS_ACTIVE_POWER);
                } else if (str2.equalsIgnoreCase("bus_q")) {
                    setMeasValue(reader, lists[3], TYPE_BUS_REACTIVE_POWER);
                } else if (str2.equalsIgnoreCase("line_from_p")) {
                    setMeasValue(reader, lists[4], TYPE_LINE_FROM_ACTIVE);
                } else if (str2.equalsIgnoreCase("line_from_q")) {
                    setMeasValue(reader, lists[5], TYPE_LINE_FROM_REACTIVE);
                } else if (str2.equalsIgnoreCase("line_to_p")) {
                    setMeasValue(reader, lists[6], TYPE_LINE_TO_ACTIVE);
                } else if (str2.equalsIgnoreCase("line_to_q")) {
                    setMeasValue(reader, lists[7], TYPE_LINE_TO_REACTIVE);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
        return lists;
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
                            int num = meas.line_i_amp_pos[k];
                            double v = meas.z.getValue(index);
                            double v_true = meas.z_true.getValue(index);
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
