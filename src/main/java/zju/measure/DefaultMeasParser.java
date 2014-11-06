package zju.measure;

import org.apache.log4j.Logger;

import java.io.*;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * this class provides methods to get system measurement information by parsing file
 *
 * @author Dong Shufeng
 *         Date: 2007-12-19
 */
@SuppressWarnings({"unchecked"})
public class DefaultMeasParser implements MeasTypeCons {
    private static Logger log = Logger.getLogger(DefaultMeasParser.class);

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
        ArrayList<MeasureInfo> list = new ArrayList<MeasureInfo>();
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
}
