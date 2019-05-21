package zju.bpamodel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.bpamodel.swir.*;

import java.io.*;
import java.util.LinkedList;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-5-21
 */
public class BpaSwiResultParser {
    private static Logger log = LogManager.getLogger(BpaSwiResultParser.class);

    public static SwiResult parseFile(String file) {
        return parseFile(new File(file));
    }

    public static SwiResult parseFile(String file, String charset) {
        return parseFile(new File(file), charset);
    }

    public static SwiResult parseFile(File file, String charset) {
        try {
            return parse(new FileInputStream(file), charset);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static SwiResult parseFile(File file) {
        try {
            return parse(new FileInputStream(file));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static SwiResult parse(InputStream in, String charset) {
        try {
            return parse(new InputStreamReader(in, charset));
        } catch (UnsupportedEncodingException e) {
            log.warn(e);
            e.printStackTrace();
        }
        return null;
    }

    public static SwiResult parse(InputStream in) {
        return parse(new InputStreamReader(in));
    }

    public static SwiResult parse(Reader in) {
        try {
            BufferedReader reader = new BufferedReader(in);
            SwiResult r = new SwiResult();
            String strLine;
            while ((strLine = reader.readLine()) != null) {
                strLine = strLine.trim();
                if (strLine.startsWith("* 计算过程中的监视曲线数据列表")) {
                    for (int i = 0; i < 3; i++) {
                        reader.readLine();
                    }
                    break;
                }
            }
            r.setMonitorDataList(new LinkedList<>());
            while ((strLine = reader.readLine()) != null) {
                if (!strLine.trim().isEmpty()) {
                    String charset = "GBK";
                    byte[] src = strLine.getBytes("GBK");
                    double time = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 0, 7)).trim());
                    RelativeAngle relativeAngle = new RelativeAngle();
                    relativeAngle.setBusName1(new String(BpaFileRwUtil.getTarget(src, 11, 19), charset).trim());
                    relativeAngle.setBaseKv1(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 20, 25)).trim()));
                    relativeAngle.setBusName2(new String(BpaFileRwUtil.getTarget(src, 26, 34), charset).trim());
                    relativeAngle.setBaseKv2(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 35, 41)).trim()));
                    relativeAngle.setRelativeAngle(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 45, 52)).trim()));
                    BusVoltage minBusVoltage = new BusVoltage();
                    minBusVoltage.setName(new String(BpaFileRwUtil.getTarget(src, 55, 64), charset).trim());
                    minBusVoltage.setBaseKv(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 65, 70)).trim()));
                    minBusVoltage.setvInPu(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 72, 78)).trim()));
                    BusVoltage maxBusVoltage = new BusVoltage();
                    maxBusVoltage.setName(new String(BpaFileRwUtil.getTarget(src, 81, 91), charset).trim());
                    maxBusVoltage.setBaseKv(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 92, 97)).trim()));
                    maxBusVoltage.setvInPu(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 99, 105)).trim()));
                    BusFreq minBusFreq = new BusFreq();
                    minBusFreq.setName(new String(BpaFileRwUtil.getTarget(src, 110, 118), charset).trim());
                    minBusFreq.setBaseKv(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 119, 124)).trim()));
                    minBusFreq.setFreq(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 127, 133)).trim()));
                    BusFreq maxBusFreq = new BusFreq();
                    maxBusFreq.setName(new String(BpaFileRwUtil.getTarget(src, 137, 145), charset).trim());
                    maxBusFreq.setBaseKv(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 146, 151)).trim()));
                    maxBusFreq.setFreq(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 155, 161)).trim()));
                    r.getMonitorDataList().add(new MonitorData(time, relativeAngle, minBusVoltage, maxBusVoltage, minBusFreq, maxBusFreq));
                } else {
                    break;
                }
            }
            return r;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
