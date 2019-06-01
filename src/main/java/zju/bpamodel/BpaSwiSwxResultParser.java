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
public class BpaSwiSwxResultParser {
    private static Logger log = LogManager.getLogger(BpaSwiSwxResultParser.class);

    public static SwiSwxResult parseFile(String file) {
        return parseFile(new File(file));
    }

    public static SwiSwxResult parseFile(String file, String charset) {
        return parseFile(new File(file), charset);
    }

    public static SwiSwxResult parseFile(File file, String charset) {
        try {
            return parse(new FileInputStream(file), charset);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static SwiSwxResult parseFile(File file) {
        try {
            return parse(new FileInputStream(file));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static SwiSwxResult parse(InputStream in, String charset) {
        try {
            return parse(new InputStreamReader(in, charset));
        } catch (UnsupportedEncodingException e) {
            log.warn(e);
            e.printStackTrace();
        }
        return null;
    }

    public static SwiSwxResult parse(InputStream in) {
        return parse(new InputStreamReader(in));
    }

    public static SwiSwxResult parse(Reader in) {
        try {
            BufferedReader reader = new BufferedReader(in);
            SwiSwxResult r = new SwiSwxResult();
            r.setGeneratorDataList(new LinkedList<>());
            r.setBusDataList(new LinkedList<>());
            r.setLineDataList(new LinkedList<>());
            String strLine;
            while ((strLine = reader.readLine()) != null) {
                if (strLine.startsWith(" * 发电机")) {
                    GeneratorData generatorData = new GeneratorData();
                    byte[] strLineBytes = strLine.getBytes("GBK");
                    generatorData.setBusName1(new String(BpaFileRwUtil.getTarget(strLineBytes, 10, 18), "GBK").trim());
                    generatorData.setBaseKv1(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(strLineBytes, 19, 25)).trim()));
                    generatorData.setBusName2(new String(BpaFileRwUtil.getTarget(strLineBytes, 46, 54), "GBK").trim());
                    generatorData.setBaseKv2(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(strLineBytes, 55, 61)).trim()));
                    generatorData.setGenOneStepDataList(new LinkedList<>());
                    for (int i = 0; i < 3; i++) {
                        reader.readLine();
                    }
                    while ((strLine = reader.readLine()) != null) {
                        if (!strLine.trim().isEmpty()) {
                            String charset = "GBK";
                            byte[] src = strLine.getBytes("GBK");
                            GenOneStepData genOneStepData = new GenOneStepData();
                            genOneStepData.setTime(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 0, 7)).trim()));
                            genOneStepData.setRelativeAngle(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 12, 20)).trim()));
                            genOneStepData.setFreqDeviation(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 33)).trim()));
                            genOneStepData.setFieldVoltage(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 39, 46)).trim()));
                            genOneStepData.setMechPower(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 50, 59)).trim()));
                            genOneStepData.setElecPower(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 63, 72)).trim()));
                            genOneStepData.setRegulatorOutput(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 78, 85)).trim()));
                            genOneStepData.setReactivePower(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 89, 98)).trim()));
                            genOneStepData.setFieldCurrent(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 104, 111)).trim()));
                            generatorData.getGenOneStepDataList().add(genOneStepData);
                        } else {
                            r.getGeneratorDataList().add(generatorData);
                            break;
                        }
                    }
                }
                if (strLine.startsWith("    * 节点")) {
                    BusData busData = new BusData();
                    byte[] strLineBytes = strLine.getBytes("GBK");
                    busData.setBusName(new String(BpaFileRwUtil.getTarget(strLineBytes, 11, 19), "GBK").trim());
                    busData.setBaseKv(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(strLineBytes, 19, 25)).trim()));
                    busData.setBusOneStepDataList(new LinkedList<>());
                    for (int i = 0; i < 3; i++) {
                        reader.readLine();
                    }
                    while ((strLine = reader.readLine()) != null) {
                        if (!strLine.trim().isEmpty()) {
                            String charset = "GBK";
                            byte[] src = strLine.getBytes("GBK");
                            BusOneStepData busOneStepData = new BusOneStepData();
                            busOneStepData.setTime(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 0, 10)).trim()));
                            busOneStepData.setPosSeqVol(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 14, 21)).trim()));
                            busOneStepData.setFreqDeviation(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 32)).trim()));
                            busData.getBusOneStepDataList().add(busOneStepData);
                        } else {
                            r.getBusDataList().add(busData);
                            break;
                        }
                    }
                }
                if (strLine.startsWith(" * 线路")) {
                    LineData lineData = new LineData();
                    byte[] strLineBytes = strLine.getBytes("GBK");
                    lineData.setBusName1(new String(BpaFileRwUtil.getTarget(strLineBytes, 8, 16), "GBK").trim());
                    lineData.setBaseKv1(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(strLineBytes, 17, 22)).trim()));
                    lineData.setBusName2(new String(BpaFileRwUtil.getTarget(strLineBytes, 23, 31), "GBK").trim());
                    lineData.setBaseKv2(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(strLineBytes, 32, 38)).trim()));
                    lineData.setLineOneStepDataList(new LinkedList<>());
                    for (int i = 0; i < 3; i++) {
                        reader.readLine();
                    }
                    while ((strLine = reader.readLine()) != null) {
                        if (!strLine.trim().isEmpty()) {
                            String charset = "GBK";
                            byte[] src = strLine.getBytes("GBK");
                            LineOneStepData lineOneStepData = new LineOneStepData();
                            lineOneStepData.setTime(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 0, 7)).trim()));
                            lineOneStepData.setActivePower(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 20)).trim()));
                            lineOneStepData.setReactivePower(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 24, 33)).trim()));
                            lineData.getLineOneStepDataList().add(lineOneStepData);
                        } else {
                            r.getLineDataList().add(lineData);
                            break;
                        }
                    }
                }
            }
            return r;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
