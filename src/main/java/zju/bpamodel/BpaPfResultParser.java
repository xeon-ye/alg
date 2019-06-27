package zju.bpamodel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.bpamodel.pfr.BranchPfResult;
import zju.bpamodel.pfr.BusPfResult;
import zju.bpamodel.pfr.PfResult;
import zju.bpamodel.pfr.TransformerPfResult;

import java.io.*;
import java.util.HashMap;

import static java.lang.Math.abs;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-14
 */
public class BpaPfResultParser {
    private static Logger log = LogManager.getLogger(BpaPfResultParser.class);

    public static PfResult parseFile(String file) {
        return parseFile(new File(file));
    }

    public static PfResult parseFile(String file, String charset) {
        return parseFile(new File(file), charset);
    }

    public static PfResult parseFile(File file, String charset) {
        try {
            return parse(new FileInputStream(file), charset);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static PfResult parseFile(File file) {
        try {
            return parse(new FileInputStream(file));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static PfResult parse(InputStream in, String charset) {
        try {
            return parse(new InputStreamReader(in, charset));
        } catch (UnsupportedEncodingException e) {
            log.warn(e);
            e.printStackTrace();
        }
        return null;
    }

    public static PfResult parse(InputStream in) {
        return parse(new InputStreamReader(in));
    }

    public static PfResult parse(Reader in) {
        try {
            BufferedReader reader = new BufferedReader(in);
            PfResult r = new PfResult();
            String strLine;
            r.setConverged(false);
            while ((strLine = reader.readLine()) != null) {
                if (strLine.trim().contains("计算结果收敛。")) {
                    r.setConverged(true);
                    break;
                }
            }
            if (!r.isConverged())
                return r;
            r.setBusData(new HashMap<>());
            r.setBranchData(new HashMap<>());
            r.setTransformerData(new HashMap<>());
            double dToAFactor = Math.PI / 180.0;
            while ((strLine = reader.readLine()) != null) {
                strLine = strLine.trim();
                if (strLine.startsWith("整个系统的数据总结"))
                    break;
                //its not a bus, but not sure the condition judgement is perfect
                if (!(strLine.endsWith(" B") || strLine.endsWith(" BQ")))
                    continue;
                BusPfResult busR = new BusPfResult();
                int index = strLine.indexOf(" ");
                String currentData = strLine.substring(0, index);
                busR.setName(currentData);
                String s = strLine.substring(index).trim();

                index = s.indexOf(" ");
                currentData = s.substring(0, index);
                s = s.substring(index).trim();
                busR.setBaseKv(Double.parseDouble(currentData));

                index = s.indexOf("/");
                currentData = s.substring(0, index);
                s = s.substring(index + 1).trim();
                busR.setvInKv(Double.parseDouble(currentData.substring(0, currentData.indexOf("kV"))));

                index = s.indexOf("度");
                currentData = s.substring(0, index);
                s = s.substring(index).trim();
                busR.setAngleInDegree(Double.parseDouble(currentData));
                busR.setAngleInArc(busR.getAngleInDegree() * dToAFactor);

                index = s.indexOf(" ");
                currentData = s.substring(0, index);
                s = s.substring(index).trim();
                busR.setArea(currentData);

                while ((index = s.indexOf(" ")) != -1) {
                    currentData = s.substring(0, index);
                    s = s.substring(index).trim();
                    if (currentData.contains("有功出力")) {
                        currentData = currentData.substring(0, currentData.indexOf("有功出力"));
                        busR.setGenP(Double.parseDouble(currentData));
                    } else if (currentData.contains("无功出力")) {
                        currentData = currentData.substring(0, currentData.indexOf("无功出力"));
                        busR.setGenQ(Double.parseDouble(currentData));
                    } else if (currentData.contains("有功负荷")) {
                        currentData = currentData.substring(0, currentData.indexOf("有功负荷"));
                        busR.setLoadP(Double.parseDouble(currentData));
                    } else if (currentData.contains("无功负荷")) {
                        currentData = currentData.substring(0, currentData.indexOf("无功负荷"));
                        busR.setLoadQ(Double.parseDouble(currentData));
                    } else if (currentData.contains("电压pu")) {
                        currentData = currentData.substring(0, currentData.indexOf("电压pu"));
                        busR.setvInPu(Double.parseDouble(currentData));
                    }
                }
                r.getBusData().put(busR.getName(), busR);

                while ((strLine = reader.readLine()) != null) {
                    String charset = "GBK";
                    byte[] src = strLine.getBytes("GBK");
                    if (! new String(BpaFileRwUtil.getTarget(src, 46, 50)).trim().equals("PNET")) {
                        double baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 21)).trim());
                        if (abs(baseKv - busR.getBaseKv()) < 1e-3) {
                            BranchPfResult branchR = new BranchPfResult();
                            branchR.setBranchName(busR.getName() + ";" + new String(BpaFileRwUtil.getTarget(src, 7, 15), charset).trim());
                            branchR.setBusName1(busR.getName());
                            branchR.setBusName2(new String(BpaFileRwUtil.getTarget(src, 7, 15), charset).trim());
                            branchR.setBaseKv1(busR.getBaseKv());
                            branchR.setBaseKv2(baseKv);
                            branchR.setCircuit((char) src[24]);
                            branchR.setZoneName(new String(BpaFileRwUtil.getTarget(src, 34, 36)).trim());
                            branchR.setBranchP(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 37, 44)).trim()));
                            branchR.setBranchQ(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 53, 60)).trim()));
                            branchR.setBranchPLoss(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 69, 76)).trim()));
                            branchR.setBranchQLoss(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 85, 92)).trim()));
                            branchR.setChargeP(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 106, 112)).trim()));
                            r.getBranchData().put(branchR.getBranchName(), branchR);
                        } else {
                            TransformerPfResult transformerR = new TransformerPfResult();
                            transformerR.setTransformerName(busR.getName() + ";" + new String(BpaFileRwUtil.getTarget(src, 7, 15), charset).trim());
                            transformerR.setBusName1(busR.getName());
                            transformerR.setBusName2(new String(BpaFileRwUtil.getTarget(src, 7, 15), charset).trim());
                            transformerR.setBaseKv1(busR.getBaseKv());
                            transformerR.setBaseKv2(baseKv);
                            transformerR.setCircuit((char) src[24]);
                            transformerR.setZoneName(new String(BpaFileRwUtil.getTarget(src, 34, 36)).trim());
                            transformerR.setTransformerP(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 37, 44)).trim()));
                            transformerR.setTransformerQ(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 53, 60)).trim()));
                            transformerR.setTransformerPLoss(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 69, 76)).trim()));
                            transformerR.setTransformerQLoss(BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 85, 92)).trim()));
                            r.getTransformerData().put(transformerR.getTransformerName(), transformerR);
                        }
                    } else {
                        break;
                    }
                }
                //log.debug(strLine);
            }
            while ((strLine = reader.readLine()) != null) {
                if (strLine.startsWith("*  低电压和过电压节点数据列表")) {
                    for (int i = 0; i < 3; i++) {
                        reader.readLine();
                    }
                    while ((strLine = reader.readLine()) != null) {
                        if (!strLine.isEmpty()) {
                            String charset = "GBK";
                            byte[] src = strLine.getBytes("GBK");
                            r.getBusData().get(new String(BpaFileRwUtil.getTarget(src, 17, 25), charset).trim()).setVoltageLimit(true);
                        } else {
                            break;
                        }
                    }
                }
                if (strLine.startsWith("*  线路负载超过额定值")) {
                    for (int i = 0; i < 3; i++) {
                        reader.readLine();
                    }
                    while ((strLine = reader.readLine()) != null) {
                        if (!strLine.isEmpty()) {
                            String charset = "GBK";
                            byte[] src = strLine.getBytes("GBK");
                            r.getBranchData().get(new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim()
                                    + ";" + new String(BpaFileRwUtil.getTarget(src, 18, 26), charset).trim()).setOverLoad(true);
                            r.getBranchData().get(new String(BpaFileRwUtil.getTarget(src, 18, 26), charset).trim()
                                    + ";" + new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim()).setOverLoad(true);
                        } else {
                            break;
                        }
                    }
                }
                if (strLine.startsWith("*  变压器负载超过额定值")) {
                    for (int i = 0; i < 3; i++) {
                        reader.readLine();
                    }
                    while ((strLine = reader.readLine()) != null) {
                        if (!strLine.isEmpty()) {
                            String charset = "GBK";
                            byte[] src = strLine.getBytes("GBK");
                            r.getTransformerData().get(new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim()
                                    + ";" + new String(BpaFileRwUtil.getTarget(src, 18, 26), charset).trim()).setOverLoad(true);
                            r.getTransformerData().get(new String(BpaFileRwUtil.getTarget(src, 18, 26), charset).trim()
                                    + ";" + new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim()).setOverLoad(true);
                        } else {
                            break;
                        }
                    }
                }
            }
            reader.close();
            in.close();
            return r;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
