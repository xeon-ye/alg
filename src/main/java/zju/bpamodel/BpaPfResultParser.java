package zju.bpamodel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.bpamodel.pfr.BusPfResult;
import zju.bpamodel.pfr.PfResult;

import java.io.*;
import java.util.HashMap;

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
            r.setBusData(new HashMap<String, BusPfResult>());
            double dToAFactor = Math.PI / 180.0;
            while ((strLine = reader.readLine()) != null) {
                strLine = strLine.trim();
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
                //log.debug(strLine);
            }

            return r;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
