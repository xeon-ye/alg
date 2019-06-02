package zju.bpamodel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.bpamodel.pf.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-12
 */
public class BpaPfModelParser {
    private static Logger log = LogManager.getLogger(BpaPfModelParser.class);

    public static ElectricIsland parseFile(String file) {
        return parseFile(new File(file));
    }

    public static ElectricIsland parseFile(String file, String charset) {
        return parseFile(new File(file), charset);
    }

    public static ElectricIsland parseFile(File file, String charset) {
        try {
            return parse(new FileInputStream(file), charset);
        } catch (FileNotFoundException e) {
            log.warn(e);
            e.printStackTrace();
            return null;
        }
    }

    public static ElectricIsland parseFile(File file) {
        try {
            return parse(new FileInputStream(file));
        } catch (FileNotFoundException e) {
            log.warn(e);
            e.printStackTrace();
            return null;
        }
    }

    public static ElectricIsland parse(InputStream in) {
        return parse(new InputStreamReader(in));
    }

    public static ElectricIsland parse(InputStream in, String charset) {
        try {
            return parse(new InputStreamReader(in, charset));
        } catch (UnsupportedEncodingException e) {
            log.warn(e);
            e.printStackTrace();
        }
        return null;
    }

    public static ElectricIsland parse(Reader in) {
        try {
            BufferedReader reader = new BufferedReader(in);
            ElectricIsland model = new ElectricIsland();
            String strLine;
            List<PowerExchange> powerExchanges = new ArrayList<>();
            List<Bus> buses = new ArrayList<>();
            List<AcLine> aclines = new ArrayList<>();
            List<Transformer> transformers = new ArrayList<>();
            HashMap<String, DataModifyInfo> zone2dataModifyInfoMap = new HashMap<>();

            while ((strLine = reader.readLine()) != null) {
                try {
                    if (strLine.startsWith("B")) {
                        if (strLine.charAt(1) == 'D') {
                            //todo: not finished.
                        } else if (strLine.charAt(1) == 'M') {
                            //todo: not finished.
                        } else
                            buses.add(Bus.createBus(strLine));
                    } else if (strLine.startsWith("L")) {
                        if (strLine.charAt(1) == 'D') {
                            //todo: not finished.
                        } else if (strLine.charAt(1) == 'M') {
                            //todo: not finished.
                        } else if (strLine.charAt(1) == '+') {
                            //todo: not finished.
                        } else
                            aclines.add(AcLine.createAcLine(strLine));
                    } else if (strLine.startsWith("T")) {
                        transformers.add(Transformer.createTransformer(strLine));
                    } else if (strLine.startsWith("P")) {
                        if (strLine.charAt(1) == 'A') {
                            //todo: not finished.
                        } else if (strLine.charAt(1) == 'Z') {
                            DataModifyInfo dataModifyInfo = DataModifyInfo.createDataModifyInfo(strLine);
                            zone2dataModifyInfoMap.put(dataModifyInfo.getZone(),dataModifyInfo);
                        } else if (strLine.charAt(1) == 'O') {
                            //todo: not finished.
                        } else if (strLine.charAt(1) == 'C') {
                            //todo: not finished.
                        } else if (strLine.charAt(1) == 'B') {
                            //todo: not finished.
                        }
                    } else if (strLine.startsWith("A") || strLine.startsWith("I")) {
                        powerExchanges.add(PowerExchange.createPowerExchange(strLine));
                    }
                } catch (NumberFormatException ex) {
                    //ex.printStackTrace();
                    log.warn("Failed to parse because NumberFormatException is found:");
                    log.warn(strLine);
                } catch (NegativeArraySizeException ex) {
                    //ex.printStackTrace();
                    log.warn("Failed to parse because NegativeArraySizeException is found:");
                    log.warn(strLine);
                }
            }
            model.setPowerExchanges(powerExchanges);
            model.setBuses(buses);
            model.setAclines(aclines);
            model.setTransformers(transformers);
            model.modifyData(zone2dataModifyInfoMap);
            model.buildTopo();
            return model;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}

