package zju.dsmodel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.devmodel.MapObject;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 * Date: 2010-7-9
 */
public class DsDeviceParser implements DsModelCons {
    private static Logger log = LogManager.getLogger(DsDeviceParser.class);

    public DsDevices parse(String f) {
        return parse(new File(f));
    }

    public DsDevices parse(File f) {
        try {
            return parse(new FileInputStream(f));
        } catch (FileNotFoundException e) {
            log.warn(e);
            return null;
        }
    }

    public DsDevices parse(InputStream stream) {//todo: not finished
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String strLine;
        try {
            strLine = reader.readLine();
            while (!strLine.startsWith("Line Segment Data"))
                strLine = reader.readLine();
            String[] strings = strLine.split("\t");
            String s = strings[1];
            //取出数值，数值即为拓扑支路数量
            ArrayList<MapObject> branches = new ArrayList<MapObject>(Integer.parseInt(s.substring(0, s.indexOf(" "))));
            while (true) {
                strLine = reader.readLine();
                log.debug(strLine);
                if (strLine.trim().equalsIgnoreCase("-999"))
                    break;
                //空行跳过。#开头行为注释行。
                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;
                MapObject obj = parseBranch(strLine, strings[2]);
                branches.add(obj);
            }

            while (!strLine.startsWith("Spot Loads"))
                strLine = reader.readLine();
            s = strLine.split("\t")[1];
            ArrayList<MapObject> spotLoads = new ArrayList<MapObject>(Integer.parseInt(s.substring(0, s.indexOf(" "))));
            while (true) {
                strLine = reader.readLine();
                log.debug(strLine);
                if (strLine.trim().equalsIgnoreCase("-999"))
                    break;
                if (strLine.trim().equals("") || strLine.startsWith("#")) //info Mode,you can disable elements with"//"
                    continue;
                MapObject obj = parseSpotLoad(strLine);
                spotLoads.add(obj);
            }

            while (!strLine.startsWith("Distributed Loads"))
                strLine = reader.readLine();
            s = strLine.split("\t")[1];
            ArrayList<MapObject> distributedLoads = new ArrayList<MapObject>(Integer.parseInt(s.substring(0, s.indexOf(" "))));
            while (true) {
                strLine = reader.readLine();
                log.debug(strLine);
                if (strLine.trim().equalsIgnoreCase("-999"))
                    break;
                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;
                MapObject obj = parseDisLoad(strLine);
                distributedLoads.add(obj);
            }

            while (!strLine.startsWith("Shunt Capacitors"))
                strLine = reader.readLine();
            s = strLine.split("\t")[1];
            ArrayList<MapObject> shuntCapacitors = new ArrayList<MapObject>(Integer.parseInt(s.substring(0, s.indexOf(" "))));
            while (true) {
                strLine = reader.readLine();
                log.debug(strLine);
                if (strLine.trim().equalsIgnoreCase("-999"))
                    break;
                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;
                MapObject obj = parseShuntCapacitor(strLine);
                shuntCapacitors.add(obj);
            }

            //变压器计数
            int tfCount = 0;
            //开关计数
            int switchCount = 0;
            //配置为Switch的branch为开关；否则，长度为0的边为变压器。
            for (MapObject obj : branches) {
                String length = obj.getProperty(KEY_LINE_LENGTH);
                if (obj.getProperty(KEY_LINE_CONF).trim().equalsIgnoreCase("Switch")) {
                    switchCount++;
                } else if (Double.parseDouble(length) < 1e-5) {
                    tfCount++;
                }
            }
            List<MapObject> transformers = new ArrayList<MapObject>(tfCount);
            List<MapObject> switches = new ArrayList<MapObject>(switchCount);
            for (MapObject obj : branches) {
                String length = obj.getProperty(KEY_LINE_LENGTH);
                if (obj.getProperty(KEY_LINE_CONF).trim().equalsIgnoreCase("Switch")) {
                    log.debug("Feeder config = Switch its a Switch");
                    //将类型Feeder改为类型Switch
                    obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_SWITCH);
                    switches.add(obj);
                } else if (Double.parseDouble(length) < 1e-5) {
                    log.debug("Feeder length = " + length + ", its a transformer");
                    //将类型Feeder改为类型Transformer
                    obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
                    transformers.add(obj);
                }
            }

            while (!strLine.startsWith("Transformer"))
                strLine = reader.readLine();
            while (true) {
                strLine = reader.readLine();
                log.debug(strLine);
                if (strLine.trim().equalsIgnoreCase("-999"))
                    break;
                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;
                setTfParameters(strLine, transformers);
            }

            while (!strLine.startsWith("Regulator"))
                strLine = reader.readLine();
            s = strLine.split("\t")[1];
            ArrayList<MapObject> regulators = new ArrayList<MapObject>(Integer.parseInt(s.substring(0, s.indexOf(" "))));
            while (true) {
                strLine = reader.readLine();
                log.debug(strLine);
                if (strLine.trim().equalsIgnoreCase("-999"))
                    break;
                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;
                MapObject obj = parseRegulator(strLine);
                regulators.add(obj);
            }

            ArrayList<MapObject> dgs = new ArrayList<MapObject>(Integer.parseInt(s.substring(0, s.indexOf(" "))));
            while (strLine != null && !strLine.startsWith("Distributed Generation"))
                strLine = reader.readLine();
            if (strLine != null) {
                while (true) {
                    strLine = reader.readLine();
                    log.debug(strLine);
                    if (strLine.trim().equalsIgnoreCase("-999"))
                        break;
                    if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                        continue;
                    MapObject obj = parseDg(strLine);
                    if (obj != null)
                        dgs.add(obj);
                }
            }

            branches.removeAll(transformers);
            branches.removeAll(switches);
            DsDevices island = new DsDevices();
            island.setSpotLoads(spotLoads);
            island.setDistributedLoads(distributedLoads);
            island.setShuntCapacitors(shuntCapacitors);
            island.setFeeders(branches);
            island.setTransformers(transformers);
            island.setSwitches(switches);
            island.setRegulators(regulators);
            island.setDispersedGens(dgs);
            return island;
        } catch (Exception e) {
            log.error("IO exception occured when trying to parse ieee data file!");
            return null;
        }
    }

    public void setTfParameters(String strLine, List<MapObject> transformers) {
        String content[] = strLine.split("\t");
        for (MapObject transformer : transformers) {//todo: not efficient
            if (content[0].equals(transformer.getProperty(KEY_LINE_CONF))) {
                setTfParameters(strLine, transformer);
                break;
            }
        }
    }

    public void setTfParameters(String strLine, MapObject transformer) {
        String content[] = strLine.split("\t");
        transformer.setProperty(KEY_KVA_S_TF, content[1]);
        transformer.setProperty(KEY_CONN_TYPE, content[2]);
        transformer.setProperty(KEY_KV_HIGH, content[3]);
        transformer.setProperty(KEY_KV_LOW, content[4]);
        transformer.setProperty(KEY_PU_R_TF, content[5]);
        transformer.setProperty(KEY_PU_X_TF, content[6]);
    }

    public MapObject parseDg(String strLine) {
        String content[] = strLine.split("\t");
        MapObject obj = new MapObject();
        obj.setProperty(KEY_CONNECTED_NODE, content[0]);
        obj.setProperty(KEY_DG_MODEL, content[1]);
        obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_DG);
        obj.setProperty(KEY_CONN_TYPE, content[2]);
        if (content[1].equals(DG_MODEL_PV)) {
            obj.setProperty(KEY_PV_P_PH1, content[3]);
            obj.setProperty(KEY_PV_P_PH2, content[4]);
            obj.setProperty(KEY_PV_P_PH3, content[5]);
            obj.setProperty(KEY_PV_V_PH1, content[6]);
            obj.setProperty(KEY_PV_V_PH2, content[7]);
            obj.setProperty(KEY_PV_V_PH3, content[8]);
        } else if (content[1].equals(DG_MODEL_IM)) {
            //Node Type   Connection  BaseKva   BaseKv   PowerOut(kW)    Rs(pu)  Xs(pu)  Rr(pu)  Xr(pu)  Xm(pu)
            obj.setProperty(KEY_IM_BASE_KVA, content[3]);
            obj.setProperty(KEY_IM_BASE_KV, content[4]);
            obj.setProperty(KEY_IM_P_OUT, content[5]);
            obj.setProperty(KEY_IM_PU_RS, content[6]);
            obj.setProperty(KEY_IM_PU_XS, content[7]);
            obj.setProperty(KEY_IM_PU_RR, content[8]);
            obj.setProperty(KEY_IM_PU_XR, content[9]);
            obj.setProperty(KEY_IM_PU_XM, content[10]);
            obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_DG);
        } else {
            log.warn("Not supported distribution generation model: " + content[2]);
            return null;
        }
        return obj;
    }

    public MapObject parseRegulator(String strLine) {
        String content[] = strLine.split("\t");
        MapObject obj = new MapObject();
        obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_REGULATOR);
        obj.setProperty(MapObject.KEY_ID, content[0]);
        obj.setProperty(KEY_CONNECTED_NODE, content[1] + ";" + content[2]);
        obj.setProperty(KEY_LOCATION, content[3]);
        obj.setProperty(KEY_PHASES, content[4]);
        obj.setProperty(KEY_CONN_TYPE, content[5]);
        obj.setProperty(KEY_MONITORING_PHASES, content[6]);
        obj.setProperty(KEY_BANDWIDTH, content[7]);
        obj.setProperty(KEY_RG_PT_NUM, content[8]);
        obj.setProperty(KEY_RG_CT_NUM, content[9]);
        obj.setProperty(KEY_RA_RG, content[10]);
        obj.setProperty(KEY_XA_RG, content[11]);
        obj.setProperty(KEY_RB_RG, content[12]);
        obj.setProperty(KEY_XB_RG, content[13]);
        obj.setProperty(KEY_RC_RG, content[14]);
        obj.setProperty(KEY_XC_RG, content[15]);
        obj.setProperty(KEY_VLA_RG, content[16]);
        obj.setProperty(KEY_VLB_RG, content[17]);
        obj.setProperty(KEY_VLC_RG, content[18]);
        return obj;
    }

    public MapObject parseShuntCapacitor(String strLine) {
        String content[] = strLine.split("\t");
        MapObject obj = new MapObject();
        obj.setProperty(KEY_CONNECTED_NODE, content[0]);
        obj.setProperty(KEY_LOAD_MODEL, LOAD_Y_Z);
        obj.setProperty(KEY_KVAR_PH1, content[1]);
        obj.setProperty(KEY_KW_PH1, "0.0");
        obj.setProperty(KEY_KVAR_PH2, content[2]);
        obj.setProperty(KEY_KW_PH2, "0.0");
        obj.setProperty(KEY_KVAR_PH3, content[3]);
        obj.setProperty(KEY_KW_PH3, "0.0");
        obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_SHUNT_CAPACITORS);
        return obj;
    }

    public MapObject parseDisLoad(String strLine) {
        String content[] = strLine.split("\t");
        MapObject obj = new MapObject();
        obj.setProperty(KEY_CONNECTED_NODE, content[0] + ";" + content[1]);
        obj.setProperty(KEY_LOAD_MODEL, content[2]);
        obj.setProperty(KEY_KW_PH1, content[3]);
        obj.setProperty(KEY_KVAR_PH1, content[4]);
        obj.setProperty(KEY_KW_PH2, content[5]);
        obj.setProperty(KEY_KVAR_PH2, content[6]);
        obj.setProperty(KEY_KW_PH3, content[7]);
        obj.setProperty(KEY_KVAR_PH3, content[8]);
        obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_DIS_LOAD);
        return obj;
    }

    public MapObject parseSpotLoad(String strLine) {
        String content[] = strLine.split("\t");
        MapObject obj = new MapObject();
        obj.setProperty(KEY_CONNECTED_NODE, content[0]);
        obj.setProperty(KEY_LOAD_MODEL, content[1]);
        obj.setProperty(KEY_KW_PH1, content[2]);
        obj.setProperty(KEY_KVAR_PH1, content[3]);
        obj.setProperty(KEY_KW_PH2, content[4]);
        obj.setProperty(KEY_KVAR_PH2, content[5]);
        obj.setProperty(KEY_KW_PH3, content[6]);
        obj.setProperty(KEY_KVAR_PH3, content[7]);
        obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_SPOT_LOAD);
        return obj;
    }

    public MapObject parseBranch(String strLine, String lenthUnit) {
        String content[] = strLine.split("\t");
        //System.out.println(strLine);
        MapObject obj = new MapObject();
        obj.setProperty(KEY_CONNECTED_NODE, content[0] + ";" + content[1]);
        obj.setProperty(KEY_LINE_LENGTH, content[2]);
        obj.setProperty(KEY_LINE_CONF, content[3]);
        obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_FEEDER);
        obj.setProperty(KEY_LENGTH_UNIT, lenthUnit);
        if (content[3].trim().equalsIgnoreCase("Switch")) {
            //设置开关状态
            if (content.length == 5)
                obj.setProperty(KEY_SWITCH_STATUS, content[4]);
        }
        return obj;
    }
}
