package zju.dsmodel;

import zju.devmodel.MapObject;

import java.io.IOException;
import java.io.InputStream;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 * Date: 2011-4-21
 */

public class IeeeDsInHand implements DsModelCons {
    private static final FeederConfMgr FEEDER_CONF = new FeederConfMgr();
    private static final FeederConfMgr FEEDER_CONF_CASE8500 = new FeederConfMgr();

    public final static DistriSys FEEDER4_DD_B;
    public final static DistriSys FEEDER4_DD_UNB;
    public final static DistriSys FEEDER4_GrYGrY_B;
    public final static DistriSys FEEDER4_GrYGrY_UNB;
    public final static DistriSys FEEDER4_DGrY_B;
    public final static DistriSys FEEDER4_DGrY_UNB;
    //public final static DistriSys FEEDER4_OPenGrYD_B;
    //public final static DistriSys FEEDER4_OPenGrYD_UNB;
    //public final static RadicalIsland RD_CASE4_DY_B;
    //public final static RadicalIsland RD_CASE4_DY_UNB;

    public final static DistriSys FEEDER13;
    public final static DistriSys FEEDER34;
    public final static DistriSys FEEDER37;
    public final static DistriSys FEEDER69;
    public final static DistriSys FEEDER123;
    public final static DistriSys FEEDER123x50;
    public final static DistriSys FEEDER8500;

    static {
        try {
            FEEDER_CONF.readImpedanceConf(IeeeDsInHand.class.getResourceAsStream("/dsieee/common/feederconfig.txt"));
            FEEDER_CONF_CASE8500.readImpedanceConf(IeeeDsInHand.class.getResourceAsStream("/dsieee/common/feederconfig8500.txt"));
        } catch (Exception e) {
            e.printStackTrace();
        }

        InputStream ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case4/case4_D-D_Bload.txt");
        FEEDER4_DD_B = createDs(ieeeFile, "1", 12.47 / sqrt3);

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case4/case4_D-D_UnBload.txt");
        FEEDER4_DD_UNB = createDs(ieeeFile, "1", 12.47 / sqrt3);

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case4/case4_Gr.Y-Gr.Y_Bload.txt");
        FEEDER4_GrYGrY_B = createDs(ieeeFile, "1", 12.47 / sqrt3);

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case4/case4_Gr.Y-Gr.Y_UnBload.txt");
        FEEDER4_GrYGrY_UNB = createDs(ieeeFile, "1", 12.47 / sqrt3);

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case4/case4_D-Gr.Y_Bload.txt");
        FEEDER4_DGrY_B = createDs(ieeeFile, "1", 12.47 / sqrt3);

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case4/case4_D-Gr.Y_UnBload.txt");
        FEEDER4_DGrY_UNB = createDs(ieeeFile, "1", 12.47 / sqrt3);

        //InputStream ieeeFile6 = IeeeDsInHand.class.getResourceAsStream("/dsieee/case4/case4_OpenGr.Y-D_Bload.txt");
        //FEEDER4_OPenGrYD_B = createDs(ieeeFile6, "1", 12.47);
        //
        //InputStream ieeeFile7 = IeeeDsInHand.class.getResourceAsStream("/dsieee/case4/case4_OpenGr.Y-D_UnBload.txt");
        //FEEDER4_OPenGrYD_UNB = createDs(ieeeFile7, "1", 12.47);

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case13/case13.txt");
        FEEDER13 = createDs(ieeeFile, "650", 4.16 / sqrt3);

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case34/case34.txt");
        FEEDER34 = createDs(ieeeFile, "800", 24.9 / sqrt3);

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case37/case37.txt");
        FEEDER37 = createDs(ieeeFile, "799", 4.8 / sqrt3);

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case69/case69.txt");
        FEEDER69 = createDs(ieeeFile, "0", 12.66 / sqrt3);

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case123/case123.txt");
        FEEDER123 = createDs(ieeeFile, "150", 4.16 / sqrt3);
        for (MapObject obj : FEEDER123.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("250;251"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("450;451"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("54;94"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("151;300"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("300;350"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/case_connection/case123x50.txt");
        FEEDER123x50 = createDs(ieeeFile, "150", 4.16 / sqrt3);
        for (MapObject obj : FEEDER123x50.getDevices().getSwitches()) {
            for (int i = 0; i < 50; i++) {
                String prefix = (i + 1) + "-";
                if (obj.getProperty(KEY_CONNECTED_NODE).equals(prefix + "250;" + prefix + "251"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                else if (obj.getProperty(KEY_CONNECTED_NODE).equals(prefix + "450;" + prefix + "451"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                else if (obj.getProperty(KEY_CONNECTED_NODE).equals(prefix + "54;" + prefix + "94"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                else if (obj.getProperty(KEY_CONNECTED_NODE).equals(prefix + "151;" + prefix + "300"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                else if (obj.getProperty(KEY_CONNECTED_NODE).equals(prefix + "300;" + prefix + "350"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);

            }
        }

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/8500node/case8500.txt");
        FEEDER8500 = createDs8500(ieeeFile, "Source", 115 * 1.05 / sqrt3);
        for (MapObject obj : FEEDER8500.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("228-979371-2_INT--193-48013;193-48013"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("228-961799-3_INT--193-46661;193-46661"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("228-1353934-4_INT--193-103041;193-103041"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("D5837361-8_INT--E182745;E182745"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("228-1048090-1_INT--193-51796;193-51796"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            // 8500节点算例中D5860423-3_INT节点与regxfmr_190-8593相连，该处有调压器，由于未处理调压器，计算配电网末端电压较低，因此断开
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("D5587291-3_INT--Q14734;Q14734"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
    }

    public static DistriSys createDs(InputStream ieeeFile, String slackCnId, double baseKv) {
        DsDevices devices = new DsDeviceParser().parse(ieeeFile);
        DistriSys dsTopo = new DistriSys();
        dsTopo.buildOrigTopo(devices);

        dsTopo.setSupplyCns(new String[]{slackCnId});
        dsTopo.setSupplyCnBaseKv(new Double[]{baseKv});
        dsTopo.setFeederConf(FEEDER_CONF);
        dsTopo.fillCnBaseKv();

        for (MapObject obj : devices.getSwitches())
            obj.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
        return dsTopo;
    }

    public static DistriSys createDs(DsDevices devices, String slackCnId, double baseKv) {
        DistriSys dsTopo = new DistriSys();
        dsTopo.buildOrigTopo(devices);

        dsTopo.setSupplyCns(new String[]{slackCnId});
        dsTopo.setSupplyCnBaseKv(new Double[]{baseKv});
        dsTopo.setFeederConf(FEEDER_CONF);
        dsTopo.fillCnBaseKv();

        for (MapObject obj : devices.getSwitches())
            obj.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
        return dsTopo;
    }

    public static DistriSys createDs8500(InputStream ieeeFile, String slackCnId, double baseKv) {
        DsDevices devices = new DsDeviceParser().parse(ieeeFile);
        DistriSys dsTopo = new DistriSys();
        dsTopo.buildOrigTopo(devices);

        dsTopo.setSupplyCns(new String[]{slackCnId});
        dsTopo.setSupplyCnBaseKv(new Double[]{baseKv});
        dsTopo.setFeederConf(FEEDER_CONF_CASE8500);
        dsTopo.fillCnBaseKv();

        for (MapObject obj : devices.getSwitches())
            obj.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
        return dsTopo;
    }
}
