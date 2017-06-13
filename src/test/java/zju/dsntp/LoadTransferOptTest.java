package zju.dsntp;

import junit.framework.TestCase;
import org.junit.Assert;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsModelCons;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

import static zju.dsmodel.IeeeDsInHand.createDs;

/**
 * Created by xcs on 2016/10/5.
 */
public class LoadTransferOptTest extends TestCase implements DsModelCons {

    Map<String, Double> supplyCap = new HashMap<String, Double>();
    Map<String, Double> load = new HashMap<String, Double>();
    LoadTransferOptResult minSwitchResult;
    Map<String, Double> maxLoadResult;
    Map<String,Double> maxFeederLoad;
    Map<String,Double> maxCircuitLoad;

    public LoadTransferOptTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testCase1() {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase1/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("L2;L9"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L3;L4"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L7;L8"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"S1", "S2", "S3"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};
        String[] ErrorSupply = new String[]{"S1"};
        int[] ErrorEdge = {0};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase1/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase1/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase1/feederCapacity.txt").getPath();
        model.doOpt(loadsPath, supplyCapacityPath, feederCapacityPath);
//        Assert.assertEquals(3, model.minSwitch);
//
//        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase1/graphtest.txt");
//        testsys = createDs(ieeeFile, "S1", 100);
//        testsys.setSupplyCns(supplyID);
//        testsys.setSupplyCnBaseKv(supplyBaseKv);
//        model = new LoadTransferOpt(testsys);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L1");
//        Assert.assertEquals(10, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L2");
//        Assert.assertEquals(50, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L3");
//        Assert.assertEquals(30, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L4");
//        Assert.assertEquals(60, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L5");
//        Assert.assertEquals(10, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L6");
//        Assert.assertEquals(10, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L7");
//        Assert.assertEquals(10, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L8");
//        Assert.assertEquals(20, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L9");
//        Assert.assertEquals(80, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L10");
//        Assert.assertEquals(60, model.maxLoad, 0.001);
    }

    public void testCase2() {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase2/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("L1;L2"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L3;L1"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L2;L3"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"S1", "S2", "S3"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};
        String[] ErrorSupply = new String[]{"S1"};
        int[] ErrorEdge = {0};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase2/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase2/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase2/feederCapacity.txt").getPath();
        model.doOpt(loadsPath, supplyCapacityPath, feederCapacityPath);
        Assert.assertEquals(1, model.minSwitch);

        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase2/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new LoadTransferOpt(testsys);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L1");
        Assert.assertEquals(10, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L2");
        Assert.assertEquals(10, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L3");
        Assert.assertEquals(10, model.maxLoad, 0.001);
    }

    public void testAllLoadMax() {
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase1/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase1/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase1/feederCapacity.txt").getPath();
        DistriSys testsys;
        String[] supplyID = new String[]{"S1", "S2", "S3"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};

        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase1/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.allLoadMax(loadsPath, supplyCapacityPath, feederCapacityPath);
    }

    public void testCase3() {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase3/graphtest.txt");
        testsys = createDs(ieeeFile, "12", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("3;4"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("3;6"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("10;11"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"12", "13", "14", "15"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100.};
        String[] ErrorSupply = new String[]{"12"};
        int[] ErrorEdge = {0};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase3/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase3/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase3/feederCapacity.txt").getPath();
        model.doOpt(loadsPath, supplyCapacityPath, feederCapacityPath);
        Assert.assertEquals(2, model.minSwitch);

        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase3/graphtest.txt");
        testsys = createDs(ieeeFile, "12", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new LoadTransferOpt(testsys);
        model.allLoadMax(loadsPath, supplyCapacityPath, feederCapacityPath);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "6");
        Assert.assertEquals(8167, model.maxLoad, 0.001);
    }

    public void testCase4() {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase4/graphtest.txt");
        testsys = createDs(ieeeFile, "24", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("8;9"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("10;12"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("13;21"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("17;18"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("18;23"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"24", "25", "26", "27", "28"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100., 100.};
        String[] ErrorSupply = new String[]{"24"};
        int[] ErrorEdge = {0};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase4/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase4/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase4/feederCapacity.txt").getPath();
        model.doOpt(loadsPath, supplyCapacityPath, feederCapacityPath);
        Assert.assertEquals(3, model.minSwitch);

        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase4/graphtest.txt");
        testsys = createDs(ieeeFile, "24", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new LoadTransferOpt(testsys);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "1");
        Assert.assertEquals(7829, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "18");
        Assert.assertEquals(1409, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "19");
        Assert.assertEquals(2559, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "23");
        Assert.assertEquals(1639, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "7");
        Assert.assertEquals(8654, model.maxLoad, 0.001);
    }

    public void testCase4AllNode() {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase4/graphtest.txt");
        testsys = createDs(ieeeFile, "24", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("8;9"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("10;12"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("13;21"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("17;18"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("18;23"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"24", "25", "26", "27", "28"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100., 100.};
        String[] ErrorSupply = new String[]{"24"};
        int[] ErrorEdge = {0};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase4/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase4/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase4/feederCapacity.txt").getPath();
        model.doOpt(loadsPath, supplyCapacityPath, feederCapacityPath);
        Assert.assertEquals(3, model.minSwitch);

        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase4/graphtest.txt");
        testsys = createDs(ieeeFile, "24", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new LoadTransferOpt(testsys);
//        model.allLoadMax(loadsPath, supplyCapacityPath, feederCapacityPath);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "1");
        Assert.assertEquals(7829, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "20");
        Assert.assertEquals(1404, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "19");
        Assert.assertEquals(2559, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "23");
        Assert.assertEquals(1639, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "7");
        Assert.assertEquals(8654, model.maxLoad, 0.001);
    }

    public void testCase5() {
        DistriSys testsys;
        String[] supplyID;
//        for(int i = 1; i < 5; i++) {
//            long start = System.currentTimeMillis();
            InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase5/graphtest.txt");
            testsys = createDs(ieeeFile, "12", 100);
            for (MapObject obj : testsys.getDevices().getSwitches()) {
                if (obj.getProperty(KEY_CONNECTED_NODE).equals("3;4"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                else if (obj.getProperty(KEY_CONNECTED_NODE).equals("3;6"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                else if (obj.getProperty(KEY_CONNECTED_NODE).equals("10;11"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            }
            supplyID = new String[]{"12", "13", "14", "15"};
            Double[] supplyBaseKv = new Double[]{100., 200., 100., 100.};
            String[] ErrorSupply = new String[]{"13"};
            int[] ErrorEdge = {0};
            testsys.setSupplyCns(supplyID);
            testsys.setSupplyCnBaseKv(supplyBaseKv);

            LoadTransferOpt model = new LoadTransferOpt(testsys);
            model.setErrorFeeder(ErrorEdge);
            model.setErrorSupply(ErrorSupply);
            String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase5/loads.txt").getPath();
            String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase5/supplyCapacity.txt").getPath();
            String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase5/feederCapacity.txt").getPath();
            model.doOpt(loadsPath, supplyCapacityPath, feederCapacityPath);
//            System.out.println((System.currentTimeMillis() - start) + "ms");
//        }

//        Assert.assertEquals(2, model.minSwitch);
//        for(int i = 1; i < 5; i++) {
//            long start = System.currentTimeMillis();
//            ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase5/graphtest.txt");
//            testsys = createDs(ieeeFile, "12", 100);
//            testsys.setSupplyCns(supplyID);
//            testsys.setSupplyCnBaseKv(supplyBaseKv);
//            model = new LoadTransferOpt(testsys);
////            model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "1");
//        model.allLoadMax(loadsPath, supplyCapacityPath, feederCapacityPath);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "6");
//        Assert.assertEquals(8167, model.maxLoad, 0.001);
//            System.out.println((System.currentTimeMillis() - start) + "ms");
//        }
    }

    public void testCase6() {
        DistriSys testsys;
        String[] supplyID;
//        for(int i = 1; i < 5; i++) {
//            long start = System.currentTimeMillis();
            InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase6/graphtest.txt");
            testsys = createDs(ieeeFile, "24", 100);
            for (MapObject obj : testsys.getDevices().getSwitches()) {
                if (obj.getProperty(KEY_CONNECTED_NODE).equals("8;9"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                else if (obj.getProperty(KEY_CONNECTED_NODE).equals("10;12"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                else if (obj.getProperty(KEY_CONNECTED_NODE).equals("13;21"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                else if (obj.getProperty(KEY_CONNECTED_NODE).equals("17;18"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                else if (obj.getProperty(KEY_CONNECTED_NODE).equals("18;23"))
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            }
            supplyID = new String[]{"24", "25", "26", "27", "28"};
            Double[] supplyBaseKv = new Double[]{100., 200., 100., 100., 100.};
            String[] ErrorSupply = new String[]{"24"};
            int[] ErrorEdge = {0};
            testsys.setSupplyCns(supplyID);
            testsys.setSupplyCnBaseKv(supplyBaseKv);

            LoadTransferOpt model = new LoadTransferOpt(testsys);
            model.setErrorFeeder(ErrorEdge);
            model.setErrorSupply(ErrorSupply);
            String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase6/loads.txt").getPath();
            String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase6/supplyCapacity.txt").getPath();
            String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase6/feederCapacity.txt").getPath();
            model.doOpt(loadsPath, supplyCapacityPath, feederCapacityPath);
//            System.out.println((System.currentTimeMillis() - start) + "ms");
//        }

//    for(int i = 1; i < 5; i++) {
//        long start = System.currentTimeMillis();
        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase6/graphtest.txt");
        testsys = createDs(ieeeFile, "24", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new LoadTransferOpt(testsys);
        model.allLoadMax(loadsPath, supplyCapacityPath, feederCapacityPath);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "1");
//        Assert.assertEquals(7829, model.maxLoad, 0.001);
//        System.out.println((System.currentTimeMillis() - start) + "ms");
//    }
    }

    public void testCase7() {
        DistriSys testsys;
        String[] supplyID;
//        for(int i = 1; i < 5; i++) {
//            long start = System.currentTimeMillis();
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase7/graphtest.txt");
        testsys = createDs(ieeeFile, "12", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("3;4"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("3;6"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("10;11"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"12", "13", "14", "15"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100.};
        String[] ErrorSupply = new String[]{"15"};
        int[] ErrorEdge = {0};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase7/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase7/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase7/feederCapacity.txt").getPath();
        model.doOpt(loadsPath, supplyCapacityPath, feederCapacityPath);
//            System.out.println((System.currentTimeMillis() - start) + "ms");
//        }

//        Assert.assertEquals(2, model.minSwitch);
//        for(int i = 1; i < 5; i++) {
//            long start = System.currentTimeMillis();
            ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase7/graphtest.txt");
            testsys = createDs(ieeeFile, "12", 100);
            testsys.setSupplyCns(supplyID);
            testsys.setSupplyCnBaseKv(supplyBaseKv);
            model = new LoadTransferOpt(testsys);
//            model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "1");
        model.allLoadMax(loadsPath, supplyCapacityPath, feederCapacityPath);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "6");
//        Assert.assertEquals(8167, model.maxLoad, 0.001);
//            System.out.println((System.currentTimeMillis() - start) + "ms");
//        }
    }

    public void testCase8() {
        DistriSys testsys;
        String[] supplyID;
//        for(int i = 1; i < 5; i++) {
//            long start = System.currentTimeMillis();
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase8/graphtest.txt");
        testsys = createDs(ieeeFile, "24", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("8;9"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("10;12"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("13;21"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("17;18"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("18;23"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"24", "25", "26", "27", "28"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100., 100.};
        String[] ErrorSupply = new String[]{"26"};
        int[] ErrorEdge = {0};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase8/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase8/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase8/feederCapacity.txt").getPath();
        model.doOpt(loadsPath, supplyCapacityPath, feederCapacityPath);
//            System.out.println((System.currentTimeMillis() - start) + "ms");
//        }

//        for(int i = 1; i < 5; i++) {
//            long start = System.currentTimeMillis();
            ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase8/graphtest.txt");
            testsys = createDs(ieeeFile, "24", 100);
            testsys.setSupplyCns(supplyID);
            testsys.setSupplyCnBaseKv(supplyBaseKv);
            model = new LoadTransferOpt(testsys);
            model.allLoadMax(loadsPath, supplyCapacityPath, feederCapacityPath);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "1");
//        Assert.assertEquals(7829, model.maxLoad, 0.001);
//            System.out.println((System.currentTimeMillis() - start) + "ms");
//        }
    }

    public void testCase9() {
        DistriSys testsys;
        String[] supplyID;
//        for(int i = 1; i < 5; i++) {
//            long start = System.currentTimeMillis();
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase9/graphtest.txt");
        testsys = createDs(ieeeFile, "SRC_7", 100);
        supplyID = new String[]{"SRC_7"};
        Double[] supplyBaseKv = new Double[]{100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.buildPathes();
    }

    public void testCase1New() throws IOException {
        DistriSys testsys;
        String[] supplyID;

        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase1/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("L2;L9"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L3;L4"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L7;L8"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"S1", "S2", "S3"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};
        String[] ErrorSupply = new String[]{"S1"};
        int[] ErrorEdge = {0};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase1/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase1/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.doOptNew();
        Assert.assertEquals(1, model.minSwitch);

        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase1/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new LoadTransferOpt(testsys);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.loadMaxNew("L1");
//        Assert.assertEquals(10, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L2");
//        Assert.assertEquals(50, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L3");
//        Assert.assertEquals(30, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L4");
//        Assert.assertEquals(60, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L5");
//        Assert.assertEquals(10, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L6");
//        Assert.assertEquals(10, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L7");
//        Assert.assertEquals(10, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L8");
//        Assert.assertEquals(20, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L9");
//        Assert.assertEquals(80, model.maxLoad, 0.001);
//        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L10");
//        Assert.assertEquals(60, model.maxLoad, 0.001);
    }

    public void testCase2New() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase2/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("L1;L2"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L3;L1"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L2;L3"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"S1", "S2", "S3"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};
        String[] ErrorSupply = new String[]{"S1"};
        int[] ErrorEdge = {0};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase2/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase2/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase2/feederCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.doOptNew();
        Assert.assertEquals(1, model.minSwitch);

        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase2/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new LoadTransferOpt(testsys);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L1");
        Assert.assertEquals(10, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L2");
        Assert.assertEquals(10, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L3");
        Assert.assertEquals(10, model.maxLoad, 0.001);
    }

    public void testCase1All() throws IOException {
        DistriSys testsys;
        String[] supplyID;

        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase1/graphtestNew.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("L2;L9"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L3;L4"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L7;L8"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"S1", "S2", "S3"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase1/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase1/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            if(minSwitchResult.getSupplyId()[i] != null) {
                System.out.println(minSwitchResult.getSupplyId()[i]);
                System.out.println(minSwitchResult.getMinSwitch()[i]);
                for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                    System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            }
        }

        model.allLoadMaxNew();
//        model.allLoadMaxN();
        this.maxLoadResult = model.maxLoadResult;
        for(int i = 0; i < maxLoadResult.size(); i++) {
            System.out.println(model.nodes.get(i).getId());
            System.out.println(maxLoadResult.get(model.nodes.get(i).getId()));
        }
        this.maxCircuitLoad = model.getMaxCircuitLoad();
        for(int i = 0; i < model.edges.size(); i++) {
            System.out.println(model.edges.get(i).getId());
            System.out.println(maxCircuitLoad.get(model.edges.get(i).getId()));
        }
    }

    public void testCase2All() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase2/graphtestNew.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("L1;L2"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L3;L1"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L2;L3"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"S1", "S2", "S3"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase2/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase2/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
           // if(minSwitchResult.getSupplyId()[i] != null) {
                for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                    System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
         //   }
        }

        model.allLoadMaxNew();
//        model.allLoadMaxN();
        this.maxLoadResult = model.getMaxLoadResult();
        for(int i = 0; i < maxLoadResult.size(); i++) {
            System.out.println(model.nodes.get(i).getId());
            System.out.println(maxLoadResult.get(model.nodes.get(i).getId()));
        }
        this.maxCircuitLoad = model.getMaxCircuitLoad();
        for(int i = 0; i < model.edges.size(); i++) {
            System.out.println(model.edges.get(i).getId());
            System.out.println(maxCircuitLoad.get(model.edges.get(i).getId()));
        }
    }

    public void testCase3All() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase3/graphtestNew.txt");
        testsys = createDs(ieeeFile, "12", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("3;4"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("4;6"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("10;11"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"12", "13", "14", "15"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase3/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase3/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMaxNew();
//        model.allLoadMaxN();
        this.maxLoadResult = model.maxLoadResult;
        for(int i = 0; i < maxLoadResult.size(); i++) {
            System.out.println(model.nodes.get(i).getId());
            System.out.println(maxLoadResult.get(model.nodes.get(i).getId()));
        }
        this.maxCircuitLoad = model.getMaxCircuitLoad();
        for(int i = 0; i < model.edges.size(); i++) {
            System.out.println(model.edges.get(i).getId());
            System.out.println(maxCircuitLoad.get(model.edges.get(i).getId()));
        }
    }

    public void testCase4All() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase4/graphtestNew.txt");
        testsys = createDs(ieeeFile, "24", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("8;9"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("10;12"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("13;21"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("17;18"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("18;23"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"24", "25", "26", "27", "28"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase4/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase4/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMaxNew();
        this.maxLoadResult = model.maxLoadResult;
        for(int i = 0; i < maxLoadResult.size(); i++) {
            System.out.println(model.nodes.get(i).getId());
            System.out.println(maxLoadResult.get(model.nodes.get(i).getId()));
        }
        this.maxCircuitLoad = model.getMaxCircuitLoad();
        for(int i = 0; i < model.edges.size(); i++) {
            System.out.println(model.edges.get(i).getId());
            System.out.println(maxCircuitLoad.get(model.edges.get(i).getId()));
        }
    }

    public void testCase11() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase11/graphtestNew.txt");
        testsys = createDs(ieeeFile, "24", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("8;9"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("10;12"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("13;21"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("17;18"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("18;23"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"24", "25", "26", "27", "28"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase11/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase11/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(21032.76);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMaxNew();
        this.maxLoadResult = model.maxLoadResult;
        String writePath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\算例校验-5联络组\\输出.txt";
        exportFile(new File(writePath), model);
    }

    public void testCase12() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase12/graphtestNew.txt");
        testsys = createDs(ieeeFile, "26", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("8;12"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("12;13"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("17;18"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("15;22"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"26", "27", "28", "29", "30"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase12/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase12/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMaxNew();
        this.maxLoadResult = model.maxLoadResult;
        String writePath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\算例校验-5联络组\\输出.txt";
        exportFile(new File(writePath), model);
    }

    public void testCase13() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase13/graphtestNew.txt");
        testsys = createDs(ieeeFile, "26", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("8;12"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("12;13"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("17;18"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("15;22"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"26", "27", "28", "29", "30"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase13/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase13/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMaxNew();
        this.maxLoadResult = model.maxLoadResult;
        String writePath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\算例校验-5联络组\\输出.txt";
        exportFile(new File(writePath), model);
    }

    public void testCase14() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase14/graphtestNew.txt");
        testsys = createDs(ieeeFile, "26", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("8;12"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("12;13"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("17;18"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("15;22"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"26", "27", "28", "29", "30"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase14/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase14/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMaxNew();
        this.maxLoadResult = model.maxLoadResult;
        String writePath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\算例校验-5联络组\\输出.txt";
        exportFile(new File(writePath), model);
    }

    public void testCase15() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase15/graphtestNew.txt");
        testsys = createDs(ieeeFile, "26", 100);
        for (MapObject obj : testsys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("8;12"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("12;13"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("17;18"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("15;22"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
        }
        supplyID = new String[]{"26", "27", "28", "29", "30"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase15/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase15/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMaxNew();
        this.maxLoadResult = model.maxLoadResult;

        String writePath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\算例校验-5联络组\\输出.txt";
        exportFile(new File(writePath), model);
    }

    //读取各节点带的负载
    public void readLoads(String path) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        String data;
        String[] newdata;
        String cnId;
        Double cnLoad;

        while ((data = br.readLine()) != null) {
            newdata = data.split(" ", 2);
            cnId = newdata[0];
            cnLoad = new Double(Double.parseDouble(newdata[1]));
            load.put(cnId, cnLoad);
        }
    }

    //读取电源容量
    public void readSupplyCapacity(String path) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        String data;
        String[] newdata;
        String supplyId;
        Double supplyLoad;

        while((data = br.readLine()) != null) {
            newdata = data.split(" ", 2);
            supplyId = newdata[0];
            supplyLoad = new Double(Double.parseDouble(newdata[1]));
            supplyCap.put(supplyId, supplyLoad);
        }
    }

    /**
     * 导出
     *
     * @param file 文件(路径+文件名)，文件不存在会自动创建
     * @return
     */
    public boolean exportFile(File file, LoadTransferOpt model) {
        boolean isSucess = false;

        FileOutputStream out = null;
        OutputStreamWriter osw = null;
        BufferedWriter bw = null;
        try {
            out = new FileOutputStream(file);
            osw = new OutputStreamWriter(out);
            bw = new BufferedWriter(osw);
            if(!maxLoadResult.isEmpty()){
                for(int i = 0; i < maxLoadResult.size(); i++) {
                        bw.write(model.nodes.get(i).getId() + "\t");
                        bw.write(maxLoadResult.get(model.nodes.get(i).getId()) + "\t");
                        bw.newLine();
                }
            }
            isSucess = true;
        } catch (Exception e) {
            isSucess = false;
        } finally {
            if(bw != null) {
                try {
                    bw.close();
                    bw = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if(osw != null) {
                try {
                    osw.close();
                    osw = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if(out!=null) {
                try {
                    out.close();
                    out = null;
                } catch(IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return isSucess;
    }
}