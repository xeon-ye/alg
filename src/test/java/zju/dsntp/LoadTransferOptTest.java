package zju.dsntp;

import junit.framework.TestCase;
import org.junit.Assert;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsModelCons;

import java.io.InputStream;

import static zju.dsmodel.IeeeDsInHand.createDs;

/**
 * Created by xcs on 2016/10/5.
 */
public class LoadTransferOptTest extends TestCase implements DsModelCons {

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
        Assert.assertEquals(3, model.minSwitch);

        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase1/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new LoadTransferOpt(testsys);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L1");
        Assert.assertEquals(10, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L2");
        Assert.assertEquals(50, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L3");
        Assert.assertEquals(30, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L4");
        Assert.assertEquals(60, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L5");
        Assert.assertEquals(10, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L6");
        Assert.assertEquals(10, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L7");
        Assert.assertEquals(10, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L8");
        Assert.assertEquals(20, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L9");
        Assert.assertEquals(80, model.maxLoad, 0.001);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath, "L10");
        Assert.assertEquals(60, model.maxLoad, 0.001);
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
}