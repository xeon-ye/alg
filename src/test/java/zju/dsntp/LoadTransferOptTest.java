package zju.dsntp;

import junit.framework.TestCase;
import org.jgrapht.UndirectedGraph;
import org.junit.Assert;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsConnectNode;
import zju.dsmodel.DsModelCons;

import java.io.*;
import java.util.*;

import static zju.dsmodel.IeeeDsInHand.createDs;

/**
 * Created by xcs on 2016/10/5.
 */
public class LoadTransferOptTest extends TestCase implements DsModelCons {

    Map<String, Double> supplyCap = new HashMap<String, Double>();
    Map<String, Double> load = new HashMap<>();
    Map<String, Double> feederCap = new HashMap<>();
    Map<String, Double> edgeCap = new HashMap<>();
    LoadTransferOptResult minSwitchResult;
    Map<String, Double> maxLoadResult;
    Map<String, Double> maxCircuitLoad;

    public LoadTransferOptTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testCase1() throws Exception {
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
        model.buildPathes();
        model.doOpt();
        Assert.assertEquals(1, model.minSwitch);

        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase1/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new LoadTransferOpt(testsys);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.loadMax("L1");
    }

    public void testCase12() throws Exception {
        DistriSys testsys;
        String[] supplyID;

        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase1/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        supplyID = new String[]{"S1", "S2", "S3"};
        testsys.setSupplyCns(supplyID);
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase1/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase1/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.setFeederCapacityConst(20000);
        model.calcTSC();
    }

    public void testCase2() throws IOException {
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
        model.doOpt();
        Assert.assertEquals(1, model.minSwitch);

        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase2/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new LoadTransferOpt(testsys);
        model.setFeederCapacityConst(20000);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.loadMax("L1");
    }

    public void testCase1All() throws Exception {
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
        UndirectedGraph<DsConnectNode, MapObject> g = testsys.getOrigGraph();
        for (MapObject e : g.edgeSet()) {
            e.setProperty(MapObject.KEY_ID, e.getProperty(KEY_CONNECTED_NODE));
        }

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase1/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase1/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase1/edgeCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        readEdgeCapacity(feederCapacityPath);
        model.setFeederCapacityConst(200);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.setFeederCap(feederCap);
        model.buildPathes();
        model.makeFeederCapArray();
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for (int i = 0; i < minSwitchResult.getFeederId().length; i++) {
            if (minSwitchResult.getFeederId()[i] != null) {
                System.out.println(minSwitchResult.getFeederId()[i]);
                System.out.println(minSwitchResult.getMinSwitch()[i]);
                for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                    System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            }
        }

        model.allLoadMax_1();
//        model.allLoadMaxN();
        this.maxLoadResult = model.maxLoadResult;
        for (int i = 0; i < maxLoadResult.size(); i++) {
            System.out.println(model.nodes.get(i).getId());
            System.out.println(maxLoadResult.get(model.nodes.get(i).getId()));
        }
        this.maxCircuitLoad = model.getMaxCircuitLoad();
        for (int i = 0; i < model.edges.size(); i++) {
            System.out.println(model.edges.get(i).getId());
            System.out.println(maxCircuitLoad.get(model.edges.get(i).getId()));
        }
    }

    public void testCase2All() throws Exception {
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
        UndirectedGraph<DsConnectNode, MapObject> g = testsys.getOrigGraph();
        for (MapObject e : g.edgeSet()) {
            e.setProperty(MapObject.KEY_ID, e.getProperty(KEY_CONNECTED_NODE));
        }

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase2/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase2/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase2/edgeCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        readEdgeCapacity(feederCapacityPath);
        model.setFeederCapacityConst(20);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.setFeederCap(feederCap);
        model.buildPathes();
        model.makeFeederCapArray();
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for (int i = 0; i < minSwitchResult.getFeederId().length; i++) {
            System.out.println(minSwitchResult.getFeederId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMax_1();
//        model.allLoadMaxN();
        this.maxLoadResult = model.getMaxLoadResult();
        for (int i = 0; i < maxLoadResult.size(); i++) {
            System.out.println(model.nodes.get(i).getId());
            System.out.println(maxLoadResult.get(model.nodes.get(i).getId()));
        }
        this.maxCircuitLoad = model.getMaxCircuitLoad();
        for (int i = 0; i < model.edges.size(); i++) {
            System.out.println(model.edges.get(i).getId());
            System.out.println(maxCircuitLoad.get(model.edges.get(i).getId()));
        }
    }

    public void testCase3All() throws Exception {
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
        UndirectedGraph<DsConnectNode, MapObject> g = testsys.getOrigGraph();
        for (MapObject e : g.edgeSet()) {
            e.setProperty(MapObject.KEY_ID, e.getProperty(KEY_CONNECTED_NODE));
        }

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase3/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase3/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase3/edgeCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        readEdgeCapacity(feederCapacityPath);
        model.setFeederCapacityConst(16302);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.setFeederCap(feederCap);
        model.buildPathes();
        model.makeFeederCapArray();
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for (int i = 0; i < minSwitchResult.getFeederId().length; i++) {
            System.out.println(minSwitchResult.getFeederId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMax_1();
//        model.allLoadMaxN();
        this.maxLoadResult = model.maxLoadResult;
        for (int i = 0; i < maxLoadResult.size(); i++) {
            System.out.println(model.nodes.get(i).getId());
            System.out.println(maxLoadResult.get(model.nodes.get(i).getId()));
        }
        this.maxCircuitLoad = model.getMaxCircuitLoad();
        for (int i = 0; i < model.edges.size(); i++) {
            System.out.println(model.edges.get(i).getId());
            System.out.println(maxCircuitLoad.get(model.edges.get(i).getId()));
        }
    }

    public void testCase4All() throws Exception {
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
        UndirectedGraph<DsConnectNode, MapObject> g = testsys.getOrigGraph();
        for (MapObject e : g.edgeSet()) {
            e.setProperty(MapObject.KEY_ID, e.getProperty(KEY_CONNECTED_NODE));
        }

        LoadTransferOpt model = new LoadTransferOpt(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase4/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase4/supplyCapacity.txt").getPath();
        String feederCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase4/edgeCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        readEdgeCapacity(feederCapacityPath);
        model.setFeederCapacityConst(19334);
        model.setLoad(load);
        model.setSupplyCap(supplyCap);
        model.setFeederCap(feederCap);
        model.buildPathes();
        model.makeFeederCapArray();


        //
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for (int i = 0; i < minSwitchResult.getFeederId().length; i++) {
            System.out.println(minSwitchResult.getFeederId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMax_1();
        this.maxLoadResult = model.maxLoadResult;
        for (int i = 0; i < maxLoadResult.size(); i++) {
            System.out.println(model.nodes.get(i).getId());
            System.out.println(maxLoadResult.get(model.nodes.get(i).getId()));
        }
        this.maxCircuitLoad = model.getMaxCircuitLoad();
        for (int i = 0; i < model.edges.size(); i++) {
            System.out.println(model.edges.get(i).getId());
            System.out.println(maxCircuitLoad.get(model.edges.get(i).getId()));
        }
    }

    public void testCase5() throws Exception {
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
        //model.checkN1();
    }

    public void testCase6() throws IOException {
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
        for (int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMax();
        this.maxLoadResult = model.maxLoadResult;
        String writePath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\算例校验-5联络组\\输出.txt";
        exportFile(new File(writePath), model);
    }

    public void testCase7() throws IOException {
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
        for (int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMax();
        this.maxLoadResult = model.maxLoadResult;
        String writePath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\算例校验-5联络组\\输出.txt";
        exportFile(new File(writePath), model);
    }

    public void testCase8() throws IOException {
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
        for (int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMax();
        this.maxLoadResult = model.maxLoadResult;
        String writePath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\算例校验-5联络组\\输出.txt";
        exportFile(new File(writePath), model);
    }

    public void testCase9() throws IOException {
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
        for (int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMax();
        this.maxLoadResult = model.maxLoadResult;
        String writePath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\算例校验-5联络组\\输出.txt";
        exportFile(new File(writePath), model);
    }

    public void testCase10() throws IOException {
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
        for (int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            // if(minSwitchResult.getSupplyId()[i] != null) {
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
            //   }
        }

        model.allLoadMax();
        this.maxLoadResult = model.maxLoadResult;

        String writePath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\算例校验-5联络组\\输出.txt";
        exportFile(new File(writePath), model);
    }

    public void testCase17() throws Exception {
        //生成系统
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase17/graphtest.txt");
        DistriSys distriSys = createDs(ieeeFile, "S1", 100);
        //设置电源节点，电源名
        String[] supplyID = new String[]{"S1", "S2", "S3", "S4", "S5", "S6"};
        distriSys.setSupplyCns(supplyID);
        //设置电源基准电压
        Double[] supplyBaseKv = new Double[]{100., 100., 100., 100., 100., 100.,};
        distriSys.setSupplyCnBaseKv(supplyBaseKv);

        //新建计算模型
        LoadTransferOpt loadTransferOpt = new LoadTransferOpt(distriSys);

        //设置电源容量
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase17/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        loadTransferOpt.setSupplyCap(supplyCap);

        //设置线路容量
        loadTransferOpt.setFeederCapacityConst(20000);

        //搜索路径
        loadTransferOpt.buildPathes();

        System.out.println("打印路径");
//        loadTransferOpt.printPathes(loadTransferOpt.getPathes());

        //读取负荷
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase17/loads.txt").getPath();
        readLoads(loadsPath);
        loadTransferOpt.setLoad(load);

        //N-1校验
        loadTransferOpt.checkN1();
    }

    /**
     * 灾后供电恢复程序测试
     * @throws Exception
     */
    public void testCase18() throws Exception {
        //生成系统
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase18/graphtest.txt");
        DistriSys distriSys = createDs(ieeeFile, "S1", 100);
        //设置电源节点，电源名
        String[] supplyID = new String[]{"S1", "S2"};
        distriSys.setSupplyCns(supplyID);
        //设置电源基准电压
        Double[] supplyBaseKv = new Double[]{100., 100.};
        distriSys.setSupplyCnBaseKv(supplyBaseKv);

        //新建计算模型
        LoadTransferOpt loadTransferOpt = new LoadTransferOpt(distriSys);

        //设置电源容量
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase18/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        loadTransferOpt.setSupplyCap(supplyCap);

        //设置线路容量
        loadTransferOpt.setFeederCapacityConst(20000);

        //搜索路径
        loadTransferOpt.buildPathes();

        System.out.println("打印路径");
        loadTransferOpt.printPathes(loadTransferOpt.getPathes());

        //读取负荷
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase18/loads.txt").getPath();
        readLoads(loadsPath);

        loadTransferOpt.setLoad(load);

        List<String> impLoadList = new LinkedList<>();
        impLoadList.add("A");
        //N-1校验
        loadTransferOpt.restoration(impLoadList);
    }

    public void testCase19() throws Exception {
        //生成系统
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase19/graphtest.txt");
        DistriSys distriSys = createDs(ieeeFile, "1", 100);
        //设置电源节点，电源名
        String[] supplyID = new String[]{"1"};
        distriSys.setSupplyCns(supplyID);
        //设置电源基准电压
        Double[] supplyBaseKv = new Double[]{100., 100.};
        distriSys.setSupplyCnBaseKv(supplyBaseKv);

        //新建计算模型
        LoadTransferOpt loadTransferOpt = new LoadTransferOpt(distriSys);

        //设置电源容量
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase19/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        loadTransferOpt.setSupplyCap(supplyCap);

        //设置支路容量
        String edgeCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase19/edgecapacity.txt").getPath();
        readEdgeCapacity(edgeCapacityPath);
        loadTransferOpt.setEdgeCap(edgeCap);

        //设置线路容量
        loadTransferOpt.setFeederCapacityConst(20000);

        //搜索路径
        loadTransferOpt.buildPathes();

        System.out.println("打印路径");
        loadTransferOpt.printPathes(loadTransferOpt.getPathes());

        //读取负荷
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase19/loads.txt").getPath();
        readLoads(loadsPath);

        loadTransferOpt.setLoad(load);

        List<String> impLoadList = new LinkedList<>();
        //设置重要负荷
//        impLoadList.add("7");impLoadList.add("8");impLoadList.add("24");impLoadList.add("25");impLoadList.add("30");
//        impLoadList.add("32");impLoadList.add("12");
        //N-1校验
        loadTransferOpt.restoration(impLoadList);
    }

    public void testLoadBalanceCase() throws Exception {
        //生成系统
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/loadbalancecase/graphtest.txt");
        DistriSys distriSys = createDs(ieeeFile, "S1", 100);
        //设置电源节点，电源名
        String[] supplyID = new String[]{"S1", "S2", "S3", "S4", "S5", "S6"};
        distriSys.setSupplyCns(supplyID);
        //设置电源基准电压
        Double[] supplyBaseKv = new Double[]{100., 100., 100., 100., 100., 100.,};
        distriSys.setSupplyCnBaseKv(supplyBaseKv);

        for (MapObject obj : distriSys.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("L21;L23"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L22;L30"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L25;L29"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L26;L31"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L27;L28"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L1;L2"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L3;L4"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("L5;L6"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);

        }

        //新建计算模型
        LoadBalance loadBalance = new LoadBalance(distriSys);

        //设置电源容量
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/loadbalancecase/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        loadBalance.setSupplyCap(supplyCap);

        //设置线路容量
        loadBalance.setFeederCapacityConst(20000);

        //搜索路径
        loadBalance.buildPathes();

        System.out.println("打印路径");
//        loadTransferOpt.printPathes(loadTransferOpt.getPathes());

        //读取负荷
        String loadsPath = this.getClass().getResource("/loadtransferfiles/loadbalancecase/loads.txt").getPath();
        readLoads(loadsPath);
        loadBalance.setLoad(load);

        //N-1校验
        loadBalance.calculate();
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
            cnLoad = Double.parseDouble(newdata[1]);
            load.put(cnId, cnLoad);
        }
    }

    //读取电源容量
    public void readSupplyCapacity(String path) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        String data;
        String[] newdata;
        String supplyId;
        double supplyLoad;

        while ((data = br.readLine()) != null) {
            newdata = data.split(" ", 2);
            supplyId = newdata[0];
            supplyLoad = Double.parseDouble(newdata[1]);
            supplyCap.put(supplyId, supplyLoad);
        }
    }

    //读取馈线容量
    public void readFeederCapacity(String path) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        String data;
        String[] newdata;
        String feederId;
        double feederCapacity;

        while((data = br.readLine()) != null) {
            newdata = data.split(" ", 3);
            feederId = newdata[0] + ";" + newdata[1];
            feederCapacity = Double.parseDouble(newdata[2]);
            feederCap.put(feederId, feederCapacity);
        }
    }

    //读取馈线容量
    public void readEdgeCapacity(String path) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        String data;
        String[] newdata;
        while((data = br.readLine()) != null) {
            newdata = data.split(" ", 2);
            edgeCap.put(newdata[0], Double.parseDouble(newdata[1])*1.732*12.66);
        }
    }

    /**
     * 导出
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
            if (!maxLoadResult.isEmpty()) {
                for (int i = 0; i < maxLoadResult.size(); i++) {
                    bw.write(model.nodes.get(i).getId() + "\t");
                    bw.write(maxLoadResult.get(model.nodes.get(i).getId()) + "\t");
                    bw.newLine();
                }
            }
            isSucess = true;
        } catch (Exception e) {
            isSucess = false;
        } finally {
            if (bw != null) {
                try {
                    bw.close();
                    bw = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (osw != null) {
                try {
                    osw.close();
                    osw = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (out != null) {
                try {
                    out.close();
                    out = null;
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return isSucess;
    }
}