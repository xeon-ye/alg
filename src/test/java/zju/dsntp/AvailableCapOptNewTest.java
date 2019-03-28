package zju.dsntp;

import junit.framework.TestCase;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

import static zju.dsmodel.DsModelCons.KEY_CONNECTED_NODE;
import static zju.dsmodel.DsModelCons.KEY_SWITCH_STATUS;
import static zju.dsmodel.DsModelCons.SWITCH_OFF;
import static zju.dsmodel.IeeeDsInHand.createDs;

public class AvailableCapOptNewTest extends TestCase {
    Map<String, Double> supplyCap = new HashMap<String, Double>();
    Map<String, Double> loads = new HashMap<String, Double>();
    LoadTransferOptResult minSwitchResult;
    Map<String, Double> maxLoadResult;

    public void testCase1() throws IOException {
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

        AvailableCapOptNew model = new AvailableCapOptNew(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase1/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase1/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoads(loads);
        model.setSupplyCap(supplyCap);
        model.buildLoops();
        model.doOpt();
//        Assert.assertEquals(1, model.minSwitch);

//        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/case1/graphtest.txt");
//        testsys = createDs(ieeeFile, "S1", 100);
//        testsys.setSupplyCns(supplyID);
//        testsys.setSupplyCnBaseKv(supplyBaseKv);
//        model = new AvailableCapOpt(testsys);
//        model.setFeederCapacityConst(20000);
//        model.setLoads(loads);
//        model.setSupplyCap(supplyCap);
//        model.buildLoops();
//        model.loadMax("L1");
//        System.out.println(model.maxLoad);
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

        AvailableCapOptNew model = new AvailableCapOptNew(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase2/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase2/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoads(loads);
        model.setSupplyCap(supplyCap);
        model.buildLoops();
//        model.doOpt();
//        Assert.assertEquals(1, model.minSwitch);

        ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase2/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new AvailableCapOptNew(testsys);
        model.setFeederCapacityConst(20000);
        model.setLoads(loads);
        model.setSupplyCap(supplyCap);
        model.buildLoops();
        model.loadMax("L1");
        System.out.println(model.maxLoad);
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

        AvailableCapOptNew model = new AvailableCapOptNew(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase1/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase1/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoads(loads);
        model.setSupplyCap(supplyCap);
        model.buildLoops();
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            if(minSwitchResult.getSupplyId()[i] != null) {
                System.out.println(minSwitchResult.getSupplyId()[i] + "\t" + minSwitchResult.getMinSwitch()[i]);
                for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++) {
                    System.out.print(minSwitchResult.getSwitchChanged().get(i)[j]);
                }
                System.out.println();
            }
        }

        model.allLoadMax();
//        model.allLoadMaxN();
        this.maxLoadResult = model.maxLoadResult;
        for(int i = 0; i < maxLoadResult.size(); i++) {
            System.out.println(model.loadNodes.get(i).getId());
            System.out.println(maxLoadResult.get(model.loadNodes.get(i).getId()));
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

        AvailableCapOptNew model = new AvailableCapOptNew(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase2/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase2/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoads(loads);
        model.setSupplyCap(supplyCap);
        model.buildLoops();
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            if(minSwitchResult.getSupplyId()[i] != null) {
                System.out.println(minSwitchResult.getSupplyId()[i] + "\t" + minSwitchResult.getMinSwitch()[i]);
                for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++) {
                    System.out.print(minSwitchResult.getSwitchChanged().get(i)[j]);
                }
                System.out.println();
            }
        }

        model.allLoadMax();
//        model.allLoadMaxN();
        this.maxLoadResult = model.maxLoadResult;
        for(int i = 0; i < maxLoadResult.size(); i++) {
            System.out.println(model.loadNodes.get(i).getId());
            System.out.println(maxLoadResult.get(model.loadNodes.get(i).getId()));
        }
    }

    public void testCase4() throws IOException {
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
        String[] ErrorSupply = new String[]{"24"};

        AvailableCapOptNew model = new AvailableCapOptNew(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase4/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase4/supplyCapacity.txt").getPath();
        model.setErrorSupply(ErrorSupply);
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(19334);
        model.setLoads(loads);
        model.setSupplyCap(supplyCap);
        model.buildLoops();
        model.doOpt();
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

        AvailableCapOptNew model = new AvailableCapOptNew(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase3/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase3/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(20000);
        model.setLoads(loads);
        model.setSupplyCap(supplyCap);
        model.buildLoops();
        model.allMinSwitch();
        this.minSwitchResult = model.getOptResult();
        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
            System.out.println(minSwitchResult.getSupplyId()[i]);
            System.out.println(minSwitchResult.getMinSwitch()[i]);
            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
        }
        model.allLoadMax();
        this.maxLoadResult = model.maxLoadResult;
        for(int i = 0; i < maxLoadResult.size(); i++) {
            System.out.println(model.nodes.get(i).getId());
            System.out.println(maxLoadResult.get(model.nodes.get(i).getId()));
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

        AvailableCapOptNew model = new AvailableCapOptNew(testsys);
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase4/loads.txt").getPath();
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase4/supplyCapacity.txt").getPath();
        readSupplyCapacity(supplyCapacityPath);
        readLoads(loadsPath);
        model.setFeederCapacityConst(8712);
        model.setLoads(loads);
        model.setSupplyCap(supplyCap);
        String[] impLoads = {"6", "19", "22"};
        model.setImpLoads(impLoads);
        String[] ErrorSupply = new String[]{"25"};
        model.setErrorSupply(ErrorSupply);
        model.buildLoops();
        model.buildImpPathes();
        long time = 0;
        for (int i = 0; i < 1; i++) {
            long startT = System.nanoTime();
//            model.doOptPathLimit();
            model.doOpt();
//            System.out.println(System.nanoTime() - startT);
//            if (i >= 10) {
//                time += System.nanoTime() - startT;
//            }
        }
//        System.out.println(time / 5);
//        model.allMinSwitch();
//        this.minSwitchResult = model.getOptResult();
//        for(int i = 0; i < minSwitchResult.getSupplyId().length; i++) {
//            System.out.println(minSwitchResult.getSupplyId()[i]);
//            System.out.println(minSwitchResult.getMinSwitch()[i]);
//            // if(minSwitchResult.getSupplyId()[i] != null) {
//            for (int j = 0; j < minSwitchResult.getMinSwitch()[i]; j++)
//                System.out.println(minSwitchResult.getSwitchChanged().get(i)[j]);
//            //   }
//        }

//        long startT = System.currentTimeMillis();
//        model.loadMax("5");
//        System.out.println("Time:");
//        System.out.println(System.currentTimeMillis() - startT);
//        System.out.println(model.maxLoad);
//        model.allLoadMax();
//        this.maxLoadResult = model.maxLoadResult;
//        for(int i = 0; i < maxLoadResult.size(); i++) {
//            System.out.println(model.loadNodes.get(i).getId());
//            System.out.println(maxLoadResult.get(model.loadNodes.get(i).getId()));
//        }
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
            loads.put(cnId, cnLoad);
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
}