package zju.dsntp;

import junit.framework.TestCase;
import org.junit.Assert;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsModelCons;
import zju.dsmodel.IeeeDsInHand;

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
        InputStream ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/mytest/testcase1/graphtest.txt");
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
        String loadsPath = "C:\\Users\\Administrator.2013-20160810IY\\IdeaProjects\\alg\\src\\main\\resources\\dsieee\\mytest\\testcase1\\loads.txt";
        String supplyCapacityPath = "C:\\Users\\Administrator.2013-20160810IY\\IdeaProjects\\alg\\src\\main\\resources\\dsieee\\mytest\\testcase1\\supplyCapacity.txt";
        String feederCapacityPath = "C:\\Users\\Administrator.2013-20160810IY\\IdeaProjects\\alg\\src\\main\\resources\\dsieee\\mytest\\testcase1\\feederCapacity.txt";
        model.doOpt(loadsPath, supplyCapacityPath, feederCapacityPath);
        Assert.assertEquals(3, model.minSwitch);

        ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/mytest/testcase1/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        model = new LoadTransferOpt(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        model.doOptLoadMax("/dsieee/mytest/testcase1/graphtest.txt", loadsPath, supplyCapacityPath, feederCapacityPath);
        Assert.assertEquals(10, model.maxLoad, 0.001);
    }

    public void testCase2() {
        String loadsPath = "C:\\Users\\Administrator.2013-20160810IY\\IdeaProjects\\alg\\src\\main\\resources\\dsieee\\mytest\\testcase1\\loads.txt";
        String supplyCapacityPath = "C:\\Users\\Administrator.2013-20160810IY\\IdeaProjects\\alg\\src\\main\\resources\\dsieee\\mytest\\testcase1\\supplyCapacity.txt";
        String feederCapacityPath = "C:\\Users\\Administrator.2013-20160810IY\\IdeaProjects\\alg\\src\\main\\resources\\dsieee\\mytest\\testcase1\\feederCapacity.txt";
        double maxLoad = 0;
        DistriSys testsys;
        String[] supplyID = new String[]{"S1", "S2", "S3"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};

        String[] ErrorSupply = new String[]{"S1"};
        int[] ErrorEdge = {0};

        InputStream ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/mytest/testcase1/graphtestLM1.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.setErrorFeeder(ErrorEdge);
        model.setErrorSupply(ErrorSupply);
        model.doOptLoadMax("/dsieee/mytest/testcase1/graphtestLM1.txt", loadsPath, supplyCapacityPath, feederCapacityPath);
        if(model.maxLoad > maxLoad)
            maxLoad = model.maxLoad;

//        System.out.printf("%.0f\n", maxLoad);
        //Assert.assertEquals(10, maxLoad, 0.001);
    }

    public void testCase3() {
        String loadsPath = "C:\\Users\\Administrator.2013-20160810IY\\IdeaProjects\\alg\\src\\main\\resources\\dsieee\\mytest\\testcase1\\loads.txt";
        String supplyCapacityPath = "C:\\Users\\Administrator.2013-20160810IY\\IdeaProjects\\alg\\src\\main\\resources\\dsieee\\mytest\\testcase1\\supplyCapacity.txt";
        String feederCapacityPath = "C:\\Users\\Administrator.2013-20160810IY\\IdeaProjects\\alg\\src\\main\\resources\\dsieee\\mytest\\testcase1\\feederCapacity.txt";
        DistriSys testsys;
        String[] supplyID = new String[]{"S1", "S2", "S3"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};

        InputStream ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/mytest/testcase1/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath);

        //Assert.assertEquals(10, model.maxLoad, 0.001);
    }

    public void testCase4() {
        String loadsPath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\ieee配电网-2013\\loads.txt";
        String supplyCapacityPath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\ieee配电网-2013\\supplyCapacity.txt";
        String feederCapacityPath = "C:\\Users\\Administrator.2013-20160810IY\\Desktop\\ieee配电网-2013\\feederCapacity.txt";
        DistriSys testsys;
        String[] supplyID = new String[]{"S1", "S2", "S3"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};

        InputStream ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsieee/mytest/graphtest.txt");
        testsys = createDs(ieeeFile, "S1", 100);
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);
        LoadTransferOpt model = new LoadTransferOpt(testsys);
        model.loadMax(loadsPath, supplyCapacityPath, feederCapacityPath);

        Assert.assertEquals(10, model.maxLoad, 0.001);
    }
}