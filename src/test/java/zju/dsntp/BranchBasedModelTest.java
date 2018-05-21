package zju.dsntp;

import junit.framework.TestCase;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.IeeeDsInHand;

import java.io.IOException;
import java.io.InputStream;

import static zju.dsmodel.DsModelCons.*;
import static zju.dsmodel.IeeeDsInHand.createDs;

/**
 * Created by xuchengsi on 2018/1/15.
 */
public class BranchBasedModelTest extends TestCase {

    public void testDscase4() {
        BranchBasedModel model = new BranchBasedModel(IeeeDsInHand.FEEDER4_DD_B);
        long startT = System.nanoTime();
        model.buildLoops();
        System.out.println(System.nanoTime() - startT);
        model.printLoop();
        assertEquals(0, model.getLoops().size());
        String[] impLoads = {"3"};
        model.setImpLoads(impLoads);
        model.buildImpPathes();
    }

    public void testDscase13() {
        BranchBasedModel model = new BranchBasedModel(IeeeDsInHand.FEEDER13);
        long startT = System.nanoTime();
        model.buildLoops();
        System.out.println(System.nanoTime() - startT);
        model.printLoop();
        assertEquals(0, model.getLoops().size());
        String[] impLoads = {"645", "675", "684"};
        model.setImpLoads(impLoads);
        model.buildImpPathes();
    }

    public void testDscase34() {
        BranchBasedModel model = new BranchBasedModel(IeeeDsInHand.FEEDER34);
        long startT = System.nanoTime();
        model.buildLoops();
        System.out.println(System.nanoTime() - startT);
        model.printLoop();
        assertEquals(0, model.getLoops().size());
        String[] impLoads = {"824", "834"};
        model.setImpLoads(impLoads);
        model.buildImpPathes();
    }

    public void testDscase37() {
        BranchBasedModel model = new BranchBasedModel(IeeeDsInHand.FEEDER37);
        model.buildLoops();
        model.printLoop();
        assertEquals(0, model.getLoops().size());
        String[] impLoads = {"711", "725"};
        model.setImpLoads(impLoads);
        model.buildImpPathes();
    }

    public void testDscase123() {
        BranchBasedModel model = new BranchBasedModel(IeeeDsInHand.FEEDER123);
        long startT = System.nanoTime();
        model.buildLoops();
        System.out.println(System.nanoTime() - startT);
//        model.printLoop();
        assertEquals(1, model.getLoops().size());
        String[] impLoads = {"93", "151"};
        model.setImpLoads(impLoads);
        model.buildImpPathes();
    }

    public void testCase() throws IOException {
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

        BranchBasedModel model = new BranchBasedModel(testsys);
        long startT = System.nanoTime();
        model.buildLoops();
        System.out.println(System.nanoTime() - startT);
//        model.printLoop();
        assertEquals(18, model.getLoops().size());
        String[] impLoads = {"4", "16", "22"};
        model.setImpLoads(impLoads);
        model.buildImpPathes();
    }

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
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        BranchBasedModel model = new BranchBasedModel(testsys);
        model.buildLoops();
        assertEquals(7, model.getLoops().size());
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
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        BranchBasedModel model = new BranchBasedModel(testsys);
        model.buildLoops();
        model.printLoop();
        assertEquals(7, model.getLoops().size());
    }

    public void testCase3() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase17/graphtest.txt");
        testsys = createDs(ieeeFile, "1", 100);
        supplyID = new String[]{"1", "6"};
        Double[] supplyBaseKv = new Double[]{100., 200.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        BranchBasedModel model = new BranchBasedModel(testsys);
        model.buildLoops();
        model.printLoop();
        assertEquals(14, model.getLoops().size());
    }

    public void testCase4() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase18/graphtest.txt");
        testsys = createDs(ieeeFile, "1", 100);
        supplyID = new String[]{"1", "6", "12"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        BranchBasedModel model = new BranchBasedModel(testsys);
        model.buildLoops();
        model.printLoop();
        assertEquals(14, model.getLoops().size());
    }

    public void testCase5() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase19/graphtest.txt");
        testsys = createDs(ieeeFile, "1", 100);
        supplyID = new String[]{"1", "6", "15"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        BranchBasedModel model = new BranchBasedModel(testsys);
        model.buildLoops();
        model.printLoop();
        assertEquals(28, model.getLoops().size());
//        UndirectedGraph<DsConnectNode, MapObject> g = model.sys.getOrigGraph();
//        for (int[] loop : model.loops) {
//            System.out.println(model.sortLoop(loop));
//            for (int i : loop) {
//                MapObject edge = model.edges.get(i);
//                System.out.print(g.getEdgeSource(edge).getId() + "-" + g.getEdgeTarget(edge).getId() + ", ");
//            }
//            System.out.println();
//        }
    }

    public void testCase6() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase20/graphtest.txt");
        testsys = createDs(ieeeFile, "1", 100);
        supplyID = new String[]{"1", "6", "15", "16"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        BranchBasedModel model = new BranchBasedModel(testsys);
        model.buildLoops();
//        model.printLoop();
        assertEquals(47, model.getLoops().size());
    }

    public void testCase7() throws IOException {
        DistriSys testsys;
        String[] supplyID;
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase21/graphtest.txt");
        testsys = createDs(ieeeFile, "1", 100);
        supplyID = new String[]{"1", "6", "15"};
        Double[] supplyBaseKv = new Double[]{100., 200., 100.};
        testsys.setSupplyCns(supplyID);
        testsys.setSupplyCnBaseKv(supplyBaseKv);

        BranchBasedModel model = new BranchBasedModel(testsys);
        model.buildLoops();
//        model.printLoop();
        assertEquals(50, model.getLoops().size());
    }
}