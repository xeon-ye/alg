package zju.dspf;

import junit.framework.TestCase;
import zju.devmodel.MapObject;
import zju.dsmodel.*;

import java.io.InputStream;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/5/8
 */
public class MeshedDsPfTest extends TestCase implements DsModelCons {

    public void testNotConverged_case33() {

        DsTopoIsland island1 = DsCase33.createOpenLoopCase33();
        for(GeneralBranch b : island1.getBranches().values()) {
            for (double[] z : ((Feeder) b).getZ_real()) {
                z[0] *= 3;
                z[1] *= 3;
                z[2] *= 3;
            }
            for (double[] z : ((Feeder) b).getZ_imag()) {
                z[0] *= 3;
                z[1] *= 3;
                z[2] *= 3;
            }
        }
        for(ThreePhaseLoad load : island1.getLoads().values()) {
            for (double[] s : ((BasicLoad)load).getConstantS()) {
                s[0] *= 1;
                s[1] *= 1;
            }
        }
        DsTopoIsland island2 = island1.clone();
        testConverged(island2, true);
        //DsPowerflowTest.printBusV(island2, true, true);
        testConverged(island1, false);
        DsPowerflowTest.assertStateEquals(island1, island2);
        DsPowerflowTest.printBusV(island1, true, true);
    }

    public void testLoopedPf_case33() {
        DsTopoIsland island1 = DsCase33.createRadicalCase33();
        DsTopoIsland island2 = DsCase33.createRadicalCase33();
        testConverged(island1, false);
        testConverged(island2, true);
        DsPowerflowTest.assertStateEquals(island1, island2);

        Map<String, DsTopoNode> tns = DsCase33.createTnMap(island2);
        DsCase33.addFeeder(island2, tns, "33 8 21 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        island2.initialIsland();
        testConverged(island2, true);
        //DsPowerflowTest.printBusV(island2, false, true);

        DsCase33.addFeeder(island2, tns, "34 9 15 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        island2.initialIsland();
        testConverged(island2, true);

        DsCase33.addFeeder(island2, tns, "35 12 22 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        island2.initialIsland();
        testConverged(island2, true);

        DsCase33.addFeeder(island2, tns, "36 18 33 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        island2.initialIsland();
        testConverged(island2, true);

        DsCase33.addFeeder(island2, tns, "37 25 29 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        island2.initialIsland();
        testConverged(island2, true);
        DsPowerflowTest.printBusV(island2, true, true);
    }

    public static void testConverged(DsTopoIsland island, boolean isBcfMethod) {
        DsPowerflow pf = new DsPowerflow();
        //pf.setMaxIter(500);
        long start = System.currentTimeMillis();
        pf.setTolerance(1e-1);
        if (isBcfMethod) {
            pf.doLcbPf(island);
        } else {
            island.initialVariables();
            pf.doRadicalPf(island);
        }
        assertTrue(pf.isConverged());
        System.out.println("计算潮流用时：" + (System.currentTimeMillis() - start) + "ms.");
        DsPowerflowTest.checkConstraints(island);
    }

    public void testLoopedPf_case123() {
        DistriSys ds = getLoopedCase123();
        DistriSys sys = ds.clone();
        sys.buildDynamicTopo();
        int branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        int busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        sys = ds.clone();
        combineTwoNode(sys.getDevices(), "85", "75");
        combineTwoNode(sys.getDevices(), "36", "57");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        combineTwoNode(sys.getDevices(), "56", "90");
        combineTwoNode(sys.getDevices(), "39", "66");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        combineTwoNode(sys.getDevices(), "23", "44");
        combineTwoNode(sys.getDevices(), "62", "101");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        combineTwoNode(sys.getDevices(), "81", "86");
        combineTwoNode(sys.getDevices(), "70", "100");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        combineTwoNode(sys.getDevices(), "9", "18");
        combineTwoNode(sys.getDevices(), "30", "47");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        combineTwoNode(sys.getDevices(), "34", "94");
        combineTwoNode(sys.getDevices(), "64", "300");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);
    }

    public static DistriSys getLoopedCase123() {
        InputStream ieeeFile = DsPowerflowTest.class.getClass().getResourceAsStream("/dsieee/case123/case123-all-PQ.txt");
        DistriSys ds = IeeeDsInHand.createDs(ieeeFile, "150", 4.16 / sqrt3);
        DsPowerflowTest.testConverged(ds, true);

        ds = IeeeDsInHand.FEEDER123.clone();
        //case1:
        for (MapObject obj : ds.getDevices().getSwitches()) {
            if (obj.getProperty(KEY_CONNECTED_NODE).equals("250;251"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("450;451"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("54;94"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("151;300"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
            else if (obj.getProperty(KEY_CONNECTED_NODE).equals("300;350"))
                obj.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
        }
        return ds;
    }

    public static void combineTwoNode(DsDevices devs, String node1, String node2) {
        for (MapObject obj : devs.getFeeders()) {
            String[] nodes = obj.getProperty(KEY_CONNECTED_NODE).split(";");
            if (nodes[0].equals(node1)) {
                obj.setProperty(KEY_CONNECTED_NODE, node2 + ";" + nodes[1]);
            } else if (nodes[1].equals(node1)) {
                obj.setProperty(KEY_CONNECTED_NODE, nodes[0] + ";" + node2);
            }
        }
        for (MapObject obj : devs.getDistributedLoads()) {
            String[] nodes = obj.getProperty(KEY_CONNECTED_NODE).split(";");
            if (nodes[0].equals(node1)) {
                obj.setProperty(KEY_CONNECTED_NODE, node2 + ";" + nodes[1]);
            } else if (nodes[1].equals(node1)) {
                obj.setProperty(KEY_CONNECTED_NODE, nodes[0] + ";" + node2);
            }
        }
        for (MapObject obj : devs.getSpotLoads()) {
            String node = obj.getProperty(KEY_CONNECTED_NODE);
            if (node.equals(node1))
                obj.setProperty(KEY_CONNECTED_NODE, node2);
        }
        for (MapObject obj : devs.getShuntCapacitors()) {
            String node = obj.getProperty(KEY_CONNECTED_NODE);
            if (node.equals(node1))
                obj.setProperty(KEY_CONNECTED_NODE, node2);
        }
    }
}
