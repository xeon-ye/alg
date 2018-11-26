package zju.dspf;

import junit.framework.TestCase;
import zju.devmodel.MapObject;
import zju.dsmodel.*;
import zju.dsntp.DsPowerflow;
import zju.dsntp.LcbPfModel;
import zju.matrix.AVector;
import zju.util.DoubleMatrixToolkit;

import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * DsPowerflow Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>10/31/2013</pre>
 */
public class DsPowerflowTest extends TestCase implements DsModelCons {
    public DsPowerflowTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testCase4() throws IOException {
        DistriSys ds = IeeeDsInHand.FEEDER4_GrYGrY_B.clone();
        testConverged(ds, false);
        testKCL(ds);
        assert_GrYGrY_B(ds);
        ds = IeeeDsInHand.FEEDER4_GrYGrY_B.clone();
        testConverged(ds, true);
        assert_GrYGrY_B(ds);

        System.out.println("开始打印");
        DsTopoIsland[] islands = ds.getActiveIslands();
        for (DsTopoIsland i : islands) {
            printBusV(i, i.isPerUnitSys(), false);
        }


        ds = IeeeDsInHand.FEEDER4_GrYGrY_UNB.clone();
        testConverged(ds, false);
        testKCL(ds);
        assert_GrYGrY_UNB(ds);
        ds = IeeeDsInHand.FEEDER4_GrYGrY_UNB.clone();
        testConverged(ds, true);
        assert_GrYGrY_UNB(ds);

        ds = IeeeDsInHand.FEEDER4_DD_B.clone();
        testConverged(ds, false);
        testKCL(ds);
        assertDD_B(ds, false);
        ds = IeeeDsInHand.FEEDER4_DD_B.clone();
        testConverged(ds, true);
        assertDD_B(ds, true);

        ds = IeeeDsInHand.FEEDER4_DD_UNB.clone();
        testConverged(ds, false);
        testKCL(ds);
        assertDD_UNB(ds, false);
        ds = IeeeDsInHand.FEEDER4_DD_UNB.clone();
        testConverged(ds, true);
        assertDD_UNB(ds, true);

        ds = IeeeDsInHand.FEEDER4_DGrY_B.clone();
        testConverged(ds, false);
        testKCL(ds);
        assert_DGrY_B(ds);
        ds = IeeeDsInHand.FEEDER4_DGrY_B.clone();
        testConverged(ds, true);
        assert_DGrY_B(ds);

        ds = IeeeDsInHand.FEEDER4_DGrY_UNB.clone();
        testConverged(ds, false);
        testKCL(ds);
        assert_DGrY_UNB(ds);
        ds = IeeeDsInHand.FEEDER4_DGrY_UNB.clone();
        testConverged(ds, true);
        assert_DGrY_UNB(ds);
    }

    public static void testKCL(DistriSys ds) {
        DsTopoIsland island = ds.getActiveIslands()[0];
        island.buildDetailedGraph();
        LcbPfModel model = new LcbPfModel(island, 0, 0);
        AVector state = model.getInitial();
        model.fillState();
        AVector z_est = model.calZ(state, false);
        int i;
        for (i = 0; i < 2 * model.getLoopSize(); i++) {
            //if(Math.abs(z_est.getValue(i)) > 0.9)
            //    System.out.println();
            assertTrue(Math.abs(z_est.getValue(i)) < 0.9);
        }
        for (; i < 2 * (model.getLoopSize() + model.getWindingEdges().size()); i++)
            assertTrue(Math.abs(z_est.getValue(i)) < 4.1);
        for (; i < 2 * (model.getLoopSize() + model.getWindingEdges().size() + model.getNonZLoadEdges().size()); i++)
            if (Math.abs(z_est.getValue(i)) > 110)
                assertTrue(Math.abs(z_est.getValue(i)) < 110);
        for (; i < z_est.getN(); i++)
            assertTrue(Math.abs(z_est.getValue(i)) < 1e-1);

        //System.out.println("支路为:");
        //for (i = 0; i < model.getNoToEdge().size(); i++) {
        //    DetailedEdge e = model.getNoToEdge().get(i);
        //    System.out.println(e.getTnNo1() + "->" + e.getTnNo2() + "_" + e.getPhase());
        //}
        //model.getB().printOnScreen();
    }

    private void assert_DGrY_UNB(DistriSys ds) {
        DsTopoIsland island;
        double[][] v_node2;
        double[][] v_node3;
        double[][] v_node4;
        double[][] busV;
        island = ds.getActiveIslands()[0];
        v_node2 = new double[][]{{12350, 29.6}, {12314, -90.4}, {12333, 149.8}};
        v_node3 = new double[][]{{2290, -32.4}, {2261, -153.8}, {2214, 85.2}};
        v_node4 = new double[][]{{2157, -34.2}, {1936, -157.0}, {1849, 73.4}};
        FeederAndLoadTest.trans_polar2rect_deg(v_node2);
        FeederAndLoadTest.trans_polar2rect_deg(v_node3);
        FeederAndLoadTest.trans_polar2rect_deg(v_node4);
        busV = transPhaseToLL(island.getBusV().get(island.getTnNoToTn().get(2)));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node2, 10));
        busV = island.getBusV().get(island.getTnNoToTn().get(3));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node3, 1));
        busV = island.getBusV().get(island.getTnNoToTn().get(4));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node4, 5));
    }

    private void assert_DGrY_B(DistriSys ds) {
        DsTopoIsland island;
        double[][] v_node2;
        double[][] v_node3;
        double[][] v_node4;
        double[][] busV;
        island = ds.getActiveIslands()[0];
        v_node2 = new double[][]{{12340, 29.7}, {12349, -90.4}, {12318, 149.6}};
        v_node3 = new double[][]{{2249, -33.7}, {2263, -153.4}, {2259, 86.4}};
        v_node4 = new double[][]{{1920, -39.1}, {2054, -158.3}, {1986, 80.9}};
        FeederAndLoadTest.trans_polar2rect_deg(v_node2);
        FeederAndLoadTest.trans_polar2rect_deg(v_node3);
        FeederAndLoadTest.trans_polar2rect_deg(v_node4);
        busV = transPhaseToLL(island.getBusV().get(island.getTnNoToTn().get(2)));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node2, 10));
        busV = island.getBusV().get(island.getTnNoToTn().get(3));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node3, 5));
        busV = island.getBusV().get(island.getTnNoToTn().get(4));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node4, 5));
    }

    private void assertDD_UNB(DistriSys ds, boolean isLoopPf) {
        DsTopoIsland island;
        double[][] v_node2;
        double[][] v_node3;
        double[][] v_node4;
        double[][] busV;
        island = ds.getActiveIslands()[0];
        v_node2 = new double[][]{{12341, 29.8}, {12370, -90.5}, {12302, 149.5}};
        v_node3 = new double[][]{{3902, 27.2}, {3972, -93.9}, {3871, 145.7}};
        v_node4 = new double[][]{{3431, 24.3}, {3647, -100.4}, {3294, 138.6}};
        FeederAndLoadTest.trans_polar2rect_deg(v_node2);
        FeederAndLoadTest.trans_polar2rect_deg(v_node3);
        FeederAndLoadTest.trans_polar2rect_deg(v_node4);
        busV = transPhaseToLL(island.getBusV().get(island.getTnNoToTn().get(2)));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node2, 10));
        if (isLoopPf)
            return;
        busV = transPhaseToLL(island.getBusV().get(island.getTnNoToTn().get(3)));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node3, 5));
        busV = transPhaseToLL(island.getBusV().get(island.getTnNoToTn().get(4)));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node4, 5));
    }

    private void assertDD_B(DistriSys ds, boolean isLoopPf) {
        DsTopoIsland island;
        double[][] v_node2;
        double[][] v_node3;
        double[][] v_node4;
        double[][] busV;
        island = ds.getActiveIslands()[0];
        v_node2 = new double[][]{{12339, 29.7}, {12349, -90.4}, {12321, 149.6}};
        v_node3 = new double[][]{{3911, 26.5}, {3914, -93.6}, {3905, 146.4}};
        v_node4 = new double[][]{{3442, 22.3}, {3497, -99.4}, {3384, 140.7}};
        FeederAndLoadTest.trans_polar2rect_deg(v_node2);
        FeederAndLoadTest.trans_polar2rect_deg(v_node3);
        FeederAndLoadTest.trans_polar2rect_deg(v_node4);
        busV = transPhaseToLL(island.getBusV().get(island.getTnNoToTn().get(2)));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node2, 5));
        if (isLoopPf)
            return;
        busV = transPhaseToLL(island.getBusV().get(island.getTnNoToTn().get(3)));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node3, 5));
        busV = transPhaseToLL(island.getBusV().get(island.getTnNoToTn().get(4)));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node4, 5));
    }

    private void assert_GrYGrY_UNB(DistriSys ds) {
        DsTopoIsland island;
        double[][] v_node2;
        double[][] v_node3;
        double[][] v_node4;
        double[][] busV;
        island = ds.getActiveIslands()[0];
        v_node2 = new double[][]{{7164, -0.1}, {7110, -120.2}, {7082, 119.3}};
        v_node3 = new double[][]{{2305, -2.3}, {2255, -123.6}, {2203, 114.8}};
        v_node4 = new double[][]{{2175, -4.1}, {1930, -126.8}, {1833, 102.8}};
        FeederAndLoadTest.trans_polar2rect_deg(v_node2);
        FeederAndLoadTest.trans_polar2rect_deg(v_node3);
        FeederAndLoadTest.trans_polar2rect_deg(v_node4);
        busV = island.getBusV().get(island.getTnNoToTn().get(2));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node2, 5));
        busV = island.getBusV().get(island.getTnNoToTn().get(3));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node3, 5));
        busV = island.getBusV().get(island.getTnNoToTn().get(4));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node4, 5));
    }

    private void assert_GrYGrY_B(DistriSys ds) {
        assertEquals(1, ds.getActiveIslands().length);
        DsTopoIsland island = ds.getActiveIslands()[0];
        double[][] v_node2 = {{7107, -0.3}, {7140, -120.3}, {7121, 119.6}};
        double[][] v_node3 = {{2247.6, -3.7}, {2269, -123.5}, {2256, 116.4}};
        double[][] v_node4 = {{1918, -9.1}, {2061, -128.3}, {1981, 110.9}};
        FeederAndLoadTest.trans_polar2rect_deg(v_node2);
        FeederAndLoadTest.trans_polar2rect_deg(v_node3);
        FeederAndLoadTest.trans_polar2rect_deg(v_node4);
        double[][] busV = island.getBusV().get(island.getTnNoToTn().get(2));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node2, 5));
        busV = island.getBusV().get(island.getTnNoToTn().get(3));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node3, 5));
        busV = island.getBusV().get(island.getTnNoToTn().get(4));
        assertTrue(DoubleMatrixToolkit.isDoubleMatrixEqual(busV, v_node4, 5));
    }

    public void testStandardCases_allPQ() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/dsieee/case13/case13-all-PQ.txt");
        DistriSys ds = IeeeDsInHand.createDs(ieeeFile, "650", 4.16 / sqrt3);
        DistriSys ds1 = ds.clone();
        DistriSys ds2 = ds.clone();
        testConverged(ds1, false);
        testKCL(ds1);
        testConverged(ds2, true);
        DsTopoIsland island1 = ds1.getActiveIslands()[0];
        DsTopoIsland island2 = ds2.getActiveIslands()[0];
        assertStateEquals(island1, island2);

        ieeeFile = this.getClass().getResourceAsStream("/dsieee/case34/case34-all-PQ.txt");
        ds = IeeeDsInHand.createDs(ieeeFile, "800", 24.9 / sqrt3);
        ds1 = ds.clone();
        ds2 = ds.clone();
        testConverged(ds1, false);
        testKCL(ds1);
        testConverged(ds2, true);
        island1 = ds1.getActiveIslands()[0];
        island2 = ds2.getActiveIslands()[0];
        assertStateEquals(island1, island2);

        ieeeFile = this.getClass().getResourceAsStream("/dsieee/case37/case37-all-PQ.txt");
        ds = IeeeDsInHand.createDs(ieeeFile, "799", 4.8 / sqrt3);
        ds1 = ds.clone();
        ds2 = ds.clone();
        testConverged(ds1, false);
        testKCL(ds1);
        testConverged(ds2, true);
        island1 = ds1.getActiveIslands()[0];
        island2 = ds2.getActiveIslands()[0];
        assertStateEquals(island1, island2);

        ieeeFile = this.getClass().getResourceAsStream("/dsieee/case123/case123-all-PQ.txt");
        ds = IeeeDsInHand.createDs(ieeeFile, "150", 4.16 / sqrt3);
        for (MapObject obj : ds.getDevices().getSwitches()) {
            switch (obj.getProperty(KEY_CONNECTED_NODE)) {
                case "250;251":
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                    break;
                case "450;451":
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                    break;
                case "54;94":
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                    break;
                case "151;300":
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                    break;
                case "300;350":
                    obj.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                    break;
            }
        }

        ds1 = ds.clone();
        ds2 = ds.clone();
        testConverged(ds1, false);
        testKCL(ds1);
        testConverged(ds2, true);
        island1 = ds1.getActiveIslands()[0];
        island2 = ds2.getActiveIslands()[0];
        assertStateEquals(island1, island2);
    }

    public void testStandardCases() throws IOException {
        DistriSys ds1 = IeeeDsInHand.FEEDER13.clone();
        DistriSys ds2 = IeeeDsInHand.FEEDER13.clone();
        testConverged(ds1, false);
        testKCL(ds1);
        testConverged(ds2, true);
        DsTopoIsland island1 = ds1.getActiveIslands()[0];
        DsTopoIsland island2 = ds2.getActiveIslands()[0];
        assertStateEquals(island1, island2);

        ds1 = IeeeDsInHand.FEEDER34.clone();
        ds2 = IeeeDsInHand.FEEDER34.clone();
        testConverged(ds1, false);
        testKCL(ds1);
        testConverged(ds2, true);
        island1 = ds1.getActiveIslands()[0];
        island2 = ds2.getActiveIslands()[0];
        assertStateEquals(island1, island2);

        ds1 = IeeeDsInHand.FEEDER37.clone();
        ds2 = IeeeDsInHand.FEEDER37.clone();
        testConverged(ds1, false);
        testKCL(ds1);
        testConverged(ds2, true);
        island1 = ds1.getActiveIslands()[0];
        island2 = ds2.getActiveIslands()[0];
        assertStateEquals(island1, island2);

//        ds1 = IeeeDsInHand.FEEDER69.clone();
//        ds2 = IeeeDsInHand.FEEDER69.clone();
//        testConverged(ds1, false);
//        testKCL(ds1);
//        testConverged(ds2, true);
//        island1 = ds1.getActiveIslands()[0];
//        island2 = ds2.getActiveIslands()[0];
//        assertStateEquals(island1, island2);

        ds1 = IeeeDsInHand.FEEDER123.clone();
        ds2 = IeeeDsInHand.FEEDER123.clone();
        testConverged(ds1, false);
        testKCL(ds1);
        testConverged(ds2, true);
        island1 = ds1.getActiveIslands()[0];
        island2 = ds2.getActiveIslands()[0];
        assertStateEquals(island1, island2);

        ds1 = IeeeDsInHand.FEEDER123x50.clone();
        ds2 = IeeeDsInHand.FEEDER123x50.clone();
        testConverged(ds1, false);
        testKCL(ds1);
        testConverged(ds2, true);
        island1 = ds1.getActiveIslands()[0];
        island2 = ds2.getActiveIslands()[0];
        assertStateEquals(island1, island2);

        ds1 = IeeeDsInHand.FEEDER8500.clone();
        ds2 = IeeeDsInHand.FEEDER8500.clone();
        testConverged(ds1, false);
        testKCL(ds1);
        testConverged(ds2, true);
        island1 = ds1.getActiveIslands()[0];
        island2 = ds2.getActiveIslands()[0];
        assertStateEquals(island1, island2);
    }

    public void testOneCase() {
        DistriSys ds1 = IeeeDsInHand.FEEDER13.clone();
        DistriSys ds2 = IeeeDsInHand.FEEDER13.clone();
        testConverged(ds1, false);
        testKCL(ds1);
        testConverged(ds2, true);
        DsTopoIsland island1 = ds1.getActiveIslands()[0];
        DsTopoIsland island2 = ds2.getActiveIslands()[0];
        assertStateEquals(island1, island2);
        printBusV(ds2.getActiveIslands()[0], false, false);
    }

    public void testRealCase() throws Exception {
        FeederConfMgr feederConfMgr = new FeederConfMgr();
        feederConfMgr.readImpedanceConf(DsPowerflowTest.class.getResourceAsStream("/dsfiles/fuzhou/feederconfig.txt"));
        InputStream ieeeFile = DsPowerflowTest.class.getResourceAsStream("/dsfiles/fuzhou/case-normal.txt");

        DistriSys ds1 = IeeeDsInHand.createDs(ieeeFile, feederConfMgr, "1", 12.66 / sqrt3);
        ieeeFile = DsPowerflowTest.class.getResourceAsStream("/dsfiles/fuzhou/case-restore.txt");
        DistriSys ds2 = IeeeDsInHand.createDs(ieeeFile, feederConfMgr, "1", 12.66 / sqrt3);
        testConverged(ds1, false);
        printBusV(ds1.getActiveIslands()[0], false, false);
        DsTopoIsland dsIsland = ds1.getActiveIslands()[0];
        double[][] headI = dsIsland.getBranchHeadI().get(dsIsland.getIdToBranch().get(1));
        double[][] headV = dsIsland.getBusV().get(dsIsland.getTnNoToTn().get(1));
        double pSum = 0;
        double pLoad = 0;
        for (int i = 0; i < 3; i++) {
            pSum += headV[i][0] * Math.cos(Math.PI / 180 * headV[i][1]) * headI[i][0] + headV[i][0] * Math.sin(Math.PI / 180 * headV[i][1]) * headI[i][1];
        }
        for (MapObject spotLoad : ds1.getDevices().getSpotLoads()) {
            pLoad += Double.parseDouble(spotLoad.getProperty("LoadKW1"));
            pLoad += Double.parseDouble(spotLoad.getProperty("LoadKW2"));
            pLoad += Double.parseDouble(spotLoad.getProperty("LoadKW3"));
        }
        System.out.println("网损为：" + (pSum - pLoad));
        testConverged(ds2, false);
        printBusV(ds2.getActiveIslands()[0], false, false);
        dsIsland = ds2.getActiveIslands()[0];
        headI = dsIsland.getBranchHeadI().get(dsIsland.getIdToBranch().get(1));
        headV = dsIsland.getBusV().get(dsIsland.getTnNoToTn().get(1));
        pSum = 0;
        pLoad = 0;
        for (int i = 0; i < 3; i++) {
            pSum += headV[i][0] * Math.cos(Math.PI / 180 * headV[i][1]) * headI[i][0] + headV[i][0] * Math.sin(Math.PI / 180 * headV[i][1]) * headI[i][1];
        }
        for (MapObject spotLoad : ds2.getDevices().getSpotLoads()) {
            pLoad += Double.parseDouble(spotLoad.getProperty("LoadKW1"));
            pLoad += Double.parseDouble(spotLoad.getProperty("LoadKW2"));
            pLoad += Double.parseDouble(spotLoad.getProperty("LoadKW3"));
        }
        System.out.println("网损为：" + (pSum - pLoad));
    }

    public static void assertStateEquals(DsTopoIsland island1, DsTopoIsland island2) {
        double[][] v1, v2;
        DsTopoNode tn2;
        for (DsTopoNode tn : island1.getBusV().keySet()) {
            v1 = island1.getBusV().get(tn);
            tn2 = island2.getTnNoToTn().get(tn.getTnNo());
            if (!island2.getBusV().containsKey(tn2))
                continue;
            v2 = island2.getBusV().get(tn2);
            boolean b = FeederAndLoadTest.isDoubleMatrixEqual(v1, v2, 1.0);
            assertTrue(b);
        }

        for (int branchNo = 1; branchNo <= island1.getBranches().size(); branchNo++) {
            MapObject obj1 = island1.getIdToBranch().get(branchNo);
            MapObject obj2 = island2.getIdToBranch().get(branchNo);
            double[][] c1 = island1.getBranchHeadI().get(obj1);
            double[][] c2 = island2.getBranchHeadI().get(obj2);
            boolean b = FeederAndLoadTest.isDoubleMatrixEqual(c1, c2, 1.0);
            assertTrue(b);

            if (island1.getBranchTailI().get(obj1) == c1)
                continue;
            c1 = island1.getBranchTailI().get(obj1);
            c2 = island2.getBranchTailI().get(obj2);
            b = FeederAndLoadTest.isDoubleMatrixEqual(c1, c2, 1.0);
            assertTrue(b);
        }
    }

    public static void testConverged(DistriSys ds, boolean isBcfMethod) {
        ds.buildDynamicTopo();

        ds.createCalDevModel();

        DsPowerflow pf = new DsPowerflow(ds);
        //pf.setMaxIter(500);
        long start = System.currentTimeMillis();
        pf.setTolerance(1e-3);
        if (isBcfMethod) {
            pf.doLcbPf();
        } else
            pf.doPf();
        assertTrue(pf.isConverged());
        System.out.println("计算潮流用时：" + (System.currentTimeMillis() - start) + "ms.");

        DsTopoIsland dsIsland = ds.getActiveIslands()[0];
        checkConstraints(dsIsland);
    }

    public static void checkConstraints(DsTopoIsland dsIsland) {
        double[][] vTemp = new double[3][2];
        DsTopoNode tmpTn;
        //计算电压方程约束 VLNABC = a * VLNabc + b * Iabc;
        for (int branchNo = 1; branchNo <= dsIsland.getBranches().size(); branchNo++) {
            MapObject obj = dsIsland.getIdToBranch().get(branchNo);
            DsTopoNode father = dsIsland.getGraph().getEdgeSource(obj);
            DsTopoNode tn = dsIsland.getGraph().getEdgeTarget(obj);
            if (father.getTnNo() > tn.getTnNo()) {
                tmpTn = tn;
                tn = father;
                father = tmpTn;
            }
            if (!dsIsland.getBusV().containsKey(tn))
                continue;
            if (dsIsland.getBranchTailI() == null
                    || dsIsland.getBranchTailI().size() < 1
                    || !dsIsland.getBranchTailI().containsKey(obj))
                continue;
            GeneralBranch generalBranch = dsIsland.getBranches().get(obj);
            generalBranch.calHeadV(dsIsland.getBusV().get(tn), dsIsland.getBranchTailI().get(obj), vTemp);
            for (int j = 0; j < 3; j++) {
                if (generalBranch instanceof Feeder) {
                    Feeder feeder = (Feeder) generalBranch;
                    if (!feeder.containsPhase(j))
                        continue;
                }
                assertTrue(Math.abs(vTemp[j][0] - dsIsland.getBusV().get(father)[j][0]) < 5);
                assertTrue(Math.abs(vTemp[j][1] - dsIsland.getBusV().get(father)[j][1]) < 5);
            }
            generalBranch.calTailV(dsIsland.getBusV().get(father), dsIsland.getBranchTailI().get(obj), vTemp);
            for (int j = 0; j < 3; j++) {
                if (generalBranch instanceof Feeder) {
                    Feeder feeder = (Feeder) generalBranch;
                    if (!feeder.containsPhase(j))
                        continue;
                }
                assertTrue(Math.abs(vTemp[j][0] - dsIsland.getBusV().get(tn)[j][0]) < 1e-4);
                assertTrue(Math.abs(vTemp[j][1] - dsIsland.getBusV().get(tn)[j][1]) < 1e-4);
            }
        }
    }

    public static double[][] transPhaseToLL(double[][] phaseV) {
        double[][] lineToLineV = new double[3][2];
        for (int i = 0; i < 3; i++) {
            lineToLineV[i][0] = phaseV[i][0] - phaseV[(i + 1) % 3][0];
            lineToLineV[i][1] = phaseV[i][1] - phaseV[(i + 1) % 3][1];
        }
        return lineToLineV;
    }

    public static void printBusV(DsTopoIsland island, boolean isPerUnit, boolean isCartesian) {
        List<DsTopoNode> tns = new ArrayList<>(island.getBusV().keySet());
        tns.sort((DsTopoNode tn1, DsTopoNode tn2) -> {
            String id1 = tn1.getConnectivityNodes().get(0).getId();
            String id2 = tn2.getConnectivityNodes().get(0).getId();
            return new Integer(id1).compareTo(new Integer(id2));
        });
        DecimalFormat df1 = new DecimalFormat("0.00000");
        DecimalFormat df2;
        if (isCartesian)
            df2 = new DecimalFormat("0.00000");
        else
            df2 = new DecimalFormat("#.#");
        SimpleDateFormat df = new SimpleDateFormat("yyyyMMddHHmmss");
        for (DsTopoNode tn : tns) {
            double[][] v = island.getBusV().get(tn);
            if (!isCartesian)
                FeederAndLoadTest.trans_rect2polar_deg(v);
            if (isPerUnit) {
                for (int i = 0; i < v.length; i++) {
                    v[i][0] /= tn.getBaseKv() * 1000.;
                    if (isCartesian)
                        v[i][1] /= tn.getBaseKv() * 1000.;
                }
            } else {
                for (int i = 0; i < v.length; i++) {
                    v[i][0] /= 1000.;
                    if (isCartesian)
                        v[i][1] /= tn.getBaseKv() * 1000.;
                }
            }
            String id = tn.getConnectivityNodes().get(0).getId();
            StringBuilder sb = new StringBuilder();
            //插入文件
            StringBuilder stringBuilder = new StringBuilder();
            sb.append(id).append("\t");
            stringBuilder.append(id).append(",");
            for (int i = 0; i < v.length; i++) {
                if (!tn.containsPhase(i)) {
                    sb.append("-\t-\t");
                    continue;
                }
                sb.append(df1.format(v[i][0])).append("\t");
                sb.append(df2.format(v[i][1])).append("\t");
                stringBuilder.append(df1.format(v[i][0])).append(",");
                stringBuilder.append(df1.format(v[i][1])).append(",");
            }
            System.out.println(sb);
            //插入文件
            stringBuilder.append("\n");
        }
    }
}
