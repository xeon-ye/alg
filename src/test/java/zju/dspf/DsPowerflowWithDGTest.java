package zju.dspf;

import junit.framework.TestCase;
import zju.devmodel.MapObject;
import zju.dsmodel.*;

import java.io.InputStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/4/8
 */
public class DsPowerflowWithDGTest extends TestCase implements DsModelCons {

    public DsDeviceParser parser;
    public CalModelBuilder cmBuilder;

    public DsPowerflowWithDGTest() {
        parser = new DsDeviceParser();
        cmBuilder = new CalModelBuilder();
    }

    public void testPf_withDg_case4() {
        DistriSys ds = IeeeDsInHand.FEEDER4_DGrY_B.clone();

        MapObject g1 = parser.parseDg("4\tPV\tY\t300\t300\t300\t2.1\t2.1\t2.1");
        ds.getDevices().getDispersedGens().add(g1);
        ds.buildOrigTopo(ds.getDevices());
        ds.fillCnBaseKv();

        DistriSys ds2 = ds.clone();
        DsPowerflowTest.testConverged(ds2, true);
    }

    /**
     * 计算丁明，郭学凤在"含多种分布式电源的弱环配电网三相潮流计算"一文中的算例
     */
    public void testPf_withDg_case33_twoLoop() {
        DsTopoIsland oriIsland = DsCase33.createRadicalCase33();
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(oriIsland);
        DsCase33.addFeeder(oriIsland, tns, "33 8 21 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(oriIsland, tns, "35 12 22 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        oriIsland.initialIsland();
        oriIsland.setRadical(false);

        //case 2: P=300kw,Q=100kvar
        DsTopoIsland island = oriIsland.clone();
        tns = DsCase33.createTnMap(island);
        MapObject obj = parser.parseSpotLoad("25\tY-PQ\t-100\t-33.3\t-100\t-33.3\t-100\t-33.3");
        DsTopoNode tn = tns.get("25");
        obj.setProperty(KEY_KV_BASE, String.valueOf(tn.getBaseKv()));
        BasicLoad load = cmBuilder.dealLoad(obj);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        island.getLoads().put(obj, load);
        island.initialIsland();
        MeshedDsPfTest.testConverged(island, true);

        //case 3: PV, P=500kw, U=12.66
        island = oriIsland.clone();
        tns = DsCase33.createTnMap(island);
        obj = parser.parseDg("9\tPV\tY\t166.67\t166.67\t166.67\t7.31\t7.31\t7.31");
        tn = tns.get("9");
        obj.setProperty(KEY_KV_BASE, String.valueOf(tn.getBaseKv()));
        DispersedGen dg = cmBuilder.dealDisGeneration(obj);
        dg.setTn(tn);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        island.getDispersedGens().put(obj, dg);
        island.initialIsland();
        MeshedDsPfTest.testConverged(island, true);

        //case 4: 异步电机, p=1000kW
        island = oriIsland.clone();
        tns = DsCase33.createTnMap(island);
        obj = parser.parseDg("9\tIM\tY\t1000\t12.66\t1000\t0.0053\t0.106\t0.007\t0.120\t4.0");
        tn = tns.get("9");
        obj.setProperty(KEY_KV_BASE, String.valueOf(tn.getBaseKv()));
        dg = cmBuilder.dealDisGeneration(obj);
        dg.setTn(tn);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        island.getDispersedGens().put(obj, dg);
        island.initialIsland();
        MeshedDsPfTest.testConverged(island, true);
        //case 5: case 3 + case 4
        obj = parser.parseDg("9\tPV\tY\t166.67\t166.67\t166.67\t7.31\t7.31\t7.31");
        dg = cmBuilder.dealDisGeneration(obj);
        dg.setTn(tn);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        island.getDispersedGens().put(obj, dg);
        island.initialIsland();
        MeshedDsPfTest.testConverged(island, true);

        //case 6: Ia=Ib=Ic=10A
        String newNodeId = "34";
        island = oriIsland.clone();
        tns = DsCase33.createTnMap(island);
        addTransformerToCase33(island, tns, newNodeId);
        obj = parser.parseSpotLoad(newNodeId + "\tY-I\t-73.1\t-73.1\t-73.1\t0\t0\t0");
        tn = tns.get(newNodeId);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        obj.setProperty(KEY_KV_BASE, String.valueOf(tn.getBaseKv()));
        load = cmBuilder.dealLoad(obj);
        island.getLoads().put(obj, load);
        island.initialIsland();
        MeshedDsPfTest.testConverged(island, true);

        //case 7: Ia=10A, Ib=Ic=0
        island = oriIsland.clone();
        tns = DsCase33.createTnMap(island);
        addTransformerToCase33(island, tns, newNodeId);
        obj = parser.parseSpotLoad(newNodeId + "\tY-I\t-73.1\t0\t0\t0\t0\t0");
        tn = tns.get(newNodeId);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        obj.setProperty(KEY_KV_BASE, String.valueOf(tn.getBaseKv()));
        load = cmBuilder.dealLoad(obj);
        island.getLoads().put(obj, load);
        island.initialIsland();
        MeshedDsPfTest.testConverged(island, true);

        //case 8: P=400kw, cos=0.85
        island = oriIsland.clone();
        tns = DsCase33.createTnMap(island);
        addTransformerToCase33(island, tns, newNodeId);
        obj = parser.parseSpotLoad(newNodeId + "\tY-PQ\t-133.33\t-133.33\t-133.33\t-82.6\t-82.6\t-82.6");
        tn = tns.get(newNodeId);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        obj.setProperty(KEY_KV_BASE, String.valueOf(tn.getBaseKv()));
        load = cmBuilder.dealLoad(obj);
        island.getLoads().put(obj, load);
        island.initialIsland();
        MeshedDsPfTest.testConverged(island, true);

        //case 10: case 6 + case 8
        obj = parser.parseSpotLoad(newNodeId + "\tY-I\t-73.1\t-73.1\t-73.1\t0\t0\t0");
        tn = tns.get(newNodeId);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        obj.setProperty(KEY_KV_BASE, String.valueOf(tn.getBaseKv()));
        load = cmBuilder.dealLoad(obj);
        island.getLoads().put(obj, load);
        island.initialIsland();
        MeshedDsPfTest.testConverged(island, true);

        //case 11: case 10 + case 2 + case 3
        obj = parser.parseSpotLoad("25\tY-PQ\t-100\t-33.3\t-100\t-33.3\t-100\t-33.3");
        tn = tns.get("25");
        obj.setProperty(KEY_KV_BASE, String.valueOf(tn.getBaseKv()));
        load = cmBuilder.dealLoad(obj);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        island.getLoads().put(obj, load);

        obj = parser.parseDg("9\tPV\tY\t166.67\t166.67\t166.67\t7.31\t7.31\t7.31");
        tn = tns.get("9");
        dg = cmBuilder.dealDisGeneration(obj);
        dg.setTn(tn);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        island.getDispersedGens().put(obj, dg);
        island.initialIsland();
        MeshedDsPfTest.testConverged(island, true);

        //case 9: 异步电机,p=300kW
        island = oriIsland.clone();
        tns = DsCase33.createTnMap(island);
        addTransformerToCase33(island, tns, newNodeId);
        obj = parser.parseDg(newNodeId + "\tIM\tY\t300\t0.38\t300\t0.0053\t0.106\t0.007\t0.120\t4.0");
        tn = tns.get(newNodeId);
        dg = cmBuilder.dealDisGeneration(obj);
        dg.setTn(tn);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        island.getDispersedGens().put(obj, dg);
        island.initialIsland();
        MeshedDsPfTest.testConverged(island, true);

        //case 4开环后潮流测试
        island = DsCase33.createOpenLoopCase33(new int[]{33, 35});
        tns = DsCase33.createTnMap(island);
        obj = parser.parseDg("9\tIM\tY\t1000\t12.66\t1000\t0.0053\t0.106\t0.007\t0.120\t4.0");
        tn = tns.get("9");
        dg = cmBuilder.dealDisGeneration(obj);
        dg.setTn(tn);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        island.getDispersedGens().put(obj, dg);
        island.initialIsland();
        DsTopoIsland island2 = island.clone();
        MeshedDsPfTest.testConverged(island, true);
        MeshedDsPfTest.testConverged(island2, false);
        DsPowerflowTest.assertStateEquals(island, island2);

        //case 9，开环后潮流测试
        newNodeId = "40";
        island = DsCase33.createOpenLoopCase33(new int[]{33, 35});
        tns = DsCase33.createTnMap(island);
        addTransformerToCase33(island, tns, newNodeId);
        obj = parser.parseDg(newNodeId + "\tIM\tY\t600\t0.38\t400\t0.0053\t0.106\t0.007\t0.120\t4.0");
        tn = tns.get(newNodeId);
        dg = cmBuilder.dealDisGeneration(obj);
        dg.setTn(tn);
        tn.getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        island.getDispersedGens().put(obj, dg);
        island.initialIsland();
        island2 = island.clone();
        MeshedDsPfTest.testConverged(island, true);
        MeshedDsPfTest.testConverged(island2, false);
        DsPowerflowTest.assertStateEquals(island, island2);
    }

    private void addTransformerToCase33(DsTopoIsland island, Map<String, DsTopoNode> tns, String newNodeId) {
        MapObject t1 = parser.parseBranch("18\t" + newNodeId + "\t0\tT1", null);
        t1.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        parser.setTfParameters("T1\t750\tGr.Y-Gr.Y\t12.66\t0.38\t1\t4", t1);

        double baseKV = tns.get("1").getBaseKv();
        DsConnectNode cn = new DsConnectNode(newNodeId);
        cn.setBaseKv(baseKV);
        cn.setConnectedObjs(new ArrayList<MapObject>());

        DsTopoNode tn = new DsTopoNode();
        tn.setBaseKv(baseKV);
        tn.setConnectivityNodes(new ArrayList<DsConnectNode>(1));
        tn.getConnectivityNodes().add(cn);
        tns.put(newNodeId, tn);

        island.getBranches().put(t1, cmBuilder.dealTransformer(t1));
        island.getGraph().addVertex(tn);
        island.getGraph().addEdge(tns.get("18"), tn, t1);
    }

    /**
     * 对于五个环的33节点系统，合环点处解环后做辐射状网络的潮流
     */
    public void testPf_withDg_case33() {
        DsTopoIsland island1 = DsCase33.createOpenLoopCase33();
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(island1);
        MapObject t1 = parser.parseBranch("18\t34\t0\tT1", null);
        t1.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        parser.setTfParameters("T1\t750\tGr.Y-Gr.Y\t12.66\t0.48\t1\t5", t1);
        MapObject dg1 = parser.parseDg("34\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0");

        double baseKV = tns.get("1").getBaseKv();
        DsConnectNode cn = new DsConnectNode("34");
        cn.setBaseKv(baseKV);
        cn.setConnectedObjs(new ArrayList<MapObject>());
        cn.getConnectedObjs().add(dg1);

        DsTopoNode tn = new DsTopoNode();
        tn.setBaseKv(baseKV);
        tn.setConnectivityNodes(new ArrayList<DsConnectNode>(1));
        tn.getConnectivityNodes().add(cn);

        island1.getBranches().put(t1, cmBuilder.dealTransformer(t1));
        DispersedGen dg = cmBuilder.dealDisGeneration(dg1);
        dg.setTn(tn);
        island1.getDispersedGens().put(dg1, dg);
        island1.getGraph().addVertex(tn);
        island1.getGraph().addEdge(tns.get("18"), tn, t1);
        island1.initialIsland();

        DsTopoIsland island2 = island1.clone();
        MeshedDsPfTest.testConverged(island2, true);
        //DsPowerflowTest.printBusV(island2, true, true);
        MeshedDsPfTest.testConverged(island1, false);
        //DsPowerflowTest.assertStateEquals(island1, island2);
    }

    public void testPf_withIM_case4() {
        DistriSys ds = IeeeDsInHand.FEEDER4_GrYGrY_B.clone();
        DsDeviceParser parser = new DsDeviceParser();

        MapObject t2 = parser.parseBranch("4\t5\t0\tT2", null);
        t2.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        ds.getDevices().getTransformers().add(t2);
        parser.setTfParameters("T2\t750\tGr.Y-Gr.Y\t4.16\t0.48\t1\t5", ds.getDevices().getTransformers());
        MapObject g1 = parser.parseDg("5\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0");
        ds.getDevices().getDispersedGens().add(g1);
        ds.buildOrigTopo(ds.getDevices());
        ds.fillCnBaseKv();

        DistriSys ds1 = ds.clone();
        DsPowerflowTest.testConverged(ds1, false);
        //DsPowerflowTest.testKCL(ds1);
        DistriSys ds2 = ds.clone();
        DsPowerflowTest.testConverged(ds2, true);
        DsTopoIsland island1 = ds1.getActiveIslands()[0];
        DsTopoIsland island2 = ds2.getActiveIslands()[0];
        DsPowerflowTest.assertStateEquals(island1, island2);

        ds = IeeeDsInHand.FEEDER4_GrYGrY_UNB.clone();
        t2 = parser.parseBranch("4\t5\t0\tT2", null);
        t2.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        ds.getDevices().getTransformers().add(t2);
        parser.setTfParameters("T2\t750\tGr.Y-Gr.Y\t4.16\t0.48\t1\t5", ds.getDevices().getTransformers());
        ds.getDevices().getDispersedGens().add(parser.parseDg("5\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0"));
        ds.buildOrigTopo(ds.getDevices());
        ds.fillCnBaseKv();

        ds1 = ds.clone();
        DsPowerflowTest.testConverged(ds1, false);
        //DsPowerflowTest.testKCL(ds1);
        ds2 = ds.clone();
        DsPowerflowTest.testConverged(ds2, true);
        island1 = ds1.getActiveIslands()[0];
        island2 = ds2.getActiveIslands()[0];
        DsPowerflowTest.assertStateEquals(island1, island2);
        printMotorState(island2, false);
        DsPowerflowTest.printBusV(island2, false, false);

        ds = IeeeDsInHand.FEEDER4_DGrY_B.clone();
        t2 = parser.parseBranch("4\t5\t0\tT2", null);
        t2.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        ds.getDevices().getTransformers().add(t2);
        parser.setTfParameters("T2\t750\tGr.Y-Gr.Y\t4.16\t0.48\t1\t5", ds.getDevices().getTransformers());
        ds.getDevices().getDispersedGens().add(parser.parseDg("5\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0"));
        ds.buildOrigTopo(ds.getDevices());
        ds.fillCnBaseKv();

        ds1 = ds.clone();
        DsPowerflowTest.testConverged(ds1, false);
        //DsPowerflowTest.testKCL(ds1);
        ds2 = ds.clone();
        DsPowerflowTest.testConverged(ds2, true);
        island1 = ds1.getActiveIslands()[0];
        island2 = ds2.getActiveIslands()[0];
        DsPowerflowTest.assertStateEquals(island1, island2);

        ds = IeeeDsInHand.FEEDER4_DGrY_UNB.clone();
        t2 = parser.parseBranch("4\t5\t0\tT2", null);
        t2.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        ds.getDevices().getTransformers().add(t2);
        parser.setTfParameters("T2\t750\tGr.Y-Gr.Y\t4.16\t0.48\t1\t5", ds.getDevices().getTransformers());
        ds.getDevices().getDispersedGens().add(parser.parseDg("5\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0"));
        ds.buildOrigTopo(ds.getDevices());
        ds.fillCnBaseKv();

        ds1 = ds.clone();
        DsPowerflowTest.testConverged(ds1, false);
        //DsPowerflowTest.testKCL(ds1);
        ds2 = ds.clone();
        DsPowerflowTest.testConverged(ds2, true);
        island1 = ds1.getActiveIslands()[0];
        island2 = ds2.getActiveIslands()[0];
        DsPowerflowTest.assertStateEquals(island1, island2);
    }

    public void testPf_withDg_case34() {
        DistriSys ds = IeeeDsInHand.FEEDER34.clone();

        MapObject t1 = parser.parseBranch("848\t891\t0\tT1", null);
        t1.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        ds.getDevices().getTransformers().add(t1);
        parser.setTfParameters("T1\t750\tGr.Y-Gr.Y\t24.9\t0.48\t1\t5", t1);
        ds.getDevices().getDispersedGens().add(parser.parseDg("891\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0"));

        ds.buildOrigTopo(ds.getDevices());
        ds.fillCnBaseKv();

        DistriSys ds1 = ds.clone();
        DsPowerflowTest.testConverged(ds1, false);
        DsPowerflowTest.testKCL(ds1);
        DistriSys ds2 = ds.clone();
        DsPowerflowTest.testConverged(ds2, true);
        DsTopoIsland island1 = ds1.getActiveIslands()[0];
        DsTopoIsland island2 = ds2.getActiveIslands()[0];
        DsPowerflowTest.assertStateEquals(island1, island2);

        MapObject t2 = parser.parseBranch("890\t892\t0\tT2", null);
        t2.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        ds.getDevices().getTransformers().add(t2);
        parser.setTfParameters("T2\t750\tGr.Y-Gr.Y\t4.16\t0.48\t1\t5", t2);
        ds.getDevices().getDispersedGens().add(parser.parseDg("892\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0"));

        ds.buildOrigTopo(ds.getDevices());
        ds.fillCnBaseKv();

        ds2 = ds.clone();
        DsPowerflowTest.testConverged(ds2, true);

        InputStream ieeeFile = this.getClass().getResourceAsStream("/dsieee/case34/case34withDG.txt");
        DistriSys ds3 = IeeeDsInHand.createDs(ieeeFile, "800", 24.9 / sqrt3);
        DsPowerflowTest.testConverged(ds3, true);
        island1 = ds2.getActiveIslands()[0];
        island2 = ds3.getActiveIslands()[0];
        DsPowerflowTest.assertStateEquals(island1, island2);

        DistriSys ds4 = ds.clone();
        MeshedDsPfTest.combineTwoNode(ds4.getDevices(), "826", "858");
        MeshedDsPfTest.combineTwoNode(ds4.getDevices(), "822", "848");
        ds4.buildOrigTopo(ds4.getDevices());
        ds4.fillCnBaseKv();
        //ds4.buildDynamicTopo();
        //int branchNum = ds4.getActiveIslands()[0].getGraph().edgeSet().size();
        //int busNum = ds4.getActiveIslands()[0].getGraph().vertexSet().size();
        //System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(ds4, true);
        printMotorState(ds4.getActiveIslands()[0], false);
        DsPowerflowTest.printBusV(ds4.getActiveIslands()[0], false, false);
    }

    public void printMotorState(DsTopoIsland island, boolean isPerUnit) {
        DecimalFormat df1 = new DecimalFormat("#.##");
        DecimalFormat df2 = new DecimalFormat("#.#");
        for (MapObject dg : island.getDispersedGens().keySet()) {
            DispersedGen dispersedGen = island.getDispersedGens().get(dg);
            InductionMachine motor = dispersedGen.getMotor();
            if (motor == null)
                continue;
            double[][] v = motor.getVLN_12();
            FeederAndLoadTest.trans_rect2polar_deg(v);
            if (isPerUnit) {
                for (int i = 0; i < v.length; i++)
                    v[i][0] /= dispersedGen.getTn().getBaseKv() * 1000.;
            }
            String id = dispersedGen.getTn().getConnectivityNodes().get(0).getId();
            StringBuilder sb = new StringBuilder();
            sb.append(id).append("\t");
            for (double[] aV : v) {
                sb.append(df1.format(aV[0])).append("\t");
                sb.append(df2.format(aV[1])).append("\t");
            }
            sb.append(motor.getSlip());
            System.out.println(sb);
        }
    }
}
