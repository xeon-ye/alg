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

        // 注：以下代码为新增（20160407），考虑分布式电源接入
        // 打印电压计算结果
        DsPowerflowTest.printBusV(ds2.getActiveIslands()[0], false, false);

        System.out.println("\n对接入了多种类型DG的4节点测试系统进行测试（变压器GrY-GrY，三相平衡负荷）：\n");

        // 1 DG类型为PQ节点
        System.out.println("1 DG类型为PQ节点");
        DistriSys node4Dg1 = IeeeDsInHand.FEEDER4_GrYGrY_B.clone();
        // 将PQ节点看成消耗功率为负的负荷
        node4Dg1.getDevices().getSpotLoads().add(parser.parseSpotLoad("4\tY-PQ\t-180\t-80\t-180\t-80\t-180\t-80"));
        node4Dg1.buildOrigTopo(node4Dg1.getDevices());
        node4Dg1.fillCnBaseKv();
        DistriSys node4Dg1Clone = node4Dg1.clone();
        System.out.println("（1）回路电流潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg1Clone, true);
        DsPowerflowTest.printBusV(node4Dg1Clone.getActiveIslands()[0], false, false);
        DistriSys node4Dg1Clone2 = node4Dg1.clone();
        System.out.println("（2）前推回推潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg1Clone2, false);
        System.out.println("");

        // 2 DG类型为恒电流型（电流幅值和功率因数恒定）
        System.out.println("2 DG类型为恒电流型（电流幅值和功率因数恒定）");
        DistriSys node4Dg2 = IeeeDsInHand.FEEDER4_GrYGrY_B.clone();
        // 看成消耗功率为负的负荷
        node4Dg2.getDevices().getSpotLoads().add(parser.parseSpotLoad("4\tY-I\t-180\t-80\t-180\t-80\t-180\t-80"));
        node4Dg2.buildOrigTopo(node4Dg2.getDevices());
        node4Dg2.fillCnBaseKv();
        DistriSys node4Dg2Clone = node4Dg2.clone();
        System.out.println("（1）回路电流潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg2Clone, true);
        DsPowerflowTest.printBusV(node4Dg2Clone.getActiveIslands()[0], false, false);
        DistriSys node4Dg2Clone2 = node4Dg2.clone();
        System.out.println("（2）前推回推潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg2Clone2, false);
        System.out.println("");

        // 3 DG类型为PV节点
        System.out.println("3 DG类型为PV节点");
        DistriSys node4Dg3 = IeeeDsInHand.FEEDER4_GrYGrY_B.clone();
        // PV节点
        node4Dg3.getDevices().getDispersedGens().add(parser.parseDg("4\tPV\tY\t180\t180\t180\t2.0\t2.0\t2.0"));
        node4Dg3.buildOrigTopo(node4Dg3.getDevices());
        node4Dg3.fillCnBaseKv();
        DistriSys node4Dg3Clone = node4Dg3.clone();
        System.out.println("（1）回路电流潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg3Clone, true);
        DsPowerflowTest.printBusV(node4Dg3Clone.getActiveIslands()[0], false, false);
        DistriSys node4Dg3Clone2 = node4Dg3.clone();
//        System.out.println("（2）前推回推潮流计算方法测试");
//        DsPowerflowTest.testConverged(node4Dg3Clone2, false);
        System.out.println("");

        // 4 DG类型为PQ节点加恒电流
        System.out.println("4 DG类型为PQ节点加恒电流");
        DistriSys node4Dg4 = IeeeDsInHand.FEEDER4_GrYGrY_B.clone();
        // PQ节点
        node4Dg4.getDevices().getSpotLoads().add(parser.parseSpotLoad("4\tY-PQ\t-180\t-80\t-180\t-80\t-180\t-80"));
        // 恒电流
        node4Dg4.getDevices().getSpotLoads().add(parser.parseSpotLoad("4\tY-I\t-180\t-80\t-180\t-80\t-180\t-80"));
        node4Dg4.buildOrigTopo(node4Dg4.getDevices());
        node4Dg4.fillCnBaseKv();
        DistriSys node4Dg4Clone = node4Dg4.clone();
        System.out.println("（1）回路电流潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg4Clone, true);
        DsPowerflowTest.printBusV(node4Dg4Clone.getActiveIslands()[0], false, false);
        DistriSys node4Dg4Clone2 = node4Dg4.clone();
        System.out.println("（2）前推回推潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg4Clone2, false);
        System.out.println("");

        // 5 DG类型为异步电机加PV节点
        System.out.println("5 DG类型为异步电机加PV节点");
        DistriSys node4Dg5 = IeeeDsInHand.FEEDER4_GrYGrY_B.clone();
        System.out.println("（异步电机（含变压器）加PV节点不能收敛，改为异步电机加PQ节点可以收敛）");
        node4Dg5.getDevices().getSpotLoads().add(parser.parseSpotLoad("4\tY-PQ\t-90\t-40\t-90\t-40\t-90\t-40")); //分布式发电为PQ型能正常收敛
        // 注意：若采用下述方案（第4节点为PV的DG、（电机用的）变压器高压侧和负荷，第5节点为变压器低压侧和异步电机），回路方法计算潮流出现NullPointerException
        // PV节点
//        node4Dg5.getDevices().getDispersedGens().add(parser.parseDg("4\tPV\tY\t180\t180\t180\t2.1\t2.1\t2.1"));
        // 异步电机
        MapObject t2 = parser.parseBranch("4\t5\t0\tT2", null);
        t2.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        node4Dg5.getDevices().getTransformers().add(t2);
        parser.setTfParameters("T2\t750\tGr.Y-Gr.Y\t4.16\t0.48\t1\t5", t2);
        MapObject motor5 = parser.parseDg("5\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0");
        node4Dg5.getDevices().getDispersedGens().add(motor5);
        // 出现NullPointerException的算例结束
        // todo fix this
        node4Dg5.buildOrigTopo(node4Dg5.getDevices());
        node4Dg5.fillCnBaseKv();
        DistriSys node4Dg5Clone = node4Dg5.clone();
        System.out.println("（1）回路电流潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg5Clone, true);
        DsPowerflowTest.printBusV(node4Dg5Clone.getActiveIslands()[0], false, false);
        DistriSys node4Dg5Clone2 = node4Dg5.clone();
        System.out.println("（2）前推回推潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg5Clone2, false);
        System.out.println("");

        System.out.println("\n含DG的4节点测试系统（变压器GrY-GrY，负荷三相平衡）潮流计算测试完毕");

//        DistriSys ds = IeeeDsInHand.FEEDER4_GrYGrY_B.clone();
//        DsDeviceParser parser = new DsDeviceParser();
//
//        MapObject t2 = parser.parseBranch("4\t5\t0\tT2", null);
//        t2.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
//        ds.getDevices().getTransformers().add(t2);
//        parser.setTfParameters("T2\t750\tGr.Y-Gr.Y\t4.16\t0.48\t1\t5", ds.getDevices().getTransformers());
//        MapObject g1 = parser.parseDg("5\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0");
//        ds.getDevices().getDispersedGens().add(g1);
//        ds.buildOrigTopo(ds.getDevices());
//        ds.fillCnBaseKv();
//
//        DistriSys ds1 = ds.clone();
//        DsPowerflowTest.testConverged(ds1, false);
    }

    public void testPf_withDg_case4_unbalanced() {
        // 以下代码基本复制自testPf_withDg_case4方法，仅修改了调用的测试系统
        System.out.println("\nIEEE4节点系统（变压器GrY-GrY，三相负荷不平衡）测试：");
        // 1 DG类型为PQ节点
        System.out.println("1 DG类型为PQ节点");
        DistriSys node4Dg1 = IeeeDsInHand.FEEDER4_GrYGrY_UNB.clone();
        // 将PQ节点看成消耗功率为负的负荷
        node4Dg1.getDevices().getSpotLoads().add(parser.parseSpotLoad("4\tY-PQ\t-180\t-80\t-180\t-80\t-180\t-80"));
        node4Dg1.buildOrigTopo(node4Dg1.getDevices());
        node4Dg1.fillCnBaseKv();
        DistriSys node4Dg1Clone = node4Dg1.clone();
        System.out.println("（1）回路电流潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg1Clone, true);
        DsPowerflowTest.printBusV(node4Dg1Clone.getActiveIslands()[0], false, false);
        DistriSys node4Dg1Clone2 = node4Dg1.clone();
        System.out.println("（2）前推回推潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg1Clone2, false);
        System.out.println("");

        // 2 DG类型为恒电流型（电流幅值和功率因数恒定）
        System.out.println("2 DG类型为恒电流型（电流幅值和功率因数恒定）");
        DistriSys node4Dg2 = IeeeDsInHand.FEEDER4_GrYGrY_UNB.clone();
        // 看成消耗功率为负的负荷
        node4Dg2.getDevices().getSpotLoads().add(parser.parseSpotLoad("4\tY-I\t-180\t-80\t-180\t-80\t-180\t-80"));
        node4Dg2.buildOrigTopo(node4Dg2.getDevices());
        node4Dg2.fillCnBaseKv();
        DistriSys node4Dg2Clone = node4Dg2.clone();
        System.out.println("（1）回路电流潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg2Clone, true);
        DsPowerflowTest.printBusV(node4Dg2Clone.getActiveIslands()[0], false, false);
        DistriSys node4Dg2Clone2 = node4Dg2.clone();
        System.out.println("（2）前推回推潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg2Clone2, false);
        System.out.println("");

        // 3 DG类型为PV节点
        System.out.println("3 DG类型为PV节点");
        DistriSys node4Dg3 = IeeeDsInHand.FEEDER4_GrYGrY_UNB.clone();
        // PV节点
        node4Dg3.getDevices().getDispersedGens().add(parser.parseDg("4\tPV\tY\t180\t180\t180\t2.0\t2.0\t2.0"));
        node4Dg3.buildOrigTopo(node4Dg3.getDevices());
        node4Dg3.fillCnBaseKv();
        DistriSys node4Dg3Clone = node4Dg3.clone();
        System.out.println("（1）回路电流潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg3Clone, true);
        DsPowerflowTest.printBusV(node4Dg3Clone.getActiveIslands()[0], false, false);
        DistriSys node4Dg3Clone2 = node4Dg3.clone();
//        System.out.println("（2）前推回推潮流计算方法测试");
//        DsPowerflowTest.testConverged(node4Dg3Clone2, false);
        System.out.println("");

        // 4 DG类型为PQ节点加恒电流
        System.out.println("4 DG类型为PQ节点加恒电流");
        DistriSys node4Dg4 = IeeeDsInHand.FEEDER4_GrYGrY_UNB.clone();
        // PQ节点
        node4Dg4.getDevices().getSpotLoads().add(parser.parseSpotLoad("4\tY-PQ\t-180\t-80\t-180\t-80\t-180\t-80"));
        // 恒电流
        node4Dg4.getDevices().getSpotLoads().add(parser.parseSpotLoad("4\tY-I\t-180\t-80\t-180\t-80\t-180\t-80"));
        node4Dg4.buildOrigTopo(node4Dg4.getDevices());
        node4Dg4.fillCnBaseKv();
        DistriSys node4Dg4Clone = node4Dg4.clone();
        System.out.println("（1）回路电流潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg4Clone, true);
        DsPowerflowTest.printBusV(node4Dg4Clone.getActiveIslands()[0], false, false);
        DistriSys node4Dg4Clone2 = node4Dg4.clone();
        System.out.println("（2）前推回推潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg4Clone2, false);
        System.out.println("");

        // 5 DG类型为异步电机加PV节点
        System.out.println("5 DG类型为异步电机加PV节点");
        DistriSys node4Dg5 = IeeeDsInHand.FEEDER4_GrYGrY_UNB.clone();
        System.out.println("（异步电机（含变压器）加PV节点不能收敛，改为异步电机加PQ节点可以收敛）");
        node4Dg5.getDevices().getSpotLoads().add(parser.parseSpotLoad("4\tY-PQ\t-90\t-40\t-90\t-40\t-90\t-40")); //分布式发电为PQ型能正常收敛
        // 注意：若采用下述方案（第4节点为PV的DG、（电机用的）变压器高压侧和负荷，第5节点为变压器低压侧和异步电机），回路方法计算潮流出现NullPointerException
        // PV节点
//        node4Dg5.getDevices().getDispersedGens().add(parser.parseDg("4\tPV\tY\t180\t180\t180\t2.1\t2.1\t2.1"));
        // 异步电机
        MapObject t2 = parser.parseBranch("4\t5\t0\tT2", null);
        t2.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        node4Dg5.getDevices().getTransformers().add(t2);
        parser.setTfParameters("T2\t750\tGr.Y-Gr.Y\t4.16\t0.48\t1\t5", t2);
        MapObject motor5 = parser.parseDg("5\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0");
        node4Dg5.getDevices().getDispersedGens().add(motor5);
        // 出现NullPointerException的算例结束
        // todo fix this
        node4Dg5.buildOrigTopo(node4Dg5.getDevices());
        node4Dg5.fillCnBaseKv();
        DistriSys node4Dg5Clone = node4Dg5.clone();
        System.out.println("（1）回路电流潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg5Clone, true);
        DsPowerflowTest.printBusV(node4Dg5Clone.getActiveIslands()[0], false, false);
        DistriSys node4Dg5Clone2 = node4Dg5.clone();
        System.out.println("（2）前推回推潮流计算方法测试");
        DsPowerflowTest.testConverged(node4Dg5Clone2, false);
        System.out.println("");

        System.out.println("\n含DG的4节点测试系统（变压器GrY-GrY，负荷三相不平衡）潮流计算测试完毕");
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

    public void testPf_withDg_case34_more() {
        System.out.println("含分布式电源的IEEE34节点系统测试（辐射型配电网）");
        System.out.println("\n案例1：846节点接入PQ型DG，Y接法，每相P=30kW，Q=20kVar");
        DistriSys node34c1 = IeeeDsInHand.FEEDER34.clone();
        node34c1.getDevices().getSpotLoads().add(parser.parseSpotLoad("846\tY-PQ\t-30\t-20\t-30\t-20\t-30\t-20"));
        node34c1.buildOrigTopo(node34c1.getDevices());
        node34c1.fillCnBaseKv();
        // 回路法测试
        DistriSys node34c1Clone = node34c1.clone();
        System.out.println("（1）回路电流潮流计算法测试");
        DsPowerflowTest.testConverged(node34c1Clone, true);
        DsPowerflowTest.printBusV(node34c1Clone.getActiveIslands()[0], false, false);
        // 前推回退法测试
        DistriSys node34c1Clone2 = node34c1.clone();
        System.out.println("（2）前推回退潮流计算法测试");
        DsPowerflowTest.testConverged(node34c1Clone2, false);
        System.out.println("");

        System.out.println("\n案例2：860节点接入I型DG，Y接法，每相P=30kW，Q=20kVar");
        DistriSys node34c2 = IeeeDsInHand.FEEDER34.clone();
        node34c2.getDevices().getSpotLoads().add(parser.parseSpotLoad("860\tY-I\t-30\t-20\t-30\t-20\t-30\t-20"));
        node34c2.buildOrigTopo(node34c2.getDevices());
        node34c2.fillCnBaseKv();
        DistriSys node34c2Clone = node34c2.clone();
        System.out.println("（1）回路电流潮流计算法测试");
        DsPowerflowTest.testConverged(node34c2Clone, true);
        DsPowerflowTest.printBusV(node34c2Clone.getActiveIslands()[0], false, false);
        // 前推回退法测试
        DistriSys node34c2Clone2 = node34c2.clone();
        System.out.println("（2）前推回退潮流计算法测试");
        DsPowerflowTest.testConverged(node34c2Clone2, false);
        System.out.println("");

        System.out.println("\n案例3：836节点接入PV型DG，Y接法，每相P=30kW，VLN=13kV"); // 24.9/1.73=14.39
        DistriSys node34c3 = IeeeDsInHand.FEEDER34.clone();
        node34c3.getDevices().getDispersedGens().add(parser.parseDg("836\tPV\tY\t30\t30\t30\t13\t13\t13"));
        node34c3.buildOrigTopo(node34c3.getDevices());
        node34c3.fillCnBaseKv();
        DistriSys node34c3Clone = node34c3.clone();
        System.out.println("（1）回路电流潮流计算法测试");
        DsPowerflowTest.testConverged(node34c3Clone, true);
        DsPowerflowTest.printBusV(node34c3Clone.getActiveIslands()[0], false, false);
        // 前推回退法测试
        DistriSys node34c3Clone2 = node34c3.clone();
        System.out.println("（2）前推回退潮流计算法测试");
        DsPowerflowTest.testConverged(node34c3Clone2, false);
        System.out.println("");

        System.out.println("\n案例4：DG配置=案例1+案例2+案例3");
        DistriSys node34c4 = IeeeDsInHand.FEEDER34.clone();
        node34c4.getDevices().getSpotLoads().add(parser.parseSpotLoad("846\tY-PQ\t-30\t-20\t-30\t-20\t-30\t-20"));
        node34c4.getDevices().getSpotLoads().add(parser.parseSpotLoad("860\tY-I\t-30\t-20\t-30\t-20\t-30\t-20"));
        node34c4.getDevices().getDispersedGens().add(parser.parseDg("836\tPV\tY\t30\t30\t30\t13\t13\t13"));
        node34c4.buildOrigTopo(node34c4.getDevices());
        node34c4.fillCnBaseKv();
        DistriSys node34c4Clone = node34c4.clone();
        System.out.println("（1）回路电流潮流计算法测试");
        DsPowerflowTest.testConverged(node34c4Clone, true);
        DsPowerflowTest.printBusV(node34c4Clone.getActiveIslands()[0], false, false);
        // 前推回退法测试
        DistriSys node34c4Clone2 = node34c4.clone();
        System.out.println("（2）前推回退潮流计算法测试");
        DsPowerflowTest.testConverged(node34c4Clone2, false);
        System.out.println("");

        System.out.println("\n案例5：DG配置=案例1+案例2+案例3+异步发电机（2台，含变压器）");
        System.out.println("异步发电机参数见：Induction Machine Test Case for the 34-Bus Test Feeder – Description");
        DistriSys node34c5 = IeeeDsInHand.FEEDER34.clone();
        node34c5.getDevices().getSpotLoads().add(parser.parseSpotLoad("846\tY-PQ\t-30\t-20\t-30\t-20\t-30\t-20"));
        node34c5.getDevices().getSpotLoads().add(parser.parseSpotLoad("860\tY-I\t-30\t-20\t-30\t-20\t-30\t-20"));
        node34c5.getDevices().getDispersedGens().add(parser.parseDg("836\tPV\tY\t30\t30\t30\t13\t13\t13"));
        // 异步发电机1
        MapObject t1 = parser.parseBranch("848\t900\t0\tT1", null);
        t1.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        node34c5.getDevices().getTransformers().add(t1);
        parser.setTfParameters("T1\t750\tGr.Y-Gr.Y\t24.9\t0.48\t1\t5", t1);
        MapObject im1 = parser.parseDg("900\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0");
        node34c5.getDevices().getDispersedGens().add(im1);
        // 异步发电机2
        MapObject t2 = parser.parseBranch("890\t902\t0\tT2", null);
        t2.setProperty(KEY_RESOURCE_TYPE, RESOURCE_TRANSFORMER);
        node34c5.getDevices().getTransformers().add(t2);
        parser.setTfParameters("T2\t750\tGr.Y-Gr.Y\t4.16\t0.48\t1\t5", t2);
        MapObject im2 = parser.parseDg("902\tIM\tY\t660\t0.48\t660\t0.0053\t0.106\t0.007\t0.120\t4.0");
        node34c5.getDevices().getDispersedGens().add(im2);
        node34c5.buildOrigTopo(node34c5.getDevices());
        node34c5.fillCnBaseKv();
        DistriSys node34c5Clone = node34c5.clone();
        System.out.println("（1）回路电流潮流计算法测试");
        DsPowerflowTest.testConverged(node34c5Clone, true);
        DsPowerflowTest.printBusV(node34c5Clone.getActiveIslands()[0], false, false);
//        // 前推回退法测试
//        DistriSys node34c5Clone2 = node34c5.clone();
//        System.out.println("（2）前推回退潮流计算法测试");
//        DsPowerflowTest.testConverged(node34c5Clone2, false); // 前推回推法不收敛，用回路法能收敛
        System.out.println("");

        System.out.println("\n含分布式电源的IEEE34节点系统测试（含有两个环的配电网）\n");
        System.out.println("案例1：将826和858节点合并、822和848节点合并，DG配置同前方案1");
        DistriSys node34c1LoopClone = node34c1.clone();
        MeshedDsPfTest.combineTwoNode(node34c1LoopClone.getDevices(), "826", "858");
        MeshedDsPfTest.combineTwoNode(node34c1LoopClone.getDevices(), "822", "848");
        node34c1LoopClone.buildOrigTopo(node34c1LoopClone.getDevices());
        node34c1LoopClone.fillCnBaseKv();
        DistriSys node34c1LoopClone2 = node34c1LoopClone.clone();
        System.out.println("（1）回路电流潮流计算法测试");
        DsPowerflowTest.testConverged(node34c1LoopClone, true);
        printMotorState(node34c1LoopClone.getActiveIslands()[0], false);
//        DsPowerflowTest.printBusV(node34c1LoopClone.getActiveIslands()[0], false, false);
//        System.out.println("（2）前推回退潮流计算法测试");
        // todo
        // 注：对环状网络不能用DsPowerflowTest.testConverged(node34c1LoopClone, false)或MeshedDsPfTest中的testConverged用前推回推法计算潮流
        System.out.println("");

        System.out.println("\n案例2：将826和858节点合并、822和848节点合并，DG配置同前方案2");
        DistriSys node34c2LoopClone = node34c2.clone();
        MeshedDsPfTest.combineTwoNode(node34c2LoopClone.getDevices(), "826", "858");
        MeshedDsPfTest.combineTwoNode(node34c2LoopClone.getDevices(), "822", "848");
        node34c2LoopClone.buildOrigTopo(node34c2LoopClone.getDevices());
        node34c2LoopClone.fillCnBaseKv();
        DistriSys node34c2LoopClone2 = node34c2LoopClone.clone();
        System.out.println("（1）回路电流潮流计算法测试");
        DsPowerflowTest.testConverged(node34c2LoopClone, true);
        printMotorState(node34c2LoopClone.getActiveIslands()[0], false);
//        DsPowerflowTest.printBusV(node34c2LoopClone.getActiveIslands()[0], false, false);
//        System.out.println("（2）前推回退潮流计算法测试");
        // todo
        System.out.println("");

        System.out.println("\n案例3：将826和858节点合并、822和848节点合并，DG配置同前方案3");
        DistriSys node34c3LoopClone = node34c3.clone();
        MeshedDsPfTest.combineTwoNode(node34c3LoopClone.getDevices(), "826", "858");
        MeshedDsPfTest.combineTwoNode(node34c3LoopClone.getDevices(), "822", "848");
        node34c3LoopClone.buildOrigTopo(node34c3LoopClone.getDevices());
        node34c3LoopClone.fillCnBaseKv();
        DistriSys node34c3LoopClone2 = node34c3LoopClone.clone();
        System.out.println("（1）回路电流潮流计算法测试");
        DsPowerflowTest.testConverged(node34c3LoopClone, true);
        printMotorState(node34c3LoopClone.getActiveIslands()[0], false);
//        DsPowerflowTest.printBusV(node34c3LoopClone.getActiveIslands()[0], false, false);
//        System.out.println("（2）前推回退潮流计算法测试");
        // todo
        System.out.println("");

        System.out.println("\n案例4：将826和858节点合并、822和848节点合并，DG配置同前方案4");
        DistriSys node34c4LoopClone = node34c4.clone();
        MeshedDsPfTest.combineTwoNode(node34c4LoopClone.getDevices(), "826", "858");
        MeshedDsPfTest.combineTwoNode(node34c4LoopClone.getDevices(), "822", "848");
        node34c4LoopClone.buildOrigTopo(node34c4LoopClone.getDevices());
        node34c4LoopClone.fillCnBaseKv();
        DistriSys node34c4LoopClone2 = node34c4LoopClone.clone();
        System.out.println("（1）回路电流潮流计算法测试");
        DsPowerflowTest.testConverged(node34c4LoopClone, true);
        printMotorState(node34c4LoopClone.getActiveIslands()[0], false);
//        DsPowerflowTest.printBusV(node34c4LoopClone.getActiveIslands()[0], false, false);
//        System.out.println("（2）前推回退潮流计算法测试");
        // todo
        System.out.println("");

        System.out.println("\n案例5：将826和858节点合并、822和848节点合并，DG配置同前方案5");
        DistriSys node34c5LoopClone = node34c5.clone();
        MeshedDsPfTest.combineTwoNode(node34c5LoopClone.getDevices(), "826", "858");
        MeshedDsPfTest.combineTwoNode(node34c5LoopClone.getDevices(), "822", "848");
        node34c5LoopClone.buildOrigTopo(node34c5LoopClone.getDevices());
        node34c5LoopClone.fillCnBaseKv();
        DistriSys node34c5LoopClone2 = node34c5LoopClone.clone();
        System.out.println("（1）回路电流潮流计算法测试");
        DsPowerflowTest.testConverged(node34c5LoopClone, true);
        printMotorState(node34c5LoopClone.getActiveIslands()[0], false);
//        DsPowerflowTest.printBusV(node34c5LoopClone.getActiveIslands()[0], false, false);
//        System.out.println("（2）前推回退潮流计算法测试");
        // todo
        System.out.println("");
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
