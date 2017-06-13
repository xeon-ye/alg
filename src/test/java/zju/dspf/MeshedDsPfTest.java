package zju.dspf;

import Jama.Matrix;
import junit.framework.TestCase;
import zju.devmodel.MapObject;
import zju.dsmodel.*;
import zju.dsntp.DsPowerflow;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/5/8
 */
public class MeshedDsPfTest extends TestCase implements DsModelCons {

    public void testNotConverged_case33() {

        DsTopoIsland island1 = DsCase33.createOpenLoopCase33();
        for (GeneralBranch b : island1.getBranches().values()) {
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
        for (ThreePhaseLoad load : island1.getLoads().values()) {
            for (double[] s : ((BasicLoad) load).getConstantS()) {
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

    //initialIsland中注释；潮流计算中注释
    public void testCase33ReconfigByDPSO() {
        DsTopoIsland island = DsCase33.createRadicalCase33();
        //tns键值为tn中的cn的id值，cn的id为原始编号
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(island);

        DsCase33.addFeeder(island, tns, "33 8 21 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(island, tns, "34 9 15 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(island, tns, "35 12 22 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(island, tns, "36 18 33 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(island, tns, "37 25 29 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        island.initialIsland();

        DPSOInReconfig dpsoInReconfig = new DPSOInReconfig(island);
        dpsoInReconfig.initial();
        dpsoInReconfig.run();
        dpsoInReconfig.showResult();
    }

    public void testCase33ReconfigStrategy() {
        //获取完整配网拓扑
        DsTopoIsland completeIsland = DsCase33.createRadicalCase33();
        //tns键值为tn中的cn的id值，cn的id为原始编号
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(completeIsland);
        DsCase33.addFeeder(completeIsland, tns, "33 8 21 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "34 9 15 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "35 12 22 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "36 18 33 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "37 25 29 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        completeIsland.initialIsland();

        //粒子群寻优
        DPSOInReconfig dpsoInReconfig = new DPSOInReconfig(completeIsland);
        dpsoInReconfig.initial();
        dpsoInReconfig.run();
        dpsoInReconfig.showResult();

        //获取初始配网拓扑
        DsTopoIsland originIsland = DsCase33.createRadicalCase33();
        originIsland.initialIsland();

        int count = 0;
        for (double[] bestPosition : DPSOInReconfig.getGlobalBestPositionList()) {
            count++;
            System.out.println("==============第"+count+"次第二层优化==============");
            //更新标号（必要操作，影响断开、闭合线路的提取。ReconfigStrategy在加入一条branch后，会进行一次更新）
            originIsland.initialIsland();

            //获取优解对应的配网拓扑
            DsTopoIsland finalIsland = completeIsland.clone();
            tns = DsCase33.createTnMap(finalIsland);
            for (int i = 0; i < bestPosition.length; i++) {
                if (bestPosition[i] == 1) {
                    String toOpenBranch = finalIsland.getIdToBranch().get(i + 1).getName();
                    DsCase33.deleteFeeder(finalIsland, tns, toOpenBranch);
                }
            }
            finalIsland.initialIsland();

            //获取切断线路
            List<String> toOpenBranches = new ArrayList<>();
            for (MapObject branch1 : originIsland.getIdToBranch().values()) {
                boolean flag = false;
                for (MapObject branch2 : finalIsland.getIdToBranch().values()) {
                    if (branch1.getName().equals(branch2.getName())) {
                        flag = true;
                        break;
                    }
                }
                if (!flag) {
                    toOpenBranches.add(branch1.getName());
                }
            }

            //获取闭合线路
            List<String> toCloseBranches = new ArrayList<>();
            for (MapObject branch1 : finalIsland.getIdToBranch().values()) {
                boolean flag = false;
                for (MapObject branch2 : originIsland.getIdToBranch().values()) {
                    if (branch1.getName().equals(branch2.getName())) {
                        flag = true;
                        break;
                    }
                }
                if (!flag) {
                    toCloseBranches.add(branch1.getName());
                }
            }

            System.out.println("断开线路");
            for (String i : toOpenBranches) {
                System.out.println(i);
            }
            System.out.println("闭合线路");
            for (String i : toCloseBranches) {
                System.out.println(i);
            }
            String result;
            result = ReconfigStrategy.getReconfigStrategy(originIsland, toOpenBranches, toCloseBranches);
            if(result != null){
                System.out.println("实施策略："+result);
                break;
            }else{
                System.out.println("==============第"+count+"次第二层优化失败==============");
            }
        }
    }

    public void testCalZMatrix() {
        Matrix[] matrices = new Matrix[3];
        double[][] ele = {{1, 0, 2, -1, 1, 1}, {0, 1, 1, 1, 1, 2}, {-1, 1, 1, 0, 1, 0}};
        matrices[0] = new Matrix(ele);
        matrices[1] = new Matrix(ele);
        matrices[2] = new Matrix(ele);

        Matrix[][] result = ReconfigStrategy.calZMatrix(3, matrices);

        for (int i = 0; i < 3; i++) {
            result[i][0].print(5, 4);
            result[i][1].print(5, 4);
        }

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
        //assertTrue(pf.isConverged());
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
