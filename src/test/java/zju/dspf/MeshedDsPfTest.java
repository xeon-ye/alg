package zju.dspf;

import Jama.Matrix;
import junit.framework.TestCase;
import zju.devmodel.MapObject;
import zju.dsmodel.*;
import zju.dsntp.DsPowerflow;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/5/8
 */
public class MeshedDsPfTest extends TestCase implements DsModelCons {

    public void testNotConverged_case33() throws IOException {

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

    public void testRadicalCase33() throws IOException {
        DsTopoIsland island = DsCase33.createRadicalCase33();
        island.initialIsland();
        DsPowerflow pf = new DsPowerflow();
        pf.setTolerance(1e-1);
        pf.doLcbPf(island);

        DsPowerflowTest.printBusV(island, true, false);

        double fitness = 0;
        for (MapObject i : island.getBranches().keySet()) {
            Feeder feeder = (Feeder) island.getBranches().get(i);
            double[][] Z_real = feeder.getZ_real();

            double[][] branchHeadI = island.getBranchHeadI().get(i);
            double loss = 0;
            for (int j = 0; j < 3; j++) {
                loss += (branchHeadI[j][0] * branchHeadI[j][0] + branchHeadI[j][1] * branchHeadI[j][1]) * Z_real[j][j];
            }
            fitness += loss;
        }
        System.out.println("fitness:"+fitness);
    }

    public void testCase69ReconfigByDPSO() throws IOException {
        //????????????????????????
        DsTopoIsland completeIsland = DsCase33.createRadicalCase69();
        //tns?????????tn??????cn???id??????cn???id???????????????
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(completeIsland);
        DsCase33.addFeeder(completeIsland, tns, "69 13 20 0.5000 0.5000 0.5000 0.5000 0.5000 0.5000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
        DsCase33.addFeeder(completeIsland, tns, "70 15 69 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
        DsCase33.addFeeder(completeIsland, tns, "71 39 48 2.0000 2.0000 2.0000 2.0000 2.0000 2.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
        DsCase33.addFeeder(completeIsland, tns, "72 11 66 0.5000 0.5000 0.5000 0.5000 0.5000 0.5000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
        DsCase33.addFeeder(completeIsland, tns, "73 27 54 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
        completeIsland.initialIsland();

        //???????????????
        DPSOInReconfig dpsoInReconfig = new DPSOInReconfig(completeIsland);
        dpsoInReconfig.initial();
        dpsoInReconfig.run();
        dpsoInReconfig.showResult();
    }

    public void testCase69ReconfigStrategy() throws IOException {
        //????????????????????????
        DsTopoIsland completeIsland = DsCase33.createRadicalCase69();
        //tns?????????tn??????cn???id??????cn???id???????????????
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(completeIsland);
        DsCase33.addFeeder(completeIsland, tns, "69 13 20 0.5000 0.5000 0.5000 0.5000 0.5000 0.5000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
        DsCase33.addFeeder(completeIsland, tns, "70 15 69 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
        DsCase33.addFeeder(completeIsland, tns, "71 39 48 2.0000 2.0000 2.0000 2.0000 2.0000 2.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
        DsCase33.addFeeder(completeIsland, tns, "72 11 66 0.5000 0.5000 0.5000 0.5000 0.5000 0.5000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
        DsCase33.addFeeder(completeIsland, tns, "73 27 54 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
        completeIsland.initialIsland();

        //???????????????
        DPSOInReconfig dpsoInReconfig = new DPSOInReconfig(completeIsland);
        dpsoInReconfig.initial();
        dpsoInReconfig.run();
        dpsoInReconfig.showResult();

        //????????????????????????
        DsTopoIsland originIsland = DsCase33.createRadicalCase69();
        originIsland.initialIsland();

        ReconfigStrategy reconfigStrategy = new ReconfigStrategy();
        int count = 0;
        for (double[] bestPosition : DPSOInReconfig.getGlobalBestPositionList()) {
            count++;
            System.out.println("==============???"+count+"??????????????????==============");
            //?????????????????????????????????????????????????????????????????????ReconfigStrategy???????????????branch??????????????????????????????
            originIsland.initialIsland();

            //?????????????????????????????????
            DsTopoIsland finalIsland = completeIsland.clone();
            tns = DsCase33.createTnMap(finalIsland);
            for (int i = 0; i < bestPosition.length; i++) {
                if (bestPosition[i] == 1) {
                    String toOpenBranch = finalIsland.getIdToBranch().get(i + 1).getName();
                    DsCase33.deleteFeeder(finalIsland, tns, toOpenBranch);
                }
            }
            finalIsland.initialIsland();

            //??????????????????
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

            //??????????????????
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

            System.out.println("????????????");
            for (String i : toOpenBranches) {
                System.out.println(i);
            }
            System.out.println("????????????");
            for (String i : toCloseBranches) {
                System.out.println(i);
            }
            //????????????????????????
            String result;
            result = reconfigStrategy.getReconfigStrategy(originIsland, toOpenBranches, toCloseBranches);
            if(result != null){
                System.out.println("???????????????"+result);
                break;
            }else{
                System.out.println("==============???"+count+"????????????????????????==============");
            }
        }
    }



    //initialIsland????????????????????????????????? DsTopoIsland???NewtonSolver???DsPowerflow
    public void testCase33ReconfigByDPSO() throws IOException {
        //????????????????????????
        DsTopoIsland completeIsland = DsCase33.createRadicalCase33();
        //tns?????????tn??????cn???id??????cn???id???????????????
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(completeIsland);
        DsCase33.addFeeder(completeIsland, tns, "33 8 21 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "34 9 15 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "35 12 22 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "36 18 33 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "37 25 29 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        completeIsland.initialIsland();

        //???????????????
        DPSOInReconfig dpsoInReconfig = new DPSOInReconfig(completeIsland);
        dpsoInReconfig.initial();
        dpsoInReconfig.run();
        dpsoInReconfig.showResult();
    }



    public void testCase33ReconfigStrategy() throws IOException {
        //????????????????????????
        DsTopoIsland completeIsland = DsCase33.createRadicalCase33();
        //tns?????????tn??????cn???id??????cn???id???????????????
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(completeIsland);
        DsCase33.addFeeder(completeIsland, tns, "33 8 21 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "34 9 15 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "35 12 22 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "36 18 33 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "37 25 29 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        completeIsland.initialIsland();

        //???????????????
        DPSOInReconfig dpsoInReconfig = new DPSOInReconfig(completeIsland);
        dpsoInReconfig.initial();
        dpsoInReconfig.run();
        dpsoInReconfig.showResult();

        //????????????????????????
        DsTopoIsland originIsland = DsCase33.createRadicalCase33();
        originIsland.initialIsland();

        ReconfigStrategy reconfigStrategy = new ReconfigStrategy();
        int count = 0;
        for (double[] bestPosition : DPSOInReconfig.getGlobalBestPositionList()) {
            count++;
            System.out.println("==============???"+count+"??????????????????==============");
            //?????????????????????????????????????????????????????????????????????ReconfigStrategy???????????????branch??????????????????????????????
            originIsland.initialIsland();

            //?????????????????????????????????
            DsTopoIsland finalIsland = completeIsland.clone();
            tns = DsCase33.createTnMap(finalIsland);
            for (int i = 0; i < bestPosition.length; i++) {
                if (bestPosition[i] == 1) {
                    String toOpenBranch = finalIsland.getIdToBranch().get(i + 1).getName();
                    DsCase33.deleteFeeder(finalIsland, tns, toOpenBranch);
                }
            }
            finalIsland.initialIsland();

            //??????????????????
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

            //??????????????????
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

            System.out.println("????????????");
            for (String i : toOpenBranches) {
                System.out.println(i);
            }
            System.out.println("????????????");
            for (String i : toCloseBranches) {
                System.out.println(i);
            }
            String result;
            result = reconfigStrategy.getReconfigStrategy(originIsland, toOpenBranches, toCloseBranches);
            if(result != null){
                System.out.println("???????????????"+result);
                break;
            }else{
                System.out.println("==============???"+count+"????????????????????????==============");
            }
        }
    }

    public void testCase33ReconfigStrategy1() throws IOException {
        //????????????????????????
        DsTopoIsland completeIsland = DsCase33.createRadicalCase33();
        //tns?????????tn??????cn???id??????cn???id???????????????
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(completeIsland);
        DsCase33.addFeeder(completeIsland, tns, "33 8 21 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "34 9 15 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "35 12 22 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "36 18 33 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        DsCase33.addFeeder(completeIsland, tns, "37 25 29 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        completeIsland.initialIsland();

        //????????????????????????
        DsTopoIsland originIsland = DsCase33.createRadicalCase33();
        originIsland.initialIsland();

        List<String> toOpenBranches = new ArrayList<>();
        List<String> toCloseBranches = new ArrayList<>();

        toOpenBranches.add("7-8");
        toOpenBranches.add("9-10");
        toOpenBranches.add("32-33");
        toOpenBranches.add("14-15");


//        //????????????????????????????????????10-11???
//        toOpenBranches.add("10-11");
//        //????????????????????????????????????14-15????????????7-8???
//        toOpenBranches.add("14-15");
//
//        toOpenBranches.add("7-8");
//        toOpenBranches.add("32-33");

        toCloseBranches.add("8-21");
        toCloseBranches.add("12-22");
        toCloseBranches.add("9-15");
        toCloseBranches.add("18-33");


//        //????????????????????????????????????9-15???
//        toCloseBranches.add("9-15");
//        //????????????????????????????????????12-22???
//        toCloseBranches.add("12-22");
//
//        toCloseBranches.add("18-33");
//        toCloseBranches.add("8-21");

        //???????????????
        Map<String,Integer> numToBranch = new HashMap<>();

        for(int i = 0; i < toCloseBranches.size();i++){
            numToBranch.put(toCloseBranches.get(i),i);
        }

        for(int i = toCloseBranches.size();i<toCloseBranches.size()+toOpenBranches.size();i++){
            numToBranch.put(toOpenBranches.get(i - toCloseBranches.size()),i);
        }

        String result;
        ReconfigStrategy reconfigStrategy = new ReconfigStrategy();
        reconfigStrategy.setNumToBranch(numToBranch);
        result = reconfigStrategy.getReconfigStrategy(originIsland, toOpenBranches, toCloseBranches);
        if(result != null){
            System.out.println("???????????????"+result);
        }else{
            System.out.println("==============?????????????????????==============");
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

    public void testLoopedPf_case33() throws IOException {
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
        System.out.println("?????????????????????" + (System.currentTimeMillis() - start) + "ms.");
        DsPowerflowTest.checkConstraints(island);
    }

    public void testLoopedPf_case123() {
        DistriSys ds = getLoopedCase123();
        DistriSys sys = ds.clone();
        sys.buildDynamicTopo();
        int branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        int busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("???????????????" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        sys = ds.clone();
        combineTwoNode(sys.getDevices(), "85", "75");
        combineTwoNode(sys.getDevices(), "36", "57");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("???????????????" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        combineTwoNode(sys.getDevices(), "56", "90");
        combineTwoNode(sys.getDevices(), "39", "66");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("???????????????" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        combineTwoNode(sys.getDevices(), "23", "44");
        combineTwoNode(sys.getDevices(), "62", "101");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("???????????????" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        combineTwoNode(sys.getDevices(), "81", "86");
        combineTwoNode(sys.getDevices(), "70", "100");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("???????????????" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        combineTwoNode(sys.getDevices(), "9", "18");
        combineTwoNode(sys.getDevices(), "30", "47");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("???????????????" + (branchNum - busNum + 1));
        DsPowerflowTest.testConverged(sys, true);

        combineTwoNode(sys.getDevices(), "34", "94");
        combineTwoNode(sys.getDevices(), "64", "300");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("???????????????" + (branchNum - busNum + 1));
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
