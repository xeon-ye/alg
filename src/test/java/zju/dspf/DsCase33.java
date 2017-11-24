package zju.dspf;

import org.jgrapht.UndirectedGraph;
import org.jgrapht.graph.SimpleGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.*;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/5/8
 */
public class DsCase33 implements DsModelCons {

    public static DsTopoIsland createOpenLoopCase33(int[] lineNos) {
        DsTopoIsland island = createRadicalCase33();
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(island);

        double baseKV = tns.get("1").getBaseKv();
        for (int lineNo : lineNos) {
            String id = String.valueOf(lineNo + 1);
            DsConnectNode cn = new DsConnectNode(id);
            cn.setBaseKv(baseKV);
            cn.setConnectedObjs(new ArrayList<MapObject>(0));
            DsTopoNode tn = new DsTopoNode();
            tn.setBaseKv(baseKV);
            tn.setConnectivityNodes(new ArrayList<DsConnectNode>(1));
            tn.getConnectivityNodes().add(cn);
            tns.put(cn.getId(), tn);
            island.getGraph().addVertex(tn);

            switch (lineNo) {
                case 33:
                    addFeeder(island, tns, "33 34 21 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
                    break;
                case 34:
                    addFeeder(island, tns, "34 9 35 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
                    break;
                case 35:
                    addFeeder(island, tns, "35 36 22 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
                    break;
                case 36:
                    addFeeder(island, tns, "36 18 37 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
                    break;
                case 37:
                    addFeeder(island, tns, "37 25 38 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
                    break;
                default:
                    break;
            }
        }
        island.initialIsland();
        for (DsTopoNode t : island.getTns())
            t.initialPhases();
        return island;
    }

    public static DsTopoIsland createOpenLoopCase33() {
        return createOpenLoopCase33(new int[]{33, 34, 35, 36, 37});
    }

    public static DsTopoIsland createMeshedCase33() {
        DsTopoIsland island = createRadicalCase33();
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(island);
        addFeeder(island, tns, "33 8 21 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        addFeeder(island, tns, "34 9 15 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        addFeeder(island, tns, "35 12 22 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        addFeeder(island, tns, "36 18 33 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        addFeeder(island, tns, "37 25 29 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        island.initialIsland();
        island.setRadical(false);
        return island;
    }

    public static DsTopoIsland createRadicalCase33() {
        return readRadicalIsland(DsCase33.class.getResourceAsStream("/other/case33.txt"), 1);
    }

    public static DsTopoIsland createRadicalCase69(){
        return readRadicalIsland(DsCase33.class.getResourceAsStream("/other/case69.txt"),1);
    }

    public static Map<String, DsTopoNode> createTnMap(DsTopoIsland island) {
        Map<String, DsTopoNode> tns = new HashMap<String, DsTopoNode>(island.getGraph().vertexSet().size());
        for (DsTopoNode tn : island.getGraph().vertexSet())
            tns.put(tn.getConnectivityNodes().get(0).getId(), tn);
        return tns;
    }

    private static DsTopoIsland readRadicalIsland(InputStream stream, int rootNum) {
        try {
            DsTopoIsland island = new DsTopoIsland();
            BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
            Map<MapObject, ThreePhaseLoad> loads = new HashMap<MapObject, ThreePhaseLoad>();
            Map<MapObject, GeneralBranch> lines = new HashMap<MapObject, GeneralBranch>();

            double baseKV = 1.0;
            String buffer;
            UndirectedGraph<DsTopoNode, MapObject> g = new SimpleGraph<DsTopoNode, MapObject>(MapObject.class);
            Map<String, DsTopoNode> tns = new HashMap<String, DsTopoNode>();

            island.setBranches(lines);
            island.setLoads(loads);
            island.setGraph(g);

            while ((buffer = reader.readLine()) != null) {
                if (buffer.equals("BaseKV")) {
                    buffer = reader.readLine();
                    baseKV = Double.parseDouble(buffer) / sqrt3;
                }
                if (buffer.equals("Buses")) {
                    while ((buffer = reader.readLine()) != null && !buffer.equals("-999")) {
                        String[] busSplit = buffer.split(" ");
                        double[][] s = new double[3][2];
                        for (int i = 0; i < 3; i++) {
                            s[i][0] = Double.parseDouble(busSplit[i * 2 + 1]) * 1000.;
                            s[i][1] = Double.parseDouble(busSplit[i * 2 + 2]) * 1000.;
                        }
                        BasicLoad load = new BasicLoad(LOAD_Y_PQ);
                        load.setConstantS(s);
                        //load.formPara(s, baseKV * 1000);
                        MapObject obj = new MapObject(busSplit[0]);
                        obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_SPOT_LOAD);
                        obj.setProperty(KEY_CONNECTED_NODE, busSplit[0]);
                        loads.put(obj, load);

                        DsConnectNode cn = new DsConnectNode(busSplit[0]);
                        cn.setBaseKv(baseKV);
                        cn.setConnectedObjs(new ArrayList<MapObject>());
                        cn.getConnectedObjs().add(obj);

                        DsTopoNode tn = new DsTopoNode();
                        tn.setBaseKv(baseKV);
                        tn.setConnectivityNodes(new ArrayList<DsConnectNode>(1));
                        tn.getConnectivityNodes().add(cn);

                        tns.put(busSplit[0], tn);
                        g.addVertex(tn);
                    }
                }
                if (buffer != null && buffer.equals("Branches")) {
                    while ((buffer = reader.readLine()) != null && !buffer.equals("-999")) {
                        addFeeder(island, tns, buffer);
                    }
                }
            }
            island.setSupplyTns(new DsTopoNode[]{tns.get(String.valueOf(rootNum))});
            island.setDispersedGens(new HashMap<MapObject, DispersedGen>(0));
            island.setPerUnitSys(false);
            island.setBalanced(false);
            island.setRadical(true);
            island.setActive(true);
            island.initialIsland();
            for (DsTopoNode tn : island.getTns())
                tn.initialPhases();
            return island;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException ioe) {
            ioe.printStackTrace();
        }
        return null;
    }

    public static void addFeeder(DsTopoIsland island, Map<String, DsTopoNode> tns, String buffer) {
        String[] branchSplit = buffer.split(" ");

        double[][] z_real = new double[3][3];
        double[][] z_image = new double[3][3];
        z_real[0][0] = Double.parseDouble(branchSplit[3]);//Zaa
        z_image[0][0] = Double.parseDouble(branchSplit[4]);
        z_real[1][1] = Double.parseDouble(branchSplit[5]);//Zbb
        z_image[1][1] = Double.parseDouble(branchSplit[6]);
        z_real[2][2] = Double.parseDouble(branchSplit[7]);//Zcc
        z_image[2][2] = Double.parseDouble(branchSplit[8]);
        z_real[0][1] = Double.parseDouble(branchSplit[9]);//Zab
        z_image[0][1] = Double.parseDouble(branchSplit[10]);
        z_real[1][0] = Double.parseDouble(branchSplit[9]);//Zba
        z_image[1][0] = Double.parseDouble(branchSplit[10]);
        z_real[0][2] = Double.parseDouble(branchSplit[11]);//Zac
        z_image[0][2] = Double.parseDouble(branchSplit[12]);
        z_real[2][0] = Double.parseDouble(branchSplit[11]);//Zca
        z_image[2][0] = Double.parseDouble(branchSplit[12]);
        z_real[1][2] = Double.parseDouble(branchSplit[13]);//Zbc
        z_image[1][2] = Double.parseDouble(branchSplit[14]);
        z_real[2][1] = Double.parseDouble(branchSplit[13]);//Zcb
        z_image[2][1] = Double.parseDouble(branchSplit[14]);
        Feeder feeder = new Feeder();
        feeder.setZ_real(z_real);
        feeder.setZ_imag(z_image);
        feeder.initialPhases();

        MapObject obj = new MapObject(branchSplit[1] + "-" + branchSplit[2]);
        obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_FEEDER);

        //tns.get(lineNodes.get(i)[0]).getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        //tns.get(lineNodes.get(i)[1]).getConnectivityNodes().get(1).getConnectedObjs().add(obj);
        island.getGraph().addEdge(tns.get(branchSplit[1]), tns.get(branchSplit[2]), obj);
        island.getBranches().put(obj, feeder);
    }

    public static void deleteFeeder(DsTopoIsland island, Map<String, DsTopoNode> tns, String buffer) {
        String[] branchSplit = buffer.split("-");

//        MapObject obj = new MapObject(branchSplit[0] + "-" + branchSplit[1]);
//        obj.setProperty(KEY_RESOURCE_TYPE, RESOURCE_FEEDER);
        //tns.get(lineNodes.get(i)[0]).getConnectivityNodes().get(0).getConnectedObjs().add(obj);
        //tns.get(lineNodes.get(i)[1]).getConnectivityNodes().get(1).getConnectedObjs().add(obj);
        if(island.getGraph().removeEdge(tns.get(branchSplit[0]), tns.get(branchSplit[1]))==null){
            System.out.println("can't remove");
        }
        MapObject obj = null;
        for(MapObject i : island.getBranches().keySet()){
            if(i.getProperties().get("name").equals(buffer)){
                obj = i;
                break;
            }
        }
        if(island.getBranches().remove(obj)==null){
            System.out.println("can't remove");
        };
    }
}
