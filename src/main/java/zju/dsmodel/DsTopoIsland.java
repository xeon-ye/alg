package zju.dsmodel;

import org.jgrapht.UndirectedGraph;
import org.jgrapht.graph.SimpleGraph;
import org.jgrapht.traverse.BreadthFirstIterator;
import zju.devmodel.MapObject;
import zju.util.JOFileUtil;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-10
 */
public class DsTopoIsland implements Serializable, DsModelCons {

    public static final String EARTH_NODE_ID = "ground-node-3";

    private boolean isActive = false;

    private boolean isRadical = false;

    private boolean isPerUnitSys = false;

    private boolean isBalanced = false;

    private DsTopoNode[] supplyTns;

    private List<DsTopoNode> tns;

    private int spotLoadNum, distriLoadNum, shuntCapacitorNum, distriGenNum;

    private Map<String, MapObject> devices;
    //三相不分离的拓扑图，不包含地节点
    private UndirectedGraph<DsTopoNode, MapObject> graph;
    //三相分离的详细拓扑图，包含地节点
    private UndirectedGraph<String, DetailedEdge> detailedG;
    //节点编号和对象之间的映射关系
    public Map<Integer, DsTopoNode> busNoToTn;
    //支路号和对象之间的映射关系
    private Map<Integer, MapObject> idToBranch;
    //支路计算模型，支路可能是：馈线，变压器，调压器
    private Map<MapObject, GeneralBranch> branches;
    //负荷计算模型，包括集中式/分布式负荷，并联电容，
    private Map<MapObject, ThreePhaseLoad> loads;
    //分布式电源计算模型
    private Map<MapObject, DispersedGen> dispersedGens;

    //下面3个是潮流结果，电流用实部和虚部表示
    public Map<MapObject, double[][]> branchHeadI;
    //流出电流，对于馈线这中流入流出电流相同的情况，不存储
    public Map<MapObject, double[][]> branchTailI;
    //节点电压
    public Map<DsTopoNode, double[][]> busV;
    //用于表征电压是实部和虚部表示，还是用幅值和相角表示
    private boolean isVCartesian = false;
    //用于表征电流是实部和虚部表示，还是用幅值和相角表示
    private boolean isICartesian = false;

    public void initialVariables() {
        initialBusV();
        branchHeadI = new HashMap<MapObject, double[][]>(getBranches().size());
        branchTailI = new HashMap<MapObject, double[][]>(getBranches().size());
        for (MapObject b : getBranches().keySet()) {
            GeneralBranch branch = getBranches().get(b);
            if (isBalanced) {
                if ((branch instanceof Feeder) && ((Feeder) branch).isShuntYNeglected()) {
                    double[][] cc = new double[1][2];
                    branchHeadI.put(b, cc);
                    branchTailI.put(b, cc);
                } else {
                    branchHeadI.put(b, new double[1][2]);
                    branchTailI.put(b, new double[1][2]);
                }
            } else {
                if ((branch instanceof Feeder) && ((Feeder) branch).isShuntYNeglected()) {
                    double[][] cc = new double[3][2];
                    branchHeadI.put(b, cc);
                    branchTailI.put(b, cc);
                } else {
                    branchHeadI.put(b, new double[3][2]);
                    branchTailI.put(b, new double[3][2]);
                }
            }
        }
    }

    public void initialBusV() {
        busV = new HashMap<DsTopoNode, double[][]>(getTns().size());
        for (DsTopoNode tn : getTns()) {
            double v = 1.0;
            if (!isPerUnitSys())
                v = tn.getBaseKv() * 1e3;
            if (isBalanced) {
                busV.put(tn, new double[][]{{v, 0}});
            } else {
                busV.put(tn, new double[][]{
                        {v, 0},
                        {v * cosB, v * sinB},
                        {v * cosC, v * sinC}});
            }
        }
    }

    /**
     * <br>建立电气岛的三相图，馈线和变压器线圈都作为单独的支路;</br>
     * <br>图中支路的名称以馈线，变压器，负荷的id + "-" 开头;</br>
     * <br>对于电源以"Supply-" 开头,"-0","-1","-2"结尾;</br>
     * <br>对于馈线，以"-0","-1","-2"结尾；</br>
     * <br>对于变压器线圈，以"-S0","-S1","-S2","-T0","-T1","-T2"等结尾;</br>
     * <Br>对于负荷，以"-0","-1","-2"结尾;</Br>
     */
    public void buildDetailedGraph() {
        devices = new HashMap<String, MapObject>(spotLoadNum + distriLoadNum + distriGenNum + shuntCapacitorNum + idToBranch.size());
        for (MapObject obj : idToBranch.values())
            devices.put(obj.getId(), obj);
        for (DsTopoNode tn : tns)
            for (MapObject obj : tn.getConnectedDev())
                if (!devices.containsKey(obj.getId()))
                    devices.put(obj.getId(), obj);
        detailedG = new SimpleGraph<String, DetailedEdge>(DetailedEdge.class);
        String id1, id2, id3;
        //第一步，加入大地节点以及电源节点
        detailedG.addVertex(EARTH_NODE_ID);
        for (DsTopoNode tn : getSupplyTns()) {
            if (tn.getType() != DsTopoNode.TYPE_SUB)
                continue;
            id1 = tn.getBusNo() + "-0";
            id2 = tn.getBusNo() + "-1";
            id3 = tn.getBusNo() + "-2";
            detailedG.addVertex(id1);
            detailedG.addVertex(id2);
            detailedG.addVertex(id3);
            DetailedEdge edge = new DetailedEdge(DetailedEdge.EDGE_TYPE_SUPPLIER, tn.getBusNo(), 0);
            detailedG.addEdge(EARTH_NODE_ID, id1, edge);
            edge = new DetailedEdge(DetailedEdge.EDGE_TYPE_SUPPLIER, tn.getBusNo(), 1);
            detailedG.addEdge(EARTH_NODE_ID, id2, edge);
            edge = new DetailedEdge(DetailedEdge.EDGE_TYPE_SUPPLIER, tn.getBusNo(), 2);
            detailedG.addEdge(EARTH_NODE_ID, id3, edge);
            break;//todo：目前只支持只有一个变电站节点
        }
        //第二步，分析所有的支路
        DsTopoNode tn1, tn2;
        int count;
        for (MapObject obj : branches.keySet()) {
            GeneralBranch branch = branches.get(obj);
            tn1 = graph.getEdgeSource(obj);
            tn2 = graph.getEdgeTarget(obj);
            //todo: 对于变压器绕组来说，默认节点编号小的为源端，这一点要注意
            if (tn1.getBusNo() > tn2.getBusNo()) {
                tn2 = graph.getEdgeSource(obj);
                tn1 = graph.getEdgeTarget(obj);
            }
            if (branch instanceof Feeder) {
                Feeder feeder = (Feeder) branch;
                count = 0;
                for (int i = 0; i < 3; i++) {
                    if (Math.abs(feeder.getZ_real()[i][i]) > ZERO_LIMIT
                            || Math.abs(feeder.getZ_imag()[i][i]) > ZERO_LIMIT)
                        count++;
                }
                DetailedEdge[] edges = new DetailedEdge[count];
                count = 0;
                for (int i = 0; i < 3; i++) {
                    if (Math.abs(feeder.getZ_real()[i][i]) > ZERO_LIMIT
                            || Math.abs(feeder.getZ_imag()[i][i]) > ZERO_LIMIT) {
                        id1 = tn1.getBusNo() + "-" + i;
                        id2 = tn2.getBusNo() + "-" + i;
                        if (!detailedG.containsVertex(id1))
                            detailedG.addVertex(id1);
                        if (!detailedG.containsVertex(id2))
                            detailedG.addVertex(id2);
                        edges[count] = new DetailedEdge(DetailedEdge.EDGE_TYPE_FEEDER, tn1.getBusNo(), tn2.getBusNo(), i, obj.getId());
                        detailedG.addEdge(id1, id2, edges[count]);
                        count++;
                    }
                }
                int count2;
                for (DetailedEdge edge : edges) {
                    DetailedEdge[] otherEdges = new DetailedEdge[count - 1];
                    count2 = 0;
                    for (DetailedEdge e : edges)
                        if (e != edge)
                            otherEdges[count2++] = e;
                    edge.setOtherEdgesOfSameFeeder(otherEdges);
                }
            } else if (branch instanceof Transformer) {
                Transformer tf = (Transformer) branch;
                for (int i = 0; i < 3; i++) {
                    id1 = tn1.getBusNo() + "-" + i;
                    id2 = tn2.getBusNo() + "-" + i;
                    if (!detailedG.containsVertex(id1))
                        detailedG.addVertex(id1);
                    if (!detailedG.containsVertex(id2))
                        detailedG.addVertex(id2);
                }
                switch (tf.getConnType()) {
                    case Transformer.CONN_TYPE_D_GrY:
                        id1 = tn1.getBusNo() + "-0";
                        id2 = tn1.getBusNo() + "-1";
                        id3 = tn1.getBusNo() + "-2";
                        DetailedEdge edge1 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 0, true, obj.getId());
                        DetailedEdge edge2 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 1, true, obj.getId());
                        DetailedEdge edge3 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 2, true, obj.getId());
                        detailedG.addEdge(id1, id2, edge1);
                        detailedG.addEdge(id2, id3, edge2);
                        detailedG.addEdge(id3, id1, edge3);
                        id1 = tn2.getBusNo() + "-0";
                        id2 = tn2.getBusNo() + "-1";
                        id3 = tn2.getBusNo() + "-2";
                        DetailedEdge edge4 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 0, false, obj.getId());
                        DetailedEdge edge5 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 1, false, obj.getId());
                        DetailedEdge edge6 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 2, false, obj.getId());
                        detailedG.addEdge(id1, EARTH_NODE_ID, edge4);
                        detailedG.addEdge(id2, EARTH_NODE_ID, edge5);
                        detailedG.addEdge(id3, EARTH_NODE_ID, edge6);
                        edge1.setOtherEdgeOfTf(edge5);
                        edge5.setOtherEdgeOfTf(edge1);
                        edge2.setOtherEdgeOfTf(edge6);
                        edge6.setOtherEdgeOfTf(edge2);
                        edge3.setOtherEdgeOfTf(edge4);
                        edge4.setOtherEdgeOfTf(edge3);
                        break;
                    case Transformer.CONN_TYPE_GrY_GrY:
                        id1 = tn1.getBusNo() + "-0";
                        id2 = tn1.getBusNo() + "-1";
                        id3 = tn1.getBusNo() + "-2";
                        edge1 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 0, true, obj.getId());
                        edge2 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 1, true, obj.getId());
                        edge3 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 2, true, obj.getId());
                        detailedG.addEdge(id1, EARTH_NODE_ID, edge1);
                        detailedG.addEdge(id2, EARTH_NODE_ID, edge2);
                        detailedG.addEdge(id3, EARTH_NODE_ID, edge3);
                        edge4 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 0, false, obj.getId());
                        edge5 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 1, false, obj.getId());
                        edge6 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 2, false, obj.getId());
                        id1 = tn2.getBusNo() + "-0";
                        id2 = tn2.getBusNo() + "-1";
                        id3 = tn2.getBusNo() + "-2";
                        detailedG.addEdge(id1, EARTH_NODE_ID, edge4);
                        detailedG.addEdge(id2, EARTH_NODE_ID, edge5);
                        detailedG.addEdge(id3, EARTH_NODE_ID, edge6);
                        edge1.setOtherEdgeOfTf(edge4);
                        edge4.setOtherEdgeOfTf(edge1);
                        edge2.setOtherEdgeOfTf(edge5);
                        edge5.setOtherEdgeOfTf(edge2);
                        edge3.setOtherEdgeOfTf(edge6);
                        edge6.setOtherEdgeOfTf(edge3);
                        break;
                    case Transformer.CONN_TYPE_Y_D:
                        id1 = tn1.getBusNo() + "-0";
                        id2 = tn1.getBusNo() + "-1";
                        id3 = tn1.getBusNo() + "-2";
                        String neutralNodeId = obj.getId() + "-3";
                        detailedG.addVertex(neutralNodeId);
                        edge1 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 0, true, obj.getId());
                        edge2 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 1, true, obj.getId());
                        edge3 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 2, true, obj.getId());
                        detailedG.addEdge(id1, neutralNodeId, edge1);
                        detailedG.addEdge(id2, neutralNodeId, edge2);
                        detailedG.addEdge(id3, neutralNodeId, edge3);
                        id1 = tn2.getBusNo() + "-0";
                        id2 = tn2.getBusNo() + "-1";
                        id3 = tn2.getBusNo() + "-2";
                        edge4 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 0, false, obj.getId());
                        edge5 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 1, false, obj.getId());
                        edge6 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 2, false, obj.getId());
                        detailedG.addEdge(id1, id2, edge4);
                        detailedG.addEdge(id2, id3, edge5);
                        detailedG.addEdge(id3, id1, edge6);
                        edge1.setOtherEdgeOfTf(edge4);
                        edge4.setOtherEdgeOfTf(edge1);
                        edge2.setOtherEdgeOfTf(edge5);
                        edge5.setOtherEdgeOfTf(edge2);
                        edge3.setOtherEdgeOfTf(edge6);
                        edge6.setOtherEdgeOfTf(edge3);
                        break;
                    case Transformer.CONN_TYPE_D_D:
                        id1 = tn1.getBusNo() + "-0";
                        id2 = tn1.getBusNo() + "-1";
                        id3 = tn1.getBusNo() + "-2";
                        edge1 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 0, true, obj.getId());
                        edge2 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 1, true, obj.getId());
                        edge3 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getBusNo(), 2, true, obj.getId());
                        detailedG.addEdge(id1, id2, edge1);
                        detailedG.addEdge(id2, id3, edge2);
                        detailedG.addEdge(id3, id1, edge3);
                        id1 = tn2.getBusNo() + "-0";
                        id2 = tn2.getBusNo() + "-1";
                        id3 = tn2.getBusNo() + "-2";
                        edge4 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 0, false, obj.getId());
                        edge5 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 1, false, obj.getId());
                        edge6 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getBusNo(), 2, false, obj.getId());
                        detailedG.addEdge(id1, id2, edge4);
                        detailedG.addEdge(id2, id3, edge5);
                        detailedG.addEdge(id3, id1, edge6);
                        edge1.setOtherEdgeOfTf(edge4);
                        edge4.setOtherEdgeOfTf(edge1);
                        edge2.setOtherEdgeOfTf(edge5);
                        edge5.setOtherEdgeOfTf(edge2);
                        edge3.setOtherEdgeOfTf(edge6);
                        edge6.setOtherEdgeOfTf(edge3);
                        break;
                    default:
                        break;
                }
            }
        }

        //第三步，分析所有的负荷
        for (DsTopoNode tn : getTns()) {
            for (MapObject obj : tn.getConnectedDev()) {
                if (!loads.containsKey(obj))
                    continue;
                BasicLoad load = (BasicLoad) loads.get(obj);
                formGraphOfLoad(tn, load, obj);
            }
        }
        //第四步，分析所有的分布式电源
        for (DsTopoNode tn : getTns()) {
            for (MapObject obj : tn.getConnectedDev()) {
                if (!dispersedGens.containsKey(obj))
                    continue;
                DispersedGen gen = dispersedGens.get(obj);
                id1 = tn.getBusNo() + "-0";
                id2 = tn.getBusNo() + "-1";
                id3 = tn.getBusNo() + "-2";
                switch (gen.getMode()) {
                    case DispersedGen.MODE_PV:
                        //todo: 认为PV节点只有接地这种模式
                        DetailedEdge e = detailedG.getEdge(id1, EARTH_NODE_ID);
                        if (e == null) {
                            e = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getBusNo(), 0);
                            detailedG.addEdge(id1, EARTH_NODE_ID, e);
                        } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING
                                || e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX)
                            e.setEdgeType(DetailedEdge.EDGE_TYPE_DG_TF_MIX);
                        else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD)
                            e.setEdgeType(DetailedEdge.EDGE_TYPE_DG);
                        e.setDgId(obj.getId());

                        e = detailedG.getEdge(id2, EARTH_NODE_ID);
                        if (e == null) {
                            e = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getBusNo(), 1);
                            detailedG.addEdge(id2, EARTH_NODE_ID, e);
                        } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING
                                || e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX)
                            e.setEdgeType(DetailedEdge.EDGE_TYPE_DG_TF_MIX);
                        else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD)
                            e.setEdgeType(DetailedEdge.EDGE_TYPE_DG);
                        e.setDgId(obj.getId());

                        e = detailedG.getEdge(id3, EARTH_NODE_ID);
                        if (e == null) {
                            e = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getBusNo(), 2);
                            e.setDevId(obj.getId());
                            detailedG.addEdge(id3, EARTH_NODE_ID, e);
                        } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING
                                || e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX)
                            e.setEdgeType(DetailedEdge.EDGE_TYPE_DG_TF_MIX);
                        else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD)
                            e.setEdgeType(DetailedEdge.EDGE_TYPE_DG);
                        e.setDgId(obj.getId());
                        break;
                    case DispersedGen.MODE_IM:
                        switch (gen.getMotor().getConnType()) {
                            case InductionMachine.CONN_TYPE_Y:
                                String neutralNodeId = tn.getBusNo() + "-motor-3";
                                detailedG.addVertex(neutralNodeId);
                                DetailedEdge edge1 = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getBusNo(), 0);
                                DetailedEdge edge2 = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getBusNo(), 1);
                                DetailedEdge edge3 = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getBusNo(), 2);
                                edge1.setDgId(obj.getId());
                                edge2.setDgId(obj.getId());
                                edge3.setDgId(obj.getId());
                                detailedG.addEdge(id1, neutralNodeId, edge1);
                                detailedG.addEdge(id2, neutralNodeId, edge2);
                                detailedG.addEdge(id3, neutralNodeId, edge3);
                                break;
                            case InductionMachine.CONN_TYPE_D:
                                System.out.println("-------------- not finished ----------------");
                                break;
                            default:
                                break;
                        }
                        break;
                    default:
                        break;
                }
            }
        }
    }

    private void formGraphOfLoad(DsTopoNode tn, BasicLoad load, MapObject obj) {
        String id1 = tn.getBusNo() + "-0";
        String id2 = tn.getBusNo() + "-1";
        String id3 = tn.getBusNo() + "-2";
        if (load.getMode().equals(LOAD_D_I)) {
            if (Math.abs(load.getConstantI()[0][0]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id1, id2);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 0);
                    e.setDevId(obj.getId());
                    e.setLoadD(true);
                    detailedG.addEdge(id1, id2, e);
                }
                setLoadEdgeI(e, load.getConstantI()[0][0], load.getConstantI()[0][1]);
            }
            if (Math.abs(load.getConstantI()[1][0]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id2, id3);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 1);
                    e.setDevId(obj.getId());
                    e.setLoadD(true);
                    detailedG.addEdge(id2, id3, e);
                }
                setLoadEdgeI(e, load.getConstantI()[1][0], load.getConstantI()[1][1]);
            }
            if (Math.abs(load.getConstantI()[2][0]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id3, id1);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 2);
                    e.setDevId(obj.getId());
                    e.setLoadD(true);
                    detailedG.addEdge(id3, id1, e);
                }
                setLoadEdgeI(e, load.getConstantI()[2][0], load.getConstantI()[2][1]);
            }
        } else if (load.getMode().equals(LOAD_D_PQ)) {
            if (Math.abs(load.getConstantS()[0][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantS()[0][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id1, id2);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 0);
                    e.setDevId(obj.getId());
                    e.setLoadD(true);
                    detailedG.addEdge(id1, id2, e);
                }
                e.setS_real(e.getS_real() + load.getConstantS()[0][0]);
                e.setS_image(e.getS_image() + load.getConstantS()[0][1]);
            }
            if (Math.abs(load.getConstantS()[1][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantS()[1][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id2, id3);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 1);
                    e.setDevId(obj.getId());
                    e.setLoadD(true);
                    detailedG.addEdge(id2, id3, e);
                }
                e.setS_real(e.getS_real() + load.getConstantS()[1][0]);
                e.setS_image(e.getS_image() + load.getConstantS()[1][1]);
            }
            if (Math.abs(load.getConstantS()[2][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantS()[2][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id3, id1);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 2);
                    e.setDevId(obj.getId());
                    e.setLoadD(true);
                    detailedG.addEdge(id3, id1, e);
                }
                e.setS_real(e.getS_real() + load.getConstantS()[2][0]);
                e.setS_image(e.getS_image() + load.getConstantS()[2][1]);
            }
        } else if (load.getMode().equals(LOAD_D_Z)) {
            if (Math.abs(load.getConstantZ()[0][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantZ()[0][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id1, id2);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 0);
                    e.setDevId(obj.getId());
                    e.setLoadD(true);
                    detailedG.addEdge(id1, id2, e);
                }
                setLoadEdgeZ(e, load.getConstantZ()[0][0], load.getConstantZ()[0][1]);
            }
            if (Math.abs(load.getConstantZ()[1][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantZ()[1][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id2, id3);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 1);
                    e.setDevId(obj.getId());
                    e.setLoadD(true);
                    detailedG.addEdge(id2, id3, e);
                }
                setLoadEdgeZ(e, load.getConstantZ()[1][0], load.getConstantZ()[1][1]);
            }
            if (Math.abs(load.getConstantZ()[2][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantZ()[2][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id3, id1);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 2);
                    e.setDevId(obj.getId());
                    e.setLoadD(true);
                    detailedG.addEdge(id3, id1, e);
                }
                setLoadEdgeZ(e, load.getConstantZ()[2][0], load.getConstantZ()[2][1]);
            }
        } else if (load.getMode().equals(LOAD_Y_I)) {
            if (Math.abs(load.getConstantI()[0][0]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id1, EARTH_NODE_ID);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 0);
                    e.setDevId(obj.getId());
                    e.setLoadD(false);
                    detailedG.addEdge(id1, EARTH_NODE_ID, e);
                } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                    e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                setLoadEdgeI(e, load.getConstantI()[0][0], load.getConstantI()[0][1]);
            }
            if (Math.abs(load.getConstantI()[1][0]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id2, EARTH_NODE_ID);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 1);
                    e.setDevId(obj.getId());
                    e.setLoadD(false);
                    detailedG.addEdge(id2, EARTH_NODE_ID, e);
                } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                    e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                setLoadEdgeI(e, load.getConstantI()[1][0], load.getConstantI()[1][1]);
            }
            if (Math.abs(load.getConstantI()[2][0]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id3, EARTH_NODE_ID);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 2);
                    e.setDevId(obj.getId());
                    e.setLoadD(false);
                    detailedG.addEdge(id3, EARTH_NODE_ID, e);
                } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                    e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                setLoadEdgeI(e, load.getConstantI()[2][0], load.getConstantI()[2][1]);
            }
        } else if (load.getMode().equals(LOAD_Y_PQ)) {
            if (Math.abs(load.getConstantS()[0][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantS()[0][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id1, EARTH_NODE_ID);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 0);
                    e.setDevId(obj.getId());
                    e.setLoadD(false);
                    detailedG.addEdge(id1, EARTH_NODE_ID, e);
                } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                    e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                e.setS_real(e.getS_real() + load.getConstantS()[0][0]);
                e.setS_image(e.getS_image() + load.getConstantS()[0][1]);
            }
            if (Math.abs(load.getConstantS()[1][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantS()[1][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id2, EARTH_NODE_ID);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 1);
                    e.setDevId(obj.getId());
                    e.setLoadD(false);
                    detailedG.addEdge(id2, EARTH_NODE_ID, e);
                } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                    e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                e.setS_real(e.getS_real() + load.getConstantS()[1][0]);
                e.setS_image(e.getS_image() + load.getConstantS()[1][1]);
            }
            if (Math.abs(load.getConstantS()[2][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantS()[2][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id3, EARTH_NODE_ID);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 2);
                    e.setDevId(obj.getId());
                    e.setLoadD(false);
                    detailedG.addEdge(id3, EARTH_NODE_ID, e);
                } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                    e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                e.setS_real(e.getS_real() + load.getConstantS()[2][0]);
                e.setS_image(e.getS_image() + load.getConstantS()[2][1]);
            }
        } else if (load.getMode().equals(LOAD_Y_Z)) {
            if (Math.abs(load.getConstantZ()[0][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantZ()[0][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id1, EARTH_NODE_ID);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 0);
                    e.setDevId(obj.getId());
                    e.setLoadD(false);
                    detailedG.addEdge(id1, EARTH_NODE_ID, e);
                } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                    e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                setLoadEdgeZ(e, load.getConstantZ()[0][0], load.getConstantZ()[0][1]);
            }
            if (Math.abs(load.getConstantZ()[1][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantZ()[1][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id2, EARTH_NODE_ID);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 1);
                    e.setDevId(obj.getId());
                    e.setLoadD(false);
                    detailedG.addEdge(id2, EARTH_NODE_ID, e);
                } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                    e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                setLoadEdgeZ(e, load.getConstantZ()[1][0], load.getConstantZ()[1][1]);
            }
            if (Math.abs(load.getConstantZ()[2][0]) > ZERO_LIMIT
                    || Math.abs(load.getConstantZ()[2][1]) > ZERO_LIMIT) {
                DetailedEdge e = detailedG.getEdge(id3, EARTH_NODE_ID);
                if (e == null) {
                    e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getBusNo(), 2);
                    e.setDevId(obj.getId());
                    e.setLoadD(false);
                    detailedG.addEdge(id3, EARTH_NODE_ID, e);
                } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                    e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                setLoadEdgeZ(e, load.getConstantZ()[2][0], load.getConstantZ()[2][1]);
            }
        }
    }

    private void setLoadEdgeI(DetailedEdge e, double i_ampl, double i_angle) {
        if (Math.abs(e.getI_ampl()) < ZERO_LIMIT) {
            e.setI_ampl(i_ampl);
            e.setI_angle(i_angle);
            return;
        }
        double x1, y1, x2, y2;
        x1 = e.getI_ampl() * Math.cos(e.getI_ampl());
        y1 = -e.getI_ampl() * Math.sin(e.getI_ampl());
        x2 = i_ampl * Math.cos(i_angle);
        y2 = -i_ampl * Math.sin(i_angle);
        e.setI_ampl(Math.sqrt((x1 + x2) * (x1 + x2) + (y1 + y2) * (y1 + y2)));
        e.setI_angle(-Math.atan2(y1 + y2, x1 + x2));
    }

    private void setLoadEdgeZ(DetailedEdge e, double zx, double zy) {
        if (Math.abs(e.getZ_real()) < ZERO_LIMIT && Math.abs(e.getZ_image()) < ZERO_LIMIT) {
            e.setZ_real(zx);
            e.setZ_image(zy);
            return;
        }
        double r1, x1, tmp, tmp1, tmp2;
        r1 = e.getZ_real();
        x1 = e.getZ_image();
        tmp = (r1 + zx) * (r1 + zx) + (x1 + zy) * (x1 + zy);
        tmp1 = r1 * zx - x1 * zy;
        tmp2 = r1 * zy + x1 * zx;
        e.setZ_real((tmp1 * (r1 + zx) + tmp2 * (x1 + zy)) / tmp);
        e.setZ_image((tmp2 * (r1 + zx) - tmp1 * (x1 + zy)) / tmp);
    }

    /**
     * 分析节点上所连接的设备，注意不包括含支路类型的设备
     */
    private void fillTnDevices() {
        int count;
        spotLoadNum = 0;
        distriLoadNum = 0;
        shuntCapacitorNum = 0;
        distriGenNum = 0;
        for (DsTopoNode tn : tns) {
            count = 0;
            for (DsConnectNode cn : tn.getConnectivityNodes()) {
                for (MapObject obj : cn.getConnectedObjs()) {
                    if (RESOURCE_SPOT_LOAD.equals(obj.getProperty(KEY_RESOURCE_TYPE))) {
                        spotLoadNum++;
                        count++;
                    }
                    if (RESOURCE_DIS_LOAD.equals(obj.getProperty(KEY_RESOURCE_TYPE))) {
                        distriLoadNum++;
                        count++;
                    }
                    if (RESOURCE_SHUNT_CAPACITORS.equals(obj.getProperty(KEY_RESOURCE_TYPE))) {
                        shuntCapacitorNum++;
                        count++;
                    }
                    if (RESOURCE_DG.equals(obj.getProperty(KEY_RESOURCE_TYPE))) {
                        distriGenNum++;
                        count++;
                    }
                }
            }
            MapObject[] devs = new MapObject[count];
            count = 0;
            for (DsConnectNode cn : tn.getConnectivityNodes()) {
                for (MapObject obj : cn.getConnectedObjs()) {
                    if (RESOURCE_SPOT_LOAD.equals(obj.getProperty(KEY_RESOURCE_TYPE)) ||
                            RESOURCE_DIS_LOAD.equals(obj.getProperty(KEY_RESOURCE_TYPE)) ||
                            RESOURCE_SHUNT_CAPACITORS.equals(obj.getProperty(KEY_RESOURCE_TYPE)) ||
                            RESOURCE_DG.equals(obj.getProperty(KEY_RESOURCE_TYPE))) {
                        devs[count++] = obj;
                    }
                }
            }
            tn.setConnectedDev(devs);
        }
        distriLoadNum /= 2;
    }

    /**
     * <br>给每个TN编号，从第一个电源点开始使用广度优先遍历从1编号</br>
     * <br>给每个支路编号</br>
     */
    public void initialIsland() {
        tns = new ArrayList<DsTopoNode>(graph.vertexSet().size());
        DsTopoNode subTn = null;
        for (DsTopoNode tn : supplyTns)
            if (tn.getType() == DsTopoNode.TYPE_SUB) {
                subTn = tn;
                break;
            }
        BreadthFirstIterator<DsTopoNode, MapObject> iter = new BreadthFirstIterator<DsTopoNode, MapObject>(graph, subTn);
        int count = 1;
        while (iter.hasNext()) {
            DsTopoNode tn = iter.next();
            tn.setBusNo(count);
            tns.add(tn);
            tn.setIsland(this);
            count++;
        }
        idToBranch = new HashMap<Integer, MapObject>(graph.edgeSet().size());
        int branchId = 1;
        for (DsTopoNode tn : getTns()) {
            int[] connected = new int[graph.degreeOf(tn)];
            count = 0;
            for (MapObject edge : graph.edgesOf(tn)) {
                DsTopoNode t1 = graph.getEdgeSource(edge);
                DsTopoNode t2 = graph.getEdgeTarget(edge);
                connected[count] = (t1 == tn) ? t2.getBusNo() : t1.getBusNo();
                if (connected[count] > tn.getBusNo()) {
                    edge.setId(String.valueOf(branchId));
                    idToBranch.put(branchId, edge);
                    branchId++;
                }
                count++;
            }
            tn.setConnectedBusNo(connected);
        }

        busNoToTn = new HashMap<Integer, DsTopoNode>(getTns().size());
        for (DsTopoNode tn : tns)
            busNoToTn.put(tn.getBusNo(), tn);
        fillTnDevices();
    }

    public List<DsTopoNode> getTns() {
        return tns;
    }

    public void setTns(List<DsTopoNode> tns) {
        this.tns = tns;
    }

    public UndirectedGraph<DsTopoNode, MapObject> getGraph() {
        return graph;
    }

    public void setGraph(UndirectedGraph<DsTopoNode, MapObject> graph) {
        this.graph = graph;
    }

    public boolean isActive() {
        return isActive;
    }

    public void setActive(boolean active) {
        isActive = active;
    }

    public boolean isRadical() {
        return isRadical;
    }

    public void setRadical(boolean radical) {
        isRadical = radical;
    }

    public DsTopoNode[] getSupplyTns() {
        return supplyTns;
    }

    public void setSupplyTns(DsTopoNode[] supplyTns) {
        this.supplyTns = supplyTns;
    }

    public boolean isPerUnitSys() {
        return isPerUnitSys;
    }

    public void setPerUnitSys(boolean perUnitSys) {
        isPerUnitSys = perUnitSys;
    }

    public Map<Integer, DsTopoNode> getBusNoToTn() {
        return busNoToTn;
    }

    public Map<Integer, MapObject> getIdToBranch() {
        return idToBranch;
    }

    public int getSpotLoadNum() {
        return spotLoadNum;
    }

    public int getDistriLoadNum() {
        return distriLoadNum;
    }

    public int getShuntCapacitorNum() {
        return shuntCapacitorNum;
    }

    public int getDistriGenNum() {
        return distriGenNum;
    }

    public Map<MapObject, GeneralBranch> getBranches() {
        return branches;
    }

    public void setBranches(Map<MapObject, GeneralBranch> branches) {
        this.branches = branches;
    }

    public Map<MapObject, ThreePhaseLoad> getLoads() {
        return loads;
    }

    public void setLoads(Map<MapObject, ThreePhaseLoad> loads) {
        this.loads = loads;
    }

    public Map<MapObject, double[][]> getBranchHeadI() {
        return branchHeadI;
    }

    public void setBranchHeadI(Map<MapObject, double[][]> branchHeadI) {
        this.branchHeadI = branchHeadI;
    }

    public Map<MapObject, double[][]> getBranchTailI() {
        return branchTailI;
    }

    public void setBranchTailI(Map<MapObject, double[][]> branchTailI) {
        this.branchTailI = branchTailI;
    }

    public Map<DsTopoNode, double[][]> getBusV() {
        return busV;
    }

    public void setBusV(Map<DsTopoNode, double[][]> busV) {
        this.busV = busV;
    }

    public boolean isBalanced() {
        return isBalanced;
    }

    public void setBalanced(boolean balanced) {
        isBalanced = balanced;
    }

    public boolean isVCartesian() {
        return isVCartesian;
    }

    public void setVCartesian(boolean VCartesian) {
        isVCartesian = VCartesian;
    }

    public boolean isICartesian() {
        return isICartesian;
    }

    public void setICartesian(boolean ICartesian) {
        isICartesian = ICartesian;
    }

    public UndirectedGraph<String, DetailedEdge> getDetailedG() {
        return detailedG;
    }

    public Map<String, MapObject> getDevices() {
        return devices;
    }

    public Map<MapObject, DispersedGen> getDispersedGens() {
        return dispersedGens;
    }

    public void setDispersedGens(Map<MapObject, DispersedGen> dispersedGens) {
        this.dispersedGens = dispersedGens;
    }

    public DsTopoIsland clone() {
           return (DsTopoIsland) JOFileUtil.cloneObj(this);
       }

}

