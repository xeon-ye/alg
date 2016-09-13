package zju.dsmodel;

import org.apache.log4j.Logger;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.graph.SimpleGraph;
import org.jgrapht.traverse.BreadthFirstIterator;
import zju.devmodel.MapObject;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.Complex;
import zju.util.JOFileUtil;

import java.io.Serializable;
import java.util.*;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-10
 */
public class DsTopoIsland implements Serializable, DsModelCons {

    private static Logger log = Logger.getLogger(DsTopoIsland.class);

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
    public Map<Integer, DsTopoNode> tnNoToTn;
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
        branchHeadI = new HashMap<>(getBranches().size());
        branchTailI = new HashMap<>(getBranches().size());
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
        busV = new HashMap<>(getTns().size());
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
        devices = new HashMap<>(spotLoadNum + distriLoadNum + distriGenNum + shuntCapacitorNum + idToBranch.size());
        for (MapObject obj : idToBranch.values())
            devices.put(obj.getId(), obj);
        for (DsTopoNode tn : tns)
            for (MapObject obj : tn.getConnectedDev())
                if (!devices.containsKey(obj.getId()))
                    devices.put(obj.getId(), obj);
        detailedG = new SimpleGraph<>(DetailedEdge.class);
        String id1, id2, id3;
        //第一步，加入大地节点以及电源节点
        detailedG.addVertex(EARTH_NODE_ID);
        for (DsTopoNode tn : getSupplyTns()) {
            if (tn.getType() != DsTopoNode.TYPE_SUB)
                continue;
            id1 = tn.getTnNo() + "-0";
            id2 = tn.getTnNo() + "-1";
            id3 = tn.getTnNo() + "-2";
            detailedG.addVertex(id1);
            detailedG.addVertex(id2);
            detailedG.addVertex(id3);
            DetailedEdge edge = new DetailedEdge(DetailedEdge.EDGE_TYPE_SUPPLIER, tn.getTnNo(), 0);
            detailedG.addEdge(EARTH_NODE_ID, id1, edge);
            edge = new DetailedEdge(DetailedEdge.EDGE_TYPE_SUPPLIER, tn.getTnNo(), 1);
            detailedG.addEdge(EARTH_NODE_ID, id2, edge);
            edge = new DetailedEdge(DetailedEdge.EDGE_TYPE_SUPPLIER, tn.getTnNo(), 2);
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
            if (tn1.getTnNo() > tn2.getTnNo()) {
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
                        id1 = tn1.getTnNo() + "-" + i;
                        id2 = tn2.getTnNo() + "-" + i;
                        if (!detailedG.containsVertex(id1))
                            detailedG.addVertex(id1);
                        if (!detailedG.containsVertex(id2))
                            detailedG.addVertex(id2);
                        edges[count] = new DetailedEdge(DetailedEdge.EDGE_TYPE_FEEDER, tn1.getTnNo(), tn2.getTnNo(), i, obj.getId());
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
                    id1 = tn1.getTnNo() + "-" + i;
                    id2 = tn2.getTnNo() + "-" + i;
                    if (!detailedG.containsVertex(id1))
                        detailedG.addVertex(id1);
                    if (!detailedG.containsVertex(id2))
                        detailedG.addVertex(id2);
                }
                switch (tf.getConnType()) {
                    case Transformer.CONN_TYPE_D_GrY:
                        id1 = tn1.getTnNo() + "-0";
                        id2 = tn1.getTnNo() + "-1";
                        id3 = tn1.getTnNo() + "-2";
                        DetailedEdge edge1 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 0, true, obj.getId());
                        DetailedEdge edge2 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 1, true, obj.getId());
                        DetailedEdge edge3 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 2, true, obj.getId());
                        detailedG.addEdge(id1, id2, edge1);
                        detailedG.addEdge(id2, id3, edge2);
                        detailedG.addEdge(id3, id1, edge3);
                        id1 = tn2.getTnNo() + "-0";
                        id2 = tn2.getTnNo() + "-1";
                        id3 = tn2.getTnNo() + "-2";
                        DetailedEdge edge4 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 0, false, obj.getId());
                        DetailedEdge edge5 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 1, false, obj.getId());
                        DetailedEdge edge6 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 2, false, obj.getId());
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
                        id1 = tn1.getTnNo() + "-0";
                        id2 = tn1.getTnNo() + "-1";
                        id3 = tn1.getTnNo() + "-2";
                        edge1 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 0, true, obj.getId());
                        edge2 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 1, true, obj.getId());
                        edge3 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 2, true, obj.getId());
                        detailedG.addEdge(id1, EARTH_NODE_ID, edge1);
                        detailedG.addEdge(id2, EARTH_NODE_ID, edge2);
                        detailedG.addEdge(id3, EARTH_NODE_ID, edge3);
                        edge4 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 0, false, obj.getId());
                        edge5 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 1, false, obj.getId());
                        edge6 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 2, false, obj.getId());
                        id1 = tn2.getTnNo() + "-0";
                        id2 = tn2.getTnNo() + "-1";
                        id3 = tn2.getTnNo() + "-2";
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
                        id1 = tn1.getTnNo() + "-0";
                        id2 = tn1.getTnNo() + "-1";
                        id3 = tn1.getTnNo() + "-2";
                        String neutralNodeId = obj.getId() + "-3";
                        detailedG.addVertex(neutralNodeId);
                        edge1 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 0, true, obj.getId());
                        edge2 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 1, true, obj.getId());
                        edge3 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 2, true, obj.getId());
                        detailedG.addEdge(id1, neutralNodeId, edge1);
                        detailedG.addEdge(id2, neutralNodeId, edge2);
                        detailedG.addEdge(id3, neutralNodeId, edge3);
                        id1 = tn2.getTnNo() + "-0";
                        id2 = tn2.getTnNo() + "-1";
                        id3 = tn2.getTnNo() + "-2";
                        edge4 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 0, false, obj.getId());
                        edge5 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 1, false, obj.getId());
                        edge6 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 2, false, obj.getId());
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
                        id1 = tn1.getTnNo() + "-0";
                        id2 = tn1.getTnNo() + "-1";
                        id3 = tn1.getTnNo() + "-2";
                        edge1 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 0, true, obj.getId());
                        edge2 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 1, true, obj.getId());
                        edge3 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn1.getTnNo(), 2, true, obj.getId());
                        detailedG.addEdge(id1, id2, edge1);
                        detailedG.addEdge(id2, id3, edge2);
                        detailedG.addEdge(id3, id1, edge3);
                        id1 = tn2.getTnNo() + "-0";
                        id2 = tn2.getTnNo() + "-1";
                        id3 = tn2.getTnNo() + "-2";
                        edge4 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 0, false, obj.getId());
                        edge5 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 1, false, obj.getId());
                        edge6 = new DetailedEdge(DetailedEdge.EDGE_TYPE_TF_WINDING, tn2.getTnNo(), 2, false, obj.getId());
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
                id1 = tn.getTnNo() + "-0";
                id2 = tn.getTnNo() + "-1";
                id3 = tn.getTnNo() + "-2";
                switch (gen.getMode()) {
                    case DispersedGen.MODE_PV:
                        //todo: 认为PV节点只有接地这种模式
                        DetailedEdge e = detailedG.getEdge(id1, EARTH_NODE_ID);
                        if (e == null) {
                            e = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getTnNo(), 0);
                            detailedG.addEdge(id1, EARTH_NODE_ID, e);
                        } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING
                                || e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX)
                            e.setEdgeType(DetailedEdge.EDGE_TYPE_DG_TF_MIX);
                        else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD)
                            e.setEdgeType(DetailedEdge.EDGE_TYPE_DG);
                        e.setDgId(obj.getId());

                        e = detailedG.getEdge(id2, EARTH_NODE_ID);
                        if (e == null) {
                            e = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getTnNo(), 1);
                            detailedG.addEdge(id2, EARTH_NODE_ID, e);
                        } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING
                                || e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX)
                            e.setEdgeType(DetailedEdge.EDGE_TYPE_DG_TF_MIX);
                        else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD)
                            e.setEdgeType(DetailedEdge.EDGE_TYPE_DG);
                        e.setDgId(obj.getId());

                        e = detailedG.getEdge(id3, EARTH_NODE_ID);
                        if (e == null) {
                            e = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getTnNo(), 2);
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
                                String neutralNodeId = tn.getTnNo() + "-motor-3";
                                detailedG.addVertex(neutralNodeId);
                                DetailedEdge edge1 = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getTnNo(), 0);
                                DetailedEdge edge2 = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getTnNo(), 1);
                                DetailedEdge edge3 = new DetailedEdge(DetailedEdge.EDGE_TYPE_DG, tn.getTnNo(), 2);
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
        String id1 = tn.getTnNo() + "-0";
        String id2 = tn.getTnNo() + "-1";
        String id3 = tn.getTnNo() + "-2";
        switch (load.getMode()) {
            case LOAD_D_I:
                if (Math.abs(load.getConstantI()[0][0]) > ZERO_LIMIT) {
                    DetailedEdge e = detailedG.getEdge(id1, id2);
                    if (e == null) {
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 0);
                        e.setDevId(obj.getId());
                        e.setLoadD(true);
                        detailedG.addEdge(id1, id2, e);
                    }
                    setLoadEdgeI(e, load.getConstantI()[0][0], load.getConstantI()[0][1]);
                }
                if (Math.abs(load.getConstantI()[1][0]) > ZERO_LIMIT) {
                    DetailedEdge e = detailedG.getEdge(id2, id3);
                    if (e == null) {
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 1);
                        e.setDevId(obj.getId());
                        e.setLoadD(true);
                        detailedG.addEdge(id2, id3, e);
                    }
                    setLoadEdgeI(e, load.getConstantI()[1][0], load.getConstantI()[1][1]);
                }
                if (Math.abs(load.getConstantI()[2][0]) > ZERO_LIMIT) {
                    DetailedEdge e = detailedG.getEdge(id3, id1);
                    if (e == null) {
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 2);
                        e.setDevId(obj.getId());
                        e.setLoadD(true);
                        detailedG.addEdge(id3, id1, e);
                    }
                    setLoadEdgeI(e, load.getConstantI()[2][0], load.getConstantI()[2][1]);
                }
                break;
            case LOAD_D_PQ:
                if (Math.abs(load.getConstantS()[0][0]) > ZERO_LIMIT
                        || Math.abs(load.getConstantS()[0][1]) > ZERO_LIMIT) {
                    DetailedEdge e = detailedG.getEdge(id1, id2);
                    if (e == null) {
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 0);
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
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 1);
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
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 2);
                        e.setDevId(obj.getId());
                        e.setLoadD(true);
                        detailedG.addEdge(id3, id1, e);
                    }
                    e.setS_real(e.getS_real() + load.getConstantS()[2][0]);
                    e.setS_image(e.getS_image() + load.getConstantS()[2][1]);
                }
                break;
            case LOAD_D_Z:
                if (Math.abs(load.getConstantZ()[0][0]) > ZERO_LIMIT
                        || Math.abs(load.getConstantZ()[0][1]) > ZERO_LIMIT) {
                    DetailedEdge e = detailedG.getEdge(id1, id2);
                    if (e == null) {
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 0);
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
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 1);
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
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 2);
                        e.setDevId(obj.getId());
                        e.setLoadD(true);
                        detailedG.addEdge(id3, id1, e);
                    }
                    setLoadEdgeZ(e, load.getConstantZ()[2][0], load.getConstantZ()[2][1]);
                }
                break;
            case LOAD_Y_I:
                if (Math.abs(load.getConstantI()[0][0]) > ZERO_LIMIT) {
                    DetailedEdge e = detailedG.getEdge(id1, EARTH_NODE_ID);
                    if (e == null) {
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 0);
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
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 1);
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
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 2);
                        e.setDevId(obj.getId());
                        e.setLoadD(false);
                        detailedG.addEdge(id3, EARTH_NODE_ID, e);
                    } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                        e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                    setLoadEdgeI(e, load.getConstantI()[2][0], load.getConstantI()[2][1]);
                }
                break;
            case LOAD_Y_PQ:
                if (Math.abs(load.getConstantS()[0][0]) > ZERO_LIMIT
                        || Math.abs(load.getConstantS()[0][1]) > ZERO_LIMIT) {
                    DetailedEdge e = detailedG.getEdge(id1, EARTH_NODE_ID);
                    if (e == null) {
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 0);
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
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 1);
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
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 2);
                        e.setDevId(obj.getId());
                        e.setLoadD(false);
                        detailedG.addEdge(id3, EARTH_NODE_ID, e);
                    } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                        e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                    e.setS_real(e.getS_real() + load.getConstantS()[2][0]);
                    e.setS_image(e.getS_image() + load.getConstantS()[2][1]);
                }
                break;
            case LOAD_Y_Z:
                if (Math.abs(load.getConstantZ()[0][0]) > ZERO_LIMIT
                        || Math.abs(load.getConstantZ()[0][1]) > ZERO_LIMIT) {
                    DetailedEdge e = detailedG.getEdge(id1, EARTH_NODE_ID);
                    if (e == null) {
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 0);
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
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 1);
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
                        e = new DetailedEdge(DetailedEdge.EDGE_TYPE_LOAD, tn.getTnNo(), 2);
                        e.setDevId(obj.getId());
                        e.setLoadD(false);
                        detailedG.addEdge(id3, EARTH_NODE_ID, e);
                    } else if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING)
                        e.setEdgeType(DetailedEdge.EDGE_TYPE_LOAD_TF_MIX);
                    setLoadEdgeZ(e, load.getConstantZ()[2][0], load.getConstantZ()[2][1]);
                }
                break;
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
        tns = new ArrayList<>(graph.vertexSet().size());
        DsTopoNode subTn = null;
        for (DsTopoNode tn : supplyTns)
            if (tn.getType() == DsTopoNode.TYPE_SUB) {
                subTn = tn;
                break;
            }
        BreadthFirstIterator<DsTopoNode, MapObject> iter = new BreadthFirstIterator<>(graph, subTn);
        int count = 1;
        while (iter.hasNext()) {
            DsTopoNode tn = iter.next();
            tn.setTnNo(count);
            tns.add(tn);
            tn.setIsland(this);
            StringBuilder sb = new StringBuilder();
            for(DsConnectNode cn : tn.getConnectivityNodes())
            sb.append(cn.getId()).append("\t");
            log.debug(sb.toString() + "---->" + tn.getTnNo());
            count++;
        }
        idToBranch = new HashMap<>(graph.edgeSet().size());
        int branchId = 1;
        for (DsTopoNode tn : getTns()) {
            int[] connected = new int[graph.degreeOf(tn)];
            count = 0;
            for (MapObject edge : graph.edgesOf(tn)) {
                DsTopoNode t1 = graph.getEdgeSource(edge);
                DsTopoNode t2 = graph.getEdgeTarget(edge);
                connected[count] = (t1 == tn) ? t2.getTnNo() : t1.getTnNo();
                if (connected[count] > tn.getTnNo()) {
                    edge.setId(String.valueOf(branchId));
                    idToBranch.put(branchId, edge);
                    log.debug(t1.getTnNo() + " ---> " + t2.getTnNo() + " =============== " + branchId);
                    branchId++;
                }
                count++;
            }
            tn.setConnectedBusNo(connected);
        }

        tnNoToTn = new HashMap<>(getTns().size());
        for (DsTopoNode tn : tns)
            tnNoToTn.put(tn.getTnNo(), tn);
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

    public Map<Integer, DsTopoNode> getTnNoToTn() {
        return tnNoToTn;
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

    /**
     * 将三相等效电路转换成IEEE潮流模型
     * @param devIdToBranch 用于记录两个模型之间线路的对应关系
     * @return IEEE潮流模型
     */
    public IEEEDataIsland toIeeeIsland(Map<String, BranchData[]> devIdToBranch, Map<String, BusData> vertexToBus) {
        return toIeeeIsland(devIdToBranch, vertexToBus, false);
    }

    /**
     *
     * 将三相等效电路转换成IEEE潮流模型
     * @param devIdToBranch 用于记录两个模型之间线路的对应关系
     * @param isP2pNeglected 是否忽略相与相之间的互阻抗
     * @return IEEE潮流模型
     */
    public IEEEDataIsland toIeeeIsland(Map<String, BranchData[]> devIdToBranch, Map<String, BusData> vertexToBus, boolean isP2pNeglected) {
        if(detailedG == null)
            buildDetailedGraph();
        IEEEDataIsland island = new IEEEDataIsland();
        ArrayList<BusData> buses = new ArrayList<>(detailedG.vertexSet().size());
        ArrayList<BranchData> branches = new ArrayList<>(detailedG.edgeSet().size());
        int index = 1;
        for(String key : detailedG.vertexSet()) {
            if(key.equals(EARTH_NODE_ID))
                continue;
            //detailedG的vertexSet，首先放入的是earth node，然后是supply node
            //所以电源节点的节点编号排在1这个位置，这点非常重要，影响后面配电网的很多算法
            BusData bus = new BusData();
            bus.setName(key);
            bus.setBusNumber(index);
            buses.add(bus);
            index++;
            vertexToBus.put(key, bus);
        }
        double baseZ = 1.0; //todo: 现在要求feeder的参数是标幺值
        //Map<Feeder, BranchData> feederToBranch = new HashMap<>(branches.size());
        //记录馈线及其对应Branch之间的关系
        for(DetailedEdge e : detailedG.edgeSet()) {
            if(e.getEdgeType() == DetailedEdge.EDGE_TYPE_FEEDER) {
                Feeder f = (Feeder) this.branches.get(this.getDevices().get(e.getDevId()));
                if(devIdToBranch.containsKey(e.getDevId()))
                    continue;
                if(isP2pNeglected) {
                    int count  = 0;
                    BranchData[] brArray = new BranchData[f.getPhases().length];
                    BranchData branch = new BranchData();
                    BusData bus1 = vertexToBus.get(detailedG.getEdgeSource(e));
                    BusData bus2 = vertexToBus.get(detailedG.getEdgeTarget(e));
                    branch.setTapBusNumber(bus1.getBusNumber());
                    branch.setZBusNumber(bus2.getBusNumber());
                    branch.setBranchR(f.getZ_real()[e.getPhase()][e.getPhase()] / baseZ);
                    branch.setBranchX(f.getZ_imag()[e.getPhase()][e.getPhase()] / baseZ);
                    brArray[count++] = branch;
                    for(DetailedEdge otherEdage : e.getOtherEdgesOfSameFeeder()) {
                        branch = new BranchData();
                        bus1 = vertexToBus.get(detailedG.getEdgeSource(otherEdage));
                        bus2 = vertexToBus.get(detailedG.getEdgeTarget(otherEdage));
                        branch.setTapBusNumber(bus1.getBusNumber());
                        branch.setZBusNumber(bus2.getBusNumber());
                        branch.setBranchR(f.getZ_real()[otherEdage.getPhase()][otherEdage.getPhase()] / baseZ);
                        branch.setBranchX(f.getZ_imag()[otherEdage.getPhase()][otherEdage.getPhase()] / baseZ);
                        brArray[count++] = branch;
                    }
                    devIdToBranch.put(e.getDevId(), brArray);
                    continue;
                }
                switch (f.getPhases().length) {
                    case 1:
                        BranchData branch = new BranchData();
                        BusData bus1 = vertexToBus.get(detailedG.getEdgeSource(e));
                        BusData bus2 = vertexToBus.get(detailedG.getEdgeTarget(e));
                        branch.setTapBusNumber(bus1.getBusNumber());
                        branch.setZBusNumber(bus2.getBusNumber());
                        branch.setBranchR(f.getZ_real()[e.getPhase()][e.getPhase()] / baseZ);
                        branch.setBranchX(f.getZ_imag()[e.getPhase()][e.getPhase()] / baseZ);
                        devIdToBranch.put(e.getDevId(), new BranchData[]{branch});
                        break;
                    case 2:
                        //由于是对称矩阵，只计算上三角矩阵
                        Complex a = new Complex(f.getZ_real()[f.getPhases()[0]][f.getPhases()[0]], f.getZ_imag()[f.getPhases()[0]][f.getPhases()[0]]);
                        Complex b = new Complex(f.getZ_real()[f.getPhases()[0]][f.getPhases()[1]], f.getZ_imag()[f.getPhases()[0]][f.getPhases()[1]]);
                        Complex d = new Complex(f.getZ_real()[f.getPhases()[1]][f.getPhases()[1]], f.getZ_imag()[f.getPhases()[1]][f.getPhases()[1]]);

                        //计算行列式
                        Complex tmp1 = Complex.subtract(Complex.multiply(a, d), Complex.multiply(b, b));
                        if(tmp1.mod() < 1e-6) {
                            log.warn("!!!Warning because mod of determinant of resistance matrix is zero. Feeder id: " + e.getDevId());
                            continue;
                        }
                        Complex y12 = Complex.divide(new Complex(-b.getRe(), -b.getIm()), tmp1);
                        Complex z11 = Complex.divide(d, tmp1).inverse();
                        Complex z22 = Complex.divide(a, tmp1).inverse();

                        BusData bus3, bus4;
                        if(e.getPhase() == f.getPhases()[0]) {
                            bus1 = vertexToBus.get(detailedG.getEdgeSource(e));
                            bus2 = vertexToBus.get(detailedG.getEdgeTarget(e));
                            bus3 = vertexToBus.get(detailedG.getEdgeSource(e.getOtherEdgesOfSameFeeder()[0]));
                            bus4 = vertexToBus.get(detailedG.getEdgeTarget(e.getOtherEdgesOfSameFeeder()[0]));
                        } else {
                            bus3 = vertexToBus.get(detailedG.getEdgeSource(e));
                            bus4 = vertexToBus.get(detailedG.getEdgeTarget(e));
                            bus1 = vertexToBus.get(detailedG.getEdgeSource(e.getOtherEdgesOfSameFeeder()[0]));
                            bus2 = vertexToBus.get(detailedG.getEdgeTarget(e.getOtherEdgesOfSameFeeder()[0]));
                        }
                        BranchData branch1 = new BranchData();
                        branch1.setTapBusNumber(bus1.getBusNumber());
                        branch1.setZBusNumber(bus2.getBusNumber());
                        branch1.setBranchR(z11.getRe() / baseZ);
                        branch1.setBranchX(z11.getIm() / baseZ);

                        BranchData branch2 = new BranchData();
                        branch2.setTapBusNumber(bus3.getBusNumber());
                        branch2.setZBusNumber(bus4.getBusNumber());
                        branch2.setBranchR(z22.getRe() / baseZ);
                        branch2.setBranchX(z22.getIm() / baseZ);

                        if(y12.mod() > 1e-6) {
                            Complex z12 = y12.inverse();
                            BranchData branch3 = new BranchData();
                            branch3.setTapBusNumber(bus1.getBusNumber());
                            branch3.setZBusNumber(bus3.getBusNumber());
                            branch3.setBranchR(-z12.getRe() / baseZ);
                            branch3.setBranchX(-z12.getIm() / baseZ);

                            BranchData branch4 = new BranchData();
                            branch4.setTapBusNumber(bus2.getBusNumber());
                            branch4.setZBusNumber(bus4.getBusNumber());
                            branch4.setBranchR(-z12.getRe() / baseZ);
                            branch4.setBranchX(-z12.getIm() / baseZ);

                            BranchData branch5 = new BranchData();
                            branch5.setTapBusNumber(bus1.getBusNumber());
                            branch5.setZBusNumber(bus4.getBusNumber());
                            branch5.setBranchR(z12.getRe() / baseZ);
                            branch5.setBranchX(z12.getIm() / baseZ);

                            BranchData branch6 = new BranchData();
                            branch6.setTapBusNumber(bus2.getBusNumber());
                            branch6.setZBusNumber(bus3.getBusNumber());
                            branch6.setBranchR(z12.getRe() / baseZ);
                            branch6.setBranchX(z12.getIm() / baseZ);
                            devIdToBranch.put(e.getDevId(), new BranchData[]{branch1, branch2, branch3, branch4, branch5, branch6});
                        } else {
                            devIdToBranch.put(e.getDevId(), new BranchData[]{branch1, branch2});
                        }
                        break;
                    case 3:

                        //由于是对称矩阵，只存储上三角矩阵
                        z11 = new Complex(f.getZ_real()[0][0], f.getZ_imag()[0][0]);
                        Complex z12 = new Complex(f.getZ_real()[0][1], f.getZ_imag()[0][1]);
                        Complex z13 = new Complex(f.getZ_real()[0][2], f.getZ_imag()[0][2]);
                        z22 = new Complex(f.getZ_real()[1][1], f.getZ_imag()[1][1]);
                        Complex z23 = new Complex(f.getZ_real()[1][2], f.getZ_imag()[1][2]);
                        Complex z33 = new Complex(f.getZ_real()[2][2], f.getZ_imag()[2][2]);

                        //计算行列式
                        tmp1 = Complex.multiply(Complex.multiply(z11, z22), z33);
                        Complex tmp2 = Complex.multiply(Complex.multiply(z12, z23), z13);
                        Complex tmp3 = Complex.multiply(Complex.multiply(z13, z12), z23);
                        Complex tmp4 = Complex.multiply(Complex.multiply(z13, z22), z13);
                        Complex tmp5 = Complex.multiply(Complex.multiply(z23, z23), z11);
                        Complex tmp6 = Complex.multiply(Complex.multiply(z33, z12), z12);

                        Complex tmp7 = Complex.add(Complex.add(tmp1, tmp2), tmp3);
                        Complex tmp8 = Complex.add(Complex.add(tmp4, tmp5), tmp6);
                        Complex tmp9 = Complex.subtract(tmp7, tmp8);


                        //由于是对称矩阵，只计算上三角矩阵
                        Complex[] Y = new Complex[6];
                        Y[0] = Complex.divide(Complex.subtract(Complex.multiply(z22, z33), Complex.multiply(z23, z23)), tmp9);
                        Y[1] = Complex.divide(Complex.subtract(Complex.multiply(z23, z13), Complex.multiply(z12, z33)), tmp9);
                        Y[2] = Complex.divide(Complex.subtract(Complex.multiply(z12, z23), Complex.multiply(z22, z13)), tmp9);
                        Y[3] = Complex.divide(Complex.subtract(Complex.multiply(z11, z33), Complex.multiply(z13, z13)), tmp9);
                        Y[4] = Complex.divide(Complex.subtract(Complex.multiply(z12, z13), Complex.multiply(z11, z23)), tmp9);
                        Y[5] = Complex.divide(Complex.subtract(Complex.multiply(z11, z22), Complex.multiply(z12, z12)), tmp9);

                        //System.out.println(z11.toString() + "\t" + z12.toString() + "\t" + z13.toString());
                        //System.out.println(z12.toString() + "\t" + z22.toString() + "\t" + z23.toString());
                        //System.out.println(z13.toString() + "\t" + z23.toString() + "\t" + z33.toString());
                        //System.out.println("-------------------------------------------------------------");
                        //System.out.println(Y[0].toString() + "\t" + Y[1].toString() + "\t" + Y[2].toString());
                        //System.out.println(Y[1].toString() + "\t" + Y[3].toString() + "\t" + Y[4].toString());
                        //System.out.println(Y[2].toString() + "\t" + Y[4].toString() + "\t" + Y[5].toString());

                        BusData[] busArr = new BusData[6];
                        busArr[0] = vertexToBus.get(e.getTnNo1() + "-0");
                        busArr[1] = vertexToBus.get(e.getTnNo1() + "-1");
                        busArr[2] = vertexToBus.get(e.getTnNo1() + "-2");
                        busArr[3] = vertexToBus.get(e.getTnNo2() + "-0");
                        busArr[4] = vertexToBus.get(e.getTnNo2() + "-1");
                        busArr[5] = vertexToBus.get(e.getTnNo2() + "-2");

                        List<BranchData> brArr = new ArrayList<>(12);
                        //分别对应a->b, a->c, a->a',a->b',a->c',b->c,b->a',b->b',b->c',c->a',c->b',c->c',a'->b', a'->c', b'->c'
                        int[] pos = new int[]{1, 2, 0, 1, 2, 4, 1, 3, 4, 2, 4, 5, 1, 2, 4};
                        double[] sign = new double[]{-1, -1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1};
                        int count = 0;
                        for(int i = 0; i < 6; i++) {
                            for(int j = i + 1; j < 6; j++) {
                                if(Y[pos[count]].mod() < 1e-6) {
                                    count++;
                                    continue;
                                }
                                Complex z = Y[pos[count]].inverse();
                                BranchData br = new BranchData();
                                br.setTapBusNumber(busArr[i].getBusNumber());
                                br.setZBusNumber(busArr[j].getBusNumber());
                                br.setBranchR(sign[count] * z.getRe() / baseZ);
                                br.setBranchX(sign[count] * z.getIm() / baseZ);
                                brArr.add(br);
                                count++;
                            }
                        }
                        devIdToBranch.put(e.getDevId(), brArr.toArray(new BranchData[]{}));
                        break;
                    default:
                        log.warn("Wrong phase number of feeder whose id is " + e.getDevId());
                }
                Collections.addAll(branches, devIdToBranch.get(e.getDevId()));
            } else if(e.getEdgeType() == DetailedEdge.EDGE_TYPE_TF_WINDING) {
                //todo
            }
        }
        island.setBuses(buses);
        island.setBranches(branches);
        return island;
    }
}

