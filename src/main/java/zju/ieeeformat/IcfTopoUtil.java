package zju.ieeeformat;

import org.apache.log4j.Logger;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.alg.ConnectivityInspector;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.DefaultWeightedEdge;
import org.jgrapht.graph.SimpleWeightedGraph;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-9-12
 */
public class IcfTopoUtil {

    private static Logger log = Logger.getLogger(IcfTopoUtil.class);

    //数据结构<图>
    private SimpleWeightedGraph<BusData, DefaultEdge> graph;
    //支路与数据结构中的边的对应关系
    private Map<BranchData, DefaultEdge> branchToEdge;
    //电气岛
    private IEEEDataIsland island;
    //母线编号和母线对象之间的映射
    public Map<Integer, BusData> busMap;
    //用于检测联通性的类
    private ConnectivityInspector<BusData, DefaultEdge> inspector;

    public static UndirectedGraph<BusData, DefaultEdge> createGraph(IEEEDataIsland island) {
        IcfTopoUtil topo = new IcfTopoUtil(island);
        topo.createGraph();
        return topo.getGraph();
    }

    public IcfTopoUtil() {
    }

    public IcfTopoUtil(IEEEDataIsland island) {
        this.island = island;
    }

    public void createGraph() {
        createGraph(true);
    }

    private void createGraph(boolean isBuildBranchIdex) {
        graph = new SimpleWeightedGraph<BusData, DefaultEdge>(DefaultWeightedEdge.class);
        for (BusData bus : island.getBuses())
            graph.addVertex(bus);
        busMap = island.getBusMap();
        if (isBuildBranchIdex)
            branchToEdge = new HashMap<BranchData, DefaultEdge>(island.getBranches().size());
        for (BranchData branch : island.getBranches()) {
            BusData bus1 = busMap.get(branch.getTapBusNumber());
            BusData bus2 = busMap.get(branch.getZBusNumber());
            DefaultEdge edge = graph.getEdge(bus1, bus2);
            //根据线路回数设置权重
            if (edge != null)
                graph.setEdgeWeight(edge, graph.getEdgeWeight(edge) + 1.0);
            else {
                edge = graph.addEdge(bus1, bus2);
                graph.setEdgeWeight(edge, 1.0);
            }
            if (isBuildBranchIdex)
                branchToEdge.put(branch, edge);
        }
    }

    /**
     * 首先判断是否联通
     *
     * @return 电气岛是否联通
     */
    public boolean isIslandConnected() {
        inspector = new ConnectivityInspector<BusData, DefaultEdge>(graph);
        return inspector.isGraphConnected();
    }

    public boolean isIslandConnected(BranchData branch) {
        return getSubIslands(branch) == null;
    }

    /**
     * 通过断开支路将网络分裂成几个子网络
     *
     * @param branch 断开的线路
     * @return 分裂后的子图
     */
    public List<Set<BusData>> getSubIslands(BranchData branch) {
        DefaultEdge edge = branchToEdge.get(branch);
        if (graph.getEdgeWeight(edge) > 1.5)
            return null;
        BusData bus1 = busMap.get(branch.getTapBusNumber());
        BusData bus2 = busMap.get(branch.getZBusNumber());
        graph.removeEdge(edge);
        ConnectivityInspector<BusData, DefaultEdge> inspector = new ConnectivityInspector<BusData, DefaultEdge>(graph);
        boolean r = inspector.isGraphConnected();
        if (r)
            return null;
        List<Set<BusData>> subIslands = inspector.connectedSets();
        graph.addEdge(bus1, bus2, edge);
        return subIslands;
    }

    /**
     * 通过断开支路将网络分裂成几个子网络
     *
     * @param branches 断开的多条线路
     * @return 分裂后的子图
     */
    public List<Set<BusData>> getSubIslands(BranchData[] branches) {
        DefaultEdge edge;
        for (BranchData branch : branches) {
            edge = branchToEdge.get(branch);
            graph.setEdgeWeight(edge, graph.getEdgeWeight(edge) - 1.0);
        }
        BusData bus1, bus2;
        for (BranchData branch : branches) {
            edge = branchToEdge.get(branch);
            if (graph.getEdgeWeight(edge) > 0.5)
                continue;
            graph.removeEdge(edge);
        }
        inspector = new ConnectivityInspector<BusData, DefaultEdge>(graph);
        boolean r = inspector.isGraphConnected();
        List<Set<BusData>> subIslands = null;
        if (!r)
            subIslands = inspector.connectedSets();
        for (BranchData branch : branches) {
            edge = branchToEdge.get(branch);
            bus1 = busMap.get(branch.getTapBusNumber());
            bus2 = busMap.get(branch.getZBusNumber());
            if (!graph.containsEdge(bus1, bus2)) {
                graph.addEdge(bus1, bus2, edge);
                graph.setEdgeWeight(edge, 1.0);
            } else
                graph.setEdgeWeight(edge, graph.getEdgeWeight(edge) + 1.0);
        }
        return subIslands;
    }

    /**
     * <br>该方法通过分析变压器支路，将由某一电压等级供电的子网络等效成负荷</br>
     * <br>该方法会影响作为数据结构的图已经IEEEDataIsland</br>
     *
     * @param baseV 需要分析的电压等级
     */
    public void doEquivalentLoad(double baseV) {
        //if (graph.txt == null)
        //    createGraph();
        //List<Set<BusData>> subgraphes;
        //Set<BusData> equivalentLoad = null;
        //List<DefaultEdge> edges = new ArrayList<DefaultEdge>();
        //BusData b1, b2, bus1, bus2, tmpBus;
        //DefaultEdge e;
        //int count = 0;
        //double p, q;
        //for (BranchData b : island.getBranches()) {
        //    if (b.getType() == BranchData.BRANCH_TYPE_TF_FIXED_TAP) {
        //        e = getBranchToEdge().get(b);
        //        b1 = graph.txt.getEdgeSource(e);
        //        b2 = graph.txt.getEdgeTarget(e);
        //        if (b1.getBaseVoltage() < b2.getBaseVoltage()) {
        //            tmpBus = b1;
        //            b1 = b2;
        //            b2 = tmpBus;
        //        }
        //        if (b1.getBaseVoltage() < (baseV - 50) || b1.getBaseVoltage() > (baseV + 50))
        //            continue;
        //
        //        //如果是和变压器中心点之间
        //        if (b2.getBaseVoltage() < 1e-2) {
        //            boolean isUpper220 = false;
        //            for (DefaultEdge edge : getGraph().edgesOf(b2)) {
        //                if (edge == e)
        //                    continue;
        //                bus1 = graph.txt.getEdgeSource(edge);
        //                bus2 = graph.txt.getEdgeTarget(edge);
        //                if (bus1 == b2 && bus2.getBaseVoltage() < b1.getBaseVoltage())
        //                    isUpper220 = true;
        //                if (bus2 == b2 && bus1.getBaseVoltage() < b1.getBaseVoltage())
        //                    isUpper220 = true;
        //            }
        //            if (isUpper220)
        //                continue;
        //            subgraphes = getSubIslands(b);
        //            assert (subgraphes.size() == 2);
        //        } else {
        //            subgraphes = getSubIslands(b);
        //            //判断网络是否解裂
        //            assert (subgraphes.size() == 2);
        //        }
        //
        //        //判断子图中哪些是主网，哪些是等效负荷
        //        for (BusData bus : subgraphes.get(0)) {
        //            if (bus == b1) {
        //                equivalentLoad = subgraphes.get(1);
        //                break;
        //            } else if (bus == b2) {
        //                equivalentLoad = subgraphes.get(0);
        //                break;
        //            }
        //        }
        //        p = 0.0;
        //        q = 0.0;
        //        edges.clear();
        //        for (BusData bus : equivalentLoad) {
        //            p += bus.getLoadMW();
        //            q += bus.getLoadMVAR();
        //            p -= bus.getGenerationMW();
        //            q -= bus.getGenerationMVAR();
        //            for (DefaultEdge edge : graph.txt.edgesOf(bus))
        //                edges.add(edge);
        //        }
        //        for (DefaultEdge edge : edges)
        //            graph.txt.removeEdge(edge);
        //        for (BusData bus : equivalentLoad)
        //            graph.txt.removeVertex(bus);
        //        b1.setLoadMW(b1.getLoadMW() + p);
        //        b1.setLoadMVAR(b1.getLoadMVAR() + q);
        //        count++;
        //    }
        //}
        //log.info("共" + count + "条变压器支路可以等效成负荷.");
        //count = 0;
        //for (DefaultEdge edge : graph.txt.edgeSet())
        //    count += graph.txt.getEdgeWeight(edge);
        //List<BranchData> branches = new ArrayList<BranchData>(count);
        //for (BranchData b : island.getBranches())
        //    if (graph.txt.containsEdge(branchToEdge.get(b)))
        //        branches.add(b);
        //island.setBuses(new ArrayList<BusData>(graph.txt.vertexSet()));
        //island.setBranches(branches);
        //createGraph();
    }

    public void setIsland(IEEEDataIsland island) {
        this.island = island;
    }

    public SimpleWeightedGraph<BusData, DefaultEdge> getGraph() {
        return graph;
    }

    public Map<BranchData, DefaultEdge> getBranchToEdge() {
        return branchToEdge;
    }

    public Map<Integer, BusData> getBusMap() {
        return busMap;
    }
}
