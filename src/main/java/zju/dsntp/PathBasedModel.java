package zju.dsntp;

import org.jgrapht.UndirectedGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsConnectNode;

import java.util.*;

/**
 * 基于路径的配电网拓扑模型,用于转供路径优化，网络重构等
 * User: Dong Shufeng
 * Date: 2016/9/12
 */
public class PathBasedModel {
    //配电系统
    DistriSys sys;
    //路径,起始点都是电源
    List<MapObject[]> pathes;
    //每个电源点在上边数组的位置
    int[] supplyStart;

    public PathBasedModel(DistriSys sys) {
        this.sys = sys;
    }

    public void buildPathes() {
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        //用于深度优先搜索的栈
        Deque<Object> stack = new ArrayDeque<>();
        pathes = new ArrayList<>();//todo: not efficient
        for(String supply : supplies) {
            DsConnectNode supplyCn = sys.getCns().get(supply);
            stack.push(supplyCn);//先将电源节点压入栈内
            for(MapObject e : g.edgesOf(supplyCn)) {
                DsConnectNode cn1 = g.getEdgeSource(e);
                DsConnectNode cn2 = g.getEdgeTarget(e);
                //todo:
            }
        }
        //sys.getOrigGraph().getEdge();
        //origGraph.getEdgeSource()
    }

    public List<MapObject[]> getPathes() {
        return pathes;
    }

    public int[] getSupplyStart() {
        return supplyStart;
    }
}
