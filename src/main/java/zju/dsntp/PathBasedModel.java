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

    private static boolean isDebug = true; //置成true, 打印调试信息

    //配电系统
    DistriSys sys;
    //路径,起始点都是电源
    List<MapObject[]> pathes;
    //每个电源点在上边数组的位置
    int[] supplyStart;
    //图中除了电源以外所有的节点
    List<DsConnectNode> nodes;
    //图中所有的边
    List<MapObject> edges;
    //以某个节点为终点的所有路径，按cns中存储节点的顺序排列
    List<MapObject[]> cnpathes;
    //每个节点在cnspathes中的位置
    int[] cnStart;
    //经过一条边的所有路径，按edges中存储的边排列
    List<MapObject[]> edgepathes;
    //每条边在edgepathes中的位置
    int[] edgeStart;
    //cnpathes中路径在pathes中对应的序号
    int[] cnpathesIndex;
    //edgepathes中路径在pathes中对应的序号
    List<Integer> edgepathesIndex;

    public PathBasedModel(DistriSys sys) {
        this.sys = sys;
    }

    public void buildPathes() {
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        //用于深度优先搜索的栈
        Deque<Object> stack = new ArrayDeque<>();
        pathes = new ArrayList<>();//todo: not efficient
        //生成的新的路径
        MapObject[] p = new MapObject[0];
        int i, count = 0;
        boolean flag1, flag2;
        supplyStart = new int[supplies.length];
        for (String supply : supplies) {
            supplyStart[count++] = pathes.size();
            DsConnectNode supplyCn = sys.getCns().get(supply);
            stack.push(supplyCn);//先将电源节点压入栈内
            while (!stack.isEmpty()) {
                DsConnectNode cn = (DsConnectNode) stack.peek();    //对栈顶元素深度搜索路径
                flag1 = true;
                for (MapObject e : g.edgesOf(cn)) {
                    DsConnectNode cn1 = g.getEdgeSource(e);
                    DsConnectNode cn2 = g.getEdgeTarget(e);
                    if (cn1.equals(cn)) {
                        flag2 = true;
                        if (stack.contains(cn2))
                            continue;
                        for (String scn : supplies)
                            if (scn.equals(cn2.getId())) {
                                flag2 = false;
                                break;
                            }
                        //将栈中的结点生成新的路径
                        i = stack.size() - 2;
                        Iterator<Object> Iter = stack.iterator();
                        p = new MapObject[stack.size()];
                        p[p.length - 1] = g.getEdge(cn1, cn2);
                        DsConnectNode temp1 = (DsConnectNode) Iter.next();
                        while (Iter.hasNext()) {
                            DsConnectNode temp2 = (DsConnectNode) Iter.next();
                            p[i] = g.getEdge(temp2, temp1);
                            temp1 = temp2;
                            i--;
                        }
                        //表中不存在某个路径与新路径完全相同
                        for (i = 0; i < pathes.size(); i++)
                            if (Arrays.equals(p, pathes.get(i))) {
                                flag2 = false;
                                break;
                            }
                        //cn2不在栈中，cn2不能是电源,且已有路径中没有生成的新路径为开头的
                        if (flag2) {
                            stack.push(cn2);    //结点入栈
                            flag1 = false;
                            //将路径加入pathes
                            pathes.add(p);  //增加一条路径
                            break;
                        }
                    } else {
                        flag2 = true;
                        if (stack.contains(cn1))
                            continue;
                        for (String scn : supplies)
                            if (scn.equals(cn1.getId())) {
                                flag2 = false;
                                break;
                            }
                        i = stack.size() - 2;
                        Iterator<Object> Iter = stack.iterator();
                        p = new MapObject[stack.size()];
                        p[p.length - 1] = g.getEdge(cn2, cn1);
                        DsConnectNode temp1 = (DsConnectNode) Iter.next();
                        while (Iter.hasNext()) {
                            DsConnectNode temp2 = (DsConnectNode) Iter.next();
                            p[i] = g.getEdge(temp2, temp1);
                            temp1 = temp2;
                            i--;
                        }
                        for (i = 0; i < pathes.size(); i++)
                            if (Arrays.equals(p, pathes.get(i))) {
                                flag2 = false;
                                break;
                            }
                        if (flag2) {  //cn2不在栈中，cn1不能是电源，且已有路径中没有生成的新路径为开头的
                            stack.push(cn1);
                            flag1 = false;
                            pathes.add(p);  //增加一条路径
                            break;
                        }
                    }
                }
                if (flag1)
                    stack.pop();
            }
        }
        //sys.getOrigGraph().getEdge();
        //origGraph.getEdgeSource()
        buildEdgesAndNodes();
        buildCnsPathes();
        buildedgePathes();
        if(isDebug) {
//            System.out.printf("\nAll the pathes started from a specific supply\n");
//            printPathes(pathes);
        }
    }

    //以某个结点为终点的所有路径
    public void buildCnsPathes() {
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        cnpathes = new ArrayList<>(pathes.size());
        cnpathesIndex = new int[pathes.size()];
        int i,k;
        String cn, lastID;
        cnStart = new int[nodes.size()];
        for (k = 0; k <= nodes.size() - 1; k++) {
            cnStart[k] = cnpathes.size();
            cn = nodes.get(k).getId();
            for (i = 0; i <= pathes.size() - 1; i++) {
                lastID = g.getEdgeTarget(pathes.get(i)[pathes.get(i).length - 1]).getId();
                if (pathes.get(i).length == 1) {
                    for (String scn : supplies) {
                        if (scn.equals(lastID)) {
                            lastID = g.getEdgeSource(pathes.get(i)[pathes.get(i).length - 1]).getId();
                            break;
                        }
                    }
                    if (cn.equals(lastID)) {
                        cnpathes.add(pathes.get(i));
                        cnpathesIndex[cnpathesIndex.length] = i;
                    }
                }
                else {
                    //如果路径上倒数第二条边有节点与lastID相同，则lastID应取最后一条边的另一个端点才是路径上的最后一个点
                    if (lastID.equals(g.getEdgeSource(pathes.get(i)[pathes.get(i).length - 2]).getId()) || lastID.equals(g.getEdgeTarget(pathes.get(i)[pathes.get(i).length - 2]).getId()))
                        lastID = g.getEdgeSource(pathes.get(i)[pathes.get(i).length - 1]).getId();
                    if (cn.equals(lastID)) {
                        cnpathes.add(pathes.get(i));
                        cnpathesIndex[cnpathesIndex.length] = i;
                    }
                }
            }
        }
        if (isDebug) {
//            System.out.printf("\nThe end of the pathes is a specific cn\n");
//            for(i = 0; i < cns.size(); i++)
//                System.out.printf("The start of the index that the pathes end in the node %s is %d\n", cns.get(i).getId(), cnStart[i]);
//            printPathes(cnpathes);
        }
    }

    //通过某条边的所有路径
    public void buildedgePathes() {
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        edgepathes = new ArrayList<>();//todo: not efficient
        edgepathesIndex = new ArrayList<>();//todo: not efficient
        int i, j, k;
        edgeStart = new int[edges.size()];
        for (k = 0; k <= edges.size() - 1; k++) {
            edgeStart[k] = edgepathes.size();
            for (i = 0; i <= pathes.size() - 1; i++) {
                for (j = 0; j <= pathes.get(i).length - 1; j++)
                    if (edges.get(k) == pathes.get(i)[j]) {
                        edgepathes.add(pathes.get(i));
                        edgepathesIndex.add(i);
                        break;
                    }
            }
        }
        if(isDebug) {
//            System.out.printf("\nThe pathes contain a specific edge\n");
//            for(i = 0; i < edges.size(); i++)
//                System.out.printf("The start of the index that the pathes contain the edge %s %s is %d\n", g.getEdgeSource(edges.get(i)).getId(), g.getEdgeTarget(edges.get(i)).getId(), edgeStart[i]);
//            printPathes(edgepathes);
        }
    }

    public List<MapObject[]> getPathes() {
        return pathes;
    }

    public int[] getSupplyStart() {
        return supplyStart;
    }

    public List<MapObject[]> getCnPathes() {
        return cnpathes;
    }

    public int[] getCnStart() {
        return cnStart;
    }

    public List<MapObject[]> getEdgePathes() {
        return edgepathes;
    }

    public int[] getEdgeStart() {
        return edgeStart;
    }

    //将图中除电源以外的节点和所有的边分别存入数组中
    public void buildEdgesAndNodes() {
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        edges = new ArrayList<>();//todo: not efficient
        nodes = new ArrayList<>();//todo: not efficient
        boolean issupply;
        DsConnectNode supply0 = sys.getCns().get(supplies[0]);  // 从某一个电源开始深度优先搜索
        DsConnectNode cn1, cn2, cn;
        Deque<Object> stack = new ArrayDeque<>();
        stack.push(supply0);
        boolean flag;
        while (!stack.isEmpty()) {
            flag = true;
            cn = (DsConnectNode) stack.peek();
            for (MapObject edge : g.edgesOf(cn)) {
                if (!edges.contains(edge))
                    edges.add(edge);
                cn1 = g.getEdgeSource(edge);
                cn2 = g.getEdgeTarget(edge);
                if (cn1 == cn) {
                    if (!nodes.contains(cn2) && !stack.contains(cn2)) {
                        stack.push(cn2);
                        flag = false;
                        break;
                    }
                } else if (!nodes.contains(cn1) && !stack.contains(cn1)) {
                    stack.push(cn1);
                    flag = false;
                    break;
                }
            }
            if (flag) {
                //节点出栈时判断是否存储：不是电源节点且还未存储过，则存入cns中
                if (!nodes.contains(stack.peek())) {
                    issupply = false;
                    for(String scn : supplies)
                        if(((DsConnectNode) stack.peek()).getId().equals(scn)) {
                            issupply = true;
                            break;
                        }
                    if(!issupply)
                        nodes.add((DsConnectNode) stack.peek());
                }
                stack.pop();
            }
        }
        if (isDebug) {
            for (DsConnectNode cn3 : nodes)
                System.out.println(cn3.getId());
            System.out.printf("%d", nodes.size());
        }
    }

    //输出每条路径，以节点连接的方式表示，测试用
    public void printPathes(List<MapObject[]> pathes) {
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        int i, j;
        boolean isSupply;
        String temp;
        System.out.println("Number of pathes is " + pathes.size());
        for (i = 0; i <= pathes.size() - 1; i++) {
            isSupply = false;
            System.out.println("Length of " + i + "th path is " + pathes.get(i).length + ":");
            for (String scn : supplies) {
                if (scn.equals(g.getEdgeSource(pathes.get(i)[0]).getId())) {
                    isSupply = true;
                    break;
                }
            }
            if (isSupply) {
                temp = g.getEdgeSource(pathes.get(i)[0]).getId();
                System.out.printf("%s", temp);
            } else {
                temp = g.getEdgeTarget(pathes.get(i)[0]).getId();
                System.out.printf("%s", temp);
            }
            for (j = 0; j <= pathes.get(i).length - 1; j++) {
                if (temp.equals(g.getEdgeTarget(pathes.get(i)[j]).getId()))
                    temp = g.getEdgeSource(pathes.get(i)[j]).getId();
                else
                    temp = g.getEdgeTarget(pathes.get(i)[j]).getId();
                System.out.printf("-%s", temp);
            }
            System.out.printf("\n");
        }
        //END
        System.out.println("-----END-----");
    }
}