package zju.dsntp;

import org.apache.log4j.Logger;
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

    private static boolean isDebug = false; //置成true, 打印调试信息

    //配电系统
    DistriSys sys;
    //路径,起始点都是电源
    List<MapObject[]> pathes;
    //每个电源点在上边数组的位置
    int[] supplyStart;

    List<DsConnectNode> cns;

    List<MapObject> edges;

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

        //START:这部分输出每条路径，测试用
        buildEdgesAndNodes();
        int j;
        boolean isSupply;
        if(isDebug) {
            System.out.println("Number of pathes is " + pathes.size());
            for (i = 0; i <= pathes.size() - 1; i++) {
                isSupply = false;
                System.out.println("Length of " + i + "th path is " + pathes.get(i).length + ":");

                String temp;
                for (String scn : supplies) {
                    if (scn.equals(g.getEdgeSource(pathes.get(i)[0]).getId())) {
                        isSupply = true;
                        break;
                    }
                }
                if (isSupply) {
                    temp = g.getEdgeSource(pathes.get(i)[0]).getId();
                    System.out.printf("%s", g.getEdgeSource(pathes.get(i)[0]).getId());
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
            if (isDebug)
                System.out.println("-----END-----");
        }

        //START:输出以某个结点为终点的所有路径
        String cn;
        for (int k = 0; k <= cns.size() - 1; k++) {
            cn = cns.get(k).getId();
            boolean isCnSupply = false;
            for (String scn : supplies) {
                if (cn.equals(scn)) {
                    isCnSupply = true;
                }
            }
            if (!isCnSupply) {
                if(isDebug)
                    System.out.println("The path ending by " + cn + ":");
                for (i = 0; i <= pathes.size() - 1; i++) {
                    boolean flag = false;
                    String lastID = g.getEdgeTarget(pathes.get(i)[pathes.get(i).length - 1]).getId();
                    if (pathes.get(i).length == 1) {
                        for (String scn : supplies) {
                            if (Objects.equals(lastID, scn)) {
                                flag = true;
                                lastID = g.getEdgeSource(pathes.get(i)[pathes.get(i).length - 1]).getId();
                                break;
                            }
                        }
                        if(cn.equals(lastID)) {
                            if (flag && isDebug)
                                System.out.println(g.getEdgeTarget(pathes.get(i)[pathes.get(i).length - 1]).getId() + "-" + lastID);
                            else if(isDebug)
                                System.out.println(g.getEdgeSource(pathes.get(i)[pathes.get(i).length - 1]).getId() + "-" + lastID);
                            continue;
                        } else
                            continue;
                    }

                    if (Objects.equals(lastID, g.getEdgeSource(pathes.get(i)[pathes.get(i).length - 2]).getId()) || Objects.equals(lastID, g.getEdgeTarget(pathes.get(i)[pathes.get(i).length - 2]).getId())) {
                        lastID = g.getEdgeSource(pathes.get(i)[pathes.get(i).length - 1]).getId();
                    }

                    if (cn.equals(lastID)) {
                        String temp;
                        isSupply = false;
                        for (String scn : supplies) {
                            if (scn.equals(g.getEdgeSource(pathes.get(i)[0]).getId())) {
                                isSupply = true;
                                break;
                            }
                        }
                        if (isSupply) {
                            temp = g.getEdgeSource(pathes.get(i)[0]).getId();
                            if(isDebug)
                                System.out.printf("%s", g.getEdgeSource(pathes.get(i)[0]).getId());
                        } else {
                            temp = g.getEdgeTarget(pathes.get(i)[0]).getId();
                            if(isDebug)
                                System.out.printf("%s", temp);
                        }

                        for (j = 0; j <= pathes.get(i).length - 1; j++) {
                            if (temp.equals(g.getEdgeTarget(pathes.get(i)[j]).getId()))
                                temp = g.getEdgeSource(pathes.get(i)[j]).getId();
                            else
                                temp = g.getEdgeTarget(pathes.get(i)[j]).getId();
                            if(isDebug)
                                System.out.printf("-%s", temp);
                        }
                        if(isDebug)
                            System.out.println();
                    }
                }
            }
        }
        //End
        if(isDebug)
            System.out.println("-----END-----");

        //START:输出通过某条边的所有路径
        for (int k = 0; k <= edges.size() - 1; k++) {
            if(isDebug)
                System.out.println("The pathes including " + g.getEdgeSource(edges.get(k)).getId() + "-" + g.getEdgeTarget(edges.get(k)).getId() + ":");
            for (i = 0; i <= pathes.size() - 1; i++) {
                boolean flag = false;
                for (j = 0; j <= pathes.get(i).length - 1; j++)
                    if (edges.get(k) == pathes.get(i)[j]) {
                        flag = true;
                        break;
                    }
                if (flag) {
                    isSupply = false;
                    String temp;
                    for (String scn : supplies) {
                        if (scn.equals(g.getEdgeSource(pathes.get(i)[0]).getId())) {
                            isSupply = true;
                            break;
                        }
                    }
                    if (isSupply) {
                        temp = g.getEdgeSource(pathes.get(i)[0]).getId();
                        if(isDebug)
                            System.out.printf("%s", g.getEdgeSource(pathes.get(i)[0]).getId());
                    } else {
                        temp = g.getEdgeTarget(pathes.get(i)[0]).getId();
                        if(isDebug)
                            System.out.printf("%s", temp);
                    }

                    for (j = 0; j <= pathes.get(i).length - 1; j++) {
                        if (temp.equals(g.getEdgeTarget(pathes.get(i)[j]).getId()))
                            temp = g.getEdgeSource(pathes.get(i)[j]).getId();
                        else
                            temp = g.getEdgeTarget(pathes.get(i)[j]).getId();
                        if(isDebug)
                            System.out.printf("-%s", temp);
                    }
                    if(isDebug)
                        System.out.printf("\n");
                }
            }
        }
        if(isDebug)
            System.out.println("-----END-----");
    }

    public List<MapObject[]> getPathes() {
        return pathes;
    }

    public int[] getSupplyStart() {
        return supplyStart;
    }

    public void buildEdgesAndNodes() {
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        edges = new ArrayList<>();//todo: not efficient
        cns = new ArrayList<>();//todo: not efficient
        DsConnectNode supply0 = sys.getCns().get(supplies[0]);
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
                    if (!cns.contains(cn2) && !stack.contains(cn2)) {
                        stack.push(cn2);
                        flag = false;
                        break;
                    }
                } else if (!cns.contains(cn1) && !stack.contains(cn1)) {
                    stack.push(cn1);
                    flag = false;
                    break;
                }
            }
            if (flag) {
                if (!cns.contains(stack.peek()))
                    cns.add((DsConnectNode) stack.peek());
                stack.pop();
            }
        }
        if(isDebug)
            for (DsConnectNode cn3 : cns)
                System.out.println(cn3.getId());
        //System.out.printf("%d", cns.size());
    }
}