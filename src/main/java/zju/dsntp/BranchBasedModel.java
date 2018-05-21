package zju.dsntp;

import org.jgrapht.UndirectedGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsConnectNode;

import java.util.*;

/**
 * Created by xuchengsi on 2018/1/13.
 */
public class BranchBasedModel {
    // 配电系统
    DistriSys sys;
    // 图中所有的节点
    List<DsConnectNode> nodes;
    // 图中的负荷节点
    List<DsConnectNode> loadNodes;
    // 图中所有的边
    List<MapObject> edges;
    // 供电环路
    List<int[]> loops;
    // 重要负荷节点
    String[] impLoads;
    // 重要负荷节点的所有路径
    List<MapObject[]> impPaths = null;
    // impPaths中负荷节点的起始位置
    int[] impPathStart;
    // impPath中边与edges的对应关系
    List<int[]> impPathToEdge;

    public BranchBasedModel(DistriSys sys) {
        this.sys = sys;
    }

    /**
     * 搜索供电环路
     */
    public void buildLoops() {
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        edges = new ArrayList<>(g.edgeSet().size());
        nodes = new ArrayList<>(g.vertexSet().size());
        loadNodes = new ArrayList<>(g.vertexSet().size() - supplies.length);
        edges.addAll(g.edgeSet());
        nodes.addAll(g.vertexSet());
        for (DsConnectNode node : nodes) {
            boolean isSupply = false;
            for (String supply : supplies) {
                if (node.getId().equals(supply)) {
                    isSupply = true;
                    break;
                }
            }
            if (!isSupply) {
                loadNodes.add(node);
            }
        }
        // 初始化节点为未访问状态
        HashMap<String, Integer> visited = new HashMap<>(nodes.size());
        for (DsConnectNode node : nodes) {
            visited.put(node.getId(), 0);
        }
        // 用于深度优先搜索的栈
        Deque<DsConnectNode> nodeStack = new ArrayDeque<>();
        // 环路的支路表示
        List<int[]> edgeInLoops = new ArrayList<>();
        loops = new ArrayList<>();

        boolean noPush;
        for (String supply : supplies) {
            List<MapObject> lastEdgeOfLoop = new ArrayList<>();
            DsConnectNode supplyCn = sys.getCns().get(supply);
            nodeStack.push(supplyCn);   // 将电源节点压入栈内
            while (!nodeStack.isEmpty()) {
                noPush = true;
                DsConnectNode cn = nodeStack.peek();
                Iterator<DsConnectNode> nodeStackIter = nodeStack.iterator();
                nodeStackIter.next();
                MapObject recentEdge = null;
                if (nodeStackIter.hasNext()) {
                    recentEdge = g.getEdge(cn, nodeStackIter.next());
                }
                // 对栈顶元素深度搜索，根据边寻找下一个顶点
                for (MapObject e : g.edgesOf(cn)) {
                    if (!e.equals(recentEdge)) {
                        DsConnectNode neighbor;
                        if (g.getEdgeSource(e).equals(cn)) {
                            neighbor = g.getEdgeTarget(e);
                        } else {
                            neighbor = g.getEdgeSource(e);
                        }
                        // 如果顶点或环的边已经被遍历过，则不进行处理
                        if (visited.get(neighbor.getId()) != 0 || lastEdgeOfLoop.contains(e)) {
                            continue;
                        }
                        // 判断顶点是否为电源节点
                        boolean isSupply = false;
                        for (String scn : supplies) {
                            if (scn.equals(neighbor.getId())) {
                                isSupply = true;
                                break;
                            }
                        }
                        // 找到环路
                        if (nodeStack.contains(neighbor) || isSupply) {
                            lastEdgeOfLoop.add(e);
                            // 存储新找到的环路
                            Deque<int[]> edgeInNewLoopQueue = new ArrayDeque<>();
                            // 深搜栈中相邻的两个节点
                            DsConnectNode cn1;
                            DsConnectNode cn2;
                            // 找出环路的起始节点在深搜栈中的位置
                            int repeatPos = 0;
                            Iterator<DsConnectNode> nodeStackDesIter = nodeStack.descendingIterator();
                            cn1 = nodeStackDesIter.next();
                            if (!isSupply) {
                                while (nodeStackDesIter.hasNext()) {
                                    cn1 = nodeStackDesIter.next();
                                    repeatPos++;
                                    if (cn1.equals(neighbor)) {
                                        break;
                                    }
                                }
                            }
                            // 存环路
                            int[] loop = new int[nodeStack.size() - repeatPos];
                            int i = 0;
                            int[] edgeInLoop = new int[edges.size()];
                            while (nodeStackDesIter.hasNext()) {
                                cn2 = nodeStackDesIter.next();
                                MapObject edgeOfLoop = g.getEdge(cn1, cn2);
                                for (int j = 0; j < edges.size(); j++) {
                                    if (edges.get(j).equals(edgeOfLoop)) {
                                        loop[i] = j;
                                        edgeInLoop[j] = 1;
                                        break;
                                    }
                                }
                                cn1 = cn2;
                                i++;
                            }
                            for (int j = 0; j < edges.size(); j++) {
                                if (edges.get(j).equals(e)) {
                                    loop[i] = j;
                                    edgeInLoop[j] = 1;
                                    break;
                                }
                            }
                            loops.add(loop);
                            edgeInNewLoopQueue.offer(edgeInLoop);
                            // 找重复环路
                            for (int[] edgeInOldLoop : edgeInLoops) {
//                                boolean hasOverlap = false;
                                int[] edgeInNewLoop = new int[edges.size()];
                                int newLoopLen = 0;
                                for (int j = 0; j < edges.size(); j++) {
                                    edgeInNewLoop[j] = edgeInLoop[j] ^ edgeInOldLoop[j];
                                    if (edgeInNewLoop[j] == 1) {
                                        newLoopLen++;
                                    }
//                                    if (edgeInLoop[j] == 1 && edgeInOldLoop[j] == 1) {
//                                        hasOverlap = true;
//                                    }
                                }
                                edgeInNewLoopQueue.offer(edgeInNewLoop);
                                // 存在重叠
//                                if (hasOverlap) {
                                int[] newLoop = new int[newLoopLen];
                                newLoopLen = 0;
                                for (int j = 0; j < edges.size(); j++) {
                                    if (edgeInNewLoop[j] == 1) {
                                        newLoop[newLoopLen] = j;
                                        newLoopLen++;
                                    }
                                }
                                if (sortLoop(newLoop)) {
                                    loops.add(newLoop);
                                }
//                                }
                            }
                            while (!edgeInNewLoopQueue.isEmpty()) {
                                edgeInLoops.add(edgeInNewLoopQueue.poll());
                            }
                            continue;
                        }
                        // 未遍历过的节点
                        nodeStack.push(neighbor);
                        noPush = false;
                        break;
                    }
                }
                if (noPush) {
                    visited.put(nodeStack.pop().getId(), 1);
                }
            }
        }
    }

    /**
     * 对环路中的支路排序
     */
    public boolean sortLoop(int[] loop) {
        if (loop.length < 0) {
            return false;
        }
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        DsConnectNode preV = g.getEdgeTarget(edges.get(loop[0]));
        int connBranchNum = 1;
        int i;
        for (i = 1; i < loop.length; i++) {
            boolean hasChange = false;
            for (int j = i; j < loop.length; j++) {
                DsConnectNode v1 = g.getEdgeSource(edges.get(loop[j]));
                DsConnectNode v2 = g.getEdgeTarget(edges.get(loop[j]));
                if (v1.equals(preV) || v2.equals(preV)) {
                    DsConnectNode nextV;
                    if (v1 == preV) {
                        nextV = v2;
                    } else {
                        nextV = v1;
                    }
                    int t = loop[i];
                    loop[i] = loop[j];
                    loop[j] = t;
                    preV = nextV;
                    connBranchNum++;
                    hasChange = true;
                    break;
                }
            }
            if (!hasChange) {
                break;
            }
        }
        if (connBranchNum < loop.length) {
            preV = g.getEdgeSource(edges.get(loop[0]));
            for (int j = 0; j < (i - 1) / 2; j++) {
                int t = loop[j];
                loop[j] = loop[i - 1 - j];
                loop[i - 1 - j] = t;
            }
            for (; i < loop.length; i++) {
                boolean hasChange = false;
                for (int j = i; j < loop.length; j++) {
                    DsConnectNode v1 = g.getEdgeSource(edges.get(loop[j]));
                    DsConnectNode v2 = g.getEdgeTarget(edges.get(loop[j]));
                    if (v1.equals(preV) || v2.equals(preV)) {
                        DsConnectNode nextV;
                        if (v1 == preV) {
                            nextV = v2;
                        } else {
                            nextV = v1;
                        }
                        int t = loop[i];
                        loop[i] = loop[j];
                        loop[j] = t;
                        preV = nextV;
                        connBranchNum++;
                        hasChange = true;
                        break;
                    }
                }
                if (!hasChange) {
                    break;
                }
            }
        }
        return connBranchNum == loop.length;
    }

    /**
     * 打印供电环路
     */
    public void printLoop() {
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        for (int[] loop : loops) {
            for (int i : loop) {
                MapObject edge = edges.get(i);
                System.out.print(g.getEdgeSource(edge).getId() + "-" + g.getEdgeTarget(edge).getId() + ", ");
            }
            System.out.println();
        }
    }

    public List<int[]> getLoops() {
        return loops;
    }

    /**
     * 搜索供电环路
     */
    public void buildLoopsFail() {
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        edges = new ArrayList<>(g.edgeSet().size());
        nodes = new ArrayList<>(g.vertexSet().size());
        edges.addAll(g.edgeSet());
        nodes.addAll(g.vertexSet());
        // 初始化节点为未访问状态
        HashMap<String, Integer> visited = new HashMap(nodes.size());
        for (DsConnectNode node : nodes) {
            visited.put(node.getId(), 0);
        }
        // 用于深度优先搜索的栈
        Deque<DsConnectNode> nodeStack = new ArrayDeque<>();
        // 环路末尾重复节点在深搜栈中的位置
        Deque<Integer> repeatPosStack = new ArrayDeque<>();
        // 深搜的起始位置
        Deque<Integer> startPosStack = new ArrayDeque<>();
        // repeatPosStack中环路在loops中的位置
        Deque<Integer> loopPosStack = new ArrayDeque<>();
        // repeatPosStack中环路的支路表示
        Deque<int[]> edgeInLoopStack = new ArrayDeque<>();
        // todo:电源分隔图的情况
        loops = new ArrayList<>();

        boolean noPush;
        for (String supply : supplies) {
            DsConnectNode supplyCn = sys.getCns().get(supply);
            nodeStack.push(supplyCn);   // 将电源节点压入栈内
            while (!nodeStack.isEmpty()) {
                noPush = true;
                DsConnectNode cn = nodeStack.peek();
                Iterator<DsConnectNode> nodeStackDesIter = nodeStack.descendingIterator();
                MapObject recentEdge = null;
                if (nodeStackDesIter.hasNext()) {
                    recentEdge = g.getEdge(cn, nodeStackDesIter.next());
                }
                // 对栈顶元素深度搜索，根据边寻找下一个顶点
                for (MapObject e : g.edgesOf(cn)) {
                    if (!e.equals(recentEdge)) {
                        DsConnectNode neighbor;
                        if (g.getEdgeSource(e).equals(cn)) {
                            neighbor = g.getEdgeTarget(e);
                        } else {
                            neighbor = g.getEdgeSource(e);
                        }
                        // 如果顶点已经被遍历过，则不进行处理
                        if (visited.get(neighbor.getId()) != 0) {
                            continue;
                        }
                        // 找到环路
                        if (nodeStack.contains(neighbor)) {
                            // 深搜栈中相邻的两个节点
                            DsConnectNode cn1 = null;
                            DsConnectNode cn2;
                            // 找出环路重复搜到的节点在深搜栈中的位置
                            int repeatPos = 0;
                            Iterator<DsConnectNode> nodeStackIter = nodeStack.iterator();
                            while (nodeStackIter.hasNext()) {
                                cn1 = nodeStackIter.next();
                                if (cn1.equals(neighbor)) {
                                    break;
                                }
                                repeatPos++;
                            }
                            repeatPosStack.push(repeatPos);
                            // 存环路
                            int[] loop = new int[nodeStack.size() - repeatPos];
                            int i = 0;
                            int[] edgeInLoop = new int[edges.size()];
                            while (nodeStackIter.hasNext()) {
                                cn2 = nodeStackIter.next();
                                MapObject edgeOfLoop = g.getEdge(cn1, cn2);
                                for (int j = 0; j < edges.size(); j++) {
                                    if (edges.get(j).equals(edgeOfLoop)) {
                                        loop[i] = j;
                                        edgeInLoop[j] = 1;
                                        break;
                                    }
                                }
                                cn1 = cn2;
                                i++;
                            }
                            for (int j = 0; j < edges.size(); j++) {
                                if (edges.get(j).equals(e)) {
                                    loop[i] = j;
                                    edgeInLoop[j] = 1;
                                    break;
                                }
                            }
                            loops.add(loop);
                            edgeInLoopStack.push(edgeInLoop);
                            // 记录环路位置
//                            loopPosStack.add(loops.size() - 1);
                            // 找重复环路
                            Deque<int[]> edgeInNewLoopQueue = new ArrayDeque<>();
                            Deque<Integer> newRepeatPosQueue = new ArrayDeque<>();
                            Iterator<Integer> startPosStackIter = startPosStack.descendingIterator();
                            Iterator<int[]> edgeInLoopStackIter = edgeInLoopStack.descendingIterator();
                            Iterator<Integer> repeatPosStackIter = repeatPosStack.descendingIterator();
                            edgeInLoopStackIter.next();
                            repeatPosStackIter.next();
                            while (startPosStackIter.hasNext()) {
                                int startPos = startPosStackIter.next();
                                if (repeatPos >= startPos) {
                                    break;
                                }
                                // 存在重复环路
                                int[] edgeInPreLoop = edgeInLoopStackIter.next();
                                int[] edgeInNewLoop = new int[edges.size()];
                                int newLoopLen = 0;
                                for (int j = 0; j < edges.size(); j++) {
                                    edgeInNewLoop[j] = edgeInLoop[j] & edgeInPreLoop[j];
                                    if (edgeInNewLoop[j] == 1) {
                                        newLoopLen++;
                                    }
                                }
                                int[] newLoop = new int[newLoopLen];
                                newLoopLen = 0;
                                for (int j = 0; j < edges.size(); j++) {
                                    if (edgeInNewLoop[j] == 1) {
                                        newLoop[newLoopLen] = j;
                                        newLoopLen++;
                                    }
                                }
                                loops.add(newLoop);
                                edgeInNewLoopQueue.offer(edgeInNewLoop);

                                int newRepeatPos = repeatPosStackIter.next();
                                if (repeatPos < newRepeatPos) {
                                    newRepeatPos = repeatPos;
                                }
                                newRepeatPosQueue.offer(newRepeatPos);
                            }
                            while (!edgeInNewLoopQueue.isEmpty()) {
                                edgeInLoopStack.push(edgeInNewLoopQueue.poll());
                                repeatPosStack.push(newRepeatPosQueue.poll());
                            }

                            continue;
                        }

                    }
                }
                if (noPush) {
                    nodeStack.pop();
                }
            }
        }
    }

    /**
     * 搜索重要负荷的所有供电路径
     */
    public void buildImpPathes() {
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        //用于深度优先搜索的栈
        Deque<DsConnectNode> stack = new ArrayDeque<>();
        //保存栈中节点已经访问过的邻接节点
        Deque<List<String>> vstNodesOfCn = new ArrayDeque<>();
        impPaths = new ArrayList<>();
        impPathToEdge = new ArrayList<>();
        MapObject[] p;  //生成的新的路径
        int i, count = 0;
        boolean hasPush, visited, isSupply, samePath;
        impPathStart = new int[impLoads.length];
        for (String impLoad : impLoads) {
            impPathStart[count++] = impPaths.size();
            for (String supply : supplies) {
                DsConnectNode supplyCn = sys.getCns().get(supply);
                stack.push(supplyCn);   //先将电源节点压入栈内
                List<String> supplyVstNodes = new ArrayList<>();
                vstNodesOfCn.push(supplyVstNodes);
                while (!stack.isEmpty()) {
                    DsConnectNode cn = stack.peek();    //对栈顶元素深度搜索路径
                    List<String> vstNodes = vstNodesOfCn.peek();
                    hasPush = false;   //判断是否有节点入栈
                    for (MapObject e : g.edgesOf(cn)) {
                        DsConnectNode nextCn = g.getEdgeSource(e);
                        if (nextCn.equals(cn)) {
                            nextCn = g.getEdgeTarget(e);
                        }
                        visited = false;
                        for (String vstNode : vstNodes) {
                            if (nextCn.getId().equals(vstNode)) {
                                visited = true;
                                break;
                            }
                        }
                        if (stack.contains(nextCn)) {
                            visited = true;
                        }
                        if (!visited) {
                            //判断是否为电源节点
                            isSupply = false;
                            for (String scn : supplies) {
                                if (scn.equals(nextCn.getId())) {
                                    isSupply = true;
                                    break;
                                }
                            }
                            if (!isSupply) {
                                //找到路径
                                if (nextCn.getId().equals(impLoad)) {
                                    //将栈中的节点生成新的路径
                                    i = stack.size() - 2;
                                    Iterator<DsConnectNode> Iter = stack.iterator();
                                    p = new MapObject[stack.size()];
                                    p[p.length - 1] = g.getEdge(cn, nextCn);
                                    DsConnectNode temp1 = (DsConnectNode) Iter.next();
                                    while (Iter.hasNext()) {
                                        DsConnectNode temp2 = (DsConnectNode) Iter.next();
                                        p[i] = g.getEdge(temp2, temp1);
                                        temp1 = temp2;
                                        i--;
                                    }
                                    samePath = false;
                                    //判断impPaths中存在某个路径与新路径完全相同
                                    for (i = 0; i < impPaths.size(); i++) {
                                        if (Arrays.equals(p, impPaths.get(i))) {
                                            samePath = true;
                                            break;
                                        }
                                    }
                                    //找到新路径
                                    if (!samePath) {
                                        impPaths.add(p);    //增加一条路径
                                        int[] pathToEdge = new int[p.length];
                                        for (i = 0; i < p.length; i++) {
                                            for (int j = 0; j < edges.size(); j++) {
                                                if (edges.get(j).equals(p[i])) {
                                                    pathToEdge[i] = j;
                                                }
                                            }
                                        }
                                        impPathToEdge.add(pathToEdge);
                                    }
                                } else {
                                    //新节点入栈
                                    stack.push(nextCn);
                                    vstNodes.add(nextCn.getId());
                                    List<String> nextVstNodes = new ArrayList<>();
                                    vstNodesOfCn.push(nextVstNodes);
                                    hasPush = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (!hasPush) {
                        //没有新节点入栈，出栈
                        stack.pop();
                        vstNodesOfCn.pop();
                        Iterator<DsConnectNode> Iter = stack.iterator();
                    }
                }
            }
        }
//        System.out.printf("\nAll the pathes started from a specific supply\n");
//        printPathes(impPaths);
    }

    /**
     * 将图中除电源以外的节点和所有的边分别存入数组中
     */
    public void buildEdgesAndNodes() {
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        edges = new ArrayList<>();
        nodes = new ArrayList<>();
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
                //节点出栈时判断是否存储：还未存储过，则存入nodes中
                if (!nodes.contains(stack.peek()))
                    nodes.add((DsConnectNode) stack.peek());
                stack.pop();
            }
        }
        //删除电源节点
        for(String scn : supplies)
            for(int i = 0; i < nodes.size(); i++)
                if(nodes.get(i).getId().equals(scn)) {
                    nodes.remove(i);
                    break;
                }
//        for (DsConnectNode cn3 : nodes)
//            System.out.println(cn3.getId());
//        for(MapObject edge : edges)
//            System.out.printf("%s %s\n", g.getEdgeSource(edge).getId(), g.getEdgeTarget(edge).getId());
    }

    public void setImpLoads(String[] impLoads) {
        this.impLoads = impLoads;
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
