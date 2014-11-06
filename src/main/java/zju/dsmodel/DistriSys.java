package zju.dsmodel;

import org.apache.log4j.Logger;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.alg.ConnectivityInspector;
import org.jgrapht.graph.SimpleGraph;
import zju.devmodel.MapObject;
import zju.util.JOFileUtil;

import java.io.Serializable;
import java.util.*;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-10-30
 */
public class DistriSys implements DsModelCons, Serializable {
    private static Logger log = Logger.getLogger(DistriSys.class);

    private static int virtual_cn_count = 1;
    //配电网络中所有的物理设备
    private DsDevices devices;

    private Map<String, DsConnectNode> cns;

    private UndirectedGraph<DsConnectNode, MapObject> origGraph;

    private Map<DsConnectNode, DsTopoNode> cnToTn;

    private List<DsTopoIsland> topoIslands;

    private String[] supplyCns;

    private Double[] supplyCnBaseKv;

    private DsTopoIsland[] activeIslands;

    private boolean isBalanced = false;

    private CalModelBuilder calModelBuilder = new CalModelBuilder();

    private void findActiveIslands() {
        //先设置每个TN的类型
        for (DsTopoNode tn : cnToTn.values()) {
            tn.setType(DsTopoNode.TYPE_LINK);
            for (DsConnectNode cn : tn.getConnectivityNodes()) {
                for (MapObject obj : cn.getConnectedObjs()) {
                    if (RESOURCE_SPOT_LOAD.equals(obj.getProperty(KEY_RESOURCE_TYPE)) ||
                            RESOURCE_DIS_LOAD.equals(obj.getProperty(KEY_RESOURCE_TYPE)) ||
                            RESOURCE_SHUNT_CAPACITORS.equals(obj.getProperty(KEY_RESOURCE_TYPE))) {
                        tn.setType(DsTopoNode.TYPE_PQ);
                        break;
                    }
                }
            }
            for (DsConnectNode cn : tn.getConnectivityNodes()) {
                for (MapObject obj : cn.getConnectedObjs()) {
                    if (RESOURCE_DG.equals(obj.getProperty(KEY_RESOURCE_TYPE))) {
                        tn.setType(DsTopoNode.TYPE_DG);
                        break;
                    }
                }
            }
        }
        List<DsTopoNode> supplyTns = new ArrayList<DsTopoNode>(supplyCns.length);
        for (String cnId : supplyCns) {
            DsTopoNode tn = getCnToTn().get(getCns().get(cnId));
            if (!supplyTns.contains(tn)) {
                supplyTns.add(tn);
                tn.setType(DsTopoNode.TYPE_SUB);
            }
        }

        //判断电气岛中是否含有SUBSTATION或DG类型的TN，如果有则认为该岛是有源的
        for (DsTopoIsland island : getTopoIslands())
            island.setActive(false);
        for (DsTopoIsland island : getTopoIslands()) {
            for (DsTopoNode tn : island.getGraph().vertexSet())
                if (tn.getType() == DsTopoNode.TYPE_SUB) {
                    island.setActive(true);
                    break;
                } else if (tn.getType() == DsTopoNode.TYPE_DG) {
                    island.setActive(true);
                    break;
                }
        }

        //将所有有源电气岛放置到数组中
        int activeIslandCount = 0;
        for (DsTopoIsland island : getTopoIslands()) {
            if (island.isActive())
                activeIslandCount++;
        }
        activeIslands = new DsTopoIsland[activeIslandCount];
        activeIslandCount = 0;
        for (DsTopoIsland island : getTopoIslands()) {
            if (island.isActive()) {
                activeIslands[activeIslandCount] = island;
                activeIslandCount++;
            }
        }
        //将每个有源电气岛中电源节点找出来
        for (DsTopoIsland activeIsland : activeIslands) {
            int supplyBusCount = 0;
            Set<DsTopoNode> tns = activeIsland.getGraph().vertexSet();
            for (DsTopoNode tn : tns) {
                if (tn.getType() == DsTopoNode.TYPE_SUB || tn.getType() == DsTopoNode.TYPE_DG)
                    supplyBusCount++;
            }
            DsTopoNode[] supplies = new DsTopoNode[supplyBusCount];
            supplyBusCount = 0;
            for (DsTopoNode tn : tns) {
                if (tn.getType() == DsTopoNode.TYPE_SUB || tn.getType() == DsTopoNode.TYPE_DG)
                    supplies[supplyBusCount++] = tn;
            }
            activeIsland.setSupplyTns(supplies);
            activeIsland.setPerUnitSys(isPerUnitSys());
            activeIsland.setBalanced(isBalanced());
            //判断电气岛是否是辐射状的
            if (tns.size() - activeIsland.getGraph().edgeSet().size() == 1)
                activeIsland.setRadical(true);
            else
                activeIsland.setRadical(false);
            activeIsland.initialIsland();
        }
    }

    public void buildOrigTopo(DsDevices devs) {
        this.devices = devs;
        cns = new HashMap<String, DsConnectNode>();
        origGraph = new SimpleGraph<DsConnectNode, MapObject>(MapObject.class);
        dealBranch(devs.getFeeders());
        dealBranch(devs.getSwitches());
        dealBranch(devs.getTransformers());

        //the order of following should be fixed or data in IeeeDsInHand would changed
        fillNode(devs.getSpotLoads());
        fillNode(devs.getDistributedLoads());
        fillNode(devs.getShuntCapacitors());
        fillNode(devs.getRegulators());
        fillNode(devs.getTransformers());
        fillNode(devs.getFeeders());
        fillNode(devs.getSwitches());
        fillNode(devs.getDispersedGens());

        //deal regulators
        //dealRegurators();//todo:
    }

    public void buildDynamicTopo() {
        cnToTn = new HashMap<DsConnectNode, DsTopoNode>(cns.size());
        //先把开关打开的边去掉
        for (MapObject aSwtich : devices.getSwitches()) {
            if (aSwtich.getProperty(KEY_SWITCH_STATUS).equals(SWITCH_OFF)) {
                String[] s = aSwtich.getProperty(KEY_CONNECTED_NODE).split(";");
                DsConnectNode cn1 = cns.get(s[0]);
                DsConnectNode cn2 = cns.get(s[1]);
                origGraph.removeEdge(cn1, cn2);
            }
        }
        //分析得到联通的子图
        ConnectivityInspector<DsConnectNode, MapObject> inspector = new ConnectivityInspector<DsConnectNode, MapObject>(origGraph);
        List<Set<DsConnectNode>> subgraphs = inspector.connectedSets();
        //把开关打开的边恢复到原图中
        for (MapObject aSwtich : devices.getSwitches()) {
            if (aSwtich.getProperty(KEY_SWITCH_STATUS).equals(SWITCH_OFF)) {
                String[] s = aSwtich.getProperty(KEY_CONNECTED_NODE).split(";");
                DsConnectNode cn1 = cns.get(s[0]);
                DsConnectNode cn2 = cns.get(s[1]);
                origGraph.addEdge(cn1, cn2, aSwtich);
            }
        }

        topoIslands = new ArrayList<DsTopoIsland>(subgraphs.size());

        for (Set<DsConnectNode> aSet : subgraphs) {
            UndirectedGraph<DsTopoNode, MapObject> g = new SimpleGraph<DsTopoNode, MapObject>(MapObject.class);
            //填充子图的顶点
            for (DsConnectNode node : aSet) {
                if (cnToTn.containsKey(node))
                    continue;
                createTopologicalNode(null, node, g);
            }
            //填充子图的边
            for (MapObject obj : origGraph.edgeSet()) {
                if (!obj.getProperty(KEY_RESOURCE_TYPE).equals(RESOURCE_SWITCH)) {
                    String[] s = obj.getProperty(KEY_CONNECTED_NODE).split(";");
                    DsConnectNode cn1 = cns.get(s[0]);
                    DsConnectNode cn2 = cns.get(s[1]);
                    if (!g.containsVertex(cnToTn.get(cn1)))//todo: may not efficient
                        continue;
                    g.addEdge(cnToTn.get(cn1), cnToTn.get(cn2), obj);
                }
            }
            DsTopoIsland topoIsland = new DsTopoIsland();
            topoIslands.add(topoIsland);
            topoIsland.setGraph(g);
        }

        findActiveIslands();
        //确保所有的ConnectivityNode的电压等级都已经设置好，接着设置TN的基准电压
        for (DsConnectNode cn : cns.values())
            cnToTn.get(cn).setBaseKv(cn.getBaseKv());
    }

    public void fillCnBaseKv() {
        DsConnectNode startCn = null;
        for (int i = 0; i < supplyCns.length; i++) {
            DsConnectNode cn = cns.get(supplyCns[i]);
            cn.setBaseKv(supplyCnBaseKv[i]);
            if (supplyCnBaseKv[i] != null)
                startCn = cn;
        }
        fillCnBaseKv(startCn);
        //对所有设备设置基值电压
        for (DsConnectNode cn : cns.values()) {
            for (MapObject obj : cn.getConnectedObjs())
                if (!RESOURCE_TRANSFORMER.equals(obj.getProperty(KEY_RESOURCE_TYPE))) {
                    obj.setProperty(KEY_KV_BASE, cn.getBaseKv().toString());
                }
        }
    }

    /**
     * 该方法通过变压器的电压等级系统，便利整个网络设定其他节点的基准电压
     *
     * @param startCn 已知电压等级的节点，如果该节点为null，
     *                则如果系统存在变压器可以通过变压器的电压等级确定整个网络所有节点的电压等级
     * @return success or failed
     */
    private boolean fillCnBaseKv(DsConnectNode startCn) {
        DsConnectNode cn1 = null, cn2;
        for (MapObject obj : devices.getTransformers()) {
            Double highKv = Double.parseDouble(obj.getProperty(KEY_KV_HIGH));
            Double lowKv = Double.parseDouble(obj.getProperty(KEY_KV_LOW));
            highKv = highKv / sqrt3;
            lowKv = lowKv / sqrt3;
            String[] nodes = obj.getProperty(KEY_CONNECTED_NODE).split(";");
            cn1 = getCns().get(nodes[0]);
            cn2 = getCns().get(nodes[1]);
            cn1.setBaseKv(highKv);
            cn2.setBaseKv(lowKv);
        }
        if (startCn == null && cn1 == null)
            return false;
        traversalBfs(startCn != null ? startCn : cn1);
        return true;
    }

    private void traversalBfs(DsConnectNode first) {
        LinkedList<DsConnectNode> queue = new LinkedList<DsConnectNode>();
        queue.addFirst(first);
        Map<DsConnectNode, Boolean> isDealed = new HashMap<DsConnectNode, Boolean>(cns.size());
        while (!queue.isEmpty()) {
            DsConnectNode last = queue.pollLast();
            isDealed.put(last, true);
            for (MapObject obj : origGraph.edgesOf(last)) {
                DsConnectNode src = origGraph.getEdgeSource(obj);
                DsConnectNode tar = origGraph.getEdgeTarget(obj);
                DsConnectNode node = (src == last ? tar : src);
                if (node.getBaseKv() == null)
                    node.setBaseKv(last.getBaseKv());
                if (!isDealed.containsKey(node)) {
                    queue.addLast(node);
                }
            }
        }
    }

    private void dealRegurators() {
        for (MapObject regulator : devices.getRegulators()) {
            String locationCnId = regulator.getProperty(KEY_LOCATION);
            String[] s = regulator.getProperty(KEY_CONNECTED_NODE).split(";");
            String anotherNode = s[0].equals(locationCnId) ? s[1] : s[0];
            DsConnectNode locationCn = cns.get(locationCnId);
            DsConnectNode anotherCn = cns.get(anotherNode);

            //新建一个虚拟节点，该虚拟节点将和LocationCn通过Regurator相连，而LocationCn和AnotherCn之间的边将
            //移到该虚拟节点和AnotherCn之间
            DsConnectNode virtualCn = new DsConnectNode("VirtualCN_" + virtual_cn_count++);
            virtualCn.setConnectedObjs(new ArrayList<MapObject>());
            anotherCn.getConnectedObjs().remove(regulator);//remove regulator from another node
            for (MapObject object : anotherCn.getConnectedObjs()) {
                String[] str = object.getProperty(KEY_CONNECTED_NODE).split(";");
                if (str.length > 1 && (str[0].equals(locationCnId) || str[1].equals(locationCnId)))
                    virtualCn.getConnectedObjs().add(object);//add all connected object on another node to virtual node
            }

            for (MapObject objOnVirtual : virtualCn.getConnectedObjs()) {
                locationCn.getConnectedObjs().remove(objOnVirtual);
                objOnVirtual.setProperty(KEY_CONNECTED_NODE, virtualCn.getId() + ";" + anotherCn.getId());
            }

            //在拓扑上先删除一条边
            MapObject branch = origGraph.removeEdge(locationCn, anotherCn);
            //在拓扑上增加一个节点，并增加两条边
            origGraph.addVertex(virtualCn);
            origGraph.addEdge(virtualCn, anotherCn, branch);
            origGraph.addEdge(locationCn, virtualCn, regulator);
            //todo:下面这行改变了Regulator的属性，注意
            regulator.setProperty(KEY_CONNECTED_NODE, locationCnId + ";" + virtualCn.getId());
            virtualCn.getConnectedObjs().add(regulator);
            cns.put(virtualCn.getId(), virtualCn);
        }
    }

    /**
     * create logical topoNode
     *
     * @param topoNode a logical topoNode
     * @param node     connectivity node in cim
     * @param g        topological graph
     */
    private void createTopologicalNode(DsTopoNode topoNode, DsConnectNode node, UndirectedGraph<DsTopoNode, MapObject> g) {
        if (topoNode == null) {
            topoNode = new DsTopoNode();
            cnToTn.put(node, topoNode);
            g.addVertex(topoNode);
            topoNode.setConnectivityNodes(new ArrayList<DsConnectNode>());
        }
        topoNode.getConnectivityNodes().add(node);
        for (MapObject obj : node.getConnectedObjs()) {
            String resourceType = obj.getProperty(KEY_RESOURCE_TYPE);
            if (resourceType.equals(RESOURCE_SWITCH)) {
                String[] s = obj.getProperty(KEY_CONNECTED_NODE).split(";");
                DsConnectNode cn1 = cns.get(s[0]);
                DsConnectNode cn2 = cns.get(s[1]);
                DsConnectNode crossingCn = cn1 == node ? cn2 : cn1;
                if (cnToTn.containsKey(crossingCn))
                    continue;
                String switchStatus = obj.getProperty(KEY_SWITCH_STATUS);
                if (switchStatus == null) {
                    log.warn("Unknown status switch: " + obj.toString());
                    return;
                }
                if (!switchStatus.equals(SWITCH_ON))
                    continue;
                cnToTn.put(crossingCn, topoNode);
                createTopologicalNode(topoNode, crossingCn, g);
            }
        }
    }

    private void dealBranch(List<MapObject> objs) {
        for (MapObject obj : objs) {
            String[] s = obj.getProperty(KEY_CONNECTED_NODE).split(";");
            for (String str : s) {
                if (!cns.containsKey(str)) {
                    DsConnectNode cn = new DsConnectNode(str);
                    cns.put(str, cn);
                    origGraph.addVertex(cn);
                }
            }
            origGraph.addEdge(cns.get(s[0]), cns.get(s[1]), obj);
        }
    }

    private void fillNode(List<MapObject> objs) {
        for (MapObject obj : objs) {
            String[] s = obj.getProperty(KEY_CONNECTED_NODE).split(";");
            for (String str : s)
                if (cns.containsKey(str)) {
                    DsConnectNode cn = cns.get(str);
                    if (cn.getConnectedObjs() == null)
                        cn.setConnectedObjs(new ArrayList<MapObject>());
                    cn.getConnectedObjs().add(obj);
                } else
                    log.warn("Not node is found at cn maps: " + str);
        }
    }

    public String[] getSupplyCns() {
        return supplyCns;
    }

    public void setSupplyCns(String[] supplyCns) {
        this.supplyCns = supplyCns;
    }

    public Double[] getSupplyCnBaseKv() {
        return supplyCnBaseKv;
    }

    public void setSupplyCnBaseKv(Double[] supplyCnBaseKv) {
        this.supplyCnBaseKv = supplyCnBaseKv;
    }

    public DsTopoIsland[] getActiveIslands() {
        return activeIslands;
    }

    public List<DsTopoIsland> getTopoIslands() {
        return topoIslands;
    }

    public Map<String, DsConnectNode> getCns() {
        return cns;
    }

    public Map<DsConnectNode, DsTopoNode> getCnToTn() {
        return cnToTn;
    }

    public DsDevices getDevices() {
        return devices;
    }

    public static int getVirtual_cn_count() {
        return virtual_cn_count;
    }

    public Double getBaseKva() {
        return calModelBuilder.getBaseKva();
    }

    public boolean isPerUnitSys() {
        return calModelBuilder.isPerUnitSys();
    }

    public void setBaseKva(Double baseKva) {
        calModelBuilder.setBaseKva(baseKva);
    }

    public void setPerUnitSys(boolean perUnitSys) {
        calModelBuilder.setPerUnitSys(perUnitSys);
    }

    public boolean isBalanced() {
        return isBalanced;
    }

    public void setBalanced(boolean balanced) {
        isBalanced = balanced;
    }

    @Override
    public DistriSys clone() {
        return (DistriSys) JOFileUtil.cloneObj(this);
    }

    public void setFeederConf(FeederConfMgr feederConf) {
        calModelBuilder.setFeederConf(feederConf);
    }

    public void createCalDevModel() {
        if (activeIslands == null)
            return;
        for (DsTopoIsland island : activeIslands)
            calModelBuilder.createCalDevModel(island);
    }
}
