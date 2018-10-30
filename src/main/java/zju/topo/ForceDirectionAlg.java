package zju.topo;

import org.jgrapht.UndirectedGraph;
import org.jgrapht.traverse.DepthFirstIterator;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsConnectNode;

import javax.swing.*;
import java.io.InputStream;
import java.util.*;

import static zju.dsmodel.DsModelCons.KEY_CONNECTED_NODE;
import static zju.dsmodel.IeeeDsInHand.createDs;

public class ForceDirectionAlg {
    private DistriSys distriSys;
    private Map<String, DsConnectNode> cns;

    private UndirectedGraph<DsConnectNode, MapObject> origGraph;
    private ArrayList<TopoNode> topoNodesList;
    private Map<DsConnectNode, TopoNode> nodeMap;

    private List<DsConnectNode> fixedNodesList;

    //引力系数
    private double K = 0.01;
    //斥力系数
    private double G = 1000;
    //自然长度
    private double naturalLength = 10;

    //图幅
    private int graphWidth = 500;
    private int graphHeight = 500;

    /**
     * 初始化
     */
    public void intial() {
        //生成系统
        InputStream ieeeFile = this.getClass().getResourceAsStream("/topo/case1/graphtest.txt");
        distriSys = createDs(ieeeFile, "S1", 100);
        cns = distriSys.getCns();

        origGraph = distriSys.getOrigGraph();

        topoNodesList = new ArrayList<>();
        nodeMap = new HashMap<>();

        fixedNodesList = new ArrayList<>();



        /*
        初始化各点
         */
        for (DsConnectNode node : distriSys.getCns().values()) {
            TopoNode topoNode = new TopoNode();
            topoNode.setCn(node);
            //名字
            topoNode.setName(node.getId());
            //质量 todo:质量
            topoNode.setMass(1);
            //出度
            topoNode.setDegree(origGraph.degreeOf(node));
            //坐标初始化
            topoNode.setxCoordinate(getRandomCoordinate());
            topoNode.setyCoordinate(getRandomCoordinate());
            //受力初始化
            topoNode.setxForce(0);
            topoNode.setyForce(0);

            topoNodesList.add(topoNode);
            nodeMap.put(node, topoNode);
        }

        /*
        固定点集合
         */
        //起始点
        DsConnectNode startCn = cns.get("S1");
        Iterator<DsConnectNode> iterator = new DepthFirstIterator<>(origGraph,startCn);

        List<Object[]> pathes = new ArrayList<>();
        Stack<DsConnectNode> cnStack = new Stack<>();

        cnStack.push(startCn);
        iterator.next();
        while (iterator.hasNext()){
            DsConnectNode node = iterator.next();
            DsConnectNode lastNode = cnStack.peek();

            while (!origGraph.containsEdge(node, lastNode)) {
                cnStack.pop();
                lastNode = cnStack.peek();
            }

            cnStack.push(node);
            pathes.add(cnStack.toArray());
        }

        Object[] longestPath = new Object[0];
        int max = 0;
        for(Object[] path : pathes){
            if(path.length>max){
                max = path.length;
                longestPath = path;
            }
        }

        for(Object obj : longestPath){
            fixedNodesList.add((DsConnectNode) obj);
        }



        graphWidth = (fixedNodesList.size() - 1)*50;


        /*
        固定点坐标控制
         */
        for(int i =0;i < fixedNodesList.size();i++){
            DsConnectNode node = fixedNodesList.get(i);
            TopoNode topoNode = nodeMap.get(node);
            topoNode.setxCoordinate(i*50-graphWidth/2);
            topoNode.setyCoordinate(0);
        }

    }

    /**
     * 布局
     */
    public void layout() {
        for (int i = 0; i < 1000; i++) {
            calculateForce();
            updateCoordinate();
        }

        for(TopoNode node:topoNodesList){
            node.setxCoordinate(node.getxCoordinate()+graphWidth/2+25);
            node.setyCoordinate(node.getyCoordinate()+graphHeight/2+25);
        }

        /*
        生成图像
         */
        TopoGraph frame = new TopoGraph(topoNodesList, origGraph);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(graphWidth+50+25, graphHeight+50);
        frame.setVisible(true);
    }

    public void calculateForce() {
        //初始化受力
        for (TopoNode node : topoNodesList) {
            node.setxForce(0);
            node.setyForce(0);
        }
        //计算引力
        for (MapObject edge : origGraph.edgeSet()) {
            String[] s = edge.getProperty(KEY_CONNECTED_NODE).split(";");
            TopoNode node1 = nodeMap.get(cns.get(s[0]));
            TopoNode node2 = nodeMap.get(cns.get(s[1]));

            double distance = Math.pow(node1.getxCoordinate() - node2.getxCoordinate(), 2) + Math.pow(node1.getyCoordinate() - node2.getyCoordinate(), 2);
            distance = Math.sqrt(distance);

            double attraction;
            double xAttraction;
            double yAttraction;

            if (distance > naturalLength) {
                attraction = K * distance;

                xAttraction = attraction * (node2.getxCoordinate() - node1.getxCoordinate()) / distance;
                yAttraction = attraction * (node2.getyCoordinate() - node1.getyCoordinate()) / distance;
            } else {
                xAttraction = 0;
                yAttraction = 0;
            }

            node1.setxForce(node1.getxForce() + xAttraction);
            node1.setyForce(node1.getyForce() + yAttraction);
            node2.setxForce(node2.getxForce() - xAttraction);
            node2.setyForce(node2.getyForce() - yAttraction);
        }
        //计算斥力
        for (int i = 0; i < topoNodesList.size(); i++) {
            for (int j = i + 1; j < topoNodesList.size(); j++) {
                TopoNode node1 = topoNodesList.get(i);
                TopoNode node2 = topoNodesList.get(j);

                double distance = Math.pow(node1.getxCoordinate() - node2.getxCoordinate(), 2) + Math.pow(node1.getyCoordinate() - node2.getyCoordinate(), 2);
                distance = Math.sqrt(distance);

                double repulsion;

                double xRepulsion;
                double yRepulsion;

                if (distance != 0) {
                    repulsion = G * node1.getMass() * node2.getMass() / Math.pow(distance, 2);

                    xRepulsion = repulsion * (node1.getxCoordinate() - node2.getxCoordinate()) / distance;
                    yRepulsion = repulsion * (node1.getyCoordinate() - node2.getyCoordinate()) / distance;
                } else {
                    xRepulsion = 100;
                    yRepulsion = 100;
                }

                node1.setxForce(node1.getxForce() + xRepulsion);
                node1.setyForce(node1.getyForce() + yRepulsion);
                node2.setxForce(node2.getxForce() - xRepulsion);
                node2.setyForce(node2.getyForce() - yRepulsion);
            }
        }
    }

    public void updateCoordinate() {
        for (TopoNode node : topoNodesList) {
            //固定点不变坐标
            if (fixedNodesList.contains(node.getCn())) {
                continue;
            }

            double x = node.getxCoordinate();
            double y = node.getyCoordinate();

            double xChange = node.getxForce() / node.getMass();
            double yChange = node.getyForce() / node.getMass();

            node.setxCoordinate(x + xChange);
            node.setyCoordinate(y + yChange);
        }

        double xMax = 1;
        double yMax = 1;
        for(TopoNode node: topoNodesList){
            double x = node.getxCoordinate();
            double y = node.getyCoordinate();

            if(Math.abs(x)>xMax){
                xMax = Math.abs(x);
            }
            if(Math.abs(y)>yMax){
                yMax = Math.abs(y);
            }
        }

        for(TopoNode node:topoNodesList){
            //固定点不变坐标
            if (fixedNodesList.contains(node.getCn())) {
                continue;
            }

            node.setxCoordinate(node.getxCoordinate()*0.5*graphWidth/xMax);
            node.setyCoordinate(node.getyCoordinate()*0.5*graphWidth/yMax);
        }

    }



    private double getRandomCoordinate() {
        double coordinate;
        Random random = new Random();
        coordinate = 20 * (random.nextDouble() - 0.5);
        return coordinate;
    }

    public static void main(String[] args) {
        ForceDirectionAlg alg = new ForceDirectionAlg();
        alg.intial();
        alg.layout();
    }
}
