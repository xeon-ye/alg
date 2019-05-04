package zju.dsntp;

import ilog.concert.IloException;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;
import org.jgrapht.UndirectedGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsConnectNode;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class IesPlan {

    // 能源系统
    DistriSys sys;
    // 图中所有的节点
    List<DsConnectNode> nodes;
    // 图中的负荷节点
    List<DsConnectNode> loadNodes;
    // 图中所有的边
    List<MapObject> edges;
    Map<String, ArrayList<Double>> loads;
    public double cs = 10000;
    public double r = 0.06;
    public int A = 20;
    public double cp = 16.8;
    public double cle = 2.94;
    public double clh = 3.42;
    public double us1 = 0.01;
    public double us2 = 0.015;
    public double ul1 = 0.03;
    public double ul2 = 0.015;
    public double tmax = 3000;
    public double cep = 0.5;
    public double Ub = 10;
    public double Rb = 0.223;
    public double Vf = 2;
    public double Gp = 150;
    public double tp = 3360;
    public double np = 0.6;
    public double lamda = 0.06;
    public double Tc = 10;
    public double db = 0.5;
    public double op = 0.05;
    public double ot = 0.009;
    public double cop = 4.2;
    public double Se = 3.5;
    public double Sh = 3;
    public double Pem = 3.5;

    public IesPlan(DistriSys sys) {
        this.sys = sys;
    }

    public void init() {
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
    }

    public void doOpt() {
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        try {
            // 开始构造线性规划模型
            double objValue[] = new double[2 * edges.size()];
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[objValue.length];
            // 状态变量上限
            double columnUpper[] = new double[objValue.length];
            // 指明变量类型
            IloNumVarType[] xt  = new IloNumVarType[objValue.length];
            // 约束方程系数
            double[][] coeff = new double[1 + nodes.size() + 2 * edges.size()][objValue.length];
            // 所有支路的通断状态
            int[] edgesStatues = new int[edges.size()];
            //记录数组中存储元素的个数
            int coeffNum = 0;

            // 负荷的功率，按loadNodes中的顺序排列
            double[][] loadArray = new double[loadNodes.size()][3];
            for (int i = 0; i < loadNodes.size(); i++) {
                loadArray[i][0] = this.loads.get(loadNodes.get(i).getId()).get(0);
                loadArray[i][1] = this.loads.get(loadNodes.get(i).getId()).get(1);
                loadArray[i][2] = this.loads.get(loadNodes.get(i).getId()).get(2);
            }

            // 支路状态变量上限为1，下限为0，都是整数
            for (int i = 0; i < edges.size(); i++) {
                columnLower[i] = 0;
                columnUpper[i] = 1;
                xt[i] = IloNumVarType.Int;
            }
            // 支路流量变量上下限
            for (int i = 0; i < edges.size(); i++) {
                columnLower[edges.size() + i] = - Pem;
                columnUpper[edges.size() + i] = Pem;
                xt[edges.size() + i] = IloNumVarType.Float;
            }

            // 设置状态变量的系数值
            for (int i = 0; i < edges.size(); i++) {
                if (edgesStatues[i] == 0) {
                    objValue[i] = 1;
                } else {
                    objValue[i] = - 1;
                }
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(2 * edges.size(), columnLower, columnUpper, xt);
            // 目标函数
            cplex.addMinimize(cplex.scalProd(x, objValue));


            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    public void setLoads(Map<String, ArrayList<Double>> loads) {
        this.loads = loads;
    }
}
