package zju.dsntp;

import ilog.concert.IloException;
import ilog.concert.IloNumExpr;
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

import static java.lang.Math.*;
import static zju.dsmodel.DsModelCons.KEY_CONNECTED_NODE;

public class IesPlan {

    // 能源系统
    DistriSys sys;
    // 图中所有的节点
    List<DsConnectNode> nodes;
    // 图中的电源节点
    List<DsConnectNode> supplyNodes;
    // 图中的负荷节点
    List<DsConnectNode> loadNodes;
    // 图中所有的边
    List<MapObject> edges;
    Map<String, ArrayList<Double>> loads;
    double minCost;
    public double cs = 10000;
    public double r = 0.06;
    public int A = 20;
    public double cp = 700;
    public double cle = 3.43;
    public double clh = 4.78;
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
    public double Se = 3200;
    public double Sh = 3000;
    public double Pem = 3200;
    public double Phm = 3000;

    public IesPlan(DistriSys sys) {
        this.sys = sys;
        init();
    }

    public void init() {
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        edges = new ArrayList<>(g.edgeSet().size());
        nodes = new ArrayList<>(g.vertexSet().size());
        supplyNodes = new ArrayList<>(supplies.length);
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
            } else {
                supplyNodes.add(node);
            }
        }
    }

    public void doOpt() {
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        String[] supplies = sys.getSupplyCns();
        double[] edgesLen = new double[edges.size()];
        for (int i = 0; i < edges.size(); i++) {
            edgesLen[i] = Double.parseDouble(edges.get(i).getProperty("LineLength"));
        }

        try {
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[supplies.length + 4 * edges.size()];
            // 状态变量上限
            double columnUpper[] = new double[columnLower.length];
            // 指明变量类型
            IloNumVarType[] xt  = new IloNumVarType[columnLower.length];
            // 约束方程系数
            double[][] coeff = new double[3 * nodes.size() + 6 * edges.size()][columnLower.length];
            //记录数组中存储元素的个数
            int coeffNum = 0;

            // 负荷的功率，按loadNodes中的顺序排列
            double[][] loadArray = new double[loadNodes.size()][3];
            for (int i = 0; i < loadNodes.size(); i++) {
                String nodeId = loadNodes.get(i).getId();
                if (loads.containsKey(nodeId)) {
                    loadArray[i][0] = this.loads.get(nodeId).get(0);
                    loadArray[i][1] = this.loads.get(nodeId).get(1);
                    loadArray[i][2] = this.loads.get(nodeId).get(2);
                }
            }

            // 能源站、支路状态变量上限为1，下限为0，都是整数
            for (int i = 0; i < supplies.length + edges.size(); i++) {
                columnLower[i] = 0;
                columnUpper[i] = 1;
                xt[i] = IloNumVarType.Bool;
            }
            // 支路流量变量上下限
            for (int i = 0; i < edges.size(); i++) {
                columnLower[supplies.length + edges.size() + i] = - Pem * 2;
                columnUpper[supplies.length + edges.size() + i] = Pem * 2;
                xt[supplies.length + edges.size() + i] = IloNumVarType.Float;
                columnLower[supplies.length + 2 * edges.size() + i] = - Pem * 2;
                columnUpper[supplies.length + 2 * edges.size() + i] = Pem * 2;
                xt[supplies.length + 2 * edges.size() + i] = IloNumVarType.Float;
                columnLower[supplies.length + 3 * edges.size() + i] = - Phm * 2;
                columnUpper[supplies.length + 3 * edges.size() + i] = Phm * 2;
                xt[supplies.length + 3 * edges.size() + i] = IloNumVarType.Float;
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(columnLower.length, columnLower, columnUpper, xt);
            // 目标函数
//            IloNumExpr obj = cplex.numExpr();
            double Cwp = Gp * Vf * tp * cep / np / 1e4;
            double coef1 = (1 + us1 + us2) * r * pow(1 + r, A) / (pow(1 + r, A) - 1) * 27.54 * 1e3 + Cwp;
            double[] objValue = new double[columnLower.length];
            for (int i = 0; i < supplies.length; i++) {
                objValue[i] = coef1;
            }
            coef1 = (1 + ul1 + ul2) * cp * r * pow(1 + r, A) / (pow(1 + r, A) - 1);
            double coef2 = tmax * cep * Rb / 3 / Ub / Ub / 10;
            double coef3 = 4 * PI * lamda * Tc * tp * cep / log((db + 2 * op + ot) / (db + 2 * op)) / cop / 1e4;
            for (int i = 0; i < edges.size(); i++) {
                objValue[supplies.length + i] += (coef1 + coef3) * edgesLen[i];
            }
//            for (int i = 0; i < edges.size(); i++) {
//                obj = cplex.sum(obj, cplex.prod(coef1 * edgesLen[i], x[supplies.length + i]));
////                IloNumExpr p2 = cplex.prod( x[supplies.length + edges.size() + i], x[supplies.length + edges.size() + i]);
////                IloNumExpr q2 = cplex.prod( x[supplies.length + 2 * edges.size() + i], x[supplies.length + 2 * edges.size() + i]);
////                obj = cplex.sum(obj, cplex.prod(coef2 * edgesLen[i], cplex.sum(p2, q2)));
//                obj = cplex.sum(obj, cplex.prod(coef3 * edgesLen[i], x[supplies.length + i]));
//            }
            cplex.addMinimize(cplex.scalProd(x, objValue));

            // 约束条件：潮流约束
            for (int i = 0; i < loadNodes.size(); i++) {
                for (MapObject e : g.edgesOf(loadNodes.get(i))) {
                    int j = edges.indexOf(e);
                    if (g.getEdgeTarget(e).equals(loadNodes.get(i))) {
                        coeff[coeffNum][supplies.length + edges.size() + j] = 1;
                        coeff[coeffNum + 1][supplies.length + 2 * edges.size() + j] = 1;
                        coeff[coeffNum + 2][supplies.length + 3 * edges.size() + j] = 1;
                    } else {
                        coeff[coeffNum][supplies.length + edges.size() + j] = - 1;
                        coeff[coeffNum + 1][supplies.length + 2 * edges.size() + j] = - 1;
                        coeff[coeffNum + 2][supplies.length + 3 * edges.size() + j] = - 1;
                    }
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), loadArray[i][0]);
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum + 1]), loadArray[i][1]);
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum + 2]), loadArray[i][2]);
                coeffNum += 3;
            }

            for (int i = 0; i < supplyNodes.size(); i++) {
                for (MapObject e : g.edgesOf(supplyNodes.get(i))) {
                    int j = edges.indexOf(e);
                    if (g.getEdgeTarget(e).equals(supplyNodes.get(i))) {
                        coeff[coeffNum][supplies.length + edges.size() + j] = 1;
                        coeff[coeffNum + 1][supplies.length + 2 * edges.size() + j] = 1;
                        coeff[coeffNum + 2][supplies.length + 3 * edges.size() + j] = 1;
                    } else {
                        coeff[coeffNum][supplies.length + edges.size() + j] = - 1;
                        coeff[coeffNum + 1][supplies.length + 2 * edges.size() + j] = - 1;
                        coeff[coeffNum + 2][supplies.length + 3 * edges.size() + j] = - 1;
                    }
                }
                coeff[coeffNum][i] = Se;
                coeff[coeffNum + 1][i] = 1800;
                coeff[coeffNum + 2][i] = Sh;
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum + 1]), 0);
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum + 2]), 0);
                coeffNum += 3;
            }

            // 约束条件：支路流量约束
            for(int i = 0; i < edges.size(); i++) {
                coeff[coeffNum][supplies.length + edges.size() + i] = 1;
                coeff[coeffNum][supplies.length + i] = - Pem * 2;
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
                coeff[coeffNum][supplies.length + edges.size() + i] = 1;
                coeff[coeffNum][supplies.length + i] = Pem * 2;
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;

                coeff[coeffNum][supplies.length + 2 * edges.size() + i] = 1;
                coeff[coeffNum][supplies.length + i] = - Pem * 2;
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
                coeff[coeffNum][supplies.length + 2 * edges.size() + i] = 1;
                coeff[coeffNum][supplies.length + i] = Pem * 2;
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;

                coeff[coeffNum][supplies.length + 3 * edges.size() + i] = 1;
                coeff[coeffNum][supplies.length + i] = - Phm * 2;
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
                coeff[coeffNum][supplies.length + 3 * edges.size() + i] = 1;
                coeff[coeffNum][supplies.length + i] = Phm * 2;
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
            }

            double[] result = new double[columnLower.length];
            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                minCost = cplex.getObjValue();
                result = cplex.getValues(x);
            }

            System.out.println("Supplies:");
            for (int i = 0; i < supplies.length; i++) {
                if (result[i] == 1) {
                    System.out.printf("%s\t", supplies[i]);
                    double Ps = 0;
                    for (MapObject e : g.edgesOf(supplyNodes.get(i))) {
                        int j = edges.indexOf(e);
                        if (g.getEdgeTarget(e).equals(loadNodes.get(i))) {
                            Ps -= result[supplies.length + edges.size() + j];
                            Ps -= result[supplies.length + 3 * edges.size() + j];
                        } else {
                            Ps += result[supplies.length + edges.size() + j];
                            Ps += result[supplies.length + 3 * edges.size() + j];
                        }
                    }
                    minCost += 2.118 * pow(Ps * cs / 1e7, 0.9198) * 1e3;
                }
            }
            System.out.println("\nEdges:");
            coef1 = r * pow(1 + r, A) / (pow(1 + r, A) - 1);
            for (int i = 0; i < edges.size(); i++) {
                if (result[supplies.length + i] == 1) {
                    System.out.println(edges.get(i).getProperty(KEY_CONNECTED_NODE));
                    minCost += (cle * abs(result[supplies.length + edges.size() + i]) +
                            clh * abs(result[supplies.length + 3 * edges.size() + i])) * edgesLen[i] * coef1 / 1e4;
                }
            }
            System.out.println("\nTotal cost:\t" + minCost);

            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    public void setLoads(Map<String, ArrayList<Double>> loads) {
        this.loads = loads;
    }
}
