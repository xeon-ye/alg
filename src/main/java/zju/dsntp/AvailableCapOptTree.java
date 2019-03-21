package zju.dsntp;

import ilog.concert.IloException;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jgrapht.UndirectedGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsConnectNode;

import java.util.HashMap;
import java.util.Map;

import static java.lang.Math.abs;
import static zju.dsmodel.DsModelCons.*;

/**
 * Created by xuchengsi on 2018/4/28.
 */
public class AvailableCapOptTree extends BranchBasedModel {

    private static Logger log = LogManager.getLogger(AvailableCapOptTree.class);

    int[] errorFeeder;

    String[] errorSupply;
    String[] errorSwitch;
    // 电源容量
    Map<String, Double> supplyCap;
    // 馈线容量
    Map<String, Double> feederCap;
    // 负荷量
    Map<String, Double> loads;
    double feederCapacityConst;
    // 开关最少次数计算结果
    int minSwitch;
    String[] switchChanged;
    // 优化结果，最大负荷量
    double maxLoad;
    // 优化后的支路流量
    double[] flows;
    // 所有负荷可装容量计算结果
    Map<String, Double> maxLoadResult;
    // 计算开关次数最少的所有结果
    LoadTransferOptResult optResult;
    // 馈线带的最大容量
    Map<String, Double> maxFeederLoad;
    // circuit的可装容量
    Map<String, Double> maxCircuitLoad;
    // 计算是否收敛的状态
    int status;

    public AvailableCapOptTree(DistriSys sys) {
        super(sys);
    }

    /**
     * 优化最小开关次数
     */
    public void doOpt() {
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        int i, j;
        double[] feederCapacity = new double[edges.size()];
        flows = new double[edges.size()];
        boolean isErrorSwitch;
        for (i = 0; i < edges.size(); i++) {
            isErrorSwitch = false;
            if(errorSwitch != null) {
                for (j = 0; j < errorSwitch.length; j++) {
                    if (edges.get(i).getId().equals(errorSwitch[j])) {
                        isErrorSwitch = true;
                        edges.get(i).setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                        break;
                    }
                }
            }
            if (isErrorSwitch)
                feederCapacity[i] = 0;
            else
                feederCapacity[i] = feederCapacityConst;
        }
        // 电源容量
        String[] supplies = sys.getSupplyCns();
        double[] supplyCapacity = new double[supplies.length];
        boolean isErrorSupply;
        for (i = 0; i < supplies.length; i++) {
            isErrorSupply = false;
            for (j = 0; j < errorSupply.length; j++) {
                if (supplies[i].equals(errorSupply[j])) {
                    isErrorSupply = true;
                    for (MapObject e : g.edgesOf(sys.getCns().get(supplies[i]))) {
                        e.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                        for (int k = 0; k < edges.size(); k++) {
                            if (edges.get(k).equals(e)) {
                                feederCapacity[k] = 0;
                                break;
                            }
                        }
                    }
                    break;
                }
            }
            if (isErrorSupply)
                supplyCapacity[i] = 0;
            else
                supplyCapacity[i] = this.supplyCap.get(supplies[i]);
        }

//        int loopsEdgeNum = 0;
//        for (int[] loop : loops) {
//            loopsEdgeNum += loop.length;
//        }
        try {
            // 开始构造线性规划模型
            double objValue[] = new double[2 * edges.size() + 2 * edges.size()];
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[objValue.length];
            // 状态变量上限
            double columnUpper[] = new double[objValue.length];
            // 指明变量类型
            IloNumVarType[] xt  = new IloNumVarType[objValue.length];
            // 约束方程系数
            double[][] coeff = new double[edges.size() + nodes.size() + nodes.size() + 2 * edges.size()][objValue.length];
            // 所有支路的通断状态
            int[] edgesStatues = new int[edges.size()];
            //记录数组中存储元素的个数
            int coeffNum = 0;

            // 负荷的功率，按loadNodes中的顺序排列
            double[] loadArray = new double[loadNodes.size()];
            for (i = 0; i < loadNodes.size(); i++) {
                loadArray[i] = this.loads.get(loadNodes.get(i).getId());
            }

            // 支路状态变量上限为1，下限为0，都是整数
            for (i = 0; i < edges.size(); i++) {
                columnLower[i] = 0;
                columnUpper[i] = 1;
                xt[i] = IloNumVarType.Int;
            }
            // 支路流量变量上下限
            for (i = 0; i < edges.size(); i++) {
                columnLower[edges.size() + i] = -feederCapacity[i];
                columnUpper[edges.size() + i] = feederCapacity[i];
                xt[edges.size() + i] = IloNumVarType.Float;
            }
            // 树层次关系变量上限为1，下限为0，都是整数
            for (i = 0; i < 2 * edges.size(); i++) {
                columnLower[2 * edges.size() + i] = 0;
                columnUpper[2 * edges.size() + i] = 1;
                xt[2 * edges.size() + i] = IloNumVarType.Int;
            }

            // 得到当前所有支路的通断
            for (i = 0; i < edges.size(); i++) {
                if (edges.get(i).getProperty(KEY_SWITCH_STATUS).equals(SWITCH_ON)) {
                    edgesStatues[i] = 1;
                }
            }
            // 设置状态变量的系数值
            for (i = 0; i < edges.size(); i++) {
                if (edgesStatues[i] == 0) {
                    objValue[i] = 1;
                } else {
                    objValue[i] = -1;
                }
            }
            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(objValue.length, columnLower, columnUpper, xt);
            // 目标函数
            cplex.addMinimize(cplex.scalProd(x, objValue));

            // 约束条件：Zij + Zji = Xij
            for (i = 0; i < edges.size(); i++) {
                coeff[coeffNum][i] = 1;
                coeff[coeffNum][2 * edges.size() + i] = -1;
                coeff[coeffNum][3 * edges.size() + i] = -1;
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
            }
            // 约束条件：负荷节点仅有1个父节点
            for(i = 0; i < loadNodes.size(); i++) {
                for (MapObject e : g.edgesOf(loadNodes.get(i))) {
                    j = 2 * edges.size() + edges.indexOf(e);
                    if (g.getEdgeSource(e).equals(loadNodes.get(i))) {
                        j = 3 * edges.size() + edges.indexOf(e);
                    }
                    coeff[coeffNum][j] = 1;
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), 1);
                coeffNum++;
            }
            // 约束条件：电源节点无父节点
            for (String supply : supplies) {
                DsConnectNode supplyNode = sys.getCns().get(supply);
                for (MapObject e : g.edgesOf(supplyNode)) {
                    j = 2 * edges.size() + edges.indexOf(e);
                    if (g.getEdgeSource(e).equals(supplyNode)) {
                        j = 3 * edges.size() + edges.indexOf(e);
                    }
                    coeff[coeffNum][j] = 1;
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
            }
            // 约束条件：潮流约束
            for(i = 0; i < loadNodes.size(); i++) {
                for (MapObject e : g.edgesOf(loadNodes.get(i))) {
                    for (j = 0; j < edges.size(); j++) {
                        if (edges.get(j).equals(e)) {
                            if (g.getEdgeTarget(e).equals(loadNodes.get(i))) {
                                coeff[coeffNum][edges.size() + j] = 1;
                            } else {
                                coeff[coeffNum][edges.size() + j] = -1;
                            }
                            break;
                        }
                    }
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), loadArray[i]);
                coeffNum++;
            }
            // 约束条件：电源容量约束
            for(i = 0; i < supplies.length; i++) {
                DsConnectNode supplyCn = sys.getCns().get(supplies[i]);
                for (MapObject e : g.edgesOf(supplyCn)) {
                    for (j = 0; j < edges.size(); j++) {
                        if (edges.get(j).equals(e)) {
                            if (g.getEdgeSource(e).equals(supplyCn)) {
                                coeff[coeffNum][edges.size() + j] = 1;
                            } else {
                                coeff[coeffNum][edges.size() + j] = -1;
                            }
                            break;
                        }
                    }
                }
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), supplyCapacity[i]);
                coeffNum++;
            }
            // 约束条件：支路流量约束
            for(i = 0; i < edges.size(); i++) {
                coeff[coeffNum][edges.size() + i] = 1;
                coeff[coeffNum][i] = -feederCapacity[i];
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
            }
            for(i = 0; i < edges.size(); i++) {
                coeff[coeffNum][edges.size() + i] = 1;
                coeff[coeffNum][i] = feederCapacity[i];
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
            }

            double[] result = new double[objValue.length];
            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                result = cplex.getValues(x);
            }

            int[] newEdgesStatues = new int[edges.size()];
            // 得到新的支路通断状态
            for (i = 0; i < edges.size(); i++) {
                if(result[i] == 1) {
                    newEdgesStatues[i] = 1;
                }
                flows[i] = abs(result[edges.size() + i]);
            }

            if (cplex.getStatus().equals(IloCplex.Status.Optimal)) {
                minSwitch = 0;
                System.out.println(errorSupply[0]);
                System.out.printf("To switch less,we can change the statues of switchs in the edges:\n");
                for (i = 0; i < edges.size(); i++) {
                    if (newEdgesStatues[i] != edgesStatues[i]) {
                        minSwitch++;
                        System.out.printf("%s ", g.getEdgeSource(edges.get(i)).getId());
                        System.out.printf("%s\n", g.getEdgeTarget(edges.get(i)).getId());
                    }
                }
                switchChanged = new String[minSwitch];
                int switchChangedCount = 0;
                for (i = 0; i < edges.size(); i++) {
                    if (newEdgesStatues[i] != edgesStatues[i]) {
                        switchChanged[switchChangedCount++] = edges.get(i).getId();
    //                    switchChanged[switchChangedCount++] = g.getEdgeSource(edges.get(i)).getId() + "-" + g.getEdgeTarget(edges.get(i)).getId();
    //                switchChanged[i] = edges.get(i).getId();
                    }
                }
            }

            //恢复原开关状态
            if (errorSwitch != null) {
                for(i = 0; i < edges.size(); i++) {
                    for (j = 0; j < errorSwitch.length; j++) {
                        if (edges.get(i).getId().equals(errorSwitch[j])) {
                            edges.get(i).setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
                            break;
                        }
                    }
                }
            }

            for (i = 0; i < errorSupply.length; i++) {
                for (MapObject e : g.edgesOf(sys.getCns().get(errorSupply[i]))) {
                    e.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
                }
            }
            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    /**
     * 优化最小开关次数，加入路径长度约束
     */
    public void doOptPathLimit() {
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        int i, j;
        double[] feederCapacity = new double[edges.size()];
        flows = new double[edges.size()];
        boolean isErrorSwitch;
        for (i = 0; i < edges.size(); i++) {
            isErrorSwitch = false;
            if(errorSwitch != null) {
                for (j = 0; j < errorSwitch.length; j++) {
                    if (edges.get(i).getId().equals(errorSwitch[j])) {
                        isErrorSwitch = true;
                        edges.get(i).setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                        break;
                    }
                }
            }
            if (isErrorSwitch)
                feederCapacity[i] = 0;
            else
                feederCapacity[i] = feederCapacityConst;
        }
        // 电源容量
        String[] supplies = sys.getSupplyCns();
        double[] supplyCapacity = new double[supplies.length];
        boolean isErrorSupply;
        for (i = 0; i < supplies.length; i++) {
            isErrorSupply = false;
            for (j = 0; j < errorSupply.length; j++) {
                if (supplies[i].equals(errorSupply[j])) {
                    isErrorSupply = true;
                    for (MapObject e : g.edgesOf(sys.getCns().get(supplies[i]))) {
                        e.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                        for (int k = 0; k < edges.size(); k++) {
                            if (edges.get(k).equals(e)) {
                                feederCapacity[k] = 0;
                                break;
                            }
                        }
                    }
                    break;
                }
            }
            if (isErrorSupply)
                supplyCapacity[i] = 0;
            else
                supplyCapacity[i] = this.supplyCap.get(supplies[i]);
        }

//        int loopsEdgeNum = 0;
//        for (int[] loop : loops) {
//            loopsEdgeNum += loop.length;
//        }
//        int impPathsEdgeNum = 0;
//        for (MapObject[] path : impPaths) {
//            impPathsEdgeNum += path.length;
//        }
        // 开始构造线性规划模型
        double objValue[] = new double[2 * edges.size() + 2 * impPaths.size() + 2 * edges.size()];
        // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double columnLower[] = new double[objValue.length];
        // 状态变量上限
        double columnUpper[] = new double[objValue.length];
        // 指明变量类型
        IloNumVarType[] xt  = new IloNumVarType[objValue.length];
        // 约束方程系数
        double[][] coeff = new double[edges.size() + nodes.size() + nodes.size() + 2 * edges.size() + impPaths.size()][objValue.length];
        // 所有支路的通断状态
        int[] edgesStatues = new int[edges.size()];
        //记录数组中存储元素的个数
        int coeffNum = 0;

        // 负荷的功率，按loadNodes中的顺序排列
        double[] loadArray = new double[loadNodes.size()];
        for (i = 0; i < loadNodes.size(); i++) {
            loadArray[i] = this.loads.get(loadNodes.get(i).getId());
        }

        // 支路状态变量上限为1，下限为0，都是整数
        for (i = 0; i < edges.size(); i++) {
            columnLower[i] = 0;
            columnUpper[i] = 1;
            xt[i] = IloNumVarType.Int;
        }
        // 支路流量变量上下限
        for (i = 0; i < edges.size(); i++) {
            columnLower[edges.size() + i] = -feederCapacity[i];
            columnUpper[edges.size() + i] = feederCapacity[i];
            xt[edges.size() + i] = IloNumVarType.Float;
        }
        // 路径状态变量上下限
        for (i = 0; i < impPaths.size(); i++) {
            columnLower[2 * edges.size() + i] = 0;
            columnUpper[2 * edges.size() + i] = 1;
            xt[2 * edges.size() + i] = IloNumVarType.Int;
        }
        // 路径辅助变量上下限
        double wUpper = 1 - 1.0 / edges.size();
        for (i = 0; i < impPaths.size(); i++) {
            columnLower[2 * edges.size() + impPaths.size() + i] = 0;
//            columnUpper[2 * edges.size() + impPaths.size() + i] = 1 - Double.MIN_VALUE;
            columnUpper[2 * edges.size() + impPaths.size() + i] = wUpper;
            xt[2 * edges.size() + impPaths.size() + i] = IloNumVarType.Float;
        }
        // 树层次关系变量上限为1，下限为0，都是整数
        for (i = 0; i < 2 * edges.size(); i++) {
            columnLower[2 * edges.size() + 2 * impPaths.size() + i] = 0;
            columnUpper[2 * edges.size() + 2 * impPaths.size() + i] = 1;
            xt[2 * edges.size() + 2 * impPaths.size() + i] = IloNumVarType.Int;
        }

        // 得到当前所有支路的通断
        for (i = 0; i < edges.size(); i++) {
            if (edges.get(i).getProperty(KEY_SWITCH_STATUS).equals(SWITCH_ON)) {
                edgesStatues[i] = 1;
            }
        }
        // 设置状态变量的系数值
        for (i = 0; i < edges.size(); i++) {
            if (edgesStatues[i] == 0) {
                objValue[i] = 1;
            } else {
                objValue[i] = -1;
            }
        }
        // sum(WL)/sum(L)
        double sumImpPathLen = 0;
        for (i = 0; i < impLoads.length; i++) {
            int endIndex;
            if (i == impLoads.length - 1) {
                endIndex = impPaths.size();
            } else {
                endIndex = impPathStart[i + 1];
            }
            for (j = impPathStart[i]; j < endIndex; j++) {
                for (MapObject edge : impPaths.get(j)) {
                    sumImpPathLen += Double.parseDouble(edge.getProperty("LineLength"));
                }
            }
        }
        for (i = 0; i < impLoads.length; i++) {
            int endIndex;
            if (i == impLoads.length - 1) {
                endIndex = impPaths.size();
            } else {
                endIndex = impPathStart[i + 1];
            }
            for (j = impPathStart[i]; j < endIndex; j++) {
                double pathLen = 0;
                for (MapObject edge : impPaths.get(j)) {
                    pathLen += Double.parseDouble(edge.getProperty("LineLength"));
                }
                objValue[2 * edges.size() + j] += pathLen/sumImpPathLen;
            }
        }
        try {
            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(objValue.length, columnLower, columnUpper, xt);
            // 目标函数
            cplex.addMinimize(cplex.scalProd(x, objValue));

            // 约束条件：Zij + Zji = Xij
            for (i = 0; i < edges.size(); i++) {
                coeff[coeffNum][i] = 1;
                coeff[coeffNum][2 * edges.size() + 2 * impPaths.size() + i] = -1;
                coeff[coeffNum][3 * edges.size() + 2 * impPaths.size() + i] = -1;
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
            }
            // 约束条件：负荷节点仅有1个父节点
            for(i = 0; i < loadNodes.size(); i++) {
                for (MapObject e : g.edgesOf(loadNodes.get(i))) {
                    j = 2 * edges.size() + 2 * impPaths.size() + edges.indexOf(e);
                    if (g.getEdgeSource(e).equals(loadNodes.get(i))) {
                        j = 3 * edges.size() + 2 * impPaths.size() + edges.indexOf(e);
                    }
                    coeff[coeffNum][j] = 1;
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), 1);
                coeffNum++;
            }
            // 约束条件：电源节点无父节点
            for (String supply : supplies) {
                DsConnectNode supplyNode = sys.getCns().get(supply);
                for (MapObject e : g.edgesOf(supplyNode)) {
                    j = 2 * edges.size() + 2 * impPaths.size() + edges.indexOf(e);
                    if (g.getEdgeSource(e).equals(supplyNode)) {
                        j = 3 * edges.size() + 2 * impPaths.size() + edges.indexOf(e);
                    }
                    coeff[coeffNum][j] = 1;
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
            }
            // 约束条件：潮流约束
            for(i = 0; i < loadNodes.size(); i++) {
                for (MapObject e : g.edgesOf(loadNodes.get(i))) {
                    for (j = 0; j < edges.size(); j++) {
                        if (edges.get(j).equals(e)) {
                            if (g.getEdgeTarget(e).equals(loadNodes.get(i))) {
                                coeff[coeffNum][edges.size() + j] = 1;
                            } else {
                                coeff[coeffNum][edges.size() + j] = -1;
                            }
                            break;
                        }
                    }
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), loadArray[i]);
                coeffNum++;
            }
            // 约束条件：电源容量约束
            for(i = 0; i < supplies.length; i++) {
                DsConnectNode supplyCn = sys.getCns().get(supplies[i]);
                for (MapObject e : g.edgesOf(supplyCn)) {
                    for (j = 0; j < edges.size(); j++) {
                        if (edges.get(j).equals(e)) {
                            if (g.getEdgeSource(e).equals(supplyCn)) {
                                coeff[coeffNum][edges.size() + j] = 1;
                            } else {
                                coeff[coeffNum][edges.size() + j] = -1;
                            }
                            break;
                        }
                    }
                }
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), supplyCapacity[i]);
                coeffNum++;
            }
            // 约束条件：支路流量约束
            for(i = 0; i < edges.size(); i++) {
                coeff[coeffNum][edges.size() + i] = 1;
                coeff[coeffNum][i] = -feederCapacity[i];
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
            }
            for(i = 0; i < edges.size(); i++) {
                coeff[coeffNum][edges.size() + i] = 1;
                coeff[coeffNum][i] = feederCapacity[i];
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
            }
            // 约束条件：路径状态与支路状态关系
            for (i = 0; i < impPaths.size(); i++) {
                for (j = 0; j < impPathToEdge.get(i).length; j++) {
                    coeff[coeffNum][impPathToEdge.get(i)[j]] = 1;
                }
                coeff[coeffNum][2 * edges.size() + i] = -impPaths.get(i).length;
                coeff[coeffNum][2 * edges.size() + impPaths.size() + i] = -impPaths.get(i).length;
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum++;
            }

            double[] result = new double[objValue.length];
            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                result = cplex.getValues(x);
            }

            int[] newEdgesStatues = new int[edges.size()];
            // 得到新的支路通断状态
            for (i = 0; i < edges.size(); i++) {
                if(result[i] == 1) {
                    newEdgesStatues[i] = 1;
                }
                flows[i] = abs(result[edges.size() + i]);
            }

            if (cplex.getStatus().equals(IloCplex.Status.Optimal)) {
                minSwitch = 0;
                System.out.println(errorSupply[0]);
                System.out.printf("To switch less,we can change the statues of switchs in the edges:\n");
                for (i = 0; i < edges.size(); i++) {
                    if (newEdgesStatues[i] != edgesStatues[i]) {
                        minSwitch++;
                        System.out.printf("%s ", g.getEdgeSource(edges.get(i)).getId());
                        System.out.printf("%s\n", g.getEdgeTarget(edges.get(i)).getId());
                    }
                }
                switchChanged = new String[minSwitch];
                int switchChangedCount = 0;
                for (i = 0; i < edges.size(); i++) {
                    if (newEdgesStatues[i] != edgesStatues[i]) {
                        switchChanged[switchChangedCount++] = edges.get(i).getId();
                        //                    switchChanged[switchChangedCount++] = g.getEdgeSource(edges.get(i)).getId() + "-" + g.getEdgeTarget(edges.get(i)).getId();
                        //                switchChanged[i] = edges.get(i).getId();
                    }
                }
            }

            //恢复原开关状态
            if (errorSwitch != null) {
                for(i = 0; i < edges.size(); i++) {
                    for (j = 0; j < errorSwitch.length; j++) {
                        if (edges.get(i).getId().equals(errorSwitch[j])) {
                            edges.get(i).setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
                            break;
                        }
                    }
                }
            }

            for (i = 0; i < errorSupply.length; i++) {
                for (MapObject e : g.edgesOf(sys.getCns().get(errorSupply[i]))) {
                    e.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
                }
            }
            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    /**
     * 分别断开每个电源，计算最小开关次数
     */
    public void allMinSwitch() {
        buildLoops();
        String[] supplyID = sys.getSupplyCns();
        int supplyNum = supplyID.length;
        optResult = new LoadTransferOptResult(supplyID.length, 0);
        for(int i = 0; i < supplyNum; i++) {
            errorSupply = new String[]{supplyID[i]};
            errorFeeder = null;
            errorSwitch = null;
            doOpt();
            optResult.setSupplyId(i, supplyID[i]);
            if (status >= 0) {
                optResult.setMinSwitch(i, minSwitch);
                optResult.setSwitchChanged(switchChanged);
            } else {
                optResult.setMinSwitch(i, -1);
                optResult.setSwitchChanged(switchChanged);
            }
        }
    }

    /**
     * 求一个节点N-1可装容量
     * @param node
     */
    public void loadMax(String node) {
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        int i, j;
        double[] feederCapacity = new double[edges.size()];
        for(i = 0; i < feederCapacity.length; i++) {
            feederCapacity[i] = feederCapacityConst;
        }
        // 负荷的功率，按nodes中的顺序排列
        double[] loadArray = new double[loadNodes.size()];
        for(i = 0; i < loadNodes.size(); i++) {
            loadArray[i] = this.loads.get(loadNodes.get(i).getId());
        }
        // 找到负荷作为变量的节点
        int loadIndex;
        for(loadIndex = 0; loadIndex < loadNodes.size(); loadIndex++) {
            if(loadNodes.get(loadIndex).getId().equals(node))
                break;
        }

        int loopsEdgeNum = 0;
        for (int[] loop : loops) {
            loopsEdgeNum += loop.length;
        }
        try {
            // 开始构造线性规划模型，变量是所有支路的通断状态和潮流，加上一个节点的负荷
            double objValue[] = new double[2 * edges.size() + 1];
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[objValue.length];
            // 状态变量上限
            double columnUpper[] = new double[objValue.length];
            // 指明变量类型
            IloNumVarType[] xt  = new IloNumVarType[2 * edges.size() + 1];
            // 约束方程系数
            double[][] coeff = new double[1 + loops.size() + nodes.size() + 2 * edges.size()][2 * edges.size() + 1];

            // 电源容量
            String[] supplies = sys.getSupplyCns();
            double[] supplyCapacity = new double[supplies.length];
            maxLoad = Double.MAX_VALUE;
            for(int count = 0; count < supplyCapacity.length; count++) {
                for(i = 0; i < supplyCapacity.length; i++) {
                    supplyCapacity[i] = this.supplyCap.get(supplies[i]);
                }
                supplyCapacity[count] = 0;
                // 记录数组中存储元素的个数
                int coeffNum = 0;
                double LMax = 0;
                for (i = 0; i < supplyCapacity.length; i++) {
                    LMax += supplyCapacity[i];
                }
                //所有支路通断状态变量上限为1，下限为0，都是整数
                for (i = 0; i < edges.size(); i++) {
                    columnLower[i] = 0;
                    columnUpper[i] = 1;
                    xt[i] = IloNumVarType.Int;
                }
                // 支路流量变量上下限
                for (i = 0; i < edges.size(); i++) {
                    columnLower[edges.size() + i] = -feederCapacity[i];
                    columnUpper[edges.size() + i] = feederCapacity[i];
                    xt[edges.size() + i] = IloNumVarType.Float;
                }
                // 负荷变量下限为读入的原负荷量
                columnLower[objValue.length - 1] = loadArray[loadIndex];
                columnUpper[objValue.length - 1] = LMax;
                xt[objValue.length - 1] = IloNumVarType.Float;
                // 设置变量的系数值
                for (i = 0; i < 2 * edges.size(); i++) {
                    objValue[i] = 0;
                }
                objValue[columnLower.length - 1] = 1;

                IloCplex cplex = new IloCplex(); // creat a model
                // 变量
                IloNumVar[] x = cplex.numVarArray(2 * edges.size() + 1, columnLower, columnUpper, xt);
                // 目标函数
                cplex.addMaximize(cplex.scalProd(x, objValue));

                // 约束条件：所有支路状态之和为节点总数减电源数量，即负荷节点数量
                for (i = 0; i < edges.size(); i++) {
                    coeff[coeffNum][i] = 1;
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), loadNodes.size());
                coeffNum++;
                // 约束条件：供电环路不连通
                for (int[] loop : loops) {
                    for (i = 0; i < loop.length; i++) {
                        coeff[coeffNum][loop[i]] = 1;
                    }
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), loop.length - 1);
                    coeffNum++;
                }
                // 约束条件：潮流约束
                for(i = 0; i < loadNodes.size(); i++) {
                    for (MapObject e : g.edgesOf(loadNodes.get(i))) {
                        for (j = 0; j < edges.size(); j++) {
                            if (edges.get(j).equals(e)) {
                                if (g.getEdgeTarget(e).equals(loadNodes.get(i))) {
                                    coeff[coeffNum][edges.size() + j] = 1;
                                } else {
                                    coeff[coeffNum][edges.size() + j] = -1;
                                }
                                break;
                            }
                        }
                    }
                    if (i == loadIndex) {
                        coeff[coeffNum][objValue.length - 1] = -1;
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), 0);
                    } else {
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), loadArray[i]);
                    }
                    coeffNum++;
                }
                // 约束条件：电源容量约束
                for(i = 0; i < supplies.length; i++) {
                    DsConnectNode supplyCn = sys.getCns().get(supplies[i]);
                    for (MapObject e : g.edgesOf(supplyCn)) {
                        for (j = 0; j < edges.size(); j++) {
                            if (edges.get(j).equals(e)) {
                                if (g.getEdgeSource(e).equals(supplyCn)) {
                                    coeff[coeffNum][edges.size() + j] = 1;
                                } else {
                                    coeff[coeffNum][edges.size() + j] = -1;
                                }
                                break;
                            }
                        }
                    }
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), supplyCapacity[i]);
                    coeffNum++;
                }
                // 约束条件：支路流量约束
                for(i = 0; i < edges.size(); i++) {
                    coeff[coeffNum][edges.size() + i] = 1;
                    coeff[coeffNum][i] = -feederCapacity[i];
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum++;
                }
                for(i = 0; i < edges.size(); i++) {
                    coeff[coeffNum][edges.size() + i] = 1;
                    coeff[coeffNum][i] = feederCapacity[i];
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum++;
                }

                double[] result = new double[2 * edges.size() + 1];
                if (cplex.solve()) {
                    cplex.output().println("Solution status = " + cplex.getStatus());
                    cplex.output().println("Solution value = " + cplex.getObjValue());
                    result = cplex.getValues(x);
                }

                if (cplex.getStatus().equals(IloCplex.Status.Optimal) && maxLoad > result[result.length - 1] - loadArray[loadIndex]) {
                    maxLoad = result[result.length - 1] - loadArray[loadIndex];
                } else if (!cplex.getStatus().equals(IloCplex.Status.Optimal)) {
                    maxLoad = 0;
                    break;
                }
                cplex.end();
            }
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    /**
     * 求所有节点的N-1可装容量
     */
    public void allLoadMax() {
        buildLoops();
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        maxLoadResult = new HashMap<>(loadNodes.size());
//        boolean isConvergent = true;
        for (DsConnectNode node : loadNodes) {
            loadMax(node.getId());
//            System.out.println(load.get(load.get(i).));
            System.out.printf("The max load in node %s is: %.2f\n", node.getId(), maxLoad);
            maxLoadResult.put(node.getId(), maxLoad);
//            if(status < 0)
//                isConvergent = false;
//            optResult = new LoadTransferOptResult(supplies.length, loadArray.length);
//            optResult.setMinSwitch(minSwitch);
        }
    }

    public void setLoads(Map<String, Double> loads) {
        this.loads = loads;
    }

    public void setSupplyCap(Map<String, Double> supplyCap) {
        this.supplyCap = supplyCap;
    }

    public void setErrorFeeder(int[] errorFeeder) {
        this.errorFeeder = errorFeeder;
    }

    public void setErrorSupply(String[] errorSupply) {
        this.errorSupply = errorSupply;
    }

    public void setFeederCapacityConst(double feederCapacityConst) {
        this.feederCapacityConst = feederCapacityConst;
    }

    public LoadTransferOptResult getOptResult() {
        return optResult;
    }
}
