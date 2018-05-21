package zju.dsntp;

import jpscpu.LinearSolver;
import org.apache.log4j.Logger;
import org.jgrapht.UndirectedGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsConnectNode;

import java.util.HashMap;
import java.util.Map;

import static java.lang.Math.abs;
import static zju.dsmodel.DsModelCons.*;

/**
 * Created by xuchengsi on 2018/1/22.
 */
public class AvailableCapOpt extends BranchBasedModel {

    private static Logger log = Logger.getLogger(AvailableCapOpt.class);

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

    public AvailableCapOpt(DistriSys sys) {
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

        int loopsEdgeNum = 0;
        for (int[] loop : loops) {
            loopsEdgeNum += loop.length;
        }

        // 开始构造线性规划模型
        // 状态变量是所有支路的通断状态
        // objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数。
        double objValue[] = new double[2 * edges.size()];
        // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double columnLower[] = new double[objValue.length];
        // 状态变量上限
        double columnUpper[] = new double[objValue.length];
        // 指明哪些是整数
        int whichInt[] = new int[edges.size()];

        // 约束下限
        double rowLower[] = new double[1 + loops.size() + nodes.size() + 2 * edges.size()];
        // 约束上限
        double rowUpper[] = new double[rowLower.length];
        // 约束中非零元系数
        double element[] = new double[edges.size() + loopsEdgeNum + 2 * edges.size() + 2 * 2 * edges.size()];
        // 上面系数对应的列
        int column[] = new int[element.length];
        // 每一行起始位置
        int starts[] = new int[rowLower.length + 1];
        // 所有支路的通断状态
        int[] edgesStatues = new int[edges.size()];
        //记录数组中存储元素的个数
        int rowLowerLen = 0, rowUpperLen = 0, elementLen = 0, columnLen = 0, startsLen = 0;

        // 负荷的功率，按loadNodes中的顺序排列
        double[] loadArray = new double[loadNodes.size()];
        for (i = 0; i < loadNodes.size(); i++) {
            loadArray[i] = this.loads.get(loadNodes.get(i).getId());
        }
        // 支路状态变量上限为1，下限为0，都是整数
        for (i = 0; i < edges.size(); i++) {
            columnLower[i] = 0;
            columnUpper[i] = 1;
            whichInt[i] = i;
        }
        // 支路流量变量上下限
        for (i = 0; i < edges.size(); i++) {
            columnLower[edges.size() + i] = -feederCapacity[i];
            columnUpper[edges.size() + i] = feederCapacity[i];
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
        // 约束条件：所有支路状态之和为节点总数减电源数量，即负荷节点数量
        starts[startsLen++] = elementLen;
        rowLower[rowLowerLen++] = loadNodes.size();
        rowUpper[rowUpperLen++] = loadNodes.size();
        for (i = 0; i < edges.size(); i++) {
            element[elementLen++] = 1;
            column[columnLen++] = i;
        }
        // 约束条件：供电环路不连通
        for (int[] loop : loops) {
            starts[startsLen++] = elementLen;
            rowLower[rowLowerLen++] = 0;
            rowUpper[rowUpperLen++] = loop.length - 1;
            for (i = 0; i < loop.length; i++) {
                element[elementLen++] = 1;
                column[columnLen++] = loop[i];
            }
        }
        // 约束条件：潮流约束
        for(i = 0; i < loadNodes.size(); i++) {
            starts[startsLen++] = elementLen;
            // 注意计算精度
            rowLower[rowLowerLen++] = loadArray[i];
            rowUpper[rowUpperLen++] = loadArray[i];
            for (MapObject e : g.edgesOf(loadNodes.get(i))) {
                if (g.getEdgeTarget(e).equals(loadNodes.get(i))) {
                    element[elementLen++] = 1;
                } else {
                    element[elementLen++] = -1;
                }
                for (j = 0; j < edges.size(); j++) {
                    if (edges.get(j).equals(e)) {
                        column[columnLen++] = edges.size() + j;
                        break;
                    }
                }
            }
        }
        // 约束条件：电源容量约束
        for(i = 0; i < supplies.length; i++) {
            starts[startsLen++] = elementLen;
            rowLower[rowLowerLen++] = 0;
            rowUpper[rowUpperLen++] = supplyCapacity[i];
            DsConnectNode supplyCn = sys.getCns().get(supplies[i]);
            for (MapObject e : g.edgesOf(supplyCn)) {
                if (g.getEdgeSource(e).equals(supplyCn)) {
                    element[elementLen++] = 1;
                } else {
                    element[elementLen++] = -1;
                }
                for (j = 0; j < edges.size(); j++) {
                    if (edges.get(j).equals(e)) {
                        column[columnLen++] = edges.size() + j;
                        break;
                    }
                }
            }
        }
        // 约束条件：支路流量约束
        for(i = 0; i < edges.size(); i++) {
            starts[startsLen++] = elementLen;
            rowLower[rowLowerLen++] = -2 * feederCapacity[i];
            rowUpper[rowUpperLen++] = 0;
            element[elementLen++] = 1;
            column[columnLen++] = edges.size() + i;
            element[elementLen++] = -feederCapacity[i];
            column[columnLen++] = i;
        }
        for(i = 0; i < edges.size(); i++) {
            starts[startsLen++] = elementLen;
            rowLower[rowLowerLen++] = 0;
            rowUpper[rowUpperLen++] = 2 * feederCapacity[i];
            element[elementLen++] = 1;
            column[columnLen++] = edges.size() + i;
            element[elementLen++] = feederCapacity[i];
            column[columnLen++] = i;
        }
        starts[startsLen++] = elementLen;
//        for(i = 0; i < columnLower.length; i++){
//            System.out.printf("%.0f %.0f %.0f %d\n", objValue[i], columnLower[i], columnUpper[i], whichInt[i]);
//        }
//        for(i = 0; i < rowLower.length; i++)
//            System.out.printf("%.0f %.0f %d\n", rowLower[i], rowUpper[i], starts[i]);
//        for(i = 0; i < element.length; i++)
//            System.out.printf("%2.0f ", element[i]);
//        System.out.printf("\n");
//        for(i = 0; i < column.length; i++)
//            System.out.printf("%2d ", column[i]);

        int numberRows = rowLower.length;
        int numberColumns = columnLower.length;
        double result[] = new double[numberColumns];
        //进行求解
        LinearSolver solver = new LinearSolver();
        solver.setDrive(LinearSolver.MLP_DRIVE_CBC);
        status = solver.solveMlp(numberColumns, numberRows, objValue,
                columnLower, columnUpper, rowLower, rowUpper, element, column, starts, whichInt, result);
        if (status < 0) {
            log.warn("计算不收敛.");
        } else { //状态位显示计算收敛
            log.info("计算结果.");
        }

//        for(i = 0; i < edges.size(); i++) {
//            System.out.printf("%s  %s  %d\n", g.getEdgeSource(edges.get(i)).getId(), g.getEdgeTarget(edges.get(i)).getId(), edgesStatues[i]);
//        }
//        for(i = 0; i < result.length; i++)
//            System.out.printf("%.0f ", result[i]);
        int[] newEdgesStatues = new int[edges.size()];
        // 得到新的支路通断状态
        for (i = 0; i < edges.size(); i++) {
            if(result[i] == 1) {
                newEdgesStatues[i] = 1;
            }
            flows[i] = abs(result[edges.size() + i]);
        }
//        for(i = 0; i < edges.size(); i++) {
//            System.out.printf("%s  %s  %d\n", g.getEdgeSource(edges.get(i)).getId(), g.getEdgeTarget(edges.get(i)).getId(), newedgesStatues[i]);
//        }
        if (status >= 0) {
            minSwitch = 0;
            System.out.println(errorSupply[0]);
            System.out.println("To switch less,we can change the statues of switchs in the edges");
            for (i = 0; i < edges.size(); i++) {
//            System.out.printf("%d ", newedgesStatues[i]);
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

//        for (i = 0; i < supplyCapacity.length; i++) {
//            for(j = 0; j < errorSupply.length; j++) {
//                if (supplies[i].equals(errorSupply[j])) {
//                    for (MapObject e : g.edgesOf(sys.getCns().get(supplies[i]))) {
//                        e.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
//                    }
//                    break;
//                }
//            }
//        }
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

        int loopsEdgeNum = 0;
        for (int[] loop : loops) {
            loopsEdgeNum += loop.length;
        }
        int impPathsEdgeNum = 0;
        for (MapObject[] path : impPaths) {
            impPathsEdgeNum += path.length;
        }

        // 开始构造线性规划模型
        // 状态变量是所有支路的通断状态
        // objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数。
        double objValue[] = new double[2 * edges.size() + 2 * impPaths.size()];
        // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double columnLower[] = new double[objValue.length];
        // 状态变量上限
        double columnUpper[] = new double[objValue.length];
        // 指明哪些是整数
        int whichInt[] = new int[edges.size() + impPaths.size()];

        // 约束下限
        double rowLower[] = new double[1 + loops.size() + nodes.size() + 2 * edges.size() + impPaths.size()];
        // 约束上限
        double rowUpper[] = new double[rowLower.length];
        // 约束中非零元系数
        double element[] = new double[edges.size() + loopsEdgeNum + 2 * edges.size() + 2 * 2 * edges.size() + (impPathsEdgeNum + 2 * impPaths.size())];
        // 上面系数对应的列
        int column[] = new int[element.length];
        // 每一行起始位置
        int starts[] = new int[rowLower.length + 1];
        // 所有支路的通断状态
        int[] edgesStatues = new int[edges.size()];
        //记录数组中存储元素的个数
        int rowLowerLen = 0, rowUpperLen = 0, elementLen = 0, columnLen = 0, startsLen = 0;

        // 负荷的功率，按loadNodes中的顺序排列
        double[] loadArray = new double[loadNodes.size()];
        for (i = 0; i < loadNodes.size(); i++) {
            loadArray[i] = this.loads.get(loadNodes.get(i).getId());
        }
        // 支路状态变量上限为1，下限为0，都是整数
        for (i = 0; i < edges.size(); i++) {
            columnLower[i] = 0;
            columnUpper[i] = 1;
            whichInt[i] = i;
        }
        // 支路流量变量上下限
        for (i = 0; i < edges.size(); i++) {
            columnLower[edges.size() + i] = -feederCapacity[i];
            columnUpper[edges.size() + i] = feederCapacity[i];
        }
        // 路径状态变量上下限
        for (i = 0; i < impPaths.size(); i++) {
            columnLower[2 * edges.size() + i] = 0;
            columnUpper[2 * edges.size() + i] = 1;
            whichInt[edges.size() + i] = 2 * edges.size() + i;
        }
        // 路径辅助变量上下限
        for (i = 0; i < impPaths.size(); i++) {
            columnLower[2 * edges.size() + impPaths.size() + i] = 0;
            columnUpper[2 * edges.size() + impPaths.size() + i] = 1 - Double.MIN_VALUE;
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
                endIndex = impPathStart[i+1];
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
                endIndex = impPathStart[i+1];
            }
            for (j = impPathStart[i]; j < endIndex; j++) {
                double pathLen = 0;
                for (MapObject edge : impPaths.get(j)) {
                    pathLen += Double.parseDouble(edge.getProperty("LineLength"));
                }
                objValue[2 * edges.size() + j] += pathLen/sumImpPathLen;
            }
        }
        // 约束条件：所有支路状态之和为节点总数减电源数量，即负荷节点数量
        starts[startsLen++] = elementLen;
        rowLower[rowLowerLen++] = loadNodes.size();
        rowUpper[rowUpperLen++] = loadNodes.size();
        for (i = 0; i < edges.size(); i++) {
            element[elementLen++] = 1;
            column[columnLen++] = i;
        }
        // 约束条件：供电环路不连通
        for (int[] loop : loops) {
            starts[startsLen++] = elementLen;
            rowLower[rowLowerLen++] = 0;
            rowUpper[rowUpperLen++] = loop.length - 1;
            for (i = 0; i < loop.length; i++) {
                element[elementLen++] = 1;
                column[columnLen++] = loop[i];
            }
        }
        // 约束条件：潮流约束
        for(i = 0; i < loadNodes.size(); i++) {
            starts[startsLen++] = elementLen;
            // 注意计算精度
            rowLower[rowLowerLen++] = loadArray[i];
            rowUpper[rowUpperLen++] = loadArray[i] + 0.0000000001;
            for (MapObject e : g.edgesOf(loadNodes.get(i))) {
                if (g.getEdgeTarget(e).equals(loadNodes.get(i))) {
                    element[elementLen++] = 1;
                } else {
                    element[elementLen++] = -1;
                }
                for (j = 0; j < edges.size(); j++) {
                    if (edges.get(j).equals(e)) {
                        column[columnLen++] = edges.size() + j;
                        break;
                    }
                }
            }
        }
        // 约束条件：电源容量约束
        for(i = 0; i < supplies.length; i++) {
            starts[startsLen++] = elementLen;
            rowLower[rowLowerLen++] = 0;
            rowUpper[rowUpperLen++] = supplyCapacity[i];
            DsConnectNode supplyCn = sys.getCns().get(supplies[i]);
            for (MapObject e : g.edgesOf(supplyCn)) {
                if (g.getEdgeSource(e).equals(supplyCn)) {
                    element[elementLen++] = 1;
                } else {
                    element[elementLen++] = -1;
                }
                for (j = 0; j < edges.size(); j++) {
                    if (edges.get(j).equals(e)) {
                        column[columnLen++] = edges.size() + j;
                        break;
                    }
                }
            }
        }
        // 约束条件：支路流量约束
        for (i = 0; i < edges.size(); i++) {
            starts[startsLen++] = elementLen;
            rowLower[rowLowerLen++] = -2 * feederCapacity[i];
            rowUpper[rowUpperLen++] = 0;
            element[elementLen++] = 1;
            column[columnLen++] = edges.size() + i;
            element[elementLen++] = -feederCapacity[i];
            column[columnLen++] = i;
        }
        for (i = 0; i < edges.size(); i++) {
            starts[startsLen++] = elementLen;
            rowLower[rowLowerLen++] = 0;
            rowUpper[rowUpperLen++] = 2 * feederCapacity[i];
            element[elementLen++] = 1;
            column[columnLen++] = edges.size() + i;
            element[elementLen++] = feederCapacity[i];
            column[columnLen++] = i;
        }
        // 约束条件：路径状态与支路状态关系
        for (i = 0; i < impPaths.size(); i++) {
            starts[startsLen++] = elementLen;
            rowLower[rowLowerLen++] = 0;
            rowUpper[rowUpperLen++] = 0;
            for (j = 0; j < impPathToEdge.get(i).length; j++) {
                element[elementLen++] = 1;
                column[columnLen++] = impPathToEdge.get(i)[j];
            }
            element[elementLen++] = -impPaths.get(i).length;
            column[columnLen++] = 2 * edges.size() + i;
            element[elementLen++] = -impPaths.get(i).length;
            column[columnLen++] = 2 * edges.size() + impPaths.size() + i;
        }
        starts[startsLen++] = elementLen;
//        for(i = 0; i < columnLower.length; i++){
//            System.out.printf("%.0f %.0f %.0f %d\n", objValue[i], columnLower[i], columnUpper[i], whichInt[i]);
//        }
//        for(i = 0; i < rowLower.length; i++)
//            System.out.printf("%.0f %.0f %d\n", rowLower[i], rowUpper[i], starts[i]);
//        for(i = 0; i < element.length; i++)
//            System.out.printf("%2.0f ", element[i]);
//        System.out.printf("\n");
//        for(i = 0; i < column.length; i++)
//            System.out.printf("%2d ", column[i]);

        int numberRows = rowLower.length;
        int numberColumns = columnLower.length;
        double result[] = new double[numberColumns];
        //进行求解
        LinearSolver solver = new LinearSolver();
        solver.setDrive(LinearSolver.MLP_DRIVE_CBC);
        status = solver.solveMlp(numberColumns, numberRows, objValue,
                columnLower, columnUpper, rowLower, rowUpper, element, column, starts, whichInt, result);
        if (status < 0) {
            log.warn("计算不收敛.");
        } else { //状态位显示计算收敛
            log.info("计算结果.");
        }

//        for(i = 0; i < edges.size(); i++) {
//            System.out.printf("%s  %s  %d\n", g.getEdgeSource(edges.get(i)).getId(), g.getEdgeTarget(edges.get(i)).getId(), edgesStatues[i]);
//        }
//        for(i = 0; i < result.length; i++)
//            System.out.printf("%.0f ", result[i]);
        int[] newEdgesStatues = new int[edges.size()];
        // 得到新的支路通断状态
        for (i = 0; i < edges.size(); i++) {
            if(result[i] == 1) {
                newEdgesStatues[i] = 1;
            }
            flows[i] = abs(result[edges.size() + i]);
        }
//        for(i = 0; i < edges.size(); i++) {
//            System.out.printf("%s  %s  %d\n", g.getEdgeSource(edges.get(i)).getId(), g.getEdgeTarget(edges.get(i)).getId(), newedgesStatues[i]);
//        }
        if (status >= 0) {
            minSwitch = 0;
            System.out.println(errorSupply[0]);
            System.out.println("To switch less,we can change the statues of switchs in the edges");
            for (i = 0; i < edges.size(); i++) {
//            System.out.printf("%d ", newedgesStatues[i]);
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

//        for (i = 0; i < supplyCapacity.length; i++) {
//            for(j = 0; j < errorSupply.length; j++) {
//                if (supplies[i].equals(errorSupply[j])) {
//                    for (MapObject e : g.edgesOf(sys.getCns().get(supplies[i]))) {
//                        e.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
//                    }
//                    break;
//                }
//            }
//        }
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
            doOptPathLimit();
            optResult.setSupplyId(i, supplyID[i]);
            if(status >= 0) {
                optResult.setMinSwitch(i, minSwitch);
                optResult.setSwitchChanged(switchChanged);
            }
            else {
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
//        int nodesEdgeNum = 0;
//        for (DsConnectNode n : nodes) {
//            nodesEdgeNum += g.edgesOf(n).size();
//        }

        // 开始构造线性规划模型，变量是所有支路的通断状态和潮流，加上一个节点的负荷
        double objValue[] = new double[2 * edges.size() + 1];
        // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double columnLower[] = new double[objValue.length];
        // 状态变量上限
        double columnUpper[] = new double[objValue.length];
        // 指明哪些是整数
        int whichInt[] = new int[edges.size()];

        // 约束下限
        double rowLower[] = new double[1 + loops.size() + nodes.size() + 2 * edges.size()];
        // 约束上限
        double rowUpper[] = new double[rowLower.length];
        // 约束中非零元系数
        double element[] = new double[edges.size() + loopsEdgeNum + 2 * edges.size() + 1 + 2 * 2 * edges.size()];
        // 上面系数对应的列
        int column[] = new int[element.length];
        // 每一行起始位置
        int starts[] = new int[rowLower.length+1];

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
            int rowLowerLen = 0, rowUpperLen = 0, elementLen = 0, columnLen = 0, startsLen = 0;
            double LMax = 0;
            for (i = 0; i < supplyCapacity.length; i++) {
                LMax += supplyCapacity[i];
            }
            //所有支路通断状态变量上限为1，下限为0，都是整数
            for (i = 0; i < edges.size(); i++) {
                columnLower[i] = 0;
                columnUpper[i] = 1;
                whichInt[i] = i;
            }
            // 支路流量变量上下限
            for (i = 0; i < edges.size(); i++) {
                columnLower[edges.size() + i] = -feederCapacity[i];
                columnUpper[edges.size() + i] = feederCapacity[i];
            }
            // 负荷变量下限为读入的原负荷量
            columnLower[objValue.length - 1] = loadArray[loadIndex];
            columnUpper[objValue.length - 1] = LMax;
            // 设置变量的系数值
            for (i = 0; i < 2 * edges.size(); i++) {
                objValue[i] = 0;
            }
            objValue[columnLower.length - 1] = -1;
            // 约束条件：所有支路状态之和为节点总数减电源数量
            starts[startsLen++] = elementLen;
            rowLower[rowLowerLen++] = loadNodes.size();
            rowUpper[rowUpperLen++] = loadNodes.size();
            for (i = 0; i < edges.size(); i++) {
                element[elementLen++] = 1;
                column[columnLen++] = i;
            }
            // 约束条件：供电环路不连通
            for (int[] loop : loops) {
                starts[startsLen++] = elementLen;
                rowLower[rowLowerLen++] = 0;
                rowUpper[rowUpperLen++] = loop.length - 1;
                for (i = 0; i < loop.length; i++) {
                    element[elementLen++] = 1;
                    column[columnLen++] = loop[i];
                }
            }
            // 约束条件：潮流约束
            for(i = 0; i < loadNodes.size(); i++) {
                starts[startsLen++] = elementLen;
                for (MapObject e : g.edgesOf(loadNodes.get(i))) {
                    for (j = 0; j < edges.size(); j++) {
                        if (edges.get(j).equals(e)) {
                            if (g.getEdgeTarget(edges.get(j)).equals(loadNodes.get(i))) {
                                element[elementLen++] = 1;
                            } else {
                                element[elementLen++] = -1;
                            }
                            column[columnLen++] = edges.size() + j;
                            break;
                        }
                    }
                }
                if (i == loadIndex) {
                    rowLower[rowLowerLen++] = 0;
                    rowUpper[rowUpperLen++] = 0 + 0.0000000001;
                    element[elementLen++] = -1;
                    column[columnLen++] = objValue.length - 1;
                } else {
                    rowLower[rowLowerLen++] = loadArray[i];
                    rowUpper[rowUpperLen++] = loadArray[i] + 0.0000000001;
                }
            }
            // 约束条件：电源容量约束
            for(i = 0; i < supplies.length; i++) {
                starts[startsLen++] = elementLen;
                rowLower[rowLowerLen++] = 0;
                rowUpper[rowUpperLen++] = supplyCapacity[i];
                DsConnectNode supplyCn = sys.getCns().get(supplies[i]);
                for (MapObject e : g.edgesOf(supplyCn)) {
                    for (j = 0; j < edges.size(); j++) {
                        if (edges.get(j).equals(e)) {
                            if (g.getEdgeSource(edges.get(j)).equals(supplyCn)) {
                                element[elementLen++] = 1;
                            } else {
                                element[elementLen++] = -1;
                            }
                            column[columnLen++] = edges.size() + j;
                            break;
                        }
                    }
                }
            }
            // 约束条件：支路流量量约束
            for(i = 0; i < edges.size(); i++) {
                starts[startsLen++] = elementLen;
                rowLower[rowLowerLen++] = -2 * feederCapacity[i];
                rowUpper[rowUpperLen++] = 0;
                element[elementLen++] = 1;
                column[columnLen++] = edges.size() + i;
                element[elementLen++] = -feederCapacity[i];
                column[columnLen++] = i;
            }
            for(i = 0; i < edges.size(); i++) {
                starts[startsLen++] = elementLen;
                rowLower[rowLowerLen++] = 0;
                rowUpper[rowUpperLen++] = 2 * feederCapacity[i];
                element[elementLen++] = 1;
                column[columnLen++] = edges.size() + i;
                element[elementLen++] = feederCapacity[i];
                column[columnLen++] = i;
            }
            starts[startsLen++] = elementLen;

//        for(i = 0; i < columnLower.length; i++){
//            System.out.printf("%.0f %.0f %.0f\n", objValue[i], columnLower[i], columnUpper[i]);
//        }
//        for(i = 0; i < whichInt.length; i++)
//            System.out.printf("%d ", whichInt[i]);
//        for(i = 0; i < rowLower.length; i++)
//            System.out.printf("%.0f %.0f %d\n", rowLower[i], rowUpper[i], starts[i]);
//        for(i = 0; i < element.length; i++)
//            System.out.printf("%2.0f ", element[i]);
//        System.out.printf("\n");
//        for(i = 0; i < column.length; i++)
//            System.out.printf("%2d ", column[i]);

            int numberRows = rowLower.length;
            int numberColumns = columnLower.length;
            double result[] = new double[numberColumns];
            //进行求解
            LinearSolver solver = new LinearSolver();
            solver.setDrive(LinearSolver.MLP_DRIVE_CBC);
            int status = solver.solveMlp(numberColumns, numberRows, objValue,
                    columnLower, columnUpper, rowLower, rowUpper, element, column, starts, whichInt, result);
            if (status < 0) {
                log.warn("计算不收敛.");
            } else { //状态位显示计算收敛
                log.info("计算结果.");
            }
            if (status >= 0 && maxLoad > result[numberColumns - 1] - loadArray[loadIndex]) {
                maxLoad = result[numberColumns - 1] - loadArray[loadIndex];
            } else if(status < 0) {
                maxLoad = 0;
                break;
            }
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
