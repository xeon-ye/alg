package zju.dsntp;

import jpscpu.LinearSolver;
import org.apache.log4j.Logger;
import org.jgrapht.UndirectedGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsConnectNode;
import zju.dsmodel.DsDevices;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

import static zju.dsmodel.DsModelCons.KEY_SWITCH_STATUS;
import static zju.dsmodel.DsModelCons.SWITCH_OFF;
import static zju.dsmodel.DsModelCons.SWITCH_ON;

/**
 * 转供路径搜索
 * @author Dong Shufeng
 * @date 2016/9/19
 */
public class LoadTransferOpt extends PathBasedModel {

    private static Logger log = Logger.getLogger(LoadTransferOpt.class);

    int[] errorFeeder;

    String[] errorSupply;
    String[] errorSwitch;
    //电源容量
    Map<String, Double> supplyCap;
    //馈线容量
    Map<String, Double> feederCap;
    //负荷量
    Map<String, Double> load;
    double feederCapacityConst;
    //开关最少次数计算结果
    int minSwitch;
    String[] switchChanged;
    double maxLoad; //优化结果，最大负荷量
    //所有负荷可装容量计算结果
    Map<String, Double> maxLoadResult;
    //计算开关次数最少的所有结果
    LoadTransferOptResult optResult;
    //馈线带的最大容量
    Map<String, Double> maxFeederLoad;
    //circuit的可装容量
    Map<String, Double> maxCircuitLoad;
    //计算是否收敛的状态
    int status;

    public LoadTransferOpt(DistriSys sys) {
        super(sys);
    }

    /**
     * 优化最小开关次数
     */
    public void doOpt() {
        //生成路径
        buildPathes();

        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();

        int i, j, k, l;
//        if(errorSwitch != null) {
//            for (i = 0; i < errorSwitch.length; i++) {
//                for (MapObject edge : edges) {
//                    if (edge.getId().equals(errorSwitch[i])) {
//                        edge.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
//                        break;
//                    }
//                }
//            }
//        }

        double[] feederCapacity = new double[edges.size()];
        boolean isErrorSwitch;
        for(i = 0; i < feederCapacity.length; i++) {
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
            if(isErrorSwitch)
                feederCapacity[i] = 0;
            else
                feederCapacity[i] = feederCapacityConst;
        }

        //电源容量
        double[] supplyCapacity = new double[supplyStart.length];
        String[] supplies = sys.getSupplyCns();
        boolean isErrorSupply;
        for(i = 0; i < supplyCapacity.length; i++) {
            isErrorSupply = false;
            for(j = 0; j < errorSupply.length; j++) {
                if (supplies[i].equals(errorSupply[j])) {
                    isErrorSupply = true;
                    for (MapObject e : g.edgesOf(sys.getCns().get(supplies[i]))) {
                        e.setProperty(KEY_SWITCH_STATUS, SWITCH_OFF);
                    }
                    break;
                }
            }
            if(isErrorSupply)
                supplyCapacity[i] = 0;
            else
                supplyCapacity[i] = this.supplyCap.get(supplies[i]);
        }

        //开始构造线性规划模型
        //状态变量是所有路径的通断状态
        //objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数。
        double objValue[] = new double[pathes.size()];
        //状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double columnLower[] = new double[objValue.length];
        //状态变量上限
        double columnUpper[] = new double[objValue.length];
        //指明哪些是整数
        int whichInt[] = new int[objValue.length];

        //约束下限
        double rowLower[] = new double[nodes.size()+(pathes.size()-supplyCnNum)+supplyStart.length+edges.size()];
        //约束上限
        double rowUpper[] = new double[rowLower.length];
        //约束中非零元系数
        double element[] = new double[cnpathes.size()+(pathes.size()-supplyCnNum)*2+pathes.size()+edgepathes.size()];
        //上面系数对应的列
        int column[] = new int[element.length];
        //每一行起始位置
        int starts[] = new int[rowLower.length+1];
        //求开关最少的次数
        //所有支路的通断状态
        int[] edgesStatues = new int[edges.size()];
        //记录数组中存储元素的个数
        int rowLowerLen = 0, rowUpperLen = 0, elementLen = 0, columnLen = 0, startsLen = 0;

        //负荷的功率，按nodes中的顺序排列
        double[] loadArray = new double[nodes.size()];
        for(i = 0; i < nodes.size(); i++) {
            loadArray[i] = this.load.get(nodes.get(i).getId());
        }
        //所有状态变量上限为1，下限为0，都是整数
        for(i = 0; i < columnLower.length; i++) {
            columnLower[i] = 0;
            columnUpper[i] = 1;
            whichInt[i] = i;
        }
        //初始化所有支路为通
        for(i = 0; i < edges.size(); i++)
            edgesStatues[i] = 1;
        //得到当前所有支路的通断
        for(i = 0; i < edges.size(); i++) {
            if(edges.get(i).getProperty(KEY_SWITCH_STATUS).equals(SWITCH_OFF)) {
                edgesStatues[i] = 0;
            }
        }
        //设置状态变量的系数值
        for(i = 0; i < edgesStatues.length; i++) {
            if(edgesStatues[i] == 0) {
                //以该支路两端节点为终点且通过该支路的所有路径的状态变量系数加1
                for(j = 0; j < nodes.size(); j++) {
                    if(nodes.get(j).equals(g.getEdgeSource(edges.get(i))) || nodes.get(j).equals(g.getEdgeTarget(edges.get(i)))) {
                        if(j == nodes.size()-1) {
                            for (k = cnStart[j]; k < cnpathes.size(); k++) {
                                if(cnpathes.get(k)[cnpathes.get(k).length-1].equals(edges.get(i)))
                                    objValue[cnpathesIndex.get(k)]++;
                            }
                        }
                        else {
                            for (k = cnStart[j]; k < cnStart[j + 1]; k++) {
                                if(cnpathes.get(k)[cnpathes.get(k).length-1].equals(edges.get(i)))
                                    objValue[cnpathesIndex.get(k)]++;
                            }
                        }
                    }
                }
            }
            else {
                //以该支路两端节点为终点且通过该支路的所有路径的状态变量系数减1
                for(j = 0; j < nodes.size(); j++) {
                    if(nodes.get(j).equals(g.getEdgeSource(edges.get(i))) || nodes.get(j).equals(g.getEdgeTarget(edges.get(i)))) {
                        if(j == nodes.size()-1) {
                            for (k = cnStart[j]; k < cnpathes.size(); k++) {
                                if(cnpathes.get(k)[cnpathes.get(k).length-1].equals(edges.get(i)))
                                    objValue[cnpathesIndex.get(k)]--;
                            }
                        }
                        else {
                            for (k = cnStart[j]; k < cnStart[j + 1]; k++) {
                                if(cnpathes.get(k)[cnpathes.get(k).length-1].equals(edges.get(i)))
                                    objValue[cnpathesIndex.get(k)]--;
                            }
                        }
                    }
                }
            }
        }
        //约束条件：对每个负荷节点，有且只有一条路径供电
        for(i = 0; i < nodes.size(); i++) {
            starts[startsLen++] = elementLen;
            //等式约束，将上下限设成相同
            rowLower[rowLowerLen++] = 1;
            rowUpper[rowUpperLen++] = 1;
            if(i == nodes.size()-1) {
                for(j = cnStart[i]; j < cnpathes.size(); j++) {
                    element[elementLen++] = 1;
                    column[columnLen++] = cnpathesIndex.get(j);
                }
            }
            else {
                for (j = cnStart[i]; j < cnStart[i + 1]; j++) {
                    element[elementLen++] = 1;
                    column[columnLen++] = cnpathesIndex.get(j);
                }
            }
        }
        //约束条件：若某路径为通路，那么包括在该路径内的任意路径也是通路
        //对pathes进行类似深度搜索的方式实现
        int endIndex;
        boolean lenEqualOne;
        for(k = 0; k < supplyStart.length; k++) {
            if(k == supplyStart.length-1)
                endIndex = pathes.size();
            else
                endIndex = supplyStart[k+1];
            for (i = supplyStart[k]+1; i < endIndex; i++) {
                lenEqualOne = false;
                starts[startsLen++] = elementLen;
                j = i - 1;
                rowLower[rowLowerLen++] = 0;
                rowUpper[rowUpperLen++] = 1;  //状态变量只取0和1，可令约束上限为1
                element[elementLen++] = 1;
                element[elementLen++] = -1;
                if (pathes.get(i).length > pathes.get(j).length) {
                    column[columnLen++] = j;
                    column[columnLen++] = i;
                }
                else {
                    while (pathes.get(i).length <= pathes.get(j).length) {
                        j--;
                        if(j < 0) {
                            lenEqualOne = true;
                            break;
                        }
                    }
                    if(lenEqualOne) {
                        startsLen--;
                        rowLowerLen--;
                        rowUpperLen--;
                        elementLen -= 2;
                        continue;
                    }
                    column[columnLen++] = j;
                    column[columnLen++] = i;
                }
            }
        }
        //约束条件：由某一电源供电的所有负荷功率之和应小于电源容量
        for(i = 0; i < supplyStart.length; i++) {
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = supplyCapacity[i];
            rowLower[rowLowerLen++] = 0;
            if(i == supplyStart.length-1)
                endIndex = pathes.size();
            else
                endIndex = supplyStart[i+1];
            for(j = supplyStart[i]; j < endIndex; j++) {
                //找出路径在cnpathes中对应的序号
                for(k = 0; k < cnpathesIndex.size(); k++) {
                    if(cnpathesIndex.get(k) == j)
                        break;
                }
                //找出路径cnpathes[k]的末尾节点
                for(l = 1; l < cnStart.length; l++) {
                    if(cnStart[l] > k)
                        break;
                }
                element[elementLen++] = loadArray[l-1];
                column[columnLen++] = j;
            }
        }
        //约束条件：每一条供电线路不能过载
        //线容量
        String lastID;
        for(i = 0; i < edges.size(); i++) {
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = feederCapacity[i];
            rowLower[rowLowerLen++] = 0;
            if(i == edgeStart.length-1)
                endIndex = edgepathes.size();
            else
                endIndex = edgeStart[i+1];
            for(j = edgeStart[i]; j < endIndex; j++) {
                //找出路径edgepathes[j]的末尾节点
                lastID = g.getEdgeTarget(edgepathes.get(j)[edgepathes.get(j).length-1]).getId();
                if (edgepathes.get(j).length == 1) {
                    for (String scn : supplies) {
                        if (scn.equals(lastID)) {
                            lastID = g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 1]).getId();
                            break;
                        }
                    }
                }
                else {
                    //如果路径上倒数第二条边有节点与lastID相同，则lastID应取最后一条边的另一个端点才是路径上的最后一个点
                    if (lastID.equals(g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 2]).getId()) || lastID.equals(g.getEdgeTarget(edgepathes.get(j)[edgepathes.get(j).length - 2]).getId()))
                        lastID = g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 1]).getId();
                }
                for(k = 0; k < nodes.size(); k++)
                    if (nodes.get(k).getId().equals(lastID))
                        break;
                element[elementLen++] = loadArray[k];
                column[columnLen++] = edgepathesIndex.get(j);
            }
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
        System.out.printf("\n");
        int[] newedgesStatues = new int[edges.size()];
        //得到新的支路通断状态
        for(i = 0; i < pathes.size(); i++) {
            if(result[i] == 1) {
                for(j = 0; j < edges.size(); j++) {
                    if(pathes.get(i)[pathes.get(i).length-1].equals(edges.get(j))) {
                        newedgesStatues[j] = 1;
                        break;
                    }
                }
            }
        }
//        for(i = 0; i < edges.size(); i++) {
//            System.out.printf("%s  %s  %d\n", g.getEdgeSource(edges.get(i)).getId(), g.getEdgeTarget(edges.get(i)).getId(), newedgesStatues[i]);
//        }
        if(status >= 0) {
            minSwitch = 0;
            System.out.printf("To switch less,we can change the statues of switchs in the edges:\n");
            for (i = 0; i < edges.size(); i++) {
//            System.out.printf("%d ", newedgesStatues[i]);
                if (newedgesStatues[i] != edgesStatues[i]) {
                    minSwitch++;
                    System.out.printf("%s ", g.getEdgeSource(edges.get(i)).getId());
                    System.out.printf("%s\n", g.getEdgeTarget(edges.get(i)).getId());
                }
            }
            switchChanged = new String[minSwitch];
            int switchChangedCount = 0;
            for (i = 0; i < edges.size(); i++) {
                if (newedgesStatues[i] != edgesStatues[i]) {
                    switchChanged[switchChangedCount++] = edges.get(i).getId();
//                    switchChanged[switchChangedCount++] = g.getEdgeSource(edges.get(i)).getId() + "-" + g.getEdgeTarget(edges.get(i)).getId();
//                switchChanged[i] = edges.get(i).getId();
                }
            }
        }

        //恢复原开关状态
        if(errorSwitch != null) {
            for(i = 0; i < feederCapacity.length; i++) {
                for (j = 0; j < errorSwitch.length; j++) {
                    if (edges.get(i).getId().equals(errorSwitch[j])) {
                        edges.get(i).setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
                        break;
                    }
                }
            }
        }

        for(i = 0; i < supplyCapacity.length; i++) {
            for(j = 0; j < errorSupply.length; j++) {
                if (supplies[i].equals(errorSupply[j])) {
                    for (MapObject e : g.edgesOf(sys.getCns().get(supplies[i]))) {
                        e.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
                    }
                    break;
                }
            }
        }
    }

    /** 求一个节点N-1可装容量
     *
     * @param node
     */
    public void loadMax(String node) {
        //生成路径
        buildPathes();

        int loadIndex;
        int i, j, k, l, endIndex;
        //找到负荷作为变量的节点
        for(loadIndex = 0; loadIndex < nodes.size(); loadIndex++) {
            if(nodes.get(loadIndex).getId().equals(node))
                break;
        }
        if(loadIndex == nodes.size()-1)
            endIndex = cnpathes.size();
        else
            endIndex = cnStart[loadIndex+1];
        //以作为变量的负荷为终点的所有路径。其中的路径状态变量乘以路径终点的负荷量替换为Z
        ArrayList<MapObject[]> loadPathes = new ArrayList<>(endIndex-cnStart[loadIndex]);
        for(i = cnStart[loadIndex]; i < endIndex; i++)
            loadPathes.add(cnpathes.get(i));
        //开始构造线性规划模型
        //状态变量是所有路径的通断状态，加上将LW替换为Z后增加的变量，加上一个节点的负荷
        //objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数。
        double objValue[] = new double[pathes.size()+loadPathes.size()+1];
        //状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double columnLower[] = new double[objValue.length];
        //状态变量上限
        double columnUpper[] = new double[objValue.length];
        //指明哪些是整数
        int whichInt[] = new int[pathes.size()];

        //约束下限
        double rowLower[] = new double[nodes.size()+(pathes.size()-supplyCnNum)+supplyStart.length+edges.size()+4*loadPathes.size()];
        //约束上限
        double rowUpper[] = new double[rowLower.length];
        //约束中非零元系数
        double element[] = new double[cnpathes.size()+(pathes.size()-supplyCnNum)*2+pathes.size()+edgepathes.size()+10*loadPathes.size()];
        //上面系数对应的列
        int column[] = new int[element.length];
        //每一行起始位置
        int starts[] = new int[rowLower.length+1];

        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        //电源容量
        String[] supplies = sys.getSupplyCns();
        double[] supplyCapacity = new double[supplyStart.length];
        maxLoad = Double.MAX_VALUE;
        for(int count = 0; count < supplyCapacity.length; count++) {
            for(i = 0; i < supplyCapacity.length; i++) {
                supplyCapacity[i] = this.supplyCap.get(supplies[i]);
            }
            supplyCapacity[count] = 0;

            //求节点可带的最大负荷
            //记录数组中存储元素的个数
            int rowLowerLen = 0, rowUpperLen = 0, elementLen = 0, columnLen = 0, startsLen = 0;

            //负荷的功率，按nodes中的顺序排列
            double[] loadArray = new double[nodes.size()];
            for(i = 0; i < nodes.size(); i++) {
                loadArray[i] = this.load.get(nodes.get(i).getId());
            }
            double LMax = 0;
            for (i = 0; i < supplyCapacity.length; i++)
                LMax += supplyCapacity[i];
            //所有路径通断状态变量上限为1，下限为0，都是整数
            for (i = 0; i < pathes.size(); i++) {
                columnLower[i] = 0;
                columnUpper[i] = 1;
                whichInt[i] = i;
            }
            //变量Z的下限为0，上限为无穷
            for (i = pathes.size(); i < columnLower.length - 1; i++) {
                columnLower[i] = 0;
                columnUpper[i] = LMax;
            }
            //负荷变量下限为读入的原负荷量
            columnLower[columnLower.length - 1] = loadArray[loadIndex];
            columnUpper[columnLower.length - 1] = LMax;
            //设置变量的系数值
            for (i = 0; i < columnLower.length - 1; i++)
                objValue[i] = 0;
            objValue[columnLower.length - 1] = -1;
            //约束条件：对每个负荷节点，有且只有一条路径供电
            for (i = 0; i < nodes.size(); i++) {
                starts[startsLen++] = elementLen;
                //等式约束，将上下限设成相同
                rowLower[rowLowerLen++] = 1;
                rowUpper[rowUpperLen++] = 1;
                if (i == nodes.size() - 1) {
                    for (j = cnStart[i]; j < cnpathes.size(); j++) {
                        element[elementLen++] = 1;
                        column[columnLen++] = cnpathesIndex.get(j);
                    }
                } else {
                    for (j = cnStart[i]; j < cnStart[i + 1]; j++) {
                        element[elementLen++] = 1;
                        column[columnLen++] = cnpathesIndex.get(j);
                    }
                }
            }
            //约束条件：若某路径为通路，那么包括在该路径内的任意路径也是通路
            //对pathes进行类似深度搜索的方式实现
            boolean lenEqualOne;
            for(k = 0; k < supplyStart.length; k++) {
                if(k == supplyStart.length-1)
                    endIndex = pathes.size();
                else
                    endIndex = supplyStart[k+1];
                for (i = supplyStart[k]+1; i < endIndex; i++) {
                    lenEqualOne = false;
                    starts[startsLen++] = elementLen;
                    j = i - 1;
                    rowLower[rowLowerLen++] = 0;
                    rowUpper[rowUpperLen++] = 1;  //状态变量只取0和1，可令约束上限为1
                    element[elementLen++] = 1;
                    element[elementLen++] = -1;
                    if (pathes.get(i).length > pathes.get(j).length) {
                        column[columnLen++] = j;
                        column[columnLen++] = i;
                    }
                    else {
                        while (pathes.get(i).length <= pathes.get(j).length) {
                            j--;
                            if(j < 0) {
                                lenEqualOne = true;
                                break;
                            }
                        }
                        if(lenEqualOne) {
                            startsLen--;
                            rowLowerLen--;
                            rowUpperLen--;
                            elementLen -= 2;
                            continue;
                        }
                        column[columnLen++] = j;
                        column[columnLen++] = i;
                    }
                }
            }
            //约束条件：由某一电源供电的所有负荷功率之和应小于电源容量
            int p;
            for (i = 0; i < supplyStart.length; i++) {
                starts[startsLen++] = elementLen;
                rowUpper[rowUpperLen++] = supplyCapacity[i];
                rowLower[rowLowerLen++] = 0;
                if (i == supplyStart.length - 1)
                    endIndex = pathes.size();
                else
                    endIndex = supplyStart[i + 1];
                for (j = supplyStart[i]; j < endIndex; j++) {
                    //找出路径在cnpathes中对应的序号
                    for (k = 0; k < cnpathesIndex.size(); k++) {
                        if (cnpathesIndex.get(k) == j)
                            break;
                    }
                    //找出路径cnpathes[k]的末尾节点
                    for (l = 1; l < cnStart.length; l++) {
                        if (cnStart[l] > k)
                            break;
                    }
                    if (l - 1 == loadIndex) {
                        //如果末尾节点是作为变量的负荷
                        for (p = 0; p < loadPathes.size(); p++) {
                            if (Arrays.equals(loadPathes.get(p), cnpathes.get(k)))
                                break;
                        }
                        element[elementLen++] = 1;
                        column[columnLen++] = pathes.size() + p;
                    } else {
                        element[elementLen++] = loadArray[l - 1];
                        column[columnLen++] = j;
                    }
                }
            }
            //约束条件：每一条供电线路不能过载
            //线容量
            String lastID;
            double[] feederCapacity = new double[edges.size()];
            for(i = 0; i < feederCapacity.length; i++) {
                feederCapacity[i] = feederCapacityConst;
            }
            for (i = 0; i < edges.size(); i++) {
                starts[startsLen++] = elementLen;
                rowUpper[rowUpperLen++] = feederCapacity[i];
                rowLower[rowLowerLen++] = 0;
                if (i == edgeStart.length - 1)
                    endIndex = edgepathes.size();
                else
                    endIndex = edgeStart[i + 1];
                for (j = edgeStart[i]; j < endIndex; j++) {
                    //找出路径edgepathes[j]的末尾节点
                    lastID = g.getEdgeTarget(edgepathes.get(j)[edgepathes.get(j).length - 1]).getId();
                    if (edgepathes.get(j).length == 1) {
                        for (String scn : supplies) {
                            if (scn.equals(lastID)) {
                                lastID = g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 1]).getId();
                                break;
                            }
                        }
                    } else {
                        //如果路径上倒数第二条边有节点与lastID相同，则lastID应取最后一条边的另一个端点才是路径上的最后一个点
                        if (lastID.equals(g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 2]).getId()) || lastID.equals(g.getEdgeTarget(edgepathes.get(j)[edgepathes.get(j).length - 2]).getId()))
                            lastID = g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 1]).getId();
                    }
                    for (k = 0; k < nodes.size(); k++)
                        if (nodes.get(k).getId().equals(lastID))
                            break;
                    if (k == loadIndex) {
                        //如果末尾节点是作为变量的负荷
                        for (p = 0; p < loadPathes.size(); p++) {
                            if (Arrays.equals(loadPathes.get(p), edgepathes.get(j)))
                                break;
                        }
                        element[elementLen++] = 1;
                        column[columnLen++] = pathes.size() + p;
                    } else {
                        element[elementLen++] = loadArray[k];
                        column[columnLen++] = edgepathesIndex.get(j);
                    }
                }
            }
            //等式Z = LW的约束条件
            for (i = 0; i < loadPathes.size(); i++) {
                //Z <= WM
                starts[startsLen++] = elementLen;
                rowUpper[rowUpperLen++] = 0;
                rowLower[rowLowerLen++] = -LMax;
                element[elementLen++] = 1;
                element[elementLen++] = -LMax;
                column[columnLen++] = pathes.size() + i;
                column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex] + i);
                //Z >= -WM
                starts[startsLen++] = elementLen;
                rowUpper[rowUpperLen++] = 2 * LMax;
                rowLower[rowLowerLen++] = 0;
                element[elementLen++] = 1;
                element[elementLen++] = LMax;
                column[columnLen++] = pathes.size() + i;
                column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex] + i);
                //Z - L <= (1-W)M       Z-L+WM <= M
                starts[startsLen++] = elementLen;
                rowUpper[rowUpperLen++] = LMax;
                rowLower[rowLowerLen++] = -LMax;
                element[elementLen++] = 1;
                element[elementLen++] = -1;
                element[elementLen++] = LMax;
                column[columnLen++] = pathes.size() + i;
                column[columnLen++] = columnLower.length - 1;
                column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex] + i);
                //Z-L >= -(1-W)M      Z-L-WM >= -M
                starts[startsLen++] = elementLen;
                rowUpper[rowUpperLen++] = 0;
                rowLower[rowLowerLen++] = -LMax;
                element[elementLen++] = 1;
                element[elementLen++] = -1;
                element[elementLen++] = -LMax;
                column[columnLen++] = pathes.size() + i;
                column[columnLen++] = columnLower.length - 1;
                column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex] + i);
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
            }
            else if(status < 0) {
                maxLoad = 0;
                break;
            }
        }
    }

    public void allMinSwitch() {
        buildPathes();
        String[] supplyID = sys.getSupplyCns();
        int supplyNum = supplyStart.length;
        optResult = new LoadTransferOptResult(supplyStart.length, 0);
        for(int i = 0; i < supplyNum; i++) {
            errorSupply = new String[]{supplyID[i]};
            errorFeeder = new int[]{0};
            errorSwitch = new String[0];
            doOpt();
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
     * 求所有节点的N-1可装容量
     */
    public void allLoadMax() {
        buildPathes();
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        maxLoadResult = new HashMap<String, Double>(nodes.size());
//        boolean isConvergent = true;
        for(int i = 0; i < nodes.size(); i++) {
            loadMax(nodes.get(i).getId());
//            System.out.println(load.get(load.get(i).));
            System.out.printf("The max load in node %s is: %.2f\n", nodes.get(i).getId(), maxLoad);
            maxLoadResult.put(nodes.get(i).getId(), maxLoad);
//            if(status < 0)
//                isConvergent = false;
//            optResult = new LoadTransferOptResult(supplies.length, loadArray.length);
//            optResult.setMinSwitch(minSwitch);
        }
//        if(isConvergent) {
            double[] maxLoadChanged = new double[nodes.size()]; //可装容量
            double[] minFeederLoadChanged = new double[edges.size()];
            for (int i = 0; i < nodes.size(); i++) {
                maxLoadChanged[i] = maxLoadResult.get(nodes.get(i).getId());

                //可装容量小于0怎么办
                if (maxLoadChanged[i] < 0)
                    maxLoadChanged[i] = 0;

            }
            String lastID;
            int endIndex, i, j, k, l;
            boolean isON;   //路径的通断状态
            maxFeederLoad = new HashMap<String, Double>(edges.size());
            maxCircuitLoad = new HashMap<String, Double>(supplies.length);
            double[] feederLoad = new double[edges.size()];
            for (i = 0; i < edges.size(); i++) {
                minFeederLoadChanged[i] = Double.MAX_VALUE;
                if (i == edgeStart.length - 1)
                    endIndex = edgepathes.size();
                else
                    endIndex = edgeStart[i + 1];
                for (j = edgeStart[i]; j < endIndex; j++) {
                    //判断路径的通断
                    isON = true;
                    for (l = 0; l < edgepathes.get(j).length; l++) {
                        if (edgepathes.get(j)[l].getProperty(KEY_SWITCH_STATUS).equals(SWITCH_OFF)) {
                            isON = false;
                            break;
                        }
                    }
                    if (isON) {
                        //找出路径edgepathes[j]的末尾节点
                        lastID = g.getEdgeTarget(edgepathes.get(j)[edgepathes.get(j).length - 1]).getId();
                        if (edgepathes.get(j).length == 1) {
                            for (String scn : supplies) {
                                if (scn.equals(lastID)) {
                                    lastID = g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 1]).getId();
                                    break;
                                }
                            }
                        } else {
                            //如果路径上倒数第二条边有节点与lastID相同，则lastID应取最后一条边的另一个端点才是路径上的最后一个点
                            if (lastID.equals(g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 2]).getId()) || lastID.equals(g.getEdgeTarget(edgepathes.get(j)[edgepathes.get(j).length - 2]).getId()))
                                lastID = g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 1]).getId();
                        }
                        for (k = 0; k < nodes.size(); k++) {
                            if (nodes.get(k).getId().equals(lastID)) {
                                break;
                            }
                        }
                        feederLoad[i] += load.get(lastID);
                        if (minFeederLoadChanged[i] > maxLoadChanged[k]) {
                            minFeederLoadChanged[i] = maxLoadChanged[k];
//                        System.out.println(12345);
//                        System.out.println(maxLoadChanged[k]);
                        }
                    }
                }
            }
            for (i = 0; i < feederLoad.length; i++) {
                maxFeederLoad.put(edges.get(i).getId(), feederLoad[i] + minFeederLoadChanged[i]);
                for (j = 0; j < supplies.length; j++) {
                    if (g.getEdgeSource(edges.get(i)).getId().equals(supplies[j]) || g.getEdgeTarget(edges.get(i)).getId().equals(supplies[j])) {
                        maxCircuitLoad.put(edges.get(i).getId(), minFeederLoadChanged[i]);
//                        maxCircuitLoad.put(edges.get(i).getId(), feederLoad[i] + minFeederLoadChanged[i]);
                        break;
                    }
                }
            }
//        }
    }

    /**
     * 节点的非N-1可装容量
     */
    public void loadMaxN(String node) {
        int loadIndex,endIndex, i, j, k, l, pathOnIndex;
        //电源容量
        String[] supplies = sys.getSupplyCns();
        double[] supplyCapacity = new double[supplyStart.length];
        for(i = 0; i < supplyCapacity.length; i++) {
            supplyCapacity[i] = this.supplyCap.get(supplies[i]);
        }
        //负荷的功率，按nodes中的顺序排列
        double[] loadArray = new double[nodes.size()];
        for(i = 0; i < nodes.size(); i++) {
            loadArray[i] = this.load.get(nodes.get(i).getId());
        }
        //找到负荷节点
        for(loadIndex = 0; loadIndex < nodes.size(); loadIndex++) {
            if(nodes.get(loadIndex).getId().equals(node))
                break;
        }
        if(loadIndex == nodes.size()-1)
            endIndex = cnpathes.size();
        else
            endIndex = cnStart[loadIndex + 1];
        //找到node的供电路径为通的那条路径
        for(i = cnStart[loadIndex]; i < endIndex; i++) {
            boolean pathStatues = true; //路径的通断状态
            for(j = 0; j < cnpathes.get(i).length; j++) {
                if(cnpathes.get(i)[j].getProperty(KEY_SWITCH_STATUS).equals(SWITCH_OFF)) {
                    pathStatues = false;
                    break;
                }
            }
            if(pathStatues == true)
                break;
        }
        pathOnIndex = cnpathesIndex.get(i);

        int supplyIndex;    //为node供电的电源序号
        for(i = 1; i < supplyStart.length; i++) {
            if(supplyStart[i] > pathOnIndex)
                break;
        }
        supplyIndex = i - 1;

        if(supplyIndex == supplyStart.length - 1)
            endIndex = pathes.size();
        else
            endIndex = supplyStart[supplyIndex + 1];

        int[] pathsStatue = new int[endIndex - supplyStart[supplyIndex]];   //该电源所有路径的通断状态

        maxLoad = supplyCapacity[supplyIndex];

        for(i = 0; i < pathsStatue.length; i++)
            pathsStatue[i] = 1;
        for(i = supplyStart[supplyIndex]; i < endIndex; i++) {
            for(j = 0; j < pathes.get(i).length; j++) {
                if(pathes.get(i)[j].getProperty(KEY_SWITCH_STATUS).equals(SWITCH_OFF)) {
                    pathsStatue[i - supplyStart[supplyIndex]] = 0;
                    break;
                }
            }
            if(pathsStatue[i - supplyStart[supplyIndex]] == 1) {
                //找出路径在cnpathes中对应的序号
                for (k = 0; k < cnpathesIndex.size(); k++) {
                    if (cnpathesIndex.get(k) == i)
                        break;
                }
                //找出路径cnpathes[k]的末尾节点
                for (l = 1; l < cnStart.length; l++) {
                    if (cnStart[l] > k)
                        break;
                }
                maxLoad -= loadArray[l - 1];
            }
        }
    }

    /**
     * 求所有节点的非N-1可装容量
     */
    public void allLoadMaxN() {
        buildPathes();
        String[] supplies = sys.getSupplyCns();
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        maxLoadResult = new HashMap<String, Double>(nodes.size());
        for(int i = 0; i < nodes.size(); i++) {
            loadMaxN(nodes.get(i).getId());
            System.out.printf("The max load in node %s is: %.2f\n", nodes.get(i).getId(), maxLoad);
            maxLoadResult.put(nodes.get(i).getId(), maxLoad);
        }

        double[] maxLoadChanged = new double[nodes.size()]; //可装容量
        double[] minFeederLoadChanged = new double[edges.size()];
        for (int i = 0; i < nodes.size(); i++) {
            maxLoadChanged[i] = maxLoadResult.get(nodes.get(i).getId());

            //可装容量小于0怎么办
            if (maxLoadChanged[i] < 0)
                maxLoadChanged[i] = 0;

        }
        String lastID;
        int endIndex, i, j, k, l;
        boolean isON;   //路径的通断状态
        maxFeederLoad = new HashMap<String, Double>(edges.size());
        maxCircuitLoad = new HashMap<String, Double>(supplies.length);
        double[] feederLoad = new double[edges.size()];
        for (i = 0; i < edges.size(); i++) {
            minFeederLoadChanged[i] = Double.MAX_VALUE;
            if (i == edgeStart.length - 1)
                endIndex = edgepathes.size();
            else
                endIndex = edgeStart[i + 1];
            for (j = edgeStart[i]; j < endIndex; j++) {
                //判断路径的通断
                isON = true;
                for (l = 0; l < edgepathes.get(j).length; l++) {
                    if (edgepathes.get(j)[l].getProperty(KEY_SWITCH_STATUS).equals(SWITCH_OFF)) {
                        isON = false;
                        break;
                    }
                }
                if (isON) {
                    //找出路径edgepathes[j]的末尾节点
                    lastID = g.getEdgeTarget(edgepathes.get(j)[edgepathes.get(j).length - 1]).getId();
                    if (edgepathes.get(j).length == 1) {
                        for (String scn : supplies) {
                            if (scn.equals(lastID)) {
                                lastID = g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 1]).getId();
                                break;
                            }
                        }
                    } else {
                        //如果路径上倒数第二条边有节点与lastID相同，则lastID应取最后一条边的另一个端点才是路径上的最后一个点
                        if (lastID.equals(g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 2]).getId()) || lastID.equals(g.getEdgeTarget(edgepathes.get(j)[edgepathes.get(j).length - 2]).getId()))
                            lastID = g.getEdgeSource(edgepathes.get(j)[edgepathes.get(j).length - 1]).getId();
                    }
                    for (k = 0; k < nodes.size(); k++) {
                        if (nodes.get(k).getId().equals(lastID)) {
                            break;
                        }
                    }
                    feederLoad[i] += load.get(lastID);
                    if (minFeederLoadChanged[i] > maxLoadChanged[k]) {
                        minFeederLoadChanged[i] = maxLoadChanged[k];
//                        System.out.println(maxLoadChanged[k]);
                    }
                }
            }
        }
        for (i = 0; i < feederLoad.length; i++) {
            maxFeederLoad.put(edges.get(i).getId(), feederLoad[i] + minFeederLoadChanged[i]);
            for (j = 0; j < supplies.length; j++) {
                if (g.getEdgeSource(edges.get(i)).getId().equals(supplies[j]) || g.getEdgeTarget(edges.get(i)).getId().equals(supplies[j])) {
                    maxCircuitLoad.put(edges.get(i).getId(), minFeederLoadChanged[i]);
//                        maxCircuitLoad.put(edges.get(i).getId(), feederLoad[i] + minFeederLoadChanged[i]);
                    break;
                }
            }
        }
    }

    /**
     * 读取各节点带的负载
     * @param loads
     * @param path  文件路径
     * @throws IOException
     */
    public void readLoads(double[] loads, String path) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        String data;
        String[] newdata;
        String cnId;
        double cnLoad;
        int i,j;

        for(i = 1; i <= nodes.size(); i++) {
            data = br.readLine();
            newdata = data.split(" ", 2);
            cnId = newdata[0];
            cnLoad = Double.parseDouble(newdata[1]);
            for(j = 0; j < nodes.size(); j++) {
                if(nodes.get(j).getId().equals(cnId)) {
                    loads[j] = cnLoad;
                    break;
                }
            }
        }
    }

    //读取馈线容量
    public void readFeederCapacity(double[] feederCapacity, String path) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        String data;
        String[] newdata;
        String cnId1, cnId2;
        double feederLoad;
        MapObject edge;
        DsConnectNode cn1, cn2;
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        int i;

        while((data = br.readLine()) != null) {
            newdata = data.split(" ", 3);
            cnId1 = newdata[0];
            cnId2 = newdata[1];
            feederLoad = Double.parseDouble(newdata[2]);
            cn1 = sys.getCns().get(cnId1);
            cn2 = sys.getCns().get(cnId2);
            if(cn1 == null || cn2 == null) {
                System.out.println("Wrong!");
                continue;
            }
            edge = g.getEdge(cn1, cn2);
            for(i = 0; i < edges.size(); i++) {
                if(edges.get(i).equals(edge)) {
                    feederCapacity[i] = feederLoad;
                    break;
                }
            }
        }
    }

    //读取电源容量
    public void readSupplyCapacity(double[] supplyCapacity, String path) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        String data;
        String[] newdata;
        String supplyId;
        double supplyLoad;
        String[] supplies = sys.getSupplyCns();
        int i;
        while((data = br.readLine()) != null) {
            newdata = data.split(" ", 2);
            supplyId = newdata[0];
            supplyLoad = Double.parseDouble(newdata[1]);
            for(i = 0; i < supplies.length; i++) {
                if(supplies[i].equals(supplyId)) {
                    supplyCapacity[i] = supplyLoad;
                    break;
                }
            }
        }
    }

    public void setErrorFeeder(int[] errorFeeder) {
        this.errorFeeder = errorFeeder;
    }

    public void setErrorSupply(String[] errorSupply) {
        this.errorSupply = errorSupply;
    }

    public void setErrorSwitch(String[] errorSwitch) {
        this.errorSwitch = errorSwitch;
    }

    public void setSupplyCap(Map<String, Double> supplyCap) {
        this.supplyCap = supplyCap;
    }

    public void setFeederCap(Map<String, Double> feederCap) {
        this.feederCap = feederCap;
    }

    public void setFeederCapacityConst(double feederCapacityConst) {
        this.feederCapacityConst = feederCapacityConst;
    }

    public void setLoad(Map<String, Double> load) {
        this.load = load;
    }

    public Map<String, Double> getMaxLoadResult() {
        return maxLoadResult;
    }

    public LoadTransferOptResult getOptResult() {
        return optResult;
    }

    public Map<String, Double> getMaxFeederLoad() {
        return maxFeederLoad;
    }

    public Map<String, Double> getMaxCircuitLoad() {
        return maxCircuitLoad;
    }
}