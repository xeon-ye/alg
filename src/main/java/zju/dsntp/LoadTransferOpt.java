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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static zju.dsmodel.DsModelCons.KEY_SWITCH_STATUS;
import static zju.dsmodel.DsModelCons.SWITCH_OFF;

/**
 * 转供路径搜索
 * @author Dong Shufeng
 * @date 2016/9/19
 */
public class LoadTransferOpt extends PathBasedModel {

    private static Logger log = Logger.getLogger(LoadTransferOpt.class);

    int[] errorFeeder;

    String[] errorSupply;
    //电源容量
    Map<String, Double> supplyCap;
    //馈线容量
    Map<String, Double> feederCap;
    //开关最少次数计算结果
    int minSwitch;
    double maxLoad;

    public LoadTransferOpt(DistriSys sys) {
        super(sys);
    }

    //loadsPath为节点负载量的文件路径，supplyCapacityPath为电源容量文件路径，feederCapacityPath为馈线容量文件路径
    public void doOpt(String loadsPath, String supplyCapacityPath, String feederCapacityPath) {
        DsDevices devices = sys.getDevices();
        //从系统中删除故障的馈线
        List<MapObject> toRemove = new ArrayList<>(errorFeeder.length);
        for(int i : errorFeeder)
            toRemove.add(devices.getFeeders().get(i));
        devices.getFeeders().removeAll(toRemove);
        //将故障的电源从supplyCn中删除
        String[] normalSupply = new String[sys.getSupplyCns().length - errorSupply.length];
        Double[] supplyBaseKv = new Double[normalSupply.length];
        int index = 0;
        boolean isNoraml;
        for(int i = 0; i < sys.getSupplyCns().length; i++) {
            String cnId = sys.getSupplyCns()[i];
            isNoraml = true;
            for (String errorS : errorSupply) {
                if (cnId.equals(errorS)) {
                    isNoraml = false;
                    break;
                }
            }
            if(isNoraml) {
                supplyBaseKv[index] = sys.getSupplyCnBaseKv()[i];
                normalSupply[index++] = cnId;
            }
        }
        sys.setSupplyCns(normalSupply);
        sys.setSupplyCnBaseKv(supplyBaseKv);
        //重新形成拓扑图
        sys.buildOrigTopo(devices);

        //生成路径
        buildPathes();

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
        double rowLower[] = new double[nodes.size()+(pathes.size()-supplyStart.length)+supplyStart.length+edges.size()];
        //约束上限
        double rowUpper[] = new double[rowLower.length];
        //约束中非零元系数
        double element[] = new double[cnpathes.size()+(pathes.size()-supplyStart.length)*2+pathes.size()+edgepathes.size()];
        //上面系数对应的列
        int column[] = new int[element.length];
        //每一行起始位置
        int starts[] = new int[rowLower.length+1];
        //求开关最少的次数
        //所有支路的通断状态
        int[] edgesStatues = new int[edges.size()];
        //记录数组中存储元素的个数
        int rowLowerLen = 0, rowUpperLen = 0, elementLen = 0, columnLen = 0, startsLen = 0;
        //路径的通断，即问题中的状态变量。按pathes中的顺序排列
        int[] w = new int[pathes.size()];
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        int i, j, k, l;
        //负荷的功率，按nodes中的顺序排列
        double[] loads = new double[nodes.size()];
        try {
            readLoads(loads, loadsPath);
        } catch (IOException e) {
            e.printStackTrace();
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
        for(k = 0; k < supplyStart.length; k++) {
            if(k == supplyStart.length-1)
                endIndex = pathes.size();
            else
                endIndex = supplyStart[k+1];
            for (i = supplyStart[k]+1; i < endIndex; i++) {
                starts[startsLen++] = elementLen;
                j = i - 1;
                rowLower[rowLowerLen++] = 0;
                rowUpper[rowUpperLen++] = 1;  //状态变量只取0和1，可令约束上限为1
                element[elementLen++] = 1;
                element[elementLen++] = -1;
                if (pathes.get(i).length > pathes.get(j).length) {
                    column[columnLen++] = j;
                    column[columnLen++] = i;
                } else {
                    while (pathes.get(i).length <= pathes.get(j).length)
                        j--;
                    column[columnLen++] = j;
                    column[columnLen++] = i;
                }
            }
        }
        //约束条件：由某一电源供电的所有负荷功率之和应小于电源容量
        //电源容量
        double[] supplyCapacity = new double[supplyStart.length];
        try {
            readSupplyCapacity(supplyCapacity, supplyCapacityPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        String[] supplies = sys.getSupplyCns();
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
                element[elementLen++] = loads[l-1];
                column[columnLen++] = j;
            }
        }
        //约束条件：每一条供电线路不能过载
        //线容量
        String lastID;
        double[] feederCapacity = new double[edges.size()];
        try {
            readFeederCapacity(feederCapacity, feederCapacityPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
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
                element[elementLen++] = loads[k];
                column[columnLen++] = edgepathesIndex.get(j);
            }
        }
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
        int status = solver.solveMlp(numberColumns, numberRows, objValue,
                columnLower, columnUpper, rowLower, rowUpper, element, column, starts, whichInt, result);
        if (status < 0) {
            log.warn("计算不收敛.");
        } else { //状态位显示计算收敛
            log.info("计算结果.");
        }

//        for(i = 0; i < result.length; i++)
//            System.out.printf("%.0f ", result[i]);
        System.out.printf("\n");
        int[] newedgesStatues = new int[edges.size()];
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
        minSwitch = 0;
        System.out.printf("To switch less,we can change the statues of switchs in the edges:\n");
        for(i = 0; i < edges.size(); i++) {
            //System.out.printf("%d ", newedgesStatues[i]);
            if(newedgesStatues[i] != edgesStatues[i]) {
                minSwitch++;
                System.out.printf("%s ", g.getEdgeSource(edges.get(i)).getId());
                System.out.printf("%s\n", g.getEdgeTarget(edges.get(i)).getId());
            }
        }
    }

    public void doOptLoadMax(String ieeeFilePath, String loadsPath, String supplyCapacityPath, String feederCapacityPath) {
        DsDevices devices = sys.getDevices();

        //从系统中删除故障的馈线
        List<MapObject> toRemove = new ArrayList<>(errorFeeder.length);
        for(int i : errorFeeder)
            toRemove.add(devices.getFeeders().get(i));
        devices.getFeeders().removeAll(toRemove);
        //将故障的电源从supplyCn中删除
        String[] normalSupply = new String[sys.getSupplyCns().length - errorSupply.length];
        Double[] supplyBaseKv = new Double[normalSupply.length];
        int index = 0;
        boolean isNoraml;
        for(int i = 0; i < sys.getSupplyCns().length; i++) {
            String cnId = sys.getSupplyCns()[i];
            isNoraml = true;
            for (String errorS : errorSupply) {
                if (cnId.equals(errorS)) {
                    isNoraml = false;
                    break;
                }
            }
            if(isNoraml) {
                supplyBaseKv[index] = sys.getSupplyCnBaseKv()[i];
                normalSupply[index++] = cnId;
            }
        }
        sys.setSupplyCns(normalSupply);
        sys.setSupplyCnBaseKv(supplyBaseKv);
        //重新形成拓扑图
        sys.buildOrigTopo(devices);

        //生成路径
        buildPathes();

        int loadIndex;
        int i, j, k, l, endIndex;
        //找到负荷作为变量的节点
        for(loadIndex = 0; loadIndex < nodes.size(); loadIndex++) {
            if(nodes.get(loadIndex).getId().equals("L10"))
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
        double rowLower[] = new double[nodes.size()+(pathes.size()-supplyStart.length)+supplyStart.length+edges.size()+4*loadPathes.size()];
        //约束上限
        double rowUpper[] = new double[rowLower.length];
        //约束中非零元系数
        double element[] = new double[cnpathes.size()+(pathes.size()-supplyStart.length)*2+pathes.size()+edgepathes.size()+10*loadPathes.size()];
        //上面系数对应的列
        int column[] = new int[element.length];
        //每一行起始位置
        int starts[] = new int[rowLower.length+1];

        //求节点可带的最大负荷
        //记录数组中存储元素的个数
        int rowLowerLen = 0, rowUpperLen = 0, elementLen = 0, columnLen = 0, startsLen = 0;
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        //负荷的功率，按nodes中的顺序排列
        double[] loads = new double[nodes.size()];
        try {
            readLoads(loads, loadsPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        //电源容量
        double[] supplyCapacity = new double[supplyStart.length];
        try {
            readSupplyCapacity(supplyCapacity, supplyCapacityPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        double LMax = 0;
        for(i = 0; i < supplyCapacity.length; i++)
            LMax += supplyCapacity[i];
        //所有路径通断状态变量上限为1，下限为0，都是整数
        for(i = 0; i < pathes.size(); i++) {
            columnLower[i] = 0;
            columnUpper[i] = 1;
            whichInt[i] = i;
        }
        //变量Z的下限为0，上限为无穷
        for(i = pathes.size(); i < columnLower.length-1; i++) {
            columnLower[i] = 0;
            columnUpper[i] = LMax;
        }
        //负荷变量下限为读入的原负荷量
        columnLower[columnLower.length-1] = loads[loadIndex];
        columnUpper[columnLower.length-1] = LMax;
        //设置变量的系数值
        for(i = 0; i < columnLower.length-1; i++)
            objValue[i] = 0;
        objValue[columnLower.length-1] = -1;
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
        for(k = 0; k < supplyStart.length; k++) {
            if(k == supplyStart.length-1)
                endIndex = pathes.size();
            else
                endIndex = supplyStart[k+1];
            for (i = supplyStart[k]+1; i < endIndex; i++) {
                starts[startsLen++] = elementLen;
                j = i - 1;
                rowLower[rowLowerLen++] = 0;
                rowUpper[rowUpperLen++] = 1;  //状态变量只取0和1，可令约束上限为1
                element[elementLen++] = 1;
                element[elementLen++] = -1;
                if (pathes.get(i).length > pathes.get(j).length) {
                    column[columnLen++] = j;
                    column[columnLen++] = i;
                } else {
                    while (pathes.get(i).length <= pathes.get(j).length)
                        j--;
                    column[columnLen++] = j;
                    column[columnLen++] = i;
                }
            }
        }
        //约束条件：由某一电源供电的所有负荷功率之和应小于电源容量
        int p;
        String[] supplies = sys.getSupplyCns();
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
                if(l-1 == loadIndex) {
                    //如果末尾节点是作为变量的负荷
                    for(p = 0; p < loadPathes.size(); p++) {
                        if(loadPathes.get(p).equals(cnpathes.get(k)))
                            break;
                    }
                    element[elementLen++] = 1;
                    column[columnLen++] = pathes.size()+p;
                }
                else {
                    element[elementLen++] = loads[l - 1];
                    column[columnLen++] = j;
                }
            }
        }
        //约束条件：每一条供电线路不能过载
        //线容量
        String lastID;
        double[] feederCapacity = new double[edges.size()];
        try {
            readFeederCapacity(feederCapacity, feederCapacityPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(i = 0; i < edges.size(); i++) {
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = feederCapacity[i];
            rowLower[rowLowerLen++] = 0;
            if(i == edgeStart.length-1)
                endIndex = edgepathes.size();
            else
                endIndex = edgeStart[i+1];
            for(j = edgeStart[i]; j < endIndex; j++) {
                //找出edgepathes中第j条路径的末尾节点
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
                if(k == loadIndex) {
                    //如果末尾节点是作为变量的负荷
                    for(p = 0; p < loadPathes.size(); p++) {
                        if(loadPathes.get(p).equals(edgepathes.get(j)))
                            break;
                    }
                    element[elementLen++] = 1;
                    column[columnLen++] = pathes.size()+p;
                }
                else {
                    element[elementLen++] = loads[k];
                    column[columnLen++] = edgepathesIndex.get(j);
                }
            }
        }
        //等式Z = LW的约束条件
        for(i = 0; i < loadPathes.size(); i++) {
            //Z <= WM
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = 0;
            rowLower[rowLowerLen++] = -LMax;
            element[elementLen++] = 1;
            element[elementLen++] = -LMax;
            column[columnLen++] = pathes.size()+i;
            column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex]+i);
            //Z >= -WM
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = 2*LMax;
            rowLower[rowLowerLen++] = 0;
            element[elementLen++] = 1;
            element[elementLen++] = LMax;
            column[columnLen++] = pathes.size()+i;
            column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex]+i);
            //Z - L <= (1-W)M       Z-L+WM <= M
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = LMax;
            rowLower[rowLowerLen++] = -LMax;
            element[elementLen++] = 1;
            element[elementLen++] = -1;
            element[elementLen++] = LMax;
            column[columnLen++] = pathes.size()+i;
            column[columnLen++] = columnLower.length-1;
            column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex]+i);
            //Z-L >= -(1-W)M      Z-L-WM >= -M
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = LMax;
            rowLower[rowLowerLen++] = -LMax;
            element[elementLen++] = 1;
            element[elementLen++] = -1;
            element[elementLen++] = -LMax;
            column[columnLen++] = pathes.size()+i;
            column[columnLen++] = columnLower.length-1;
            column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex]+i);
        }

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

        maxLoad = result[numberColumns-1];
        for(i = 0; i < result.length; i++)
            System.out.printf("%.0f ", result[i]);
//        System.out.printf("\n");
//        int[] newedgesStatues = new int[edges.size()];
//        for(i = 0; i < pathes.size(); i++) {
//            if(result[i] == 1) {
//                for(j = 0; j < edges.size(); j++) {
//                    if(pathes.get(i)[pathes.get(i).length-1].equals(edges.get(j))) {
//                        newedgesStatues[j] = 1;
//                        break;
//                    }
//                }
//            }
//        }
//        for(i = 0; i < edges.size(); i++)
//            System.out.printf("%d ", newedgesStatues[i]);
//        System.out.printf("\n");
    }

    public void doOptLoadMax1(String loadsPath, String supplyCapacityPath, String feederCapacityPath, String node, String errorSupplyID) {

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
        double rowLower[] = new double[nodes.size()+(pathes.size()-supplyStart.length)+supplyStart.length+edges.size()+4*loadPathes.size()];
        //约束上限
        double rowUpper[] = new double[rowLower.length];
        //约束中非零元系数
        double element[] = new double[cnpathes.size()+(pathes.size()-supplyStart.length)*2+pathes.size()+edgepathes.size()+10*loadPathes.size()];
        //上面系数对应的列
        int column[] = new int[element.length];
        //每一行起始位置
        int starts[] = new int[rowLower.length+1];

        //求节点可带的最大负荷
        //记录数组中存储元素的个数
        int rowLowerLen = 0, rowUpperLen = 0, elementLen = 0, columnLen = 0, startsLen = 0;
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();

        //负荷的功率，按nodes中的顺序排列
        double[] loads = new double[nodes.size()];
        try {
            readLoads(loads, loadsPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        //电源容量
        String[] supplies = sys.getSupplyCns();
        double[] supplyCapacity = new double[supplyStart.length];
        try {
            readSupplyCapacity(supplyCapacity, supplyCapacityPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(i = 0; i < supplyCapacity.length; i++) {
            if(supplies[i].equals(errorSupplyID)) {
                supplyCapacity[i] = 0;
                break;
            }
        }
        double LMax = 0;
        for(i = 0; i < supplyCapacity.length; i++)
            LMax += supplyCapacity[i];
        //所有路径通断状态变量上限为1，下限为0，都是整数
        for(i = 0; i < pathes.size(); i++) {
            columnLower[i] = 0;
            columnUpper[i] = 1;
            whichInt[i] = i;
        }
        //变量Z的下限为0，上限为无穷
        for(i = pathes.size(); i < columnLower.length-1; i++) {
            columnLower[i] = 0;
            columnUpper[i] = LMax;
        }
        //负荷变量下限为读入的原负荷量
        columnLower[columnLower.length-1] = loads[loadIndex];
        columnUpper[columnLower.length-1] = LMax;
        //设置变量的系数值
        for(i = 0; i < columnLower.length-1; i++)
            objValue[i] = 0;
        objValue[columnLower.length-1] = -1;
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
        for(k = 0; k < supplyStart.length; k++) {
            if(k == supplyStart.length-1)
                endIndex = pathes.size();
            else
                endIndex = supplyStart[k+1];
            for (i = supplyStart[k]+1; i < endIndex; i++) {
                starts[startsLen++] = elementLen;
                j = i - 1;
                rowLower[rowLowerLen++] = 0;
                rowUpper[rowUpperLen++] = 1;  //状态变量只取0和1，可令约束上限为1
                element[elementLen++] = 1;
                element[elementLen++] = -1;
                if (pathes.get(i).length > pathes.get(j).length) {
                    column[columnLen++] = j;
                    column[columnLen++] = i;
                } else {
                    while (pathes.get(i).length <= pathes.get(j).length)
                        j--;
                    column[columnLen++] = j;
                    column[columnLen++] = i;
                }
            }
        }
        //约束条件：由某一电源供电的所有负荷功率之和应小于电源容量
        int p;
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
                if(l-1 == loadIndex) {
                    //如果末尾节点是作为变量的负荷
                    for(p = 0; p < loadPathes.size(); p++) {
                        if(loadPathes.get(p).equals(cnpathes.get(k)))
                            break;
                    }
                    element[elementLen++] = 1;
                    column[columnLen++] = pathes.size()+p;
                }
                else {
                    element[elementLen++] = loads[l - 1];
                    column[columnLen++] = j;
                }
            }
        }
        //约束条件：每一条供电线路不能过载
        //线容量
        String lastID;
        double[] feederCapacity = new double[edges.size()];
        try {
            readFeederCapacity(feederCapacity, feederCapacityPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
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
                if(k == loadIndex) {
                    //如果末尾节点是作为变量的负荷
                    for(p = 0; p < loadPathes.size(); p++) {
                        if(loadPathes.get(p).equals(edgepathes.get(j)))
                            break;
                    }
                    element[elementLen++] = 1;
                    column[columnLen++] = pathes.size()+p;
                }
                else {
                    element[elementLen++] = loads[k];
                    column[columnLen++] = edgepathesIndex.get(j);
                }
            }
        }
        //等式Z = LW的约束条件
        for(i = 0; i < loadPathes.size(); i++) {
            //Z <= WM
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = 0;
            rowLower[rowLowerLen++] = -LMax;
            element[elementLen++] = 1;
            element[elementLen++] = -LMax;
            column[columnLen++] = pathes.size()+i;
            column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex]+i);
            //Z >= -WM
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = 2*LMax;
            rowLower[rowLowerLen++] = 0;
            element[elementLen++] = 1;
            element[elementLen++] = LMax;
            column[columnLen++] = pathes.size()+i;
            column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex]+i);
            //Z - L <= (1-W)M       Z-L+WM <= M
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = LMax;
            rowLower[rowLowerLen++] = -LMax;
            element[elementLen++] = 1;
            element[elementLen++] = -1;
            element[elementLen++] = LMax;
            column[columnLen++] = pathes.size()+i;
            column[columnLen++] = columnLower.length-1;
            column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex]+i);
            //Z-L >= -(1-W)M      Z-L-WM >= -M
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = LMax;
            rowLower[rowLowerLen++] = -LMax;
            element[elementLen++] = 1;
            element[elementLen++] = -1;
            element[elementLen++] = -LMax;
            column[columnLen++] = pathes.size()+i;
            column[columnLen++] = columnLower.length-1;
            column[columnLen++] = cnpathesIndex.get(cnStart[loadIndex]+i);
        }

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
        if(maxLoad < result[numberColumns-1])
            maxLoad = result[numberColumns-1];
//        for(i = 0; i < result.length; i++)
//            System.out.printf("%.0f ", result[i]);
//        System.out.printf("\n");
//        int[] newedgesStatues = new int[edges.size()];
//        for(i = 0; i < pathes.size(); i++) {
//            if(result[i] == 1) {
//                for(j = 0; j < edges.size(); j++) {
//                    if(pathes.get(i)[pathes.get(i).length-1].equals(edges.get(j))) {
//                        newedgesStatues[j] = 1;
//                        break;
//                    }
//                }
//            }
//        }
//        for(i = 0; i < edges.size(); i++)
//            System.out.printf("%d ", newedgesStatues[i]);
//        System.out.printf("\n");
    }

    public void loadMax(String loadsPath, String supplyCapacityPath, String feederCapacityPath) {
        //生成路径
        buildPathes();
        String[] supplies = sys.getSupplyCns();
        for(int i = 0; i < nodes.size(); i++) {
            maxLoad = 0;
            for (String errorSupplyID : supplies)
                doOptLoadMax1(loadsPath, supplyCapacityPath, feederCapacityPath, nodes.get(i).getId(), errorSupplyID);
            System.out.printf("The max load in node %s is: %.0f\n", nodes.get(i).getId(), maxLoad);
        }
    }

    //读取各节点带的负载
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

    public void setSupplyCap(Map<String, Double> supplyCap) {
        this.supplyCap = supplyCap;
    }

    public void setFeederCap(Map<String, Double> feederCap) {
        this.feederCap = feederCap;
    }
}