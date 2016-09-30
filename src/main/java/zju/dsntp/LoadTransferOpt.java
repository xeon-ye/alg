package zju.dsntp;

import jpscpu.LinearSolver;
import org.apache.log4j.Logger;
import org.jgrapht.UndirectedGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsConnectNode;
import zju.dsmodel.DsDevices;

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

    public LoadTransferOpt(DistriSys sys) {
        super(sys);
    }

    public void doOpt() {
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

        //开始构造线性规划模型
        //状态变量是所有路径的通断状态
        //objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数。
        double objValue[] = new double[pathes.size()];
        //状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double columnLower[] = new double[objValue.length];
        //状态变量上限
        double columnUpper[] = new double[objValue.length];
        //指明那些是整数
        int whichInt[] = new int[objValue.length];

        //约束下限
        double rowLower[] = new double[nodes.size()+pathes.size()+supplyStart.length+edges.size()];
        //约束上限
        double rowUpper[] = new double[rowLower.length];
        //约束中非零元系数
        double element[] = new double[cnpathes.size()+pathes.size()*2+pathes.size()+edgepathes.size()];
        //上面系数对应的列
        int column[] = new int[element.length];
        //每一行起始位置
        int starts[] = new int[rowLower.length + 1];
        starts[0] = 0;

        //todo:对上面这些参数赋值
        //求开关最少的次数
        //所有支路的通断状态
        int rowLowerLen = 0, rowUpperLen = 0, elementLen = 0, columnLen = 0, startsLen = 0;
        int[] edgesStatues = new int[edges.size()];
        //路径的通断，即问题中的状态变量。按pathes中的顺序排列
        int[] w = new int[pathes.size()];
        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        int i, j, k, l, offNum = 0;
        //负荷的功率，按nodes中的顺序排列
        double[] loads = new double[nodes.size()]; //todo 读入
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
                offNum++;
                //以该支路两端节点为终点的所有路径的状态变量系数加1
                for(j = 0; j < nodes.size(); j++) {
                    if(nodes.get(j).equals(g.getEdgeSource(edges.get(i))) || nodes.get(j).equals(g.getEdgeTarget(edges.get(i)))) {
                        if(j == nodes.size()-1) {
                            for (k = cnStart[j]; k < cnpathes.size(); k++) {
                                objValue[cnpathesIndex.get(k)]++;
                            }
                        }
                        else {
                            for (k = cnStart[j]; k < cnStart[j + 1]; k++) {
                                objValue[cnpathesIndex.get(k)]++;
                            }
                        }
                    }
                }
            }
            else {
                //以该支路两端节点为终点的所有路径的状态变量系数减1
                for(j = 0; j < nodes.size(); j++) {
                    if(nodes.get(j).equals(g.getEdgeSource(edges.get(i))) || nodes.get(j).equals(g.getEdgeTarget(edges.get(i)))) {
                        if(j == nodes.size()-1) {
                            for (k = cnStart[j]; k < cnpathes.size(); k++) {
                                objValue[cnpathesIndex.get(k)]--;
                            }
                        }
                        else {
                            for (k = cnStart[j]; k < cnStart[j + 1]; k++) {
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
                    column[elementLen++] = cnpathesIndex.get(j);
                }
            }
            else {
                for (j = cnStart[i]; j < cnStart[i + 1]; j++) {
                    element[elementLen++] = 1;
                    column[elementLen++] = cnpathesIndex.get(j);
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
        double[] supplyCapacity = new double[supplyStart.length];   //todo 读入
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
                if(cnStart[l] > k)
                    element[elementLen++] = loads[l-1];
                else
                    element[elementLen++] = loads[cnStart.length-1];
                column[columnLen++] = j;
            }
        }
        //约束条件：每一条供电线路不能过载
        //线容量
        String lastID;
        double[] feederCapacity = new double[edges.size()]; //todo 读入
        for(i = 0; i < edges.size(); i++) {
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = feederCapacity[i];
            rowLower[rowLowerLen++] = 0;
            if(i == edgeStart.length-1)
                endIndex = pathes.size();
            else
                endIndex = edgeStart[i+1];
            for(j = edgeStart[i]; j < endIndex; j++) {
                //找出路径edgepathes[j]的末尾节点
                lastID = g.getEdgeTarget(edgepathes.get(j)[edgepathes.get(j).length-1]).getId();
                if (edgepathes.get(i).length == 1) {
                    for (String scn : supplies) {
                        if (scn.equals(lastID)) {
                            lastID = g.getEdgeSource(pathes.get(i)[pathes.get(i).length - 1]).getId();
                            break;
                        }
                    }
                }
                else {
                    //如果路径上倒数第二条边有节点与lastID相同，则lastID应取最后一条边的另一个端点才是路径上的最后一个点
                    if (lastID.equals(g.getEdgeSource(pathes.get(i)[pathes.get(i).length - 2]).getId()) || lastID.equals(g.getEdgeTarget(pathes.get(i)[pathes.get(i).length - 2]).getId()))
                        lastID = g.getEdgeSource(pathes.get(i)[pathes.get(i).length - 1]).getId();
                }
                for(k = 0; k < nodes.size(); k++)
                    if(lastID == nodes.get(k).getId())
                        break;
                element[elementLen++] = loads[k];
                column[columnLen++] = edgepathesIndex.get(j);
            }
        }


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
