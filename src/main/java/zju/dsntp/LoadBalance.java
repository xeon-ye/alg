package zju.dsntp;

import jpscpu.LinearSolver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jgrapht.UndirectedGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsConnectNode;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import static zju.dsmodel.DsModelCons.KEY_SWITCH_STATUS;
import static zju.dsmodel.DsModelCons.SWITCH_OFF;
import static zju.dsmodel.IeeeDsInHand.createDs;

public class LoadBalance extends LoadTransferOpt {
    private static Logger log = LogManager.getLogger(LoadTransferOpt.class);

    private static double LOAD_RATE_UPPER = 0.816;
    private static double LOAD_RATE_LOWER = 0.596;

    public LoadBalance(DistriSys sys) {
        super(sys);
    }

    public void calculate() {
        long time;

        int endIndex;
        int i, j, k, l;

        //开始构造线性规划模型
        //状态变量是所有路径的通断状态，加上将LW替换为Z后增加的变量，加上一个节点的负荷
        //objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数。
        double objValue[] = new double[pathes.size()];
        double objConstValue = edges.size();

        //状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double columnLower[] = new double[objValue.length];
        //状态变量上限
        double columnUpper[] = new double[objValue.length];
        //指明哪些是整数
        int whichInt[] = new int[pathes.size()];

        //约束下限
        double rowLower[] = new double[nodes.size() + (pathes.size() - supplyCnNum) + supplyStart.length + edges.size()];
        //约束上限
        double rowUpper[] = new double[rowLower.length];
        //约束中非零元系数
        double element[] = new double[cnpathes.size() + (pathes.size() - supplyCnNum) * 2 + pathes.size() + edgepathes.size()];
        //上面系数对应的列
        int column[] = new int[element.length];
        //每一行起始位置
        int starts[] = new int[rowLower.length + 1];

        //所有支路的通断状态
        int[] edgesStatues = new int[edges.size()];

        UndirectedGraph<DsConnectNode, MapObject> g = sys.getOrigGraph();
        //电源容量
        String[] supplies = sys.getSupplyCns();
        double[] supplyCapacity = new double[supplyStart.length];
        maxLoad = Double.MAX_VALUE;

        //求节点可带的最大负荷
        //数组游标
        int rowLowerLen = 0, rowUpperLen = 0, elementLen = 0, columnLen = 0, startsLen = 0;

        //负荷的功率，按nodes中的顺序排列
        double[] loadArray = new double[nodes.size()];
        for (i = 0; i < nodes.size(); i++) {
            loadArray[i] = this.load.get(nodes.get(i).getId());
        }

        for (i = 0; i < supplyCapacity.length; i++) {
            supplyCapacity[i] = this.supplyCap.get(supplies[i]);
        }

        //所有电源容量之和
        double LMax = 0;
        for (i = 0; i < supplyCapacity.length; i++)
            LMax += supplyCapacity[i];

        //所有路径通断状态变量上限为1，下限为0，都是整数
        for (i = 0; i < pathes.size(); i++) {
            columnLower[i] = 0;
            columnUpper[i] = 1;
            whichInt[i] = i;
        }

        //初始化所有支路为通
        for (i = 0; i < edges.size(); i++)
            edgesStatues[i] = 1;
        //得到当前所有支路的通断，1通，0断
        for (i = 0; i < edges.size(); i++) {
            if (edges.get(i).getProperty(KEY_SWITCH_STATUS).equals(SWITCH_OFF)) {
                edgesStatues[i] = 0;
                objConstValue--;
            }
        }
        //设置状态变量的系数值
        for (i = 0; i < edgesStatues.length; i++) {
            if (edgesStatues[i] == 0) {
                //以该支路两端节点为终点且通过该支路的所有路径的状态变量系数加1
                for (j = 0; j < nodes.size(); j++) {
                    if (nodes.get(j).equals(g.getEdgeSource(edges.get(i))) || nodes.get(j).equals(g.getEdgeTarget(edges.get(i)))) {
                        //遍历所有以j节点为终点的cnpathes
                        if (j == nodes.size() - 1) {
                            for (k = cnStart[j]; k < cnpathes.size(); k++) {
                                if (cnpathes.get(k)[cnpathes.get(k).length - 1].equals(edges.get(i)))
                                    objValue[cnpathesIndex.get(k)]++;
                            }
                        } else {
                            for (k = cnStart[j]; k < cnStart[j + 1]; k++) {
                                if (cnpathes.get(k)[cnpathes.get(k).length - 1].equals(edges.get(i)))
                                    objValue[cnpathesIndex.get(k)]++;
                            }
                        }
                    }
                }
            } else {
                //以该支路两端节点为终点且通过该支路的所有路径的状态变量系数减1
                for (j = 0; j < nodes.size(); j++) {
                    if (nodes.get(j).equals(g.getEdgeSource(edges.get(i))) || nodes.get(j).equals(g.getEdgeTarget(edges.get(i)))) {
                        if (j == nodes.size() - 1) {
                            for (k = cnStart[j]; k < cnpathes.size(); k++) {
                                if (cnpathes.get(k)[cnpathes.get(k).length - 1].equals(edges.get(i)))
                                    objValue[cnpathesIndex.get(k)]--;
                            }
                        } else {
                            for (k = cnStart[j]; k < cnStart[j + 1]; k++) {
                                if (cnpathes.get(k)[cnpathes.get(k).length - 1].equals(edges.get(i)))
                                    objValue[cnpathesIndex.get(k)]--;
                            }
                        }
                    }
                }
            }
        }


        //约束条件：对每个负荷节点，有且只有一条路径供电
        for (i = 0; i < nodes.size(); i++) {
            starts[startsLen++] = elementLen;
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
        for (k = 0; k < supplyStart.length; k++) {
            if (k == supplyStart.length - 1)
                endIndex = pathes.size();
            else
                endIndex = supplyStart[k + 1];
            for (i = supplyStart[k] + 1; i < endIndex; i++) {
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
                } else {
                    while (pathes.get(i).length <= pathes.get(j).length) {
                        j--;
                        if (j < 0) {
                            //i、j路径长度均为1
                            lenEqualOne = true;
                            break;
                        }
                    }
                    if (lenEqualOne) {
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

        //约束条件：由某一电源供电的所有负荷功率之和应小于电源容量乘以负载率约束
        for (i = 0; i < supplyStart.length; i++) {
            starts[startsLen++] = elementLen;
            rowUpper[rowUpperLen++] = LOAD_RATE_UPPER * supplyCapacity[i];
            rowLower[rowLowerLen++] = LOAD_RATE_LOWER * supplyCapacity[i];
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
                element[elementLen++] = loadArray[l - 1];
                column[columnLen++] = j;
            }
        }

        //约束条件：每一条供电线路不能过载
        //线容量
        String lastID;
        double[] feederCapacity = new double[edges.size()];
        //设置线路容量
        for (i = 0; i < edges.size(); i++) {
//            MapObject edge = edges.get(i);
//            String target = g.getEdgeTarget(edge).getId();
//            String source = g.getEdgeSource(edge).getId();

            feederCapacity[i] = feederCapacityConst;

//            for(String key : edgeCap.keySet()){
//                if(key.equals(target+";"+source)||key.equals(source+";"+target)){
//                    feederCapacity[i] = edgeCap.get(key);
//                    break;
//                }
//            }
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
                //路径只有一条边
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
                element[elementLen++] = loadArray[k];
                column[columnLen++] = edgepathesIndex.get(j);
            }
        }

        starts[startsLen++] = elementLen;

        int numberRows = rowLower.length;
        int numberColumns = columnLower.length;
        double result[] = new double[numberColumns];
        //求解
        LinearSolver solver = new LinearSolver();
        solver.setDrive(LinearSolver.MLP_DRIVE_CBC);

        time = System.currentTimeMillis();
        int status = solver.solveMlp(numberColumns, numberRows, objValue,
                columnLower, columnUpper, rowLower, rowUpper, element, column, starts, whichInt, result);


        double resultValue = 0;
        for (i = 0; i < result.length; i++) {
            resultValue += result[i] * objValue[i];
        }


        //打印时间
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        System.out.println(dateFormat.format(new Date()));
        System.out.println("求解耗时：" + (System.currentTimeMillis() - time));


        if (status < 0) {
            log.warn("计算不收敛");
        } else {
            log.info("计算收敛");
            System.out.println("最优值：" + (resultValue + objConstValue));
            //打印路径
            for (i = 0; i < result.length; i++) {
                if (Math.abs(result[i] - 1) < 0.001) {
                    System.out.print("闭合路径：");
                    List<MapObject[]> tempList = new ArrayList<>();
                    tempList.add(pathes.get(i));
                    printPathes(tempList);
                }
            }
        }
    }

}
