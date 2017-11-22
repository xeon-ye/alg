package zju.dspf;

import Jama.Matrix;
import org.jgrapht.alg.ConnectivityInspector;
import zju.devmodel.MapObject;
import zju.dsmodel.DsConnectNode;
import zju.dsmodel.DsTopoIsland;
import zju.dsmodel.DsTopoNode;
import zju.dsntp.DsPowerflow;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Created by zjq on 2017/4/26.
 */
public class ReconfigStrategy {
    private static Map<String, String> branchesInfo = new HashMap<>();
    private static double maxCurrent = 58;

    static {
        branchesInfo.put("1-2", "1 1 2 0.0935 0.0477 0.0933 0.0475 0.0931 0.0474 0.0009 0.0004 0.0013 0.0007 0.0011 0.0005");
        branchesInfo.put("2-3", "2 2 3 0.5003 0.2548 0.4989 0.2541 0.4979 0.2536 0.0049 0.0025 0.0073 0.0037 0.0059 0.0030");
        branchesInfo.put("3-4", "3 3 4 0.3714 0.1891 0.3704 0.1886 0.3696 0.1882 0.0036 0.0018 0.0054 0.0027 0.0043 0.0022");
        branchesInfo.put("4-5", "4 4 5 0.3868 0.1970 0.3856 0.1964 0.3849 0.1960 0.0038 0.0019 0.0057 0.0029 0.0045 0.0023");
        branchesInfo.put("5-6", "5 5 6 0.8312 0.7176 0.8288 0.7154 0.8271 0.7140 0.0081 0.0070 0.0122 0.0106 0.0098 0.0084");
        branchesInfo.put("6-7", "6 6 7 0.1900 0.6280 0.1894 0.6262 0.1890 0.6249 0.0018 0.0061 0.0028 0.0092 0.0022 0.0074");
        branchesInfo.put("7-8", "7 7 8 0.7220 0.2386 0.7199 0.2379 0.7185 0.2374 0.0071 0.0023 0.0106 0.0035 0.0085 0.0028");
        branchesInfo.put("8-9", "8 8 9 1.0454 0.7510 1.0423 0.7488 1.0403 0.7473 0.0103 0.0074 0.0154 0.0110 0.0123 0.0088");
        branchesInfo.put("9-10", "9 9 10 1.0596 0.7510 1.0565 0.7488 1.0544 0.7473 0.0104 0.0074 0.0156 0.0110 0.0125 0.0088");
        branchesInfo.put("10-11", "10 10 11 0.1995 0.0659 0.1989 0.0657 0.1985 0.0656 0.0019 0.0006 0.0029 0.0009 0.0023 0.0007");
        branchesInfo.put("11-12", "11 11 12 0.3800 0.1256 0.3788 0.1252 0.3781 0.1250 0.0037 0.0012 0.0056 0.0018 0.0044 0.0014");
        branchesInfo.put("12-13", "12 12 13 1.4900 1.1723 1.4856 1.1688 1.4826 1.1665 0.0146 0.0115 0.0220 0.0173 0.0176 0.0138");
        branchesInfo.put("13-14", "13 13 14 0.5497 0.7235 0.5480 0.7214 0.5470 0.7200 0.0054 0.0071 0.0081 0.0106 0.0064 0.0085");
        branchesInfo.put("14-15", "14 14 15 0.5998 0.5338 0.5980 0.5323 0.5969 0.5312 0.0059 0.0052 0.0088 0.0078 0.0070 0.0063");
        branchesInfo.put("15-16", "15 15 16 0.7514 0.5531 0.7491 0.5515 0.7477 0.5504 0.0074 0.0054 0.0111 0.0081 0.0088 0.0065");
        branchesInfo.put("16-17", "16 16 17 1.3083 1.7468 1.3044 1.7416 1.3018 1.7382 0.0128 0.0172 0.0193 0.0258 0.0154 0.0206");
        branchesInfo.put("17-18", "17 17 18 0.7429 0.5826 0.7407 0.5808 0.7393 0.5797 0.0073 0.0057 0.0109 0.0086 0.0087 0.0068");
        branchesInfo.put("2-19", "18 2 19 0.1664 0.1588 0.1659 0.1583 0.1656 0.1580 0.0016 0.0015 0.0024 0.0023 0.0019 0.0018");
        branchesInfo.put("19-20", "19 19 20 1.5267 1.3757 1.5222 1.3716 1.5192 1.3689 0.0150 0.0135 0.0225 0.0203 0.0180 0.0162");
        branchesInfo.put("20-21", "20 20 21 0.4156 0.4855 0.4144 0.4841 0.4135 0.4831 0.0040 0.0047 0.0061 0.0071 0.0049 0.0057");
        branchesInfo.put("21-22", "21 21 22 0.7195 0.9513 0.7174 0.9485 0.7159 0.9466 0.0070 0.0093 0.0106 0.0140 0.0085 0.0112");
        branchesInfo.put("3-23", "22 3 23 0.4579 0.3129 0.4566 0.3119 0.4557 0.3113 0.0045 0.0030 0.0067 0.0046 0.0054 0.0036");
        branchesInfo.put("23-24", "23 23 24 0.9114 0.7197 0.9087 0.7176 0.9069 0.7161 0.0089 0.0070 0.0134 0.0106 0.0107 0.0085");
        branchesInfo.put("24-25", "24 24 25 0.9094 0.7116 0.9067 0.7095 0.9049 0.7081 0.0089 0.0070 0.0134 0.0105 0.0107 0.0084");
        branchesInfo.put("6-26", "25 6 26 0.2060 0.1049 0.2054 0.1046 0.2050 0.1044 0.0020 0.0010 0.0030 0.0015 0.0024 0.1044");
        branchesInfo.put("26-27", "26 26 27 0.2884 0.1468 0.2876 0.1464 0.2870 0.1461 0.0028 0.0014 0.0042 0.0021 0.0034 0.0017");
        branchesInfo.put("27-28", "27 27 28 1.0748 0.9477 1.0717 0.9449 1.0695 0.9430 0.0105 0.0093 0.0158 0.0140 0.0127 0.0112");
        branchesInfo.put("28-29", "28 28 29 0.8162 0.7111 0.8138 0.7090 0.8122 0.7076 0.0080 0.0070 0.0120 0.0105 0.0096 0.0084");
        branchesInfo.put("29-30", "29 29 30 0.5151 0.2623 0.5135 0.2616 0.5125 0.2610 0.0050 0.0025 0.0076 0.0038 0.0060 0.0031");
        branchesInfo.put("30-31", "30 30 31 0.9890 0.9774 0.9860 0.9745 0.9841 0.9726 0.0097 0.0096 0.0146 0.0144 0.0116 0.0115");
        branchesInfo.put("31-32", "31 31 32 0.3151 0.3637 0.3142 0.3662 0.3136 0.3655 0.0031 0.0036 0.0046 0.0054 0.0037 0.0043");
        branchesInfo.put("32-33", "32 32 33 0.3461 0.5381 0.3450 0.5365 0.3444 0.5355 0.0034 0.0053 0.0051 0.0079 0.0040 0.0063");
        branchesInfo.put("8-21", "33 8 21 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        branchesInfo.put("9-15", "34 9 15 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        branchesInfo.put("12-22", "35 12 22 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
        branchesInfo.put("18-33", "36 18 33 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
        branchesInfo.put("25-29", "37 25 29 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
    }

    public static String getReconfigStrategy(DsTopoIsland calIsland, List<String> toOpenBranches, List<String> toCloseBranches) {
        String strategy = null;
        String toCloseBranch = null;
        String toOpenBranch = null;
        //配网拓扑点的Map
        Map<String, DsTopoNode> tns = DsCase33.createTnMap(calIsland);


        //合环校验
        for (int i = 0; i < toCloseBranches.size(); i++) {
            //提取边的两端节点编号
            toCloseBranch = toCloseBranches.get(i);
//            String[] nodeNames = toCloseBranch.split("-");
//            int[] nodes = new int[2];
//            //nodename从1开始编号，故减1
//            nodes[0] = Integer.parseInt(nodeNames[0]) - 1;
//            nodes[1] = Integer.parseInt(nodeNames[1]) - 1;

            //增加线路
            DsCase33.addFeeder(calIsland, tns, branchesInfo.get(toCloseBranch));

            calIsland.initialIsland();
            DsPowerflow pf = new DsPowerflow();
            pf.setTolerance(1e-1);
            pf.doLcbPf(calIsland);

            //获取线路对应的key
            MapObject branchKey = null;
            for (MapObject j : calIsland.getBranches().keySet()) {
                if (j.getName().equals(toCloseBranch)) {
                    branchKey = j;
                    break;
                }
            }
            //得稳态电流
            double[][] branchI = calIsland.getBranchHeadI().get(branchKey);

            double[] surgeCurrentAmp = new double[3];
            for (int j = 0; j < 3; j++) {
                surgeCurrentAmp[j] = 2 * Math.sqrt(branchI[j][0] * branchI[j][0] + branchI[j][1] * branchI[j][1]);
            }

            System.out.println("尝试闭合："+toCloseBranch);
            System.out.println("A相冲击电流幅值："+surgeCurrentAmp[0]);
            System.out.println("B相冲击电流幅值："+surgeCurrentAmp[1]);
            System.out.println("C相冲击电流幅值："+surgeCurrentAmp[2]);
            System.out.println("平均冲击电流幅值："+(surgeCurrentAmp[0]+surgeCurrentAmp[2]+surgeCurrentAmp[2])/3);
//            //计算三相电压差
//            double[][] nodeV1 = null;
//            double[][] nodeV2 = null;
//            //电压差
//            double[][] nodeVDifference = new double[3][2];
//            for (DsTopoNode dsTopoNode : calIsland.getBusV().keySet()) {
//                if (dsTopoNode.getConnectivityNodes().get(0).getId().equals(nodeNames[0])) {
//                    nodeV1 = calIsland.getBusV().get(dsTopoNode);
//                } else if (dsTopoNode.getConnectivityNodes().get(0).getId().equals(nodeNames[1])) {
//                    nodeV2 = calIsland.getBusV().get(dsTopoNode);
//                }
//            }
//            for (int j = 0; j < 3; j++) {
//                nodeVDifference[j][0] = nodeV1[j][0] - nodeV2[j][0];
//                nodeVDifference[j][1] = nodeV1[j][1] - nodeV2[j][1];
//            }
//
//            //阻抗矩阵
//            int size = calIsland.getGraph().vertexSet().size();
//            Matrix[][] zMatrix = calZMatrix(size,yMatrix.getyMatrix());
//
//                yMatrix.print(5,2);
//
//
//            for(int j = 0 ; j<3;j++){
//                System.out.println("实部");
//                zMatrix[j][0].print(5,2);
//                System.out.println("虚部");
//                zMatrix[j][1].print(5,2);
//            }
//
//            //戴维南等效阻抗
//            double[][] z = new double[3][2];
//            for (int j = 0; j < 3; j++) {
//                z[j][0] = zMatrix[j][0].get(nodes[0], nodes[0]) + zMatrix[j][0].get(nodes[1], nodes[1])
//                        - 2 * zMatrix[j][0].get(nodes[0],nodes[1]);
//                z[j][1] = zMatrix[j][1].get(nodes[0],  nodes[0] ) + zMatrix[j][1].get(nodes[1], nodes[1])
//                        - 2 * zMatrix[j][1].get(nodes[0], nodes[1]);
//            }
//
//            //冲击电流
//            double[] surgeCurrentAmp = new double[3];
//            for (int j = 0; j < 3; j++) {
//                double vAmp = Math.sqrt(nodeVDifference[j][0] * nodeVDifference[j][0] + nodeVDifference[j][1] * nodeVDifference[j][1]);
//                double zAmp = Math.sqrt(z[j][0] * z[j][0] + z[j][1] * z[j][1]);
//                surgeCurrentAmp[j] = vAmp / zAmp;
//            }

            if (surgeCurrentAmp[0] < maxCurrent && surgeCurrentAmp[1] < maxCurrent && surgeCurrentAmp[2] < maxCurrent) {
                System.out.println(toCloseBranch+"合环校验通过");
                //默认肯定可以找到对应的断开开关,且有多种断开的可能方式
                for (int j = 0; j < toOpenBranches.size(); j++) {
                    toOpenBranch = toOpenBranches.get(j);
                    DsCase33.deleteFeeder(calIsland, tns, toOpenBranch);
                    //检查线路辐射状
                    ConnectivityInspector inspector = new ConnectivityInspector<>(calIsland.getGraph());
                    List<Set<DsConnectNode>> subGraphs = inspector.connectedSets();
                    if (subGraphs.size() == 1) {
                        //跳出找断开开关
                        //break;
                        System.out.println("尝试断开：" + toOpenBranch+"\n");
                        //去除列表中的内容
                        toCloseBranches.remove(toCloseBranch);
                        toOpenBranches.remove(toOpenBranch);

                        //判断是否为递归结束情况
                        if (toCloseBranches.size() == 0 && toOpenBranches.size() == 0) {
                            return "闭合" + toCloseBranch + "断开" + toOpenBranch;
                        } else {
                            strategy = getReconfigStrategy(calIsland.clone(), toOpenBranches, toCloseBranches);
                        }

                        //判断子递归是否有解
                        if (strategy != null) {
                            return "闭合" + toCloseBranch + "断开" + toOpenBranch + strategy;
                        }else {
                            //还原
                            System.out.println("尝试失败!\n");
                            toCloseBranches.add(i, toCloseBranch);
                            toOpenBranches.add(j, toOpenBranch);
                            DsCase33.addFeeder(calIsland, tns, branchesInfo.get(toOpenBranch));
                        }

                    } else {
                        DsCase33.addFeeder(calIsland, tns, branchesInfo.get(toOpenBranch));
                    }
                }
                DsCase33.deleteFeeder(calIsland, tns, toCloseBranch);
            } else {
                System.out.println(toCloseBranch+"冲击电流幅值过大，合环校验不通过！\n");
                DsCase33.deleteFeeder(calIsland, tns, toCloseBranch);
            }
        }
        return strategy;
    }

    public static Matrix[][] calZMatrix(int size, Matrix[] yMatrix) {
        Matrix[][] zMatrix = new Matrix[3][2];
        for (int j = 0; j < 3; j++) {
            Matrix y_real = new Matrix(size, size);
            Matrix y_imag = new Matrix(size, size);
            for (int row = 0; row < size; row++) {
                for (int column = 0; column < size; column++) {
                    y_real.set(row, column, yMatrix[j].get(row, 2 * column));
                    y_imag.set(row, column, yMatrix[j].get(row, 2 * column + 1));
                }
            }
            Matrix temp = y_real.plus(y_imag.times(y_real.inverse()).times(y_imag)).inverse();
            zMatrix[j][0] = temp.copy();
            temp = y_real.inverse().times(y_imag).times(temp).times(-1);
            zMatrix[j][1] = temp.copy();
        }
        return zMatrix;
    }

}
