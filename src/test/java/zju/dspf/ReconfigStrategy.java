package zju.dspf;

import Jama.Matrix;
import org.jgrapht.alg.ConnectivityInspector;
import zju.devmodel.MapObject;
import zju.dsmodel.DsConnectNode;
import zju.dsmodel.DsTopoIsland;
import zju.dsmodel.DsTopoNode;
import zju.dsntp.DsPowerflow;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Created by zjq on 2017/4/26.
 */
public class ReconfigStrategy {
    private static Map<String,String> branchesInfo33 = new HashMap<>();
    private static Map<String,String> branchesInfo69 = new HashMap<>();

    private static double maxCurrent = 58;

    public ReconfigStrategy() throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(DsCase33.class.getResource("/other/case33.txt").getPath()));
        String buffer;
        try {
            //读取33节点算例
            buffer = reader.readLine();
            while (buffer!=null){
                buffer = reader.readLine();
                if(buffer.equals("Branches")){
                    break;
                }
            }

            buffer = reader.readLine();
            while (!buffer.equals("-999")){
                String[] messages = buffer.split(" ");
                String branchName = messages[1]+"-"+messages[2];
                branchesInfo33.put(branchName,buffer);
                buffer=reader.readLine();
            }
            branchesInfo33.put("8-21","33 8 21 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
            branchesInfo33.put("9-15","34 9 15 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
            branchesInfo33.put("12-22","35 12 22 2.0 2.0 2.0 2.0 2.0 2.0 0. 0. 0. 0. 0. 0.");
            branchesInfo33.put("18-33","36 18 33 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");
            branchesInfo33.put("25-29","37 25 29 0.5 0.5 0.5 0.5 0.5 0.5 0. 0. 0. 0. 0. 0.");

            //读取69节点算例
            reader = new BufferedReader(new FileReader(DsCase33.class.getResource("/other/case69.txt").getPath()));
            buffer = reader.readLine();
            while (buffer!=null){
                buffer = reader.readLine();
                if(buffer.equals("Branches")){
                    break;
                }
            }

            buffer = reader.readLine();
            while (!buffer.equals("-999")){
                String[] messages = buffer.split(" ");
                String branchName = messages[1]+"-"+messages[2];
                branchesInfo69.put(branchName,buffer);
                buffer=reader.readLine();
            }
            branchesInfo69.put("13-20","69 13 20 0.5000 0.5000 0.5000 0.5000 0.5000 0.5000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
            branchesInfo69.put("15-69","70 15 69 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
            branchesInfo69.put("39-48","71 39 48 2.0000 2.0000 2.0000 2.0000 2.0000 2.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
            branchesInfo69.put("11-66","72 11 66 0.5000 0.5000 0.5000 0.5000 0.5000 0.5000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
            branchesInfo69.put("27-54","73 27 54 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public String getReconfigStrategy(DsTopoIsland calIsland, List<String> toOpenBranches, List<String> toCloseBranches) {
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
            DsCase33.addFeeder(calIsland, tns, branchesInfo69.get(toCloseBranch));

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
                            DsCase33.addFeeder(calIsland, tns, branchesInfo69.get(toOpenBranch));
                        }

                    } else {
                        DsCase33.addFeeder(calIsland, tns, branchesInfo69.get(toOpenBranch));
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
