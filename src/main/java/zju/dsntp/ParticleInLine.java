package zju.dsntp;

import org.jgrapht.UndirectedGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsConnectNode;

import java.io.*;
import java.util.*;

/**
 * 粒子类
 * 求解函数
 *
 * @author
 */
class ParticleInLine {
    //粒子属性值
    //粒子的位置
    private double[] position;
    //粒子的速度，维数同位置
    private double[] velocity;
    //粒子的适应度
    private double fitness;
    //粒子的历史最好位置
    private double[] pBestPosition;
    //历史最优适应值
    private double pBestFitness;

    //维数
    private int dimension;

    //公式参数设置
    //产生随机变量
    private static Random rnd;
    //声明权重
    private static double w;
    //声明学习系数
    private static double c1;
    private static double c2;

    //廊道系统
    private DistriSys distriSys;
    private LoadTransferOpt loadTransferOpt;
    //虚拟负荷点数
    private int virtualLoadsNumber;

    //廊道长度
    private Map<String, Double> roadLengthMap;

    //电源容量
    private Map<String, Double> supplyCapMap;
    //负荷功率
    private Map<String, Double> loadsMap;

    //线路容量常值
    private static final double FEEDER_CAPACITY_CONST = 100000;

    /**
     * 初始化粒子
     *
     * @param dis 表示粒子的维数,即可安装量测的位置
     * @param number 表示虚拟负荷点的数目
     */
    public void initial(DistriSys dis,int number) {
        distriSys = dis.clone();
        virtualLoadsNumber = number;
        //搜索路径
        loadTransferOpt = new LoadTransferOpt(distriSys);
        loadTransferOpt.buildPathes();
        //粒子维数
        dimension = loadTransferOpt.getPathes().size();

        position = new double[dimension];
        velocity = new double[dimension];
        fitness = 1e10;

        pBestPosition = new double[dimension];
        pBestFitness = 1e10;

        rnd = new Random();

        for (int i = 0; i < dimension; ++i) {
            //位置初始化
            position[i] = getRandomPostion();
            //将初始化后的位置赋值给“粒子的历史最好位置”
            pBestPosition[i] = position[i];
            //速度的初始化0
            velocity[i] = 0;
        }
    }

    /**
     * 读取廊道长度
     *
     * @param filePath 廊道价格文件路径
     */
    public void readRoadLength(String filePath) {
        BufferedReader bufferedReader = null;
        try {
            bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("File not found!");
        }

        roadLengthMap = new LinkedHashMap<>();
        String data;
        String[] dataArray;
        try {
            while ((data = bufferedReader.readLine()) != null) {
                dataArray = data.split(" ");
                roadLengthMap.put(dataArray[0], Double.parseDouble(dataArray[1]));
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error occurred in reading!");
        }
    }


    /**
     * 读取电源容量
     *
     * @param filePath 电源容量文件路径
     */
    public void readSupplyCap(String filePath) {
        BufferedReader bufferedReader = null;
        try {
            bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("File not found!");
        }

        supplyCapMap = new LinkedHashMap<>();
        String data;
        String[] dataArray;
        try {
            while ((data = bufferedReader.readLine()) != null) {
                dataArray = data.split(" ");
                supplyCapMap.put(dataArray[0], Double.parseDouble(dataArray[1]));
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error occurred in reading!");
        }
    }

    /**
     * 读取负荷
     * @param filePath 负荷文件路径
     */
    public void readLoads(String filePath) {
        BufferedReader bufferedReader = null;
        try {
            bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("File not found!");
        }

        loadsMap = new LinkedHashMap<>();
        String data;
        String[] dataArray;
        try {
            while ((data = bufferedReader.readLine()) != null) {
                dataArray = data.split(" ");
                loadsMap.put(dataArray[0], Double.parseDouble(dataArray[1]));
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Error occurred in reading!");
        }
    }


    /**
     * 评估函数值,同时记录历史最优位置
     */
    public void evaluateFitness() {
        //适应度的计算公式
        fitness = 0;

        Iterator iterator = roadLengthMap.keySet().iterator();
        String roadName;
        for (int i = 0; i < dimension; i++) {
            roadName = (String) iterator.next();
            if (position[i] == 1) {
                fitness += roadLengthMap.get(roadName);
            }
        }

        //todo:



        //传入路径状态信息，判断该路径状态是否满足
        boolean isPassed = judgeFeasibility(position);
        if(!isPassed){
            fitness = 1e10;
        }

        if (pBestFitness - fitness > 1e-2) {
            //假如计算出的适应值比历史最优适应值好，则用新计算出的适应值函数替代历史最优适应值
            pBestFitness = fitness;
            for (int i = 0; i < dimension; ++i) {
                pBestPosition[i] = position[i];
            }
        }
    }

    /**
     * 更新速度和位置
     */
    public void updatePosAndVel() {
        for (int j = 0; j < dimension; j++) {
            velocity[j] = w * velocity[j] + c1 * rnd.nextDouble() * (pBestPosition[j] - position[j])
                    + c2 * rnd.nextDouble() * (PSOInTSC.globalBestPosition[j] - position[j]);//速度更新
            double threshold = 1 / (1 + Math.exp(-velocity[j]));
            if (rnd.nextDouble() < threshold) {//若随机数小于速度的sig函数值
                position[j] = 1;//位置等于1
            } else {
                position[j] = 0;//位置等于0
            }
        }
    }

    /**
     * 返回low—uper之间的数
     *
     * @param low  下限
     * @param uper 上限
     * @return 返回low—uper之间的数
     */
    double getRandomValue(double low, double uper) {//一个方法
        rnd = new Random();//一个对象
        return rnd.nextDouble() * (uper - low) + low;//返回一个随机数
    }

    /**
     * 生成一个随机坐标
     */
    private double getRandomPostion() {
        double i = rnd.nextDouble();
        if (i < 0.5) {
            return 0;
        } else {
            return 1;
        }
    }

    /**
     * 根据传入的路径状态信息，判断该种路径通断状态是否具备可行性
     * @param routeState
     * @return 可行性校验结果
     */
    private boolean judgeFeasibility(double[] routeState) {
        //电源顺序按照supplyCns中的存放顺序
        String[] suppliesArray = distriSys.getSupplyCns();
        //电源容量
        double[] supplyCapacityArray = new double[supplyCapMap.size()];
        for (int i = 0; i < supplyCapacityArray.length; i++) {
            supplyCapacityArray[i] = supplyCapMap.get(suppliesArray[i]);
        }

        //负荷的功率，按nodes中的顺序排列
        double[] loadsArray = new double[loadTransferOpt.getNodes().size()];
        for (int i = 0; i < loadTransferOpt.getNodes().size(); i++) {
            loadsArray[i] = loadsMap.get(loadTransferOpt.getNodes().get(i).getId());
        }

        UndirectedGraph<DsConnectNode, MapObject> g = distriSys.getOrigGraph();

        //约束条件：对每个实际负荷节点，有且只有一条路径供电
        double sum;
        for (int i = 0; i < (loadTransferOpt.getNodes().size()-virtualLoadsNumber); i++) {
            //路径状态的累加和
            sum = 0;
            if (i == loadTransferOpt.getNodes().size() - 1) {
                for (int j = loadTransferOpt.getCnStart()[i]; j < loadTransferOpt.getCnPathes().size(); j++) {
                    sum += routeState[loadTransferOpt.getCnpathesIndex().get(j)];
                }
            } else {
                for (int j = loadTransferOpt.getCnStart()[i]; j < loadTransferOpt.getCnStart()[i + 1]; j++) {
                    sum += routeState[loadTransferOpt.getCnpathesIndex().get(j)];
                }
            }
            if (sum != 1) {
                return false;
            }
        }

        //约束条件：对每个虚拟负荷节点，无任何路径供电
        for (int i = (loadTransferOpt.getNodes().size()-virtualLoadsNumber); i < loadTransferOpt.getNodes().size(); i++) {
            //路径状态的累加和
            sum = 0;
            if (i == loadTransferOpt.getNodes().size() - 1) {
                for (int j = loadTransferOpt.getCnStart()[i]; j < loadTransferOpt.getCnPathes().size(); j++) {
                    sum += routeState[loadTransferOpt.getCnpathesIndex().get(j)];
                }
            } else {
                for (int j = loadTransferOpt.getCnStart()[i]; j < loadTransferOpt.getCnStart()[i + 1]; j++) {
                    sum += routeState[loadTransferOpt.getCnpathesIndex().get(j)];
                }
            }
            if (sum != 0) {
                return false;
            }
        }


        //约束条件：若某路径为通路，那么包括在该路径内的任意路径也是通路
        //对pathes进行类似深度搜索的方式实现
        boolean lenEqualOne;
        int endIndex;
        for (int k = 0; k < loadTransferOpt.getSupplyStart().length; k++) {
            if (k == loadTransferOpt.getSupplyStart().length - 1)
                endIndex = loadTransferOpt.getPathes().size();
            else
                endIndex = loadTransferOpt.getSupplyStart()[k + 1];
            for (int i = loadTransferOpt.getSupplyStart()[k] + 1; i < endIndex; i++) {
                lenEqualOne = false;
                int j = i - 1;
                if (loadTransferOpt.getPathes().get(i).length > loadTransferOpt.getPathes().get(j).length) {
                    //若某路径是通路，而包含路径不是，返回false
                    if (routeState[j] < routeState[i]) {
                        return false;
                    }
                } else {
                    while (loadTransferOpt.getPathes().get(i).length <= loadTransferOpt.getPathes().get(j).length) {
                        j--;
                        if (j < 0) {
                            //i、j路径长度均为1
                            lenEqualOne = true;
                            break;
                        }
                    }
                    if (lenEqualOne) {
                        continue;
                    }
                    //若某路径是通路，而包含路径不是，返回false
                    if (routeState[j] < routeState[i]) {
                        return false;
                    }
                }
            }
        }

        //约束条件：由某一电源供电的所有负荷功率之和应小于电源容量
        for (int i = 0; i < loadTransferOpt.getSupplyStart().length; i++) {
            //找电源路径的末尾索引
            if (i == loadTransferOpt.getSupplyStart().length - 1)
                endIndex = loadTransferOpt.getPathes().size();
            else
                endIndex = loadTransferOpt.getSupplyStart()[i + 1];
            double loadSum = 0;
            for (int j = loadTransferOpt.getSupplyStart()[i]; j < endIndex; j++) {
                //如果该路径为通
                if (routeState[j] == 1) {
                    //找出路径在负荷路径中对应的序号
                    int k;
                    for (k = 0; k < loadTransferOpt.getCnpathesIndex().size(); k++) {
                        if (loadTransferOpt.getCnpathesIndex().get(k) == j)
                            break;
                    }
                    //找出第k条负荷路径对应的负荷点+1
                    int l;
                    for (l = 1; l < loadTransferOpt.getCnStart().length; l++) {
                        if (loadTransferOpt.getCnStart()[l] > k)
                            break;
                    }
                    loadSum += loadsArray[l - 1];
                }
            }
            if (loadSum > supplyCapacityArray[i]) {
                return false;
            }
        }

        //约束条件：每一条供电线路不能过载
        //线路容量
        String lastID;
        double[] feederCapacityArray = new double[loadTransferOpt.getEdges().size()];
        //设置线路容量
        for (int i = 0; i < feederCapacityArray.length; i++) {
            feederCapacityArray[i] = FEEDER_CAPACITY_CONST;
        }
        //第i条边的容量约束
        for (int i = 0; i < loadTransferOpt.getEdges().size(); i++) {
            if (i == loadTransferOpt.getEdgeStart().length - 1)
                endIndex = loadTransferOpt.getEdgePathes().size();
            else
                endIndex = loadTransferOpt.getEdgeStart()[i + 1];
            double loadSum = 0;
            for (int j = loadTransferOpt.getEdgeStart()[i]; j < endIndex; j++) {
                //若路径状态为通，负荷累加
                if (routeState[loadTransferOpt.getEdgepathesIndex().get(j)] == 1) {
                    //找出路径edgepathes[j]的末尾节点
                    lastID = g.getEdgeTarget(loadTransferOpt.getEdgePathes().get(j)[loadTransferOpt.getEdgePathes().get(j).length - 1]).getId();
                    //路径只有一条边
                    if (loadTransferOpt.getEdgePathes().get(j).length == 1) {
                        for (String scn : distriSys.getSupplyCns()) {
                            if (scn.equals(lastID)) {
                                lastID = g.getEdgeSource(loadTransferOpt.getEdgePathes().get(j)[loadTransferOpt.getEdgePathes().get(j).length - 1]).getId();
                                break;
                            }
                        }
                    } else {
                        //如果路径上倒数第二条边有节点与lastID相同，则lastID应取最后一条边的另一个端点才是路径上的最后一个点
                        if (lastID.equals(g.getEdgeSource(loadTransferOpt.getEdgePathes().get(j)[loadTransferOpt.getEdgePathes().get(j).length - 2]).getId())
                                || lastID.equals(g.getEdgeTarget(loadTransferOpt.getEdgePathes().get(j)[loadTransferOpt.getEdgePathes().get(j).length - 2]).getId()))
                            lastID = g.getEdgeSource(loadTransferOpt.getEdgePathes().get(j)[loadTransferOpt.getEdgePathes().get(j).length - 1]).getId();
                    }

                    int k;
                    for (k = 0; k < loadTransferOpt.getNodes().size(); k++) {
                        if (loadTransferOpt.getNodes().get(k).getId().equals(lastID))
                            break;
                    }

                    loadSum += loadsArray[k];
                }
            }
            if (loadSum > feederCapacityArray[i]) {
                return false;
            }
        }

        //通过所有的校验
        return true;
    }


    public double[] getPosition() {
        return position;
    }

    public void setPosition(double[] position) {
        this.position = position;
    }

    public double[] getVelocity() {
        return velocity;
    }

    public void setVelocity(double[] velocity) {
        this.velocity = velocity;
    }

    public double getFitness() {
        return fitness;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    public double[] getpBestPosition() {
        return pBestPosition;
    }

    public void setpBestPosition(double[] pBestPosition) {
        this.pBestPosition = pBestPosition;
    }

    public double getpBestFitness() {
        return pBestFitness;
    }

    public void setpBestFitness(double pBestFitness) {
        this.pBestFitness = pBestFitness;
    }

    public static Random getRnd() {
        return rnd;
    }

    public static void setRnd(Random rnd) {
        ParticleInLine.rnd = rnd;
    }

    public static double getW() {
        return w;
    }

    public static void setW(double w) {
        ParticleInLine.w = w;
    }

    public static double getC1() {
        return c1;
    }

    public static void setC1(double c1) {
        ParticleInLine.c1 = c1;
    }

    public static double getC2() {
        return c2;
    }

    public static void setC2(double c2) {
        ParticleInLine.c2 = c2;
    }
}



