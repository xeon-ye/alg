package zju.dsntp;

import org.jgrapht.UndirectedGraph;
import org.jgrapht.alg.ConnectivityInspector;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DsTopoNode;

import java.io.*;
import java.util.*;

/**
 * 粒子类
 * 求解函数
 *
 * @author
 */
class ParticleInRoad {
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

    //廊道成本
    private Map<String, Double> roadPriceMap;

    /**
     * 初始化粒子
     *
     * @param dimen 表示粒子的维数,即可安装量测的位置
     */
    public void initial(int dimen) {
        dimension = dimen;
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
     * 读取廊道价格
     *
     * @param filePath 廊道价格文件路径
     */
    public void readRoadPrice(String filePath) {
        BufferedReader bufferedReader = null;
        try {
            bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("File not found!");
        }

        roadPriceMap = new LinkedHashMap<>();
        String data;
        String[] dataArray;
        try {
            while ((data = bufferedReader.readLine()) != null) {
                dataArray = data.split(" ");
                roadPriceMap.put(dataArray[0], Double.parseDouble(dataArray[2]));
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

        Iterator iterator = roadPriceMap.keySet().iterator();
        String roadName;
        for (int i = 0; i < dimension; i++) {
            roadName = (String) iterator.next();
            if (position[i] == 1) {
                fitness += roadPriceMap.get(roadName);
            }
        }

        //利用拓扑文件判断图的连通性
        boolean isConnected = judgeConnection(this.getClass().getResource("/roadplanning/graph.txt").getPath());
        //若连通
        if(isConnected){
        //todo:




        }else {
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
     * 读取指定路径文件的负荷数据
     *
     * @param path
     * @return load
     */
    private LinkedList<String> readLoads(String path) {
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("Can't find the file!");
            return null;
        }
        //行数据
        String data;
        String[] newdata;
        LinkedList<String> load = new LinkedList<>();

        try {
            while ((data = br.readLine()) != null) {
                newdata = data.split(" ", 2);
                load.add(newdata[0]);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Can't read the file!");
            return null;
        }
        return load;
    }

    /**
     * 读取电源容量
     *
     * @param path
     * @return supplyCapacity
     */
    public Map<String, Double> readSupplyCapacity(String path) {
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(path)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("Can't find the file!");
            return null;
        }
        String data;
        String[] newdata;
        String supplyId;
        Double supplyLoad;
        Map<String, Double> supplyCap = new HashMap<>();

        try {
            while ((data = br.readLine()) != null) {
                newdata = data.split(" ", 2);
                supplyId = newdata[0];
                supplyLoad = new Double(Double.parseDouble(newdata[1]));
                supplyCap.put(supplyId, supplyLoad);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Can't read the file!");
            return null;
        }
        return supplyCap;
    }

    /**
     * 连通性检测
     * @param filePath
     * @return isConnected
     */
    private boolean judgeConnection(String filePath) {
        boolean result = false;
        BufferedReader bufferedReader = null;
        try {
            bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(filePath)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("File Not Found!");
        }
        //网络拓扑图
        UndirectedGraph<Node, DefaultEdge> undirectedGraph = new SimpleGraph<>(DefaultEdge.class);

        String data;
        try {
            //读起始行
            while (!(data = bufferedReader.readLine()).startsWith("Line Segment Data")) {
            }
            //读至“-999”
            while (!(data = bufferedReader.readLine()).equals("-999")) {
                String[] dataArray = data.split("\t");
                boolean isExist1 = false;
                boolean isExist2 = false;
                Node node1 = null;
                Node node2 = null;
                //判断点是否已经存在
                for (Node i : undirectedGraph.vertexSet()) {
                    if (i.getId().equals(dataArray[1])) {
                        isExist1 = true;
                        //存在直接赋值
                        node1 = i;
                    } else if (i.getId().equals(dataArray[2])) {
                        isExist2 = true;
                        //存在直接赋值
                        node2 = i;
                    }
                }
                //不存在则新建并添加
                if (!isExist1) {
                    node1 = new Node(dataArray[1]);
                    undirectedGraph.addVertex(node1);
                }
                //不存在则新建并添加
                if (!isExist2) {
                    node2 = new Node(dataArray[2]);
                    undirectedGraph.addVertex(node2);
                }
                //添加边
                undirectedGraph.addEdge(node1,node2);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Can not read the File!");
        }
        //连通性检测
        ConnectivityInspector inspector = new ConnectivityInspector(undirectedGraph);
        result = inspector.isGraphConnected();
        return result;
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
        ParticleInRoad.rnd = rnd;
    }

    public static double getW() {
        return w;
    }

    public static void setW(double w) {
        ParticleInRoad.w = w;
    }

    public static double getC1() {
        return c1;
    }

    public static void setC1(double c1) {
        ParticleInRoad.c1 = c1;
    }

    public static double getC2() {
        return c2;
    }

    public static void setC2(double c2) {
        ParticleInRoad.c2 = c2;
    }
}

//fixme:临时解决方法
class Node {
    String Id;

    public Node(String id) {
        Id = id;
    }

    public String getId() {
        return Id;
    }

    public void setId(String id) {
        Id = id;
    }
}

