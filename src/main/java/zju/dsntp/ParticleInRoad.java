package zju.dsntp;

import org.jgrapht.UndirectedGraph;
import org.jgrapht.alg.ConnectivityInspector;
import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.graph.SimpleGraph;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsDevices;

import java.io.*;
import java.util.*;

import static zju.dsmodel.IeeeDsInHand.createDs;

/**
 * 粒子类
 * 求解函数
 *
 * @author meditation
 */
class ParticleInRoad {
    //粒子属性值
    //粒子的位置，对应于graph文件中至上而下的各条廊道
    private double[] position;
    //粒子的速度，维数同位置
    private double[] velocity;
    //粒子的适应度
    private double fitness;
    //廊道成本
    private double roadCost;
    //线路铺设成本
    private double lineCost;
    //网络损耗成本
    private double lineLossCost;

    //粒子的历史最好位置
    private double[] pBestPosition;

    //下层粒子的最好位置
    private double[] pBelowBestPosition;
    //下层路径
    private List<MapObject[]> pathes;

    //历史最优适应值
    private double pBestFitness;

    //维数
    private int dimension;

    //公式参数设置
    //产生随机变量
    private Random rnd;
    //声明权重
    private static double w;
    //声明学习系数
    private static double c1;
    private static double c2;

    //廊道成本
    private Map<String, Double> roadPriceMap;
    //折旧率
    private static final double Depreciation_Rate = 0.03;

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

        //位置初始化
        for (int i = 0; i < dimension; ++i) {
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
    public void evaluateFitness() throws Exception {
        //适应度的计算公式
        fitness = 0;
        //廊道费用
        Iterator iterator = roadPriceMap.keySet().iterator();
        String roadName;
        for (int i = 0; i < dimension; i++) {
            roadName = (String) iterator.next();
            if (position[i] == 1) {
                fitness += roadPriceMap.get(roadName);
            }
        }
        //廊道成本，考虑折旧率
        roadCost = fitness*(1+Depreciation_Rate);
        fitness = roadCost;

        //利用拓扑文件判断图的连通性并判断是否包含必要点
        //todo:设置必须包含的点
//       String[] necessaryNodesArray = {"S1", "L1", "L2", "L3"};
//       String[] necessaryNodesArray = {"S650", "L645", "L632", "L646","L633","L684","L671","L652","L611","L680","L675"};
//        String[] necessaryNodesArray = {"S", "L801", "L802", "L803","L804","L805","L806","L807","L808","L809","L810",
//                                        "L811", "L812", "L813","L814","L815","L816","L817","L818","L819","L820",
//                                        "L821", "L822", "L823","L824","L825","L826","L827","L828","L829","L830","L831"};
        String[] necessaryNodesArray = {"S", "L801", "L802", "L803","L804","L805","L806","L807","L808","L809","L810",
                "L811", "L812", "L813","L814","L815","L816","L817","L818","L819","L820", "L821"};

        //fixme:改成配置文件的形式
        boolean isConnected = judgeConnection(this.getClass().getResource("/roadplanning/21nodes/graph.txt").getPath(), necessaryNodesArray);
        //若连通
        PsoInLine psoInLine = null;
        if (isConnected) {
            //建立路径搜索程序所需的DistriSys对象
            // fixme:改成配置文件的形式
            InputStream file = this.getClass().getResourceAsStream("/roadplanning/21nodes/graph.txt");
            //传入文件输入流，根据position读取支路信息
            DsDevices devices = parse(file);
            //todo:修改
            DistriSys distriSys = createDs(devices, "S", 4.16);

            psoInLine = new PsoInLine();
            //源graph文件默认所有的边均为switch
            //todo:改下层粒子维数
            psoInLine.initial(500, distriSys);
            psoInLine.run();
            //上层适应度值加下层适应度值
            fitness += psoInLine.getGlobalBestFitness();
            //更新线路成本
            lineCost = psoInLine.getGlobalBestLineCost();
            //更新网络损耗成本
            lineLossCost = psoInLine.getGlobalBestLineLossCost();
            //更新路径
            pathes = psoInLine.getPathes();
        } else {
            fitness = 1e10;
        }

        //假如计算出的适应值比历史最优适应值好，则用新计算出的适应值函数替代历史最优适应值，同时更新最优位置和对应的下层位置
        if (pBestFitness - fitness > 1e-2) {
            pBestFitness = fitness;
            for (int i = 0; i < dimension; ++i) {
                pBestPosition[i] = position[i];
            }

            pBelowBestPosition = new double[PsoInLine.getGlobalBestPosition().length];
            for(int i =0; i<pBelowBestPosition.length;i++){
                pBelowBestPosition[i]=PsoInLine.getGlobalBestPosition()[i];
            }
        }
    }

    /**
     * 更新速度和位置
     */
    public void updatePosAndVel() {
        for (int j = 0; j < dimension; j++) {
            velocity[j] = w * velocity[j] + c1 * rnd.nextDouble() * (pBestPosition[j] - position[j])
                    + c2 * rnd.nextDouble() * (PsoInRoad.getGlobalBestPosition()[j] - position[j]);//速度更新
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
     * 连通性检测，并判断是否包含必要点。必要点包括电源点、真负荷点。
     *
     * @param filePath  文件路径
     * @param neceNodes 必须包含的点
     * @return isConnected
     */
    private boolean judgeConnection(String filePath, String[] neceNodes) {
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
        //维数游标
        int count = 0;
        try {
            //读起始行
            while (!(data = bufferedReader.readLine()).startsWith("Line Segment Data")) {
            }
            //读至“-999”
            while (!(data = bufferedReader.readLine()).equals("-999")) {
                //建造廊道
                if (position[count] == 1) {
                    count++;
                    String[] dataArray = data.split("\t");
                    boolean isExist1 = false;
                    boolean isExist2 = false;
                    Node node1 = null;
                    Node node2 = null;
                    //判断点是否已经存在
                    for (Node i : undirectedGraph.vertexSet()) {
                        if (i.getId().equals(dataArray[0])) {
                            isExist1 = true;
                            //存在直接赋值
                            node1 = i;
                        } else if (i.getId().equals(dataArray[1])) {
                            isExist2 = true;
                            //存在直接赋值
                            node2 = i;
                        }
                    }
                    //不存在则新建并添加
                    if (!isExist1) {
                        node1 = new Node(dataArray[0]);
                        undirectedGraph.addVertex(node1);
                    }
                    //不存在则新建并添加
                    if (!isExist2) {
                        node2 = new Node(dataArray[1]);
                        undirectedGraph.addVertex(node2);
                    }
                    //添加边
                    undirectedGraph.addEdge(node1, node2);
                    //不建造廊道
                } else {
                    count++;
                    continue;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Can not read the File!");
        }

        //调用库方法，对undirectedGraph进行连通性检测
        ConnectivityInspector inspector = new ConnectivityInspector(undirectedGraph);
        //如果不连通返回false
        if (!inspector.isGraphConnected()) {
            return false;
        }

        for (String j : neceNodes) {
            boolean isExist = false;
            for (Node i : undirectedGraph.vertexSet()) {
                if (i.getId().equals(j)) {
                    isExist = true;
                    break;
                }
            }
            //任意一个必要点不被包含，则返回false
            if (!isExist) {
                return false;
            }
        }
        return true;
    }

    /**
     * 传入文件输入流，根据position读取支路信息。
     *
     * @param stream graph文件输入流
     * @return dsDevices 返回筛选后的设备信息
     */
    public DsDevices parse(InputStream stream) {
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String strLine;
        try {
            //起始行
            strLine = reader.readLine();
            while (!strLine.startsWith("Line Segment Data")) {
                strLine = reader.readLine();
            }
            String[] strings = strLine.split("\t");
            String s = strings[1];
            //取出数值，数值即为拓扑支路数量
            ArrayList<MapObject> branches = new ArrayList<MapObject>(Integer.parseInt(s.substring(0, s.indexOf(" "))));
            //支路游标
            int count = 0;
            //设置支路属性，添加支路
            while (true) {
                strLine = reader.readLine();
                if (strLine.trim().equalsIgnoreCase("-999"))
                    break;
                //空行跳过。#开头行为注释行。
                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;
                //建设廊道
                if (position[count] == 1) {
                    count++;
                    String content[] = strLine.split("\t");
                    //存储边属性
                    MapObject obj = new MapObject();
                    obj.setProperty("ConnectedNode", content[0] + ";" + content[1]);
                    obj.setProperty("LineLength", content[2]);
                    obj.setProperty("LineConfigure", content[3]);
                    obj.setProperty("ResourceType", "Feeder");
                    obj.setProperty("LengthUnit", strings[2]);
                    branches.add(obj);
                    //不建设廊道
                } else {
                    count++;
                    continue;
                }
            }

            //Spot Loads
            while (!strLine.startsWith("Spot Loads"))
                strLine = reader.readLine();
            ArrayList<MapObject> spotLoads = new ArrayList<MapObject>(0);

            //Distributed Loads
            while (!strLine.startsWith("Distributed Loads"))
                strLine = reader.readLine();
            ArrayList<MapObject> distributedLoads = new ArrayList<MapObject>(0);

            //Shunt Capacitors
            while (!strLine.startsWith("Shunt Capacitors"))
                strLine = reader.readLine();
            ArrayList<MapObject> shuntCapacitors = new ArrayList<MapObject>(0);

            int tfCount = 0;
            int switchCount = 0;
            //配置为Switch的branch为开关；否则，长度为0的边为变压器。
            for (MapObject obj : branches) {
                String length = obj.getProperty("LineLength");
                if (obj.getProperty("LineConfigure").trim().equalsIgnoreCase("Switch")) {
                    switchCount++;
                } else if (Double.parseDouble(length) < 1e-5) {
                    tfCount++;
                }
            }
            List<MapObject> transformers = new ArrayList<MapObject>(tfCount);
            List<MapObject> switches = new ArrayList<MapObject>(switchCount);
            for (MapObject obj : branches) {
                String length = obj.getProperty("LineLength");
                if (obj.getProperty("LineConfigure").trim().equalsIgnoreCase("Switch")) {
                    //将类型Feeder改为类型Switch
                    obj.setProperty("ResourceType", "Switch");
                    switches.add(obj);
                } else if (Double.parseDouble(length) < 1e-5) {
                    //将类型Feeder改为类型Transformer
                    obj.setProperty("ResourceType", "Transformer");
                    transformers.add(obj);
                }
            }

            //Transformer
            while (!strLine.startsWith("Transformer"))
                strLine = reader.readLine();

            //Regulator
            while (!strLine.startsWith("Regulator"))
                strLine = reader.readLine();
            ArrayList<MapObject> regulators = new ArrayList<MapObject>(0);

            //Distributed Generation
            while (strLine != null && !strLine.startsWith("Distributed Generation"))
                strLine = reader.readLine();
            ArrayList<MapObject> dgs = new ArrayList<MapObject>(0);

            //-999
            while (strLine != null && !strLine.startsWith("-999"))
                strLine = reader.readLine();

            branches.removeAll(transformers);
            branches.removeAll(switches);
            DsDevices dsDevices = new DsDevices();
            dsDevices.setSpotLoads(spotLoads);
            dsDevices.setDistributedLoads(distributedLoads);
            dsDevices.setShuntCapacitors(shuntCapacitors);
            dsDevices.setFeeders(branches);
            dsDevices.setTransformers(transformers);
            dsDevices.setSwitches(switches);
            dsDevices.setRegulators(regulators);
            dsDevices.setDispersedGens(dgs);

            return dsDevices;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
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

    public double[] getpBelowBestPosition() {
        return pBelowBestPosition;
    }

    public List<MapObject[]> getPathes() {
        return pathes;
    }

    public double getRoadCost() {
        return roadCost;
    }

    public double getLineCost() {
        return lineCost;
    }

    public double getLineLossCost() {
        return lineLossCost;
    }
}

//todo:当前为临时解决方法
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

