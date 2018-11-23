package zju.dsntp;

import zju.dsmodel.*;

import java.io.*;
import java.util.*;

import static zju.dsmodel.IeeeDsInHand.createDs;

/**
 * 粒子类
 * 求解函数
 *
 * @author
 */
class ParticleInTSC {
    //粒子属性值
    //粒子的位置
    private double[] position;
    //粒子的速度，维数同位置
    private double[] velocity;
    //粒子的适应度
    private double fitness;
    //粒子的历史最好位置
    private double[] pBestPositon;
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

    //计算模型
    LoadTransferOpt loadTransferOpt;
    //负荷名
    LinkedList<String> loadName;

//    //是否完成N-1校验
//    boolean isFinish = false;
//    //是否通过N-1校验
//    boolean isPass = false;
//    //是否计算超时
//    boolean isTooLong = false;
//
//    //计时器
//    TimekeeperThread timekeeperThread;
//
//    CheckN1Task checkN1Task;

    /**
     * 初始化粒子
     *
     * @param dimen 表示粒子的维数,即可安装量测的位置
     * @return
     */
    public void initial(int dimen) throws Exception {
        dimension = dimen;
        position = new double[dimension];
        velocity = new double[dimension];
        fitness = 0;

        pBestPositon = new double[dimension];
        pBestFitness = 0;

        rnd = new Random();

        for (int i = 0; i < dimension; ++i) {
            //内部调参
            position[i] = getRandomPostion();
            pBestPositon[i] = position[i];
            velocity[i] = 0;
        }

        //生成系统
        //todo:
        InputStream ieeeFile = this.getClass().getResourceAsStream("/loadtransferfiles/testcase17/graphtest.txt");

        DistriSys distriSys = createDs(ieeeFile, "S1", 100);
        //设置电源节点，电源名
        //todo:
        String[] supplyID = new String[]{"S1", "S2", "S3", "S4", "S5", "S6"};
        distriSys.setSupplyCns(supplyID);
        //设置电源基准电压
        Double[] supplyBaseKv = new Double[]{100., 100., 100., 100., 100., 100.,};
        distriSys.setSupplyCnBaseKv(supplyBaseKv);

        //新建计算模型
        loadTransferOpt = new LoadTransferOpt(distriSys);


        //设置电源容量
        //todo:
        String supplyCapacityPath = this.getClass().getResource("/loadtransferfiles/testcase17/supplyCapacity.txt").getPath();
        Map<String, Double> supplyCap = readSupplyCapacity(supplyCapacityPath);
        loadTransferOpt.setSupplyCap(supplyCap);

        //设置线路容量
        //todo:
        loadTransferOpt.setFeederCapacityConst(20000);

        //搜索路径
        loadTransferOpt.buildPathes(5000);
    }

    /**
     * 评估函数值,同时记录历史最优位置
     */
    public void evaluateFitness() {
        //适应度的计算公式
        fitness = 0;
        for (double i : position) {
            fitness += i;
        }

        //N-1隐性条件
        //设置负荷
        //读取负荷名
        //todo:
        String loadsPath = this.getClass().getResource("/loadtransferfiles/testcase17/loads.txt").getPath();
        loadName = readLoads(loadsPath);
        Map<String, Double> load = new HashMap<>();
        int count = 0;
        while (!loadName.isEmpty()) {
            load.put(loadName.pop(), position[count++]);
        }
        loadTransferOpt.setLoad(load);

        //N-1校验
        if(!loadTransferOpt.checkN1()){
            fitness = 0;
        }

        //正常运行状态控制
        //todo:
        boolean isTooBig;
        isTooBig = position[0]+position[6]+position[7]>40;
        if (isTooBig){
            fitness= 0;
        }
        isTooBig = position[1]+position[8]+position[9]+position[20]+position[21]+position[32]+position[33]>40;
        if(isTooBig){
            fitness=0;
        }
        isTooBig=position[2]+position[10]+position[11]+position[12]+position[22]+position[23]+position[24]+position[34]+position[35]+position[36]>40;
        if (isTooBig){
            fitness=0;
        }
        isTooBig=position[3]+position[13]+position[14]+position[25]+position[26]+position[37]+position[38]>40;
        if (isTooBig){
            fitness=0;
        }
        isTooBig=position[4]+position[15]+position[16]+position[17]+position[27]+position[28]+position[29]+position[39]+position[40]+position[41]>63;
        if (isTooBig){
            fitness=0;
        }
        isTooBig=position[5]+position[18]+position[19]+position[30]+position[31]+position[42]+position[43]>63;
        if (isTooBig){
            fitness=0;
        }
//        checkN1Task = new CheckN1Task(loadTransferOpt,this);
//        checkN1Task.start();
//
//        timekeeperThread = new TimekeeperThread(200000, new TimeoutException("超时"),this);
//        timekeeperThread.start();
//        //初始设置未结束
//        timekeeperThread.updateTime();
//        isFinish = false;
//        while (!isFinish && !isTooLong) {
//            //System.out.println(".");
//        }
//        //checkN1Task.interrupt();
//        //timekeeperThread.interrupt();
//        //System.out.println("关闭成功");
//
//        //未通过N-1校验
//        if (!isPass) {
//            fitness = 0;
//        }
//        isPass = false;
//
//        //超时
//        if(!isTooLong){
//            fitness = 0;
//        }
//        isTooLong = false;

        if (fitness - pBestFitness > 1e-2) {
            //假如计算出的适应值比历史最优适应值好，则用新计算出的适应值函数替代历史最优适应值
            pBestFitness = fitness;
            for (int i = 0; i < dimension; ++i) {
                pBestPositon[i] = position[i];
            }
        }
    }

    /**
     * 更新速度和位置
     */
    public void updatePosAndVel() {
        for (int j = 0; j < dimension; j++) {
            velocity[j] = w * velocity[j] + c1 * rnd.nextDouble() * (pBestPositon[j] - position[j])
                    + c2 * rnd.nextDouble() * (PSOInTSC.globalBestPosition[j] - position[j]);//速度更新
            position[j] = position[j] + velocity[j] + rnd.nextDouble()*0.5;
            //坐标默认大于等于0
            if (position[j] < 0) {
                position[j] = 0;
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
    double getRandomPostion() {
        //todo:
        return rnd.nextDouble() * 6;
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
                supplyLoad = Double.parseDouble(newdata[1]);
                supplyCap.put(supplyId, supplyLoad);
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("Can't read the file!");
            return null;
        }
        return supplyCap;
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

    public double[] getpBestPositon() {
        return pBestPositon;
    }

    public void setpBestPositon(double[] pBestPositon) {
        this.pBestPositon = pBestPositon;
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
        ParticleInTSC.rnd = rnd;
    }

    public static double getW() {
        return w;
    }

    public static void setW(double w) {
        ParticleInTSC.w = w;
    }

    public static double getC1() {
        return c1;
    }

    public static void setC1(double c1) {
        ParticleInTSC.c1 = c1;
    }

    public static double getC2() {
        return c2;
    }

    public static void setC2(double c2) {
        ParticleInTSC.c2 = c2;
    }
}

//class CheckN1Task extends Thread {
//
//    private LoadTransferOpt loadTransferOpt;
//    private ParticleInTSC particleInTSC;
//
//    CheckN1Task(LoadTransferOpt model,ParticleInTSC particle) {
//        super();
//        loadTransferOpt = model;
//        particleInTSC = particle;
//    }
//
//    @Override
//    public void run() {
//        if (loadTransferOpt.checkN1()) {
//            particleInTSC.isPass = true;
//        } else {
//            particleInTSC.isPass = false;
//        }
//        particleInTSC.isFinish = true;
//    }
//}


