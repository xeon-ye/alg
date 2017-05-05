package zju.dspf;

import org.jgrapht.alg.ConnectivityInspector;
import zju.devmodel.MapObject;
import zju.dsmodel.DsConnectNode;
import zju.dsmodel.DsTopoIsland;
import zju.dsmodel.DsTopoNode;
import zju.dsmodel.Feeder;
import zju.dsntp.DsPowerflow;

import java.util.*;

/**
 * 粒子类
 * 求解函数 f(x)=x1^2+(x2-x3)^2 的最大值
 *
 * @author
 */
class ParticleInReconfig {
    //粒子属性值
    private double[] position;//粒子的位置，求解问题多少维，则此数组为多少维
    private double[] velocity;//粒子的速度，维数同位置
    private double fitness;//粒子的适应度
    private double[] pBestPositon;//粒子的历史最好位置
    private double pBestFitness;//历史最优适应值

    //全局约束
    private static int oneNumber;//粒子为1的个数限制
    private static int dimension;//声明维数

    //公式参数设置
    private static Random rnd;//产生随机变量
    private static double w;//声明权重
    private static double c1;//声明学习系数
    private static double c2;

    //原始拓扑
    private static DsTopoIsland originIsland;
    //计算拓扑
    private DsTopoIsland calIsland;

    //全部节点
    private Map<String, DsTopoNode> tns;

    /**
     * 初始化粒子
     *
     * @param dimension 表示粒子的维数,即可安装量测的位置
     */
    public void initial(int dimension) {
        //
        //tns = DsCase33.createTnMap(originIsland);

//        for(MapObject i : originIsland.getGraph().edgeSet()){
//            System.out.println(originIsland.getGraph().getEdgeSource(i).getConnectivityNodes().get(0).getId()+ " "+
//                    originIsland.getGraph().getEdgeTarget(i).getConnectivityNodes().get(0).getId());
//
//        }

        position = new double[dimension];//开辟存储位置的数组的内存
        velocity = new double[dimension];//开辟存储位置的数组的内存
        fitness = 1e6;//初始粒子适应值是一个1*10的6次方，赋值的过程

        pBestPositon = new double[dimension];//开辟存储历史最优适应值的内存
        pBestFitness = 1e6;//初始历史最优的适应值是1*10的6次方，赋值的过程

        ParticleInReconfig.dimension = dimension;//维数=输入的维数，赋值的过程

        for (int i = 0; i < dimension; ++i) {
            //位置初始化
            position[i] = getRandomPostion();
            //将初始化后的位置赋值给“粒子的历史最好位置”
            pBestPositon[i] = position[i];
            //速度的初始化，取（-1，1）之间的随机数
            velocity[i] = getRandomValue(-1, 1);
        }
        //位置为1的总数
        int sum = 0;
        Random rd = new Random();
        for (double i : position) {
            sum += i;
        }
        //去1操作
        if (sum > this.oneNumber) {
            int[] oneSetPlace = new int[sum];
            int index = 0;
            for (int i = 0; i < ParticleInReconfig.dimension; i++) {
                if (position[i] == 1) {
                    oneSetPlace[index] = i;
                    index++;
                }
            }
            Set toSetZero = new HashSet();
            for (int i = 0; i < (sum - oneNumber); i++) {
                int rdNumber;
                do {
                    rdNumber = oneSetPlace[rd.nextInt(sum)];
                } while (toSetZero.contains(rdNumber));
                toSetZero.add(rdNumber);
            }
            for (Object i : toSetZero) {
                position[(int) i] = 0;
            }
        }else if(sum < this.oneNumber ){
            //补1操作
            int[] zeroSetPlace = new int[this.dimension - sum];
            int index = 0;
            for(int i = 0 ; i < ParticleInReconfig.dimension ; i ++){
                if(position[i] == 0){
                    zeroSetPlace[index] = i;
                    index++;
                }
            }
            Set toSetOne = new HashSet();
            for(int i = 0 ; i < (oneNumber - sum) ; i++){
                int rdNumber;
                do{
                    rdNumber = zeroSetPlace[rd.nextInt(this.dimension - sum)];
                }while(toSetOne.contains(rdNumber));
                toSetOne.add(rdNumber);
            }
            for(Object i : toSetOne){
                position[(int) i] = 1;
            }
        }
    }

    /**
     * 评估函数值,同时记录历史最优位置
     */
    public void evaluateFitness() {
        //todo:修改适应度的计算公式
        fitness = 0;
        calIsland = originIsland.clone();
        calIsland.initialIsland();
        tns = DsCase33.createTnMap(calIsland);
        String[] toOpenBranch = new String[5];
        int index = 0;
        for(int i = 0; i < this.dimension ; i++){
            if(position[i] == 1){
                //idToBranch从1开始
                toOpenBranch[index] = calIsland.getIdToBranch().get(i+1).getName();
                index++;
            }
        }
        for(int i = 0; i < 5 ; i++){
            DsCase33.deleteFeeder(calIsland,tns,toOpenBranch[i]);
        }

        ConnectivityInspector inspector = new ConnectivityInspector<>(calIsland.getGraph());
        List<Set<DsConnectNode>> subGraphs = inspector.connectedSets();
        if(subGraphs.size() > 1){
            fitness = 1e6;
        }else{
            calIsland.initialIsland();
            //System.out.println("开始计算");
            DsPowerflow pf = new DsPowerflow();
            //long start = System.currentTimeMillis();
            pf.setTolerance(1e-1);
            pf.doLcbPf(calIsland);
            //System.out.println("计算潮流用时：" + (System.currentTimeMillis() - start) + "ms.");
            if(pf.isConverged() == true){
                //DsPowerflowTest.printBusV(calIsland, true, false);
                for (MapObject i : calIsland.getBranches().keySet()) {
                    Feeder feeder = (Feeder) calIsland.getBranches().get(i);
                    double[][] Z_real = feeder.getZ_real();

                    double[][] branchHeadI = calIsland.getBranchHeadI().get(i);
                    double loss = 0;
                    for (int j = 0; j < 3; j++) {
                        loss += (branchHeadI[j][0] * branchHeadI[j][0] + branchHeadI[j][1] * branchHeadI[j][1]) * Z_real[j][j];
                    }
                    fitness += loss;
                }
            }else{
                fitness = 1e6;
            }
        }
        if (fitness < pBestFitness) {
            pBestFitness = fitness;//假如计算出的适应值比历史最优适应值好，则用新计算出的适应值函数替代历史最优适应值
            for (int i = 0; i < dimension; ++i) {
                pBestPositon[i] = position[i];
            }//这里是方法中的，记录最优的历史位置，只有历史最优适应值被替代，历史最优位置才会进行更新
        }
    }

    /**
     * 更新速度和位置
     */
    public void updatePosAndVel() {
        for (int j = 0; j < dimension; ++j) {
            velocity[j] = w * velocity[j] + c1 * rnd.nextDouble() * (pBestPositon[j] - position[j])
                    + c2 * rnd.nextDouble() * (DPSOInReconfig.globalBestPosition[j] - position[j]);//速度更新

            double threshold = 1 / (1 + Math.exp(-velocity[j]));//sig函数
            //System.out.println(threshold);
            if (rnd.nextDouble() < threshold) {//若随机数小于速度的sig函数值
                position[j] = 1;//位置等于1
            } else {
                position[j] = 0;//位置等于0
            }
        }

        int sum = 0;
        Random rd = new Random();
        for (double i : position) {
            sum += i;
        }
        //去1操作
        if (sum > this.oneNumber) {
            int[] oneSetPlace = new int[sum];
            int index = 0;
            for (int i = 0; i < dimension; i++) {
                if (position[i] == 1) {
                    oneSetPlace[index] = i;
                    index++;
                }
            }
            Set toSetZero = new HashSet();
            for (int i = 0; i < (sum - oneNumber); i++) {
                int rdNumber;
                do {
                    rdNumber = oneSetPlace[rd.nextInt(sum)];
                } while (toSetZero.contains(rdNumber));
                toSetZero.add(rdNumber);
            }
            for (Object i : toSetZero) {
                position[(int) i] = 0;
            }
        }else if(sum < this.oneNumber ){
            //补1操作
            int[] zeroSetPlace = new int[this.dimension - sum];
            int index = 0;
            for(int i = 0 ; i < ParticleInReconfig.dimension ; i ++){
                if(position[i] == 0){
                    zeroSetPlace[index] = i;
                    index++;
                }
            }
            Set toSetOne = new HashSet();
            for(int i = 0 ; i < (oneNumber - sum) ; i++){
                int rdNumber;
                do{
                    rdNumber = zeroSetPlace[rd.nextInt(this.dimension - sum)];
                }while(toSetOne.contains(rdNumber));
                toSetOne.add(rdNumber);
            }
            for(Object i : toSetOne){
                position[(int) i] = 1;
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

    double getRandomPostion() {
        rnd = new Random();
        if (rnd.nextDouble() < 0.5) {
            return 0;
        } else {
            return 1;
        }
    }//一个方法，关于获取一个随机的位置。

    public static void setDimension(int[] candPos) {
        ParticleInReconfig.dimension = candPos.length;
    }//一个设置维度的方法，粒子类的维度

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

    public static int getOneNumber() {
        return oneNumber;
    }

    public static void setOneNumber(int oneNumber) {
        ParticleInReconfig.oneNumber = oneNumber;
    }

    public static int getDimension() {
        return dimension;
    }

    public static void setDimension(int dimension) {
        ParticleInReconfig.dimension = dimension;
    }

    public static Random getRnd() {
        return rnd;
    }

    public static void setRnd(Random rnd) {
        ParticleInReconfig.rnd = rnd;
    }

    public static double getW() {
        return w;
    }

    public static void setW(double w) {
        ParticleInReconfig.w = w;
    }

    public static double getC1() {
        return c1;
    }

    public static void setC1(double c1) {
        ParticleInReconfig.c1 = c1;
    }

    public static double getC2() {
        return c2;
    }

    public static void setC2(double c2) {
        ParticleInReconfig.c2 = c2;
    }

    public static DsTopoIsland getOriginIsland() {
        return originIsland;
    }

    public static void setOriginIsland(DsTopoIsland originIsland) {
        ParticleInReconfig.originIsland = originIsland;
    }

    public  DsTopoIsland getCalIsland() {
        return calIsland;
    }

    public  Map<String, DsTopoNode> getTns() {
        return tns;
    }

}