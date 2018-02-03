package zju.dsntp;

import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by meditation on 2018/1/25.
 */

class PsoInLine {
    //粒子群
    private ParticleInLine[] particles;
    //全局最优解
    private double globalBestFitness;
    //全局最优解对应线路铺设费用
    private double globalBestLineCost;
    //全局最优解对应网络损耗费用
    private double globalBestLineLossCost;

    //全局最优解对应位置
    private static double[] globalBestPosition;
    //全局最优解历史
    private List<double[]> globalBestPositionList;
    private List<Double> globalBestPositionFitnessList;
    private List<Integer> iterationCountList;
    //粒子的数量
    private int particlesAmount;
    //粒子维度
    private int dimension;

    //路径
    private List<MapObject[]> pathes;

    //配网模型
    DistriSys distriSys;
    LoadTransferOpt loadTransferOpt;

    public void initial(int amount, DistriSys dis) {
        //修改参数
        //类的静态成员的初始化
        ParticleInLine.setC1(2);
        ParticleInLine.setC2(2);
        ParticleInLine.setW(0.8);

        //粒子个数
        particlesAmount = amount;
        particles = new ParticleInLine[particlesAmount];

        distriSys = dis.clone();
        //搜索路径
        loadTransferOpt = new LoadTransferOpt(distriSys);
        loadTransferOpt.buildPathes();
        //粒子维数
        dimension = loadTransferOpt.getPathes().size();
        //路径
        pathes = loadTransferOpt.getPathes();

        //全局最优适应值及其相应值初始化
        globalBestFitness = 1e10;
        globalBestLineCost = 1e10;
        globalBestLineLossCost = 1e10;

        //全局最优位置
        globalBestPosition = new double[dimension];
        globalBestPositionList = new ArrayList<>();
        globalBestPositionFitnessList = new ArrayList<>();
        iterationCountList = new ArrayList<>();

        //最优位置索引
        int index = -1;
        for (int i = 0; i < particlesAmount; ++i) {
            //System.out.println("新建线路粒子" + i);
            particles[i] = new ParticleInLine();
            particles[i].initial(dis);

            //文件路径
            //fixme:改成配置文件的形式
            particles[i].readRoadLength(this.getClass().getResource("/roadplanning/21nodes/roadmessage.txt").getPath());
            particles[i].readSupplyCap(this.getClass().getResource("/roadplanning/21nodes/supplycapacity.txt").getPath());
            particles[i].readLoads(this.getClass().getResource("/roadplanning/21nodes/load.txt").getPath());

            particles[i].evaluateFitness();
            if (globalBestFitness > particles[i].getFitness()) {
                globalBestFitness = particles[i].getFitness();
                index = i;
            }
        }

        //
        if (index == -1) {
            for (int i = 0; i < dimension; i++) {
                globalBestPosition[i] = 1;
            }
        } else {
            for (int i = 0; i < dimension; ++i) {
                globalBestPosition[i] = particles[index].getPosition()[i];
            }
        }
    }

    /**
     * 粒子群的运行
     */
    public void run() {
        int runTimes = 1;
        int index;
        //设置最大迭代次数
        while (runTimes <= 1000) {
            index = -1;
            //每个粒子更新位置和适应值
            for (int i = 0; i < particlesAmount; ++i) {
                particles[i].updatePosAndVel();//更新每个粒子的位置和速度
                particles[i].evaluateFitness();//更新每个粒子的适应值
                if (globalBestFitness - particles[i].getFitness() > 1e-2) {
                    globalBestFitness = particles[i].getFitness();
                    index = i;//更新全局最优适应值，并记录最优的粒子
                }
            }
            //发现更好的解
            if (index != -1) {
                double[] position = new double[dimension];
                for (int i = 0; i < dimension; ++i) {
                    globalBestPosition[i] = particles[index].getPosition()[i];
                    position[i] = particles[index].getPosition()[i];
                }
                //保存最优解历史，始终放在第一位
                globalBestPositionList.add(0, position);
                globalBestPositionFitnessList.add(0, globalBestFitness);
                iterationCountList.add(0, runTimes);

                globalBestLineCost = particles[index].getLineCost();
                globalBestLineLossCost = particles[index].getLineLossCost();
                //打印结果
//                System.out.println("第" + runTimes + "次迭代发现更好解：");
//                System.out.println("globalBestFitness:" + globalBestFitness);
//                for (int i = 0; i < dimension; i++) {
//                    System.out.println("ROAD" + (i + 1) + " = " + getGlobalBestPosition()[i]);
//                }
            }
            runTimes++;
        }

//        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
//        System.out.println(dateFormat.format(new Date()));
    }

    /**
     * 显示程序求解结果
     */
    public void showResult() {
        //打印全局最优解
        System.out.println("全局最大值：" + globalBestFitness);
        System.out.println("全局最大值时坐标：");
        for (int i = 0; i < dimension; i++) {
            System.out.println("ROAD" + (i + 1) + " = " + getGlobalBestPosition()[i]);
        }
        //打印最优解list
        System.out.println("最优解List：");
        for (int i = 0; i < globalBestPositionList.size(); i++) {
            System.out.println("在第" + iterationCountList.get(i) + "次迭代中发现第" + i + "优解：" + globalBestPositionFitnessList.get(i));
            for (int j = 0; j < dimension; j++) {
                System.out.println("ROAD" + (j + 1) + " = " + globalBestPositionList.get(i)[j]);
            }
        }
    }

    public static double[] getGlobalBestPosition() {
        return globalBestPosition;
    }

    public double getGlobalBestFitness() {
        return globalBestFitness;
    }

    public List<MapObject[]> getPathes() {
        return pathes;
    }

    public double getGlobalBestLineCost() {
        return globalBestLineCost;
    }

    public double getGlobalBestLineLossCost() {
        return globalBestLineLossCost;
    }
}