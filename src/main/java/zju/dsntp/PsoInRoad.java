package zju.dsntp;

import zju.devmodel.MapObject;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * 粒子群类
 *
 * @author
 */
class PsoInRoad {
    //粒子群
    private ParticleInRoad[] particles;
    //全局最优解
    private double globalBestFitness;
    //全局最优解对应位置
    private static double[] globalBestPosition;
    //全局最优解对应的下层路径状态
    private static double[] globalBestBelowPositon;
    //全局最优解历史
    private List<double[]> globalBestPositionList;
    private List<Double> globalBestPositionFitnessList;
    //全局最优解对应廊道成本历史
    private List<Double> globalBestRoadCostList;
    //全局最优解对应线路成本历史
    private List<Double> globalBestLineCostList;
    //全局最优解对应网络损耗成本历史
    private List<Double> globalBestLineLossCostList;

    //迭代次数
    private List<Integer> iterationCountList;
    private List<double[]> globalBestBelowPositionList;
    private List<List<MapObject[]>> pathesList;

    //粒子的数量
    private int particlesAmount;
    //粒子维度
    private int dimension;

    public void initial(int amount, int dimen) throws Exception {
        //修改参数
        //类的静态成员的初始化
        ParticleInRoad.setC1(2);
        ParticleInRoad.setC2(2);
        ParticleInRoad.setW(0.8);

        //粒子个数
        particlesAmount = amount;
        particles = new ParticleInRoad[particlesAmount];
        //粒子维度
        dimension = dimen;
        //全局最优适应值
        globalBestFitness = 1e10;
        //全局最优位置
        globalBestPosition = new double[dimension];
        globalBestPositionList = new ArrayList<>();
        globalBestPositionFitnessList = new ArrayList<>();
        globalBestRoadCostList = new ArrayList<>();
        globalBestLineCostList = new ArrayList<>();
        globalBestLineLossCostList = new ArrayList<>();
        iterationCountList = new ArrayList<>();
        globalBestBelowPositionList = new ArrayList<>();
        pathesList = new ArrayList<>();

        //最优位置索引
        int index = -1;
        for (int i = 0; i < particlesAmount; ++i) {
            System.out.println("新建廊道粒子" + i);
            particles[i] = new ParticleInRoad();
            //初始化
            particles[i].initial(dimension);
            //fixme:配置形式
            particles[i].readRoadPrice(this.getClass().getResource("/roadplanning/21nodes/roadmessage.txt").getPath());
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
    public void run() throws Exception {
        int runTimes = 1;
        int index;
        //设置最大迭代次数
        while (runTimes <= 200) {
            System.out.println("迭代次数："+runTimes);
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

                //每次都新建，添加至list
                globalBestBelowPositon = new double[particles[index].getpBelowBestPosition().length];
                for (int i = 0; i < particles[index].getpBelowBestPosition().length; i++) {
                    globalBestBelowPositon[i] = particles[index].getpBelowBestPosition()[i];
                }

                //保存最优解历史，始终放在第一位
                globalBestPositionList.add(0, position);
                globalBestPositionFitnessList.add(0, globalBestFitness);
                globalBestRoadCostList.add(0,particles[index].getRoadCost());
                globalBestLineCostList.add(0,particles[index].getLineCost());
                globalBestLineLossCostList.add(0,particles[index].getLineLossCost());
                iterationCountList.add(0, runTimes);

                globalBestBelowPositionList.add(0, globalBestBelowPositon);
                pathesList.add(0,particles[index].getPathes());

                //打印结果
                System.out.println("第" + runTimes + "次迭代发现更好解：");
                System.out.println("globalBestFitness:" + globalBestFitness);
                for (int i = 0; i < dimension; i++) {
                    System.out.println("ROAD" + (i + 1) + " = " + getGlobalBestPosition()[i]);
                }
            }
            runTimes++;
        }

        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        System.out.println("运行结束 " + dateFormat.format(new Date()));
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
            System.out.println("在第" + iterationCountList.get(i) + "次迭代中发现第" + i + "优解：" + globalBestPositionFitnessList.get(i)
                    + " (Road: "+globalBestRoadCostList.get(i)+" Line: "+globalBestLineCostList.get(i)+" LineLoss: "+globalBestLineLossCostList.get(i)+")");
            for (int j = 0; j < dimension; j++) {
                System.out.println("ROAD" + (j + 1) + " = " + globalBestPositionList.get(i)[j]);
            }
            for (int j = 0; j < globalBestBelowPositionList.get(i).length; j++) {
                System.out.println("ROUTE" + (j + 1) + " = " + globalBestBelowPositionList.get(i)[j]);
            }
            for (MapObject[] j : pathesList.get(i)) {
                System.out.print("路径：");
                for (MapObject k : j) {
                    System.out.print(k.getProperty("ConnectedNode") + "-");
                }
                System.out.println();
            }

        }
    }

    public static double[] getGlobalBestPosition() {
        return globalBestPosition;
    }
}