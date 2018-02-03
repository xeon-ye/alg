package zju.dsntp;

import zju.dsmodel.DsTopoIsland;

import java.io.FileWriter;
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
class PSOInTSC {
    //粒子群
    ParticleInTSC[] particles;
    //全局最优解
    double globalBestFitness;
    //全局最优解对应位置
    static double[] globalBestPosition;
    //全局最优解历史
    List<double[]> globalBestPositionList;
    List<Double> globalBestPositionFitnessList;
    List<Integer> iterationCount;
    //粒子的数量
    int particlesAmount;
    //粒子维度
    int dimension;
    //迭代次数
    int iterationMaxTime;

    /**
     * 粒子群初始化
     * @param amount 粒子数量
     * @param dimen 粒子维数
     * @param maxTime 最大迭代次数
     */
    public void initial(int amount,int dimen, int maxTime) {
        //修改参数
        //类的静态成员的初始化
        ParticleInTSC.setC1(2);
        ParticleInTSC.setC2(2);
        ParticleInTSC.setW(0.8);

        iterationMaxTime = maxTime;

        //粒子个数
        particlesAmount = amount;
        particles = new ParticleInTSC[particlesAmount];
        //粒子维度
        dimension = dimen;
        //全局最优适应值
        globalBestFitness = 0;
        //全局最优位置
        globalBestPosition = new double[dimension];
        globalBestPositionList = new ArrayList<>();
        globalBestPositionFitnessList = new ArrayList<>();
        iterationCount = new ArrayList<>();

        //最优位置索引
        int index = -1;
        for (int i = 0; i < particlesAmount; ++i) {
            System.out.println("新建粒子"+i);
            particles[i] = new ParticleInTSC();
            particles[i].initial(dimension);

            particles[i].evaluateFitness();
            if (globalBestFitness < particles[i].getFitness()) {
                globalBestFitness = particles[i].getFitness();
                index = i;
            }
        }

        for (int i = 0; i < dimension; ++i) {
            globalBestPosition[i] = particles[index].getPosition()[i];
        }
    }

    /**
     * 粒子群的运行
     */
    public void run() throws IOException {
        int runTimes = 1;
        int index;
        //设置最大迭代次数
        while (runTimes <= iterationMaxTime) {
            index = -1;
            //每个粒子更新位置和适应值
            for (int i = 0; i < particlesAmount; ++i) {
                particles[i].updatePosAndVel();
                particles[i].evaluateFitness();
                if (particles[i].getFitness() - globalBestFitness > 1e-2) {
                    globalBestFitness = particles[i].getFitness();
                    index = i;
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
                iterationCount.add(0, runTimes);

                //打印结果
                System.out.println("第" + runTimes + "次迭代发现更好解：");
                System.out.println("globalBestFitness:" + globalBestFitness);
                for (int i = 0; i < dimension; i++) {
                    System.out.println("负荷" + (i+1) + " = " + globalBestPosition[i]);
                }
            }
            runTimes++;
        }

        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
        System.out.println(dateFormat.format(new Date()));
    }

    /**
     * 显示程序求解结果
     */
    public void showResult() {
        //打印全局最优解
        System.out.println("全局最大值：" + globalBestFitness);
        System.out.println("全局最大值时坐标：");
        for (int i = 0; i < dimension; i++) {
            System.out.println("负荷" + i + " = " + getGlobalBestPosition()[i]);
        }
        //打印最优解list
        System.out.println("最优解List：");
        for (int i = 0; i < globalBestPositionList.size(); i++) {
            System.out.println("在第" + iterationCount.get(i) + "次迭代中发现第" + i + "优解：" + globalBestPositionFitnessList.get(i));
            for (int j = 0; j < dimension; j++) {
                System.out.println("负荷" + (j+1) + " = " + globalBestPositionList.get(i)[j]);
            }
        }
    }

    public static double[] getGlobalBestPosition() {
        return globalBestPosition;
    }
}