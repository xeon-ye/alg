package zju.pso;

/**
 * @Description: 粒子群算法的核心
 * @Author: Fang Rui
 * @Date: 2018/6/7
 * @Time: 16:58
 */

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Vector;

public class PsoProcess implements PsoConstants {
    private Vector<Particle> swarm = new Vector<>();
    private double[] pBest; // 各粒子个体最优适应度值
    private Vector<Location> pBestLocation = new Vector<>(); // 各粒子个体最优位置
    private double gBest; // 全局最优适应度值
    private Location gBestLocation; // 全局最优位置
    private double[] fitnessValueList; // 各粒子当前适应度值
    private boolean[] feasibleList; // 各粒子是否在可行域内
    private boolean hasFeasibleOne; // 当前是否存在粒子在可行域内
    private double[] maxViolation; // 本次迭代过程中的各不等式最大偏差

    private OptModel optModel;
    private final int swarmSize;

    Random generator = new Random();

    public PsoProcess(OptModel optModel, int swarmSize) {
        this.optModel = optModel;
        this.swarmSize = swarmSize;
        this.fitnessValueList = new double[swarmSize];
        this.feasibleList = new boolean[swarmSize];
        this.pBest = new double[swarmSize];
        this.maxViolation = new double[optModel.getDimentions()];
        this.hasFeasibleOne = false;
    }

    public void execute() {
        initializeSwarm();

        int iterNum = 0;
        int n = optModel.getDimentions();
        int maxIter = optModel.getMaxIter();

        double err = 9999;
        double w; // 惯性权重

        while (iterNum < maxIter && err > optModel.getTol()) {

            w = W_UPPERBOUND - (((double) iterNum) / maxIter) * (W_UPPERBOUND - W_LOWERBOUND); // 惯性逐渐减小

            for (int i = 0; i < swarmSize; i++) {
                double r1 = generator.nextDouble();
                double r2 = generator.nextDouble();

                Particle p = swarm.get(i);

                // 步骤一：更新速度
                double[] newVel = new double[n];
                for (int j = 0; j < n; j++) {
                    newVel[j] = (w * p.getVelocity().getVel()[j]) +
                            (r1 * C1) * (pBestLocation.get(i).getLoc()[j] - p.getLocation().getLoc()[j]) +
                            (r2 * C2) * (gBestLocation.getLoc()[j] - p.getLocation().getLoc()[j]);
                }
                Velocity vel = new Velocity(newVel);
                p.setVelocity(vel);

                // 步骤二：更新位置
                double[] newLoc = new double[n];
                for (int j = 0; j < n; j++) {
                    newLoc[j] = p.getLocation().getLoc()[j] + newVel[j];
                }
                Location loc = new Location(newLoc);
                p.setLocation(loc);
            }

            // 步骤三：更新pBest
            for (int i = 0; i < swarmSize; i++) {
                // FIXME: 2018/6/8 未完成
                // 先计算约束
                double[] constrValueList = optModel.evalConstr(swarm.get(i).getLocation());
                if (PsoUtil.isFeasible(constrValueList)) {

                    if (fitnessValueList[i] < pBest[i]) {
                        pBest[i] = fitnessValueList[i];
                        pBestLocation.set(i, swarm.get(i).getLocation());
                    }
                }
            }

            // 步骤四：更新gBest
            int bestParticleIndex = PsoUtil.getMinPos(fitnessValueList);
            if (iterNum == 0 || fitnessValueList[bestParticleIndex] < gBest) {
                gBest = fitnessValueList[bestParticleIndex];
                gBestLocation = swarm.get(bestParticleIndex).getLocation();
            }

            err = optModel.evalObj(gBestLocation) - 0; // minimizing the functions means it's getting closer to 0

            System.out.println("ITERATION " + iterNum + ": ");
            System.out.println("     Best X: " + gBestLocation.getLoc()[0]);
            System.out.println("     Best Y: " + gBestLocation.getLoc()[1]);
            System.out.println("     Value: " + optModel.evalObj(gBestLocation));

            iterNum++;
            updateFitnessList();
        }

        System.out.println("\nSolution found at iteration " + (iterNum - 1) + ", the solutions is:");
        System.out.println("     Best X: " + gBestLocation.getLoc()[0]);
        System.out.println("     Best Y: " + gBestLocation.getLoc()[1]);
    }

    // 初始化粒子群
    private void initializeSwarm() {
        Particle p;
        List<double[]> constrViolationList = new ArrayList<>();
        // 在粒子已知可行范围内初始化粒子群
        int n = optModel.getDimentions();
        double[] minLoc = optModel.getMinLoc();
        double[] maxLoc = optModel.getMaxLoc();
        double[] minVel = optModel.getMinVel();
        double[] maxVel = optModel.getMaxVel();

        for (int i = 0; i < swarmSize; i++) {
            p = new Particle(optModel);

            double[] loc = new double[n];
            double[] vel = new double[n];

            // 随机化粒子的位置和速度
            for (int j = 0; j < n; j++) {
                loc[j] = minLoc[j] + generator.nextDouble() * (maxLoc[j] - minLoc[j]);
                // TODO: 2018/6/8 粒子速度问题
                vel[j] = minVel[j] + generator.nextDouble() * (maxVel[j] - minVel[j]);
            }
            Location location = new Location(loc);
            Velocity velocity = new Velocity(vel);

            p.setLocation(location);
            p.setVelocity(velocity);
            swarm.add(p);

            // 获得约束向量
            double[] constrViolation = optModel.evalConstr(location);
            constrViolationList.add(constrViolation);
            // 如果是满足约束的粒子
            if (PsoUtil.isFeasible(constrViolation)) {
                feasibleList[i] = true;
                fitnessValueList[i] = p.getFitnessValue();
                pBest[i] = fitnessValueList[i];
                if (hasFeasibleOne) {
                    if (gBest > pBest[i]) {
                        gBest = pBest[i];
                        gBestLocation = swarm.get(i).getLocation();
                    }
                } else {
                    gBest = pBest[i];
                    gBestLocation = swarm.get(i).getLocation();
                    hasFeasibleOne = true;
                }

            } else {
                feasibleList[i] = false;
                PsoUtil.maxViolationArray(maxViolation, constrViolation, n);
            }
            // 初始化时当前状态是该粒子的最好位置
            pBestLocation.add(swarm.get(i).getLocation());
        }
        for (int i = 0; i < swarmSize; i++) {
            // 如果粒子不在可行域内就计算适应度值
            if (!feasibleList[i]) {
                fitnessValueList[i] = PsoUtil.violationFitness(maxViolation, constrViolationList.get(i), n);
                pBest[i] = fitnessValueList[i];
            }
        }
        // 如果每一个粒子都不在可行域内，要找到一个gBest
        if (!hasFeasibleOne) {
            // TODO: 2018/6/8 断言
            for (int i = 0; i < swarmSize; i++) {
                assert !feasibleList[i];
            }
            int bestParticleIndex = PsoUtil.getMinPos(fitnessValueList);
            gBest = fitnessValueList[bestParticleIndex];
            gBestLocation = swarm.get(bestParticleIndex).getLocation();
        }
    }

    // 更新粒子群的适应度值
    public void updateFitnessList() {
        for (int i = 0; i < swarmSize; i++) {
            fitnessValueList[i] = swarm.get(i).getFitnessValue();
        }
    }
}
