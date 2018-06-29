package zju.pso;

import java.util.*;

/**
 * 粒子群算法的核心
 *
 * @Author: Fang Rui
 * @Date: 2018/6/7
 * @Time: 16:58
 */

public class PsoProcess implements PsoConstants {
    private Vector<Particle> swarm = new Vector<>();
    private double[] pBest; // 各粒子个体最优适应度值
    private Vector<Location> pBestLocation = new Vector<>(); // 各粒子个体最优位置
    private double gBest; // 全局最优适应度值
    private Location gBestLocation; // 全局最优位置
    private double[] fitnessValueList; // 各粒子当前适应度值
    private boolean[] feasibleList; // 各粒子是否曾经在可行域内
    private boolean[] currentFeasibleList; // 各粒子本次迭代是否在可行域内
    private boolean hasFeasibleOne = false; // 当前是否存在粒子在可行域内
    private boolean isGBestfeasible = false; // gBest是否在可行域内
    private double[] maxViolation; // 历史最大偏差

    private OptModel optModel;
    private final int swarmSize;

    private Random generator = new Random();

    public PsoProcess(OptModel optModel, int swarmSize) {
        this.optModel = optModel;
        this.swarmSize = swarmSize;
        this.fitnessValueList = new double[swarmSize];
        this.feasibleList = new boolean[swarmSize];
        this.currentFeasibleList = new boolean[swarmSize];
        this.pBest = new double[swarmSize];
    }

    public void execute() {
        initializeSwarm();

        int iterNum = 0;
        int n = optModel.getDimentions();
        int maxIter = optModel.getMaxIter();
        double[] minLoc = optModel.getMinLoc();
        double[] maxLoc = optModel.getMaxLoc();
        double[] minVel = optModel.getMinVel();
        double[] maxVel = optModel.getMaxVel();


        double tol = 99999;
        double w; // 惯性权重

        while (iterNum < maxIter && tol >= 0) {

            w = W_UPPERBOUND - (((double) iterNum) / maxIter) * (W_UPPERBOUND - W_LOWERBOUND); // 惯性逐渐减小

            hasFeasibleOne = false; // 先把hasFeasibleOne默认赋值为false
            for (int i = 0; i < swarmSize; i++) {
                double r1 = generator.nextDouble();
                double r2 = generator.nextDouble();

                Particle p = swarm.get(i);

                // 步骤一：更新速度
                double[] newVel = new double[n];
                for (int j = 0; j < n; j++) {
                    double tempVel = (w * p.getVelocity().getVel()[j]) +
                            (r1 * C1) * (pBestLocation.get(i).getLoc()[j] - p.getLocation().getLoc()[j]) +
                            (r2 * C2) * (gBestLocation.getLoc()[j] - p.getLocation().getLoc()[j]);
                    newVel[j] = PsoUtil.restrictByBoundary(tempVel, maxVel[j], minVel[j]);
                }
                Velocity vel = new Velocity(newVel);
                p.setVelocity(vel);

                // 步骤二：更新位置
                double[] newLoc = new double[n];
                for (int j = 0; j < n; j++) {
                    double previousLoc = p.getLocation().getLoc()[j];
                    double tempLoc = previousLoc + newVel[j];
                    newLoc[j] = PsoUtil.restrictByBoundary(tempLoc, maxLoc[j], minLoc[j], previousLoc);
                }
                Location loc = new Location(newLoc);
                p.setLocation(loc);

                // 步骤三：更新pBest
                // 先计算约束
                double[] constrViolation = optModel.evalConstr(loc);
                loc.setConstrViolation(constrViolation); // 在该位置下，对应唯一的不等式约束值

                boolean isFeasible = PsoUtil.isFeasible(constrViolation);
                currentFeasibleList[i] = isFeasible;
                // 当前粒子进入可行域但是之前不在可行域内
                if (isFeasible && !feasibleList[i]) {
                    feasibleList[i] = true;
                    fitnessValueList[i] = optModel.evalObj(loc);
                    pBest[i] = fitnessValueList[i];
                    pBestLocation.set(i, loc);
                    hasFeasibleOne = true;
                } else if (isFeasible && feasibleList[i]) {
                    fitnessValueList[i] = optModel.evalObj(loc);
                    if (fitnessValueList[i] < pBest[i]) {
                        pBest[i] = fitnessValueList[i];
                        pBestLocation.set(i, loc);
                    }
                    hasFeasibleOne = true;
                } else if (!isFeasible && !feasibleList[i]) {
                    PsoUtil.maxViolationArray(maxViolation, constrViolation);
                }
            }
            for (int i = 0; i < swarmSize; i++) {
                // 如果粒子之前都不在可行域内就计算适应度值
                if (!feasibleList[i]) {
                    fitnessValueList[i] = PsoUtil.violationFitness(maxViolation, swarm.get(i).getLocation().getConstrViolation());
                    pBest[i] = PsoUtil.violationFitness(maxViolation, pBestLocation.get(i).getConstrViolation());// 基值发生变化，pBest也要重算
                    if (fitnessValueList[i] < pBest[i]) {
                        pBest[i] = fitnessValueList[i];
                        pBestLocation.set(i, swarm.get(i).getLocation());
                    }
                }
            }

            // 步骤四：更新gBest
            if (isGBestfeasible && hasFeasibleOne) { // 都在可行域内
                int bestParticleIndex = PsoUtil.getMinPos(fitnessValueList, currentFeasibleList, true);
                assert bestParticleIndex != -1;
                if (fitnessValueList[bestParticleIndex] < gBest) {
                    gBest = fitnessValueList[bestParticleIndex];
                    gBestLocation = swarm.get(bestParticleIndex).getLocation();
                }
            } else if (!isGBestfeasible && hasFeasibleOne) { // 之前不在现在在
                isGBestfeasible = true;
                int bestParticleIndex = PsoUtil.getMinPos(fitnessValueList, currentFeasibleList, true);
                assert bestParticleIndex != -1;
                gBest = fitnessValueList[bestParticleIndex];
                gBestLocation = swarm.get(bestParticleIndex).getLocation();
            } else if (!isGBestfeasible && !hasFeasibleOne) { // 都不在
                int bestParticleIndex = PsoUtil.getMinPos(fitnessValueList, currentFeasibleList, false);
                assert bestParticleIndex != -1;
                gBest = PsoUtil.violationFitness(maxViolation, gBestLocation.getConstrViolation());// 基值发生变化，gBest也要重算
                if (fitnessValueList[bestParticleIndex] < gBest) {
                    gBest = fitnessValueList[bestParticleIndex];
                    gBestLocation = swarm.get(bestParticleIndex).getLocation();
                }
            }

            // 如果全局粒子在可行域内，如果已经达到模型的要求，是一个足够好的适应度值那么就结束寻优
            if (isGBestfeasible)
                tol = gBest - optModel.getTolFitness(); // minimizing the functions means it's getting closer to 0

            System.out.println("ITERATION " + iterNum + ": ");
            System.out.println("     Best Location: " + Arrays.toString(gBestLocation.getLoc()));
            System.out.println("     Value: " + gBest + "  " + isGBestfeasible);

            iterNum++;
        }

        if (isGBestfeasible) {
            System.out.println("\nSolution found at iteration " + iterNum + ", the solutions is:");
            System.out.println("     Best Location: " + Arrays.toString(gBestLocation.getLoc()));
        } else {
            System.out.println("Solution not found");
        }
    }

    // 初始化粒子群
    private void initializeSwarm() {
        Particle p;
        // 在粒子已知可行范围内初始化粒子群
        int n = optModel.getDimentions();
        double[] minLoc = optModel.getMinLoc();
        double[] maxLoc = optModel.getMaxLoc();
        double[] minVel = optModel.getMinVel();
        double[] maxVel = optModel.getMaxVel();

        for (int i = 0; i < swarmSize; i++) {
            p = new Particle();

            double[] loc = new double[n];
            double[] vel = new double[n];

            // 随机化粒子的位置和速度
            for (int j = 0; j < n; j++) {
                loc[j] = minLoc[j] + generator.nextDouble() * (maxLoc[j] - minLoc[j]); // 初始时粒子一定满足显式约束
                vel[j] = minVel[j] + generator.nextDouble() * (maxVel[j] - minVel[j]);
            }
            Location location = new Location(loc);
            Velocity velocity = new Velocity(vel);

            p.setLocation(location);
            p.setVelocity(velocity);
            swarm.add(p);

            // 获得约束向量
            double[] constrViolation = optModel.evalConstr(location);
            if (i == 0)
                maxViolation = new double[constrViolation.length]; // 第一次获取约束向量的时候初始化maxViolation的长度
            location.setConstrViolation(constrViolation);
            boolean isFeasible = PsoUtil.isFeasible(constrViolation);
            currentFeasibleList[i] = isFeasible;
            // 如果是满足约束的粒子
            if (isFeasible) {
                feasibleList[i] = true;
                fitnessValueList[i] = optModel.evalObj(location);
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
                isGBestfeasible = true;
            } else {
                feasibleList[i] = false;
                PsoUtil.maxViolationArray(maxViolation, constrViolation);
            }
            // 初始化时当前状态是该粒子的最好位置
            pBestLocation.add(swarm.get(i).getLocation());
        }
        for (int i = 0; i < swarmSize; i++) {
            // 如果粒子不在可行域内就计算适应度值
            if (!feasibleList[i]) {
                fitnessValueList[i] = PsoUtil.violationFitness(maxViolation, swarm.get(i).getLocation().getConstrViolation());
                pBest[i] = fitnessValueList[i];
            }
        }
        // 如果每一个粒子都不在可行域内，要找到一个gBest
        if (!hasFeasibleOne) {
            int bestParticleIndex = PsoUtil.getMinPos(fitnessValueList);
            gBest = fitnessValueList[bestParticleIndex];
            gBestLocation = swarm.get(bestParticleIndex).getLocation();
        }
    }

    public boolean isGBestfeasible() {
        return isGBestfeasible;
    }

    // 获得最优情况下的适应度值
    public double getgBest() {
        return gBest;
    }

    public Location getgBestLocation() {
        return gBestLocation;
    }
}
