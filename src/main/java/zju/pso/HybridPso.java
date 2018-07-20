package zju.pso;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.*;

/**
 * 在传统粒子群算法中引入杂交和变异
 *
 * @Author: Fang Rui
 * @Date: 2018/6/29
 * @Time: 17:05
 */
public class HybridPso extends AbstractPso implements PsoConstants {

    private static Logger logger = LogManager.getLogger(HybridPso.class);

    private double[] fitness; // 各粒子当前适应度值
    private boolean isGBestfeasible = false; // gBest是否在可行域内

    public HybridPso(OptModel optModel, int swarmSize) {
        super(optModel, swarmSize);
        this.fitness = new double[swarmSize];
    }

    public HybridPso(OptModel optModel, int swarmSize, double[] initVariableState) {
        super(optModel, swarmSize, initVariableState);
        this.fitness = new double[swarmSize];
    }

    @Override
    protected void initializeSwarm() {
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
            if (isWarmStart) {
                for (int j = 0; j < n; j++) {
                    assert initVariableState.length == n;
                    loc[j] = initVariableState[j]; // 初始时粒子一定满足显式约束
                    vel[j] = minVel[j] + generator.nextDouble() * (maxVel[j] - minVel[j]);
                }
            } else {
                for (int j = 0; j < n; j++) {
                    loc[j] = minLoc[j] + generator.nextDouble() * (maxLoc[j] - minLoc[j]); // 初始时粒子一定满足显式约束
                    vel[j] = minVel[j] + generator.nextDouble() * (maxVel[j] - minVel[j]);
                }
            }
            Location location = new Location(loc);
            Velocity velocity = new Velocity(vel);

            p.setLocation(location);
            p.setVelocity(velocity);
            swarm.add(p);

            // 获得约束向量
            double[] constrViolation = optModel.evalConstr(location);
            location.setConstrViolation(constrViolation);

            // 计算适应度值
            boolean isFeasible = PsoUtil.isFeasible(constrViolation);
            if (isFeasible) {
                fitness[i] = optModel.evalObj(location);
                isGBestfeasible = true;
            } else {
                fitness[i] = PsoUtil.evalInfeasibleFitness(constrViolation);
            }

            pBest[i] = fitness[i];
            pBestLocation.add(location);
        }

        // 找到gBest
        int bestParticleIndex = PsoUtil.getMinPos(fitness);
        gBest = fitness[bestParticleIndex];
        gBestLocation = swarm.get(bestParticleIndex).getLocation();
    }

    @Override
    public void execute() {
        initializeSwarm();

        int iterNum = 0;
        int n = optModel.getDimentions();
        int maxIter = optModel.getMaxIter();
        double[] minLoc = optModel.getMinLoc();
        double[] maxLoc = optModel.getMaxLoc();
        double[] minVel = optModel.getMinVel();
        double[] maxVel = optModel.getMaxVel();


        double tol = 1e6;
        double w; // 惯性权重

        while (iterNum < maxIter && tol > 0) {

            w = W_UPPERBOUND - (((double) iterNum) / maxIter) * (W_UPPERBOUND - W_LOWERBOUND); // 惯性逐渐减小

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
                p.setVelocity(new Velocity(newVel));

                // 步骤二：更新位置
                double[] newLoc = new double[n];
                for (int j = 0; j < n; j++) {
                    double previousLoc = p.getLocation().getLoc()[j];
                    double tempLoc = previousLoc + newVel[j];
                    newLoc[j] = PsoUtil.restrictByBoundary(tempLoc, maxLoc[j], minLoc[j], previousLoc);
                }
                p.setLocation(new Location(newLoc));
            }

            // 步骤三：进行杂交
            Set<Particle> pool = new HashSet<>();
            int hybridPoolSize = (int) (swarmSize * HYBRID_PROBABILITY);
            if ((hybridPoolSize & 1) == 1) // 如果是奇数
                hybridPoolSize++;
            while (pool.size() < hybridPoolSize) {
                Particle p = swarm.get((int) (Math.random() * swarmSize));
                pool.add(p);
            }
            assert pool.size() == hybridPoolSize;
            Iterator iter = pool.iterator();
            while (iter.hasNext()) {
                Particle p1 = (Particle) iter.next();
                Particle p2 = (Particle) iter.next();

                double velNorm1 = PsoUtil.getVecNorm(p1.getVelocity().getVel()); // |v1|
                double velNorm2 = PsoUtil.getVecNorm(p2.getVelocity().getVel()); // |v2|
                double[] tempVel = new double[n];
                for (int i = 0; i < p1.getVelocity().getVel().length; i++) {
                    tempVel[i] = p1.getVelocity().getVel()[i] + p2.getVelocity().getVel()[i];
                }
                double tempNorm = PsoUtil.getVecNorm(tempVel); // |v1 + v2|

                double coefficient1 = velNorm1 / tempNorm; // |v1| / |v1 + v2|
                double coefficient2 = velNorm2 / tempNorm; // |v2| / |v1 + v2|

                double[] loc1 = new double[n];
                double[] loc2 = new double[n];
                double[] vel1 = new double[n];
                double[] vel2 = new double[n];
                for (int i = 0; i < n; i++) {
                    double pb = Math.random();
                    loc1[i] = pb * p1.getLocation().getLoc()[i] + (1 - pb) * p2.getLocation().getLoc()[i];
                    loc2[i] = pb * p2.getLocation().getLoc()[i] + (1 - pb) * p1.getLocation().getLoc()[i];
                    vel1[i] = PsoUtil.restrictByBoundary(tempVel[i] * coefficient1, maxVel[i], minVel[i]);
                    vel2[i] = PsoUtil.restrictByBoundary(tempVel[i] * coefficient2, maxVel[i], minVel[i]);
                }
                p1.setLocation(new Location(loc1));
                p1.setVelocity(new Velocity(vel1));
                p2.setLocation(new Location(loc2));
                p2.setVelocity(new Velocity(vel2));
            }

            // 步骤四：进行变异
            pool.clear();
            double mutationProbability = MUTATION_UPPERBOUND - (((double) iterNum) / maxIter) * (MUTATION_UPPERBOUND - MUTATION_LOWERBOUND);
            int mutationPoolSize = (int) (swarmSize * mutationProbability);
            while (pool.size() < mutationPoolSize) {
                Particle p = swarm.get((int) (Math.random() * swarmSize));
                pool.add(p);
            }
            assert pool.size() == mutationPoolSize;
            iter = pool.iterator();
            // 计算高斯变异算子
            double[] mutationCoeff = new double[n];
            for (int i = 0; i < mutationCoeff.length; i++) {
                double sigma = (maxLoc[i] - minLoc[i]) * 0.1;
                mutationCoeff[i] = generator.nextGaussian() * sigma;
            }
            while (iter.hasNext()) {
                Particle p = (Particle) iter.next();
                double[] mutationLoc = new double[n];
                for (int i = 0; i < mutationLoc.length; i++) {
                    double previousLoc = p.getLocation().getLoc()[i];
                    double tempLoc = previousLoc + mutationCoeff[i];
                    mutationLoc[i] = PsoUtil.restrictByBoundary(tempLoc, maxLoc[i], minLoc[i], previousLoc);
                    mutationLoc[i] = tempLoc;
                }
                p.setLocation(new Location(mutationLoc));
            }

            for (int i = 0; i < swarmSize; i++) {
                Location location = swarm.get(i).getLocation();

                // 步骤六：更新适应度值
                double[] constrViolation = optModel.evalConstr(location);
                location.setConstrViolation(constrViolation);
                boolean isFeasible = PsoUtil.isFeasible(constrViolation);
                if (isFeasible) {
                    fitness[i] = optModel.evalObj(location);
                    isGBestfeasible = true;
                } else {
                    fitness[i] = PsoUtil.evalInfeasibleFitness(constrViolation);
                }

                // 步骤七：更新pBest
                if (fitness[i] < pBest[i]) {
                    pBest[i] = fitness[i];
                    pBestLocation.set(i, location);
                }
            }

            // 步骤八：更新gBest
            int bestParticleIndex = PsoUtil.getMinPos(fitness);
            if (fitness[bestParticleIndex] < gBest) {
                gBest = fitness[bestParticleIndex];
                gBestLocation = swarm.get(bestParticleIndex).getLocation();
            }

            // 如果全局粒子在可行域内，如果已经达到模型的要求，是一个足够好的适应度值那么就结束寻优
            if (isGBestfeasible)
                tol = gBest - optModel.getTolFitness();

            logger.debug("ITERATION " + iterNum + ": Value: " + gBest + "  " + isGBestfeasible);
            iterNum++;
        }

        if (isGBestfeasible) {
            logger.info("Solution found at iteration " + iterNum + ", best fitness value: " + gBest);
        } else {
            logger.warn("Solution not found");
        }
    }

    public boolean isGBestfeasible() {
        return isGBestfeasible;
    }
}
