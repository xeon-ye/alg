package zju.pso;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

/**
 * 在传统粒子群算法中引入杂交和变异
 *
 * @Author: Fang Rui
 * @Date: 2018/6/29
 * @Time: 17:05
 */
public class BinaryPso extends AbstractPso implements PsoConstants {

    private static Logger logger = LogManager.getLogger(BinaryPso.class);

    private OptModel optModel;
    private double[] fitness; // 各粒子当前适应度值
    private boolean isGBestfeasible = false; // gBest是否在可行域内

    public BinaryPso(OptModel optModel) {
        this(optModel, (int) (10 + 2 * Math.sqrt(optModel.getDimentions())));
    }

    public BinaryPso(OptModel optModel, int swarmSize) {
        super(swarmSize);
        this.optModel = optModel;
        this.fitness = new double[swarmSize];
    }

    public BinaryPso(OptModel optModel, double[] initVariableState) {
        this(optModel, (int) (10 + 2 * Math.sqrt(optModel.getDimentions())), initVariableState);
    }

    public BinaryPso(OptModel optModel, int swarmSize, double[] initVariableState) {
        super(swarmSize, initVariableState);
        this.optModel = optModel;
        this.fitness = new double[swarmSize];
    }

    @Override
    protected void initializeSwarm() {
        Particle p;
        // 在粒子已知可行范围内初始化粒子群
        int n = optModel.getDimentions();

        for (int i = 0; i < swarmSize; i++) {
            p = new Particle();

            double[] loc = new double[n];
            double[] vel = new double[n];

            // 随机化粒子的位置和速度
            if (isWarmStart) {
                for (int j = 0; j < n; j++) {
                    assert initVariableState.length == n;
                    loc[j] = initVariableState[j]; // 初始时粒子一定满足显式约束
                }
            } else {
                for (int j = 0; j < n; j++) {
                    loc[j] = PsoUtil.randomBool(generator.nextDouble());
                }
            }
            Location location = new Location(loc);
            Velocity velocity = new Velocity(vel);

            p.setLocation(location);
            p.setVelocity(velocity);
            swarm.add(p);

            // 获得约束向量
            double violation = optModel.evalConstr(location);
            if (violation > 0) {
                fitness[i] = violation + PUNISHMENT;
            } else {
                fitness[i] = optModel.evalObj(location);
                isGBestfeasible = true;
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


        double tol = 1e6;
        double w; // 惯性权重

        while (iterNum < maxIter && tol > 0) {
            w = W_UPPERBOUND - (((double) iterNum) / maxIter) * (W_UPPERBOUND - W_LOWERBOUND); // 惯性逐渐减小

            for (int i = 0; i < swarmSize; i++) {
                double r1 = generator.nextDouble();
                double r2 = generator.nextDouble();

                Particle p = swarm.get(i);

                // 步骤一：更新速度和位置
                double[] loc = new double[n];
                double[] vel = new double[n];
                for (int j = 0; j < n; j++) {
                    vel[j] = (w * p.getVelocity().getVel()[j]) +
                            (r1 * C1) * (pBestLocation.get(i).getLoc()[j] - p.getLocation().getLoc()[j]) +
                            (r2 * C2) * (gBestLocation.getLoc()[j] - p.getLocation().getLoc()[j]);
                    loc[j] = PsoUtil.sigmoid(vel[j], generator.nextDouble());
                }
                p.setLocation(new Location(loc));
                p.setVelocity(new Velocity(vel));

            }

            // 步骤二：进行杂交
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
                    vel1[i] = tempVel[i] * coefficient1;
                    vel2[i] = tempVel[i] * coefficient2;
                    loc1[i] = PsoUtil.sigmoid(vel1[i], generator.nextDouble());
                    loc2[i] = PsoUtil.sigmoid(vel2[i], generator.nextDouble());
                }
                p1.setLocation(new Location(loc1));
                p1.setVelocity(new Velocity(vel1));
                p2.setLocation(new Location(loc2));
                p2.setVelocity(new Velocity(vel2));
            }

            for (int i = 0; i < swarmSize; i++) {
                Location location = swarm.get(i).getLocation();

                // 步骤三：更新适应度值
                double violation = optModel.evalConstr(location);
                if (violation > 0) {
                    fitness[i] = violation + PUNISHMENT;
                } else {
                    fitness[i] = optModel.evalObj(location);
                    isGBestfeasible = true;
                }

                // 步骤四：更新pBest
                if (fitness[i] < pBest[i]) {
                    pBest[i] = fitness[i];
                    pBestLocation.set(i, location);
                }
            }

            // 步骤五：更新gBest
            int bestParticleIndex = PsoUtil.getMinPos(fitness);
            if (fitness[bestParticleIndex] < gBest) {
                gBest = fitness[bestParticleIndex];
                gBestLocation = swarm.get(bestParticleIndex).getLocation();
            }

            // 如果全局粒子在可行域内，如果已经达到模型的要求，是一个足够好的适应度值那么就结束寻优
            if (isGBestfeasible)
                tol = gBest - optModel.getTolFitness();
            logger.info("ITERATION " + iterNum + ": Value: " + gBest + "  " + isGBestfeasible);
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
