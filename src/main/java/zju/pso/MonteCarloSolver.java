package zju.pso;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;

public class MonteCarloSolver {

    private static Logger logger = LogManager.getLogger(MonteCarloSolver.class);

    private OptModel optModel;
    private boolean isGBestfeasible = false; // 最优解是否在可行域内
    private double gBest = Double.MAX_VALUE; // 最优解是否在可行域内
    private double[] gBestLoc; // 最优解是否在可行域内
    private int frequency = 1000000;

    public MonteCarloSolver(OptModel optModel) {
        this.optModel = optModel;
    }

    public MonteCarloSolver(OptModel optModel, int frequency) {
        this(optModel);
        this.frequency = frequency;
    }

    public void execute() {
        ExecutorService pool = Executors.newFixedThreadPool(10);
        List<Future<double[]>> futureList = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            Future<double[]> future = pool.submit(() -> simulate(frequency / 10));
            futureList.add(future);
        }

        for (Future<double[]> future : futureList) {
            try {
                double[] loc = future.get();
                Location location = new Location(loc);
                double obj = optModel.evalObj(location);
                if (optModel.evalConstr(location) > 0)
                    obj += PsoConstants.PUNISHMENT;
                if (obj < gBest) {
                    gBest = obj;
                    gBestLoc = loc;
                }
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        pool.shutdown();

        if (gBest < PsoConstants.PUNISHMENT)
            isGBestfeasible = true;

        if (isGBestfeasible) {
            logger.info("Solution found , best fitness value: " + gBest);
        } else {
            logger.warn("Solution not found");
        }
    }

    private double[] simulate(int frequency) {
        double[] min = optModel.getMinLoc();
        double[] max = optModel.getMaxLoc();

        double minObj = Double.MAX_VALUE;
        int n = optModel.getDimentions();
        double[] minLoc = new double[n];

        Random generator = new Random(); // 随机数产生器

        for (int i = 0; i < frequency; i++) {
            double[] loc = new double[n];
            for (int j = 0; j < n; j++) {
                loc[j] = min[j] + (max[j] - min[j]) * generator.nextDouble();
                Location location = new Location(loc);
                double obj = optModel.evalObj(location);
                double violation = optModel.evalConstr(location);
                if (violation > 0)
                    obj += PsoConstants.PUNISHMENT;
                if (obj < minObj) {
                    minObj = obj;
                    minLoc = loc;
                }
            }
        }
        return minLoc;
    }

    public boolean isGBestfeasible() {
        return isGBestfeasible;
    }

    public double getgBest() {
        return gBest;
    }

    public double[] getgBestLoc() {
        return gBestLoc;
    }
}
