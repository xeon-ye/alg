package zju.pso;

import java.util.Random;
import java.util.Vector;

/**
 * 粒子群算法的抽象父类
 *
 * @Author: Fang Rui
 * @Date: 2018/6/29
 * @Time: 9:36
 */
public abstract class AbstractPso {
    protected OptModel optModel;
    protected Vector<Particle> swarm = new Vector<>();
    protected final int swarmSize;
    protected double[] pBest; // 各粒子个体最优适应度值
    protected Vector<Location> pBestLocation = new Vector<>(); // 各粒子个体最优位置
    protected double gBest; // 全局最优适应度值
    protected Location gBestLocation; // 全局最优位置
    protected Random generator = new Random(); // 随机数产生器
    protected boolean isWarmStart = false;
    protected double[] initVariableState;

    protected AbstractPso(OptModel optModel, int swarmSize) {
        this.optModel = optModel;
        this.swarmSize = swarmSize;
        this.pBest = new double[swarmSize];
    }

    protected AbstractPso(OptModel optModel, int swarmSize, double[] initVariableState) {
        this(optModel, swarmSize);
        this.isWarmStart = true;
        this.initVariableState = initVariableState;
    }

    protected abstract void initializeSwarm();

    protected abstract void execute();

    public double getgBest() {
        return gBest;
    }

    public Location getgBestLocation() {
        return gBestLocation;
    }

}
