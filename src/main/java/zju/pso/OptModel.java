package zju.pso;

import java.util.Arrays;

/**
 * @Description: 优化模型接口定义
 * @Author: Fang Rui
 * @Date: 2018/6/7
 * @Time: 22:30
 */
public interface OptModel {


    /**
     * 根据当前位置计算目标函数值
     *
     * @param location
     * @return 目标函数值
     */
    double evalObj(Location location);

    /**
     * 根据当前位置计算约束函数值，其中等式约束作不等式约束处理
     *
     * @param location
     * @return 约束函数值
     */
    default double evalConstr(Location location) {
        return 0;
    }

    /**
     * @return 粒子位置的最小值
     */
    double[] getMinLoc();

    /**
     * @return 粒子位置的最大值
     */
    double[] getMaxLoc();

    /**
     * @return 粒子速度的最小值
     */
    double[] getMinVel();

    /**
     * @return 粒子速度的最大值
     */
    double[] getMaxVel();


    /**
     * @return 获取状态变量的个数
     */
    int getDimentions();

    /**
     * @return 最大迭代次数，默认100次
     */
    default int getMaxIter() {
        return 100;
    }

    /**
     * @return 收敛精度
     */
    default double getTolFitness() {
        return -99999;
    }

}
