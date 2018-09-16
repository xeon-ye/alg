package zju.pso;

/**
 * @Description: 优化模型接口定义
 * @Author: Fang Rui
 * @Date: 2018/6/7
 * @Time: 22:30
 */
public interface ParallelOptModel {


    /**
     * 根据当前位置计算目标函数值
     *
     * @param location
     * @return 目标函数值
     */
    double paraEvalObj(float[] location, int offset);

    /**
     * 根据当前位置计算约束函数值，其中等式约束作不等式约束处理
     *
     * @param location
     * @return 约束函数值
     */
    default float paraEvalConstr(float[] location, int offset) {
        return 0;
    }

    /**
     * @return 粒子位置的最小值
     */
    float[] paraGetMinLoc();

    /**
     * @return 粒子位置的最大值
     */
    float[] paraGetMaxLoc();

    /**
     * @return 粒子速度的最小值
     */
    float[] paraGetMinVel();

    /**
     * @return 粒子速度的最大值
     */
    float[] paraGetMaxVel();


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
