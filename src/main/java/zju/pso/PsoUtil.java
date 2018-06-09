package zju.pso;

/**
 * @Description: 工具类
 * @Author: Fang Rui
 * @Date: 2018/6/7
 * @Time: 17:43
 */
public class PsoUtil {

    /**
     * 找到数组中的最小元素
     *
     * @param list 数组
     * @return 下标
     */
    public static int getMinPos(double[] list) {
        int pos = 0;
        double minValue = list[0];
        for (int i = 0; i < list.length; i++) {
            if (list[i] < minValue) {
                pos = i;
                minValue = list[i];
            }
        }
        return pos;
    }

    /**
     * 找到指定元素中的最小元素
     * @param list 数组
     * @param feasibleList 可行序列
     * @param isFeasible 是否可行
     * @return 下标
     */
    public static int getMinPos(double[] list, boolean[] feasibleList, boolean isFeasible) {
        int pos = -1;
        double minValue = 0;
        assert list.length == feasibleList.length;
        for (int i = 0; i < list.length; i++) {
            if (!feasibleList[i] && isFeasible) // 当两者不一致时，跳过这次循环
                continue;
            if (pos == -1) {
                pos = i;
                minValue = list[i];
            } else {
                if (list[i] < minValue) {
                    pos = i;
                    minValue = list[i];
                }
            }
        }
        return pos;
    }

    /**
     * 判断粒子是否在可行域内
     *
     * @param constrValueList 约束向量
     * @return 是否在可行域内
     */
    public static boolean isFeasible(double[] constrValueList) {
        boolean isFeasible = true;
        for (double aConstrValue : constrValueList) {
            if (aConstrValue > 0)
                isFeasible = false;
        }
        return isFeasible;
    }

    /**
     * 选出最大的不等式偏差
     *
     * @param maxViolation     最大偏差数组
     * @param currentViolation 当前不等式偏差
     * @param dimension        维度
     */
    public static synchronized void maxViolationArray(double[] maxViolation, double[] currentViolation, int dimension) {
        for (int i = 0; i < dimension; i++) {
            if (currentViolation[i] > maxViolation[i])
                maxViolation[i] = currentViolation[i];
        }
    }

    /**
     * 计算不在可行域内的粒子的适应度值
     *
     * @param maxViolation    最大偏差数组
     * @param constrViolation 粒子约束偏差
     * @param dimension       维度
     * @return
     */
    public static double violationFitness(double[] maxViolation, double[] constrViolation, int dimension) {
        double fitness = 0;
        for (int i = 0; i < dimension; i++) {
            // 如果第i个分量在可行域内，直接跳到下一个分量。为了防止0/0的情况
            if (constrViolation[i] <= 0)
                continue;
            fitness += constrViolation[i] / maxViolation[i];
        }
        return fitness;
    }
}
