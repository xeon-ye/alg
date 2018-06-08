package zju.pso;

/* author: gandhi - gandhi.mtm [at] gmail [dot] com - Depok, Indonesia */

// just a simple utility class to find a minimum position on a list

public class PsoUtil {

    /**
     * 找到最优的粒子
     *
     * @param list 适应度值列表
     * @return 适应度值最小元素的下表
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
    public static void maxViolationArray(double[] maxViolation, double[] currentViolation, int dimension) {
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
            fitness += constrViolation[i] / maxViolation[i];
        }
        return fitness;
    }
}
