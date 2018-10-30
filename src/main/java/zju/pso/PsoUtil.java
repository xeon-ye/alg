package zju.pso;

/**
 * @Description: 工具类
 * @Author: Fang Rui
 * @Date: 2018/6/7
 * @Time: 17:43
 */
public class PsoUtil implements PsoConstants {

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
     *
     * @param list         数组
     * @param feasibleList 可行序列
     * @param isFeasible   是否可行
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
     */
    public static synchronized void maxViolationArray(double[] maxViolation, double[] currentViolation) {
        assert maxViolation.length == currentViolation.length;
        for (int i = 0; i < maxViolation.length; i++) {
            if (currentViolation[i] > maxViolation[i])
                maxViolation[i] = currentViolation[i];
        }
    }

    /**
     * 计算不在可行域内的粒子的适应度值
     *
     * @param maxViolation    最大偏差数组
     * @param constrViolation 粒子约束偏差
     * @return
     */
    public static double violationFitness(double[] maxViolation, double[] constrViolation) {
        assert maxViolation.length == constrViolation.length;
        double fitness = 0;
        for (int i = 0; i < maxViolation.length; i++) {
            // 如果第i个分量在可行域内，直接跳到下一个分量。为了防止0/0的情况
            if (constrViolation[i] <= 0)
                continue;
            fitness += constrViolation[i] / maxViolation[i];
        }
        return fitness;
    }

    /**
     * 将值限制在范围内
     *
     * @param val   输入值
     * @param upper 上限
     * @param lower 下限
     * @return
     */
    public static double restrictByBoundary(double val, double upper, double lower) {
        if (val < lower)
            return lower;
        else if (val > upper)
            return upper;
        else
            return val;
    }

    /**
     * 将值限制在中点处
     *
     * @param val      输入值
     * @param upper    上限
     * @param lower    下限
     * @param previous 下限
     * @return
     */
    public static double restrictByBoundary(double val, double upper, double lower, double previous) {
        if (val < lower)
            return (lower + previous) / 2;
        else if (val > upper)
            return (upper + previous) / 2;
        else
            return val;
    }

    /**
     * 计算不可行粒子的适应度值，利用惩罚的方式
     *
     * @param constrViolation
     * @return
     */
    public static double evalInfeasibleFitness(double[] constrViolation) {
        double fitness = PUNISHMENT;
        for (double aConstrViolation : constrViolation) {
            if (aConstrViolation > 0)
                fitness += aConstrViolation;
        }
        return fitness;
    }


    /**
     * 计算向量的模
     *
     * @param vector 向量
     * @return 模长
     */
    public static double getVecNorm(double[] vector) {
        double norm = 0;
        for (double aVector : vector) {
            norm += aVector * aVector;
        }
        norm = Math.sqrt(norm);
        return norm;
    }

    public static float[] doubleArr2floatArr(double[] array) {
        float[] result = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = (float) array[i];
        }
        return result;
    }

    public static double[] floatArr2doubleArr(float[] array) {
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }
}
