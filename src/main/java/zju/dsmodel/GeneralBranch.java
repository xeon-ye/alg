package zju.dsmodel;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-4-14
 */
public interface GeneralBranch extends Serializable {

    /**
     * @param headV 单位为伏特或标幺值
     * @param tailI 单位是安培或标幺值
     * @param tailV 单位是伏特或标幺值
     */
    void calTailV(double[][] headV, double[][] tailI, double[][] tailV);

    /**
     * @param tailV 单位为伏特或标幺值
     * @param tailI 单位是安培或标幺值
     * @param headV 单位是伏特或标幺值
     */
    void calHeadV(double[][] tailV, double[][] tailI, double[][] headV);

    /**
     * @param tailV 单位为伏特或标幺值
     * @param tailI 单位是安培或标幺值
     * @param headI 单位是安培或标幺值
     */
    void calHeadI(double[][] tailV, double[][] tailI, double[][] headI);

     /**
     * 包括下面约束
     * VLNABC = a * LNabc + b * Iabc
     * IABC = c * LNabc + d * Iabc
     *
     * @return jacobian矩阵中非零元的个数
     */
    int getNonZeroNumOfJac();

    /**
     * 该方法用于判断线路是否包含某一相
     *
     * @param phase 相位
     * @return 是否包含某相
     */
    boolean containsPhase(int phase);
}
