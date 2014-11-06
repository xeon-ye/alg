package zju.common;

import cern.colt.matrix.DoubleMatrix2D;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-8-9
 */
public interface NewtonModel {
    /**
     * @return max interation number
     */
    int getMaxIter();

    /**
     * @return tolerance denoting whether that algorithm is converged
     */
    double getTolerance();

    boolean  isTolSatisfied(double[] delta);

    /**
     * @return initial value of state variables
     * 同时返回的变量也将是最终存储计算结果的数组
     */
    AVector getInitial();

    /**
     * @param state value of state variables
     * @return get jocobian matrix under a certain state
     */
    DoubleMatrix2D getJocobian(AVector state);

    ASparseMatrixLink2D getJacobianStruc();

    /**
     * @return measurement value
     */
    AVector getZ();

    /**
     * @return 用于存储计算结构的数组
     */
    double[] getDeltaArray();

    /**
     * @param state value of state variables
     * @return computing value of measurement point
     */
    AVector calZ(AVector state);

    boolean isJacStrucChange();
}

