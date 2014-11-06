package zju.common;

import zju.matrix.AVector;

/**
 * this model is used to calculate weighted least square problem
 *
 * @author Dong Shufeng
 *         Date: 2008-1-17
 */
public interface NewtonWlsModel extends NewtonModel {

    /**
     * @return weight of measurement
     */
    AVector getWeight();

    /**
     *
     * @return 是否Jacobian矩阵是线性的
     */
    boolean isJacLinear();
}
