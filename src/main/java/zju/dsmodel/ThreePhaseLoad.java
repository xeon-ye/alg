package zju.dsmodel;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-3-27
 */
public interface ThreePhaseLoad extends Serializable {

    /**
     * @param v 电压，单位为伏特或标幺值
     * @param c 电流，单位为安培或标幺值
     */
    public void calI(double[][] v, double[][] c);  //V-A character
}
