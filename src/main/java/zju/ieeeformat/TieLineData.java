package zju.ieeeformat;

import java.io.Serializable;

/**
 * Class TieLineData
 * <p> tie line data in ieee common data format</P>
 *
 * @author Dong Shufeng
 * @version 1.0
 * @Date: 2006-7-17
 */
public class TieLineData implements Serializable, Cloneable {
    private int meteredBusNum;
    private int meteredAreaNum;
    private int nonmeteredBusNum;
    private int nonmeteredAreaNum;
    private int circuitNum;

    public int getMeteredBusNum() {
        return meteredBusNum;
    }

    public void setMeteredBusNum(int meteredBusNum) {
        this.meteredBusNum = meteredBusNum;
    }

    public int getMeteredAreaNum() {
        return meteredAreaNum;
    }

    public void setMeteredAreaNum(int meteredAreaNum) {
        this.meteredAreaNum = meteredAreaNum;
    }

    public int getNonmeteredBusNum() {
        return nonmeteredBusNum;
    }

    public void setNonmeteredBusNum(int nonmeteredBusNum) {
        this.nonmeteredBusNum = nonmeteredBusNum;
    }

    public int getNonmeteredAreaNum() {
        return nonmeteredAreaNum;
    }

    public void setNonmeteredAreaNum(int nonmeteredAreaNum) {
        this.nonmeteredAreaNum = nonmeteredAreaNum;
    }

    public int getCircuitNum() {
        return circuitNum;
    }

    public void setCircuitNum(int circuitNum) {
        this.circuitNum = circuitNum;
    }

    @SuppressWarnings({"CloneDoesntDeclareCloneNotSupportedException"})
    public TieLineData clone() {
        try {
            return (TieLineData) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return null;
    }
}
