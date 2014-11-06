package zju.ieeeformat;

import java.io.Serializable;

/**
 * Class LossZoneData
 * <p> loss zone data in ieee common data format</P>
 *
 * @author Dong Shufeng
 * @version 1.0
 * @Date: 2006-7-17
 */
public class LossZoneData implements Serializable, Cloneable {
    private int lossZoneNumber;
    private String lossZoneName;

    public int getLossZoneNumber() {
        return lossZoneNumber;
    }

    public void setLossZoneNumber(int lossZoneNumber) {
        this.lossZoneNumber = lossZoneNumber;
    }

    public String getLossZoneName() {
        return lossZoneName;
    }

    public void setLossZoneName(String lossZoneName) {
        this.lossZoneName = lossZoneName;
    }

    @SuppressWarnings({"CloneDoesntDeclareCloneNotSupportedException"})
    public LossZoneData clone() {
        try {
            return (LossZoneData) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return null;
    }
}