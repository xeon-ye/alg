package zju.measure;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-6-21
 */
public class DiscreteInfo implements Serializable, Comparable, Cloneable {
    int pointNum;

    String id; //needn't to be output

    String positionId;

    String type = "";//type of measure in cim

    int measureType = -1;

    int value;

    int value_est;

    int value_true;

    public int getMeasureType() {
        return measureType;
    }

    public void setMeasureType(int measureType) {
        this.measureType = measureType;
    }

    public int getPointNum() {
        return pointNum;
    }

    public void setPointNum(int pointNum) {
        this.pointNum = pointNum;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getPositionId() {
        return positionId;
    }

    public void setPositionId(String positionId) {
        this.positionId = positionId;
    }

    public int getValue() {
        return value;
    }

    public void setValue(int value) {
        this.value = value;
    }

    public int getValue_est() {
        return value_est;
    }

    public void setValue_est(int value_est) {
        this.value_est = value_est;
    }

    public int getValue_true() {
        return value_true;
    }

    public void setValue_true(int value_true) {
        this.value_true = value_true;
    }

    public int compareTo(Object o) {
        if (o instanceof MeasureInfo)
            return new Integer(measureType).compareTo(((DiscreteInfo) o).getMeasureType());
        return 0;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }
}
