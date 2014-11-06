package zju.measure;

import java.io.Serializable;

/**
 * this class containing information of measurement
 *
 * @author Dong Shufeng
 *         Date: 2007-11-18
 */
public class MeasureInfo implements Serializable, Comparable, Cloneable {

    int pointNum;

    String id; //needn't to be output

    String positionId;

    int measureType = -1;

    String type = "";//type of measure in cim

    double value;

    double value_est;

    double value_true;

    boolean isEligible;

    int linePowerType = -1;

    int powerType = -1;

    double weight = 1;

    double sigma = 1;

    double genMVA = 0.0;

    double upperLimit = 9999;

    double lowLimit = -9999;

    String deviceId;

    public MeasureInfo() {
    }

    /**
     * @param positionId  id of position which measurement is located
     * @param measureType measurement type defined in interface @zju.measure.MeasTypeCons
     * @param value       measurement value
     */
    public MeasureInfo(String positionId, int measureType, double value) {
        this.positionId = positionId;
        this.measureType = measureType;
        this.value = value;
    }

    /**
     * @param id    id of measurement and notice that this id is not the position id of mesurement
     * @param type  type of measurement and notice this type is not the measure type define in @zju.measure.MeasTypeCons
     * @param value measurement value
     */
    public MeasureInfo(String id, String type, double value) {
            this.id = id;
            this.type = type;
            this.value = value;
        }

    /**
     * @return id of this measurement
     */
    public String getId() {
        return id;
    }

    /**
     * @param id id of measurement
     */
    public void setId(String id) {
        this.id = id;
    }

    /**
     * @return position id that this measurement is located, like bus number, line's id and or so
     */
    public String getPositionId() {
        return positionId;
    }

    public int getPositionIdOfIntFormat() {
        String[] v = positionId.split("_");
        return Integer.parseInt(v[0]);
    }

    /**
     * @param positionId position id that measurement is located
     */
    public void setPositionId(String positionId) {
        this.positionId = positionId;
    }

    public int getMeasureType() {
        return measureType;
    }

    public void setMeasureType(int measureType) {
        this.measureType = measureType;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }


    public int getLinePowerType() {
        return linePowerType;
    }

    public void setLinePowerType(int linePowerType) {
        this.linePowerType = linePowerType;
    }

    public int compareTo(Object o) {
        if (o instanceof MeasureInfo)
            return new Integer(measureType).compareTo(((MeasureInfo) o).getMeasureType());
        return 0;
    }

    /**
     * @return this type indicates whether the power measurement is generator of load
     */
    public int getPowerType() {
        return powerType;
    }

    /**
     * the parameter is effective when this measurement is power measurement
     *
     * @param powerType this type indicates whether the power measurement is generator of load
     */
    public void setPowerType(int powerType) {
        this.powerType = powerType;
    }

    public boolean equals(Object obj) {//todo this is not perfect...
        if (obj instanceof MeasureInfo) {
            MeasureInfo info = (MeasureInfo) obj;
            return this.getMeasureType() == info.getMeasureType() &&
                    this.getPositionId() == info.getPositionId() && this.getLinePowerType() == info.getLinePowerType();

        }
        return false;
    }

    /**
     * @return weight of measurement indicating the degree of this measurement's accuracy
     */
    public double getWeight() {
        return weight;
    }

    /**
     * @param weight indicating the degree of this measurement's accuracy
     */
    public void setWeight(double weight) {
        this.weight = weight;
    }

    public double getSigma() {
        return sigma;
    }

    public void setSigma(double sigma) {
        this.sigma = sigma;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public int getPointNum() {
        return pointNum;
    }

    public void setPointNum(int pointNum) {
        this.pointNum = pointNum;
    }

    public double getValue_est() {
        return value_est;
    }

    public void setValue_est(double value_est) {
        this.value_est = value_est;
    }

    public boolean isEligible() {
        return isEligible;
    }

    public void setEligible(boolean eligible) {
        isEligible = eligible;
    }

    public double getGenMVA() {
        return genMVA;
    }

    public void setGenMVA(double genMVA) {
        this.genMVA = genMVA;
    }

    public String getDeviceId() {
        return deviceId;
    }

    public void setDeviceId(String deviceId) {
        this.deviceId = deviceId;
    }

    public double getValue_true() {
        return value_true;
    }

    public void setValue_true(double value_true) {
        this.value_true = value_true;
    }

    public double getUpperLimit() {
        return upperLimit;
    }

    public void setUpperLimit(double upperLimit) {
        this.upperLimit = upperLimit;
    }

    public double getLowLimit() {
        return lowLimit;
    }

    public void setLowLimit(double lowLimit) {
        this.lowLimit = lowLimit;
    }

    @SuppressWarnings({"CloneDoesntDeclareCloneNotSupportedException"})
    public MeasureInfo clone() {
        try {
            return (MeasureInfo) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return null;
    }
}

