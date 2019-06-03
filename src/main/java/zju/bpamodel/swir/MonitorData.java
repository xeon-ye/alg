package zju.bpamodel.swir;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-5-21
 */
public class MonitorData {
    private double time;
    private RelativeAngle relativeAngle;
    private BusVoltage minBusVoltage;
    private BusVoltage maxBusVoltage;
    private BusFreq minBusFreq;
    private BusFreq maxBusFreq;

    public MonitorData(double time, RelativeAngle relativeAngle, BusVoltage minBusVoltage, BusVoltage maxBusVoltage,
                       BusFreq minBusFreq, BusFreq maxBusFreq) {
        this.time = time;
        this.relativeAngle = relativeAngle;
        this.minBusVoltage = minBusVoltage;
        this.maxBusVoltage = maxBusVoltage;
        this.minBusFreq = minBusFreq;
        this.maxBusFreq = maxBusFreq;
    }

    public double getTime() {
        return time;
    }

    public void setTime(double time) {
        this.time = time;
    }

    public RelativeAngle getRelativeAngle() {
        return relativeAngle;
    }

    public void setRelativeAngle(RelativeAngle relativeAngle) {
        this.relativeAngle = relativeAngle;
    }

    public BusVoltage getMinBusVoltage() {
        return minBusVoltage;
    }

    public void setMinBusVoltage(BusVoltage minBusVoltage) {
        this.minBusVoltage = minBusVoltage;
    }

    public BusVoltage getMaxBusVoltage() {
        return maxBusVoltage;
    }

    public void setMaxBusVoltage(BusVoltage maxBusVoltage) {
        this.maxBusVoltage = maxBusVoltage;
    }

    public BusFreq getMinBusFreq() {
        return minBusFreq;
    }

    public void setMinBusFreq(BusFreq minBusFreq) {
        this.minBusFreq = minBusFreq;
    }

    public BusFreq getMaxBusFreq() {
        return maxBusFreq;
    }

    public void setMaxBusFreq(BusFreq maxBusFreq) {
        this.maxBusFreq = maxBusFreq;
    }
}
