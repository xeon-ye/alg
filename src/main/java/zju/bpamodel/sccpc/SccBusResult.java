package zju.bpamodel.sccpc;


import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-11-4
 */
public class SccBusResult implements Serializable {
    public final static int FAULT_TYPE_SINGLE_PHASE = 1;
    public final static int FAULT_TYPE_THREE_PHASE = 2;

    private String busName;
    private double baseKv;
    private String area;
    private int faultType;
    private double breakCurrent; //unit is kA
    private double shuntCurrent; //unit is kA
    private double shuntCapacity; //unit is MVA
    private double positiveSequenceR;//unit is ohm
    private double positiveSequenceX;//unit is ohm
    private double negativeSequenceR;//unit is ohm
    private double negativeSequenceX;//unit is ohm
    private double zeroSequenceR;//unit is ohm
    private double zeroSequenceX;//unit is ohm

    public static SccBusResult createBusResult(String content) {
        SccBusResult sccBusResult = new SccBusResult();
        sccBusResult.parseString(content);
        return sccBusResult;
    }

    public void parseString(String strLine) {
        int index = strLine.indexOf("\"");
        int index2 = strLine.indexOf("\"", index + 1);
        String currentData = strLine.substring(index + 1, index2);
        setBusName(currentData);
        String s = strLine.substring(index2 + 1).trim();

        index = s.indexOf(" ");
        currentData = s.substring(0, index);
        s = s.substring(index).trim();
        setBaseKv(Double.parseDouble(currentData));

        index = s.indexOf(" ");
        s = s.substring(index).trim();

        index = s.indexOf(" ");
        s = s.substring(index).trim();

        index = s.indexOf(" ");
        currentData = s.substring(0, index);
        s = s.substring(index).trim();
        setFaultType(Integer.parseInt(currentData));

        index = s.indexOf(" ");
        currentData = s.substring(0, index);
        s = s.substring(index).trim();
        setBreakCurrent(Double.parseDouble(currentData));

        index = s.indexOf(" ");
        currentData = s.substring(0, index);
        s = s.substring(index).trim();
        setShuntCurrent(Double.parseDouble(currentData));

        index = s.indexOf(" ");
        currentData = s.substring(0, index);
        s = s.substring(index).trim();
        setShuntCapacity(Double.parseDouble(currentData));

        index = s.indexOf(" ");
        currentData = s.substring(0, index);
        s = s.substring(index).trim();
        setPositiveSequenceR(Double.parseDouble(currentData));

        index = s.indexOf(" ");
        currentData = s.substring(0, index);
        s = s.substring(index).trim();
        setPositiveSequenceX(Double.parseDouble(currentData));

        index = s.indexOf(" ");
        currentData = s.substring(0, index);
        s = s.substring(index).trim();
        setNegativeSequenceR(Double.parseDouble(currentData));

        index = s.indexOf(" ");
        currentData = s.substring(0, index);
        s = s.substring(index).trim();
        setNegativeSequenceX(Double.parseDouble(currentData));

        index = s.indexOf(" ");
        currentData = s.substring(0, index);
        s = s.substring(index).trim();
        setZeroSequenceR(Double.parseDouble(currentData));

        index = s.indexOf(" ");
        currentData = s.substring(0, index);
        s = s.substring(index).trim();
        setZeroSequenceX(Double.parseDouble(currentData));

        if(s.length() > 0) {
            setArea(s.replace("\"", ""));
        }
    }

    public String getBusName() {
        return busName;
    }

    public void setBusName(String busName) {
        this.busName = busName;
    }

    public double getBaseKv() {
        return baseKv;
    }

    public void setBaseKv(double baseKv) {
        this.baseKv = baseKv;
    }

    public String getArea() {
        return area;
    }

    public void setArea(String area) {
        this.area = area;
    }

    public int getFaultType() {
        return faultType;
    }

    public void setFaultType(int faultType) {
        this.faultType = faultType;
    }

    public double getBreakCurrent() {
        return breakCurrent;
    }

    public void setBreakCurrent(double breakCurrent) {
        this.breakCurrent = breakCurrent;
    }

    public double getShuntCurrent() {
        return shuntCurrent;
    }

    public void setShuntCurrent(double shuntCurrent) {
        this.shuntCurrent = shuntCurrent;
    }

    public double getShuntCapacity() {
        return shuntCapacity;
    }

    public void setShuntCapacity(double shuntCapacity) {
        this.shuntCapacity = shuntCapacity;
    }

    public double getPositiveSequenceR() {
        return positiveSequenceR;
    }

    public void setPositiveSequenceR(double positiveSequenceR) {
        this.positiveSequenceR = positiveSequenceR;
    }

    public double getPositiveSequenceX() {
        return positiveSequenceX;
    }

    public void setPositiveSequenceX(double positiveSequenceX) {
        this.positiveSequenceX = positiveSequenceX;
    }

    public double getNegativeSequenceR() {
        return negativeSequenceR;
    }

    public void setNegativeSequenceR(double negativeSequenceR) {
        this.negativeSequenceR = negativeSequenceR;
    }

    public double getNegativeSequenceX() {
        return negativeSequenceX;
    }

    public void setNegativeSequenceX(double negativeSequenceX) {
        this.negativeSequenceX = negativeSequenceX;
    }

    public double getZeroSequenceR() {
        return zeroSequenceR;
    }

    public void setZeroSequenceR(double zeroSequenceR) {
        this.zeroSequenceR = zeroSequenceR;
    }

    public double getZeroSequenceX() {
        return zeroSequenceX;
    }

    public void setZeroSequenceX(double zeroSequenceX) {
        this.zeroSequenceX = zeroSequenceX;
    }
}
