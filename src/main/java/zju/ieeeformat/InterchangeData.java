package zju.ieeeformat;

import java.io.Serializable;

/**
 * Class InterchangeData
 * <p> interchange data in ieee common data format</P>
 *
 * @author Dong Shufeng
 * @version 1.0
 * @Date: 2006-7-17
 */
public class InterchangeData implements Serializable, Cloneable {
    private int areaNumber; //no zeros!
    private int slackBusNumber;
    private String alternateSwingBusName;
    private double areaExport; //area interchange export, MW
    private double areaTolerance; //area interchange tolerance, MW
    private String areaCode;
    private String areaName;

    public int getAreaNumber() {
        return areaNumber;
    }

    public void setAreaNumber(int areaNumber) {
        this.areaNumber = areaNumber;
    }

    public int getSlackBusNumber() {
        return slackBusNumber;
    }

    public void setSlackBusNumber(int slackBusNumber) {
        this.slackBusNumber = slackBusNumber;
    }

    public String getAlternateSwingBusName() {
        return alternateSwingBusName;
    }

    public void setAlternateSwingBusName(String alternateSwingBusName) {
        this.alternateSwingBusName = alternateSwingBusName;
    }

    public double getAreaExport() {
        return areaExport;
    }

    public void setAreaExport(double areaExport) {
        this.areaExport = areaExport;
    }

    public double getAreaTolerance() {
        return areaTolerance;
    }

    public void setAreaTolerance(double areaTolerance) {
        this.areaTolerance = areaTolerance;
    }

    public String getAreaCode() {
        return areaCode;
    }

    public void setAreaCode(String areaCode) {
        this.areaCode = areaCode;
    }

    public String getAreaName() {
        return areaName;
    }

    public void setAreaName(String areaName) {
        this.areaName = areaName;
    }

    @SuppressWarnings({"CloneDoesntDeclareCloneNotSupportedException"})
    public InterchangeData clone() {
        try {
            return (InterchangeData) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return null;
    }
}
