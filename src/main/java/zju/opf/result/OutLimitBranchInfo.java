package zju.opf.result;

import java.io.Serializable;

public class OutLimitBranchInfo implements Serializable {
    private String name;
    private String tapBus;
    private String zBus;
    private String type;

    private double settingValue;
    private double currentValue;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getTapBus() {
        return tapBus;
    }

    public void setTapBus(String tapBus) {
        this.tapBus = tapBus;
    }

    public String getzBus() {
        return zBus;
    }

    public void setZBus(String zBus) {
        this.zBus = zBus;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public double getSettingValue() {
        return settingValue;
    }

    public void setSettingValue(double settingValue) {
        this.settingValue = settingValue;
    }

    public double getCurrentValue() {
        return currentValue;
    }

    public void setCurrentValue(double currentValue) {
        this.currentValue = currentValue;
    }
}
