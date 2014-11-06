package zju.opf.result;

import java.io.Serializable;

public class TransformerCommand implements Serializable {
    private String name;
    private String tapBusName;
    private String zBusName;
    private double finalTurnTap;
    private double r;
    private double x;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getTapBusName() {
        return tapBusName;
    }

    public void setTapBusName(String tapBusName) {
        this.tapBusName = tapBusName;
    }

    public String getzBusName() {
        return zBusName;
    }

    public void setZBusName(String zBusName) {
        this.zBusName = zBusName;
    }

    public double getFinalTurnTap() {
        return finalTurnTap;
    }

    public void setFinalTurnTap(double finalTurnTap) {
        this.finalTurnTap = finalTurnTap;
    }

    public double getR() {
        return r;
    }

    public void setR(double r) {
        this.r = r;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }
}
