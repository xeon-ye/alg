package zju.opf.result;

import java.io.Serializable;


public class PilotNodeInfo implements Serializable {
    private String busName;
    private double p;
    private double q;
    private double angel;
    private double voltage;

    public String getBusName() {
        return busName;
    }

    public double getP() {
        return p;
    }

    public void setP(double p) {
        this.p = p;
    }

    public double getQ() {
        return q;
    }

    public void setQ(double q) {
        this.q = q;
    }

    public double getAngel() {
        return angel;
    }

    public void setAngel(double angel) {
        this.angel = angel;
    }

    public double getVoltage() {
        return voltage;
    }

    public void setVoltage(double v) {
        this.voltage = v;
    }

    public void setBusName(String busName) {
        this.busName = busName;
    }
}
