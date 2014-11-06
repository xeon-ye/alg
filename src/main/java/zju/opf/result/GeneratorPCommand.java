package zju.opf.result;

import java.io.Serializable;

public class GeneratorPCommand implements Serializable {
    private String name;
    private String busName;
    private double pGen;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getBusName() {
        return busName;
    }

    public void setBusName(String busName) {
        this.busName = busName;
    }

    public double getpGen() {
        return pGen;
    }

    public void setPGen(double pGen) {
        this.pGen = pGen;
    }
}
