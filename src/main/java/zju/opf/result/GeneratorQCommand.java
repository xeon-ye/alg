package zju.opf.result;

import java.io.Serializable;

public class GeneratorQCommand implements Serializable {
    private String name;
    private String busName;
    private double qGen;

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

    public double getqGen() {
        return qGen;
    }

    public void setQGen(double qGen) {
        this.qGen = qGen;
    }
}
