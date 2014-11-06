package zju.opf.result;

import java.io.Serializable;

public class CompensatorCommand implements Serializable {
    private String name;
    private String busName;
    private String type;
    private double finalSection;

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

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public double getFinalSection() {
        return finalSection;
    }

    public void setFinalSection(double finalSection) {
        this.finalSection = finalSection;
    }
}
