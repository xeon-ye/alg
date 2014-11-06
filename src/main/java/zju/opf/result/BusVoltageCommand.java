package zju.opf.result;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-1-14
 */
public class BusVoltageCommand implements Serializable {
    private String name;
    private double voltage;

    public String getName() {
        return name;
    }

    public double getVoltage() {
        return voltage;
    }

    public void setVoltage(double v) {
        this.voltage = v;
    }

    public void setName(String name) {
        this.name = name;
    }
}

