package zju.bpamodel.swir;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-5-21
 */
public class BusVoltage {
    private String name;
    private double baseKv;
    private double vInPu;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public double getBaseKv() {
        return baseKv;
    }

    public void setBaseKv(double baseKv) {
        this.baseKv = baseKv;
    }

    public double getvInPu() {
        return vInPu;
    }

    public void setvInPu(double vInPu) {
        this.vInPu = vInPu;
    }
}
