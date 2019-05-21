package zju.bpamodel.pfr;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-14
 */
public class BusPfResult {
    private String type;
    private String name;
    private String area;
    private double baseKv;
    private double vInKv;
    private double angleInDegree;
    private double vInPu;
    private double angleInArc;
    private double genP;
    private double genQ;
    private double loadP;
    private double loadQ;
    private boolean isVoltageLimit = false;

    public BusPfResult() {
    }

    public BusPfResult(String name) {
        this.name = name;
    }

    public String getArea() {
        return area;
    }

    public void setArea(String area) {
        this.area = area;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

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

    public double getvInKv() {
        return vInKv;
    }

    public void setvInKv(double vInKv) {
        this.vInKv = vInKv;
    }

    public double getAngleInDegree() {
        return angleInDegree;
    }

    public void setAngleInDegree(double angleInDegree) {
        this.angleInDegree = angleInDegree;
    }

    public double getvInPu() {
        return vInPu;
    }

    public void setvInPu(double vInPu) {
        this.vInPu = vInPu;
    }

    public double getAngleInArc() {
        return angleInArc;
    }

    public void setAngleInArc(double angleInArc) {
        this.angleInArc = angleInArc;
    }

    public double getGenP() {
        return genP;
    }

    public void setGenP(double genP) {
        this.genP = genP;
    }

    public double getGenQ() {
        return genQ;
    }

    public void setGenQ(double genQ) {
        this.genQ = genQ;
    }

    public double getLoadP() {
        return loadP;
    }

    public void setLoadP(double loadP) {
        this.loadP = loadP;
    }

    public double getLoadQ() {
        return loadQ;
    }

    public void setLoadQ(double loadQ) {
        this.loadQ = loadQ;
    }

    public boolean isVoltageLimit() {
        return isVoltageLimit;
    }

    public void setVoltageLimit(boolean voltageLimit) {
        isVoltageLimit = voltageLimit;
    }
}
