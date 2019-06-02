package zju.bpamodel.pfr;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-5-21
 */
public class TransformerPfResult implements Serializable {
    private String transformerName;
    private String busName1 = "";
    private String busName2 = "";
    private double baseKv1;
    private double baseKv2;
    private char circuit = ' ';
    private String zoneName = "";
    private double transformerP;
    private double transformerQ;
    private double transformerPLoss;
    private double transformerQLoss;
    private boolean isOverLoad = false;

    public String getTransformerName() {
        return transformerName;
    }

    public void setTransformerName(String transformerName) {
        this.transformerName = transformerName;
    }

    public String getBusName1() {
        return busName1;
    }

    public void setBusName1(String busName1) {
        this.busName1 = busName1;
    }

    public String getBusName2() {
        return busName2;
    }

    public void setBusName2(String busName2) {
        this.busName2 = busName2;
    }

    public double getBaseKv1() {
        return baseKv1;
    }

    public void setBaseKv1(double baseKv1) {
        this.baseKv1 = baseKv1;
    }

    public double getBaseKv2() {
        return baseKv2;
    }

    public void setBaseKv2(double baseKv2) {
        this.baseKv2 = baseKv2;
    }

    public char getCircuit() {
        return circuit;
    }

    public void setCircuit(char circuit) {
        this.circuit = circuit;
    }

    public String getZoneName() {
        return zoneName;
    }

    public void setZoneName(String zoneName) {
        this.zoneName = zoneName;
    }

    public double getTransformerP() {
        return transformerP;
    }

    public void setTransformerP(double transformerP) {
        this.transformerP = transformerP;
    }

    public double getTransformerQ() {
        return transformerQ;
    }

    public void setTransformerQ(double transformerQ) {
        this.transformerQ = transformerQ;
    }

    public double getTransformerPLoss() {
        return transformerPLoss;
    }

    public void setTransformerPLoss(double transformerPLoss) {
        this.transformerPLoss = transformerPLoss;
    }

    public double getTransformerQLoss() {
        return transformerQLoss;
    }

    public void setTransformerQLoss(double transformerQLoss) {
        this.transformerQLoss = transformerQLoss;
    }

    public boolean isOverLoad() {
        return isOverLoad;
    }

    public void setOverLoad(boolean overLoad) {
        isOverLoad = overLoad;
    }
}
