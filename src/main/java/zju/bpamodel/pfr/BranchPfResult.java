package zju.bpamodel.pfr;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-14
 */
public class BranchPfResult implements Serializable {
    private String branchName;
    private String busName1 = "";
    private String busName2 = "";
    private double baseKv1;
    private double baseKv2;
    private char circuit = ' ';
    private String zoneName = "";
    private double branchP;
    private double branchQ;
    private double branchPLoss;
    private double branchQLoss;
    private double chargeP;
    private boolean isOverLoad = false;

    public String getBranchName() {
        return branchName;
    }

    public void setBranchName(String branchName) {
        this.branchName = branchName;
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

    public double getBranchP() {
        return branchP;
    }

    public void setBranchP(double branchP) {
        this.branchP = branchP;
    }

    public double getBranchQ() {
        return branchQ;
    }

    public void setBranchQ(double branchQ) {
        this.branchQ = branchQ;
    }

    public double getBranchPLoss() {
        return branchPLoss;
    }

    public void setBranchPLoss(double branchPLoss) {
        this.branchPLoss = branchPLoss;
    }

    public double getBranchQLoss() {
        return branchQLoss;
    }

    public void setBranchQLoss(double branchQLoss) {
        this.branchQLoss = branchQLoss;
    }

    public double getChargeP() {
        return chargeP;
    }

    public void setChargeP(double chargeP) {
        this.chargeP = chargeP;
    }

    public boolean isOverLoad() {
        return isOverLoad;
    }

    public void setOverLoad(boolean overLoad) {
        isOverLoad = overLoad;
    }
}
