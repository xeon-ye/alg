package zju.bpamodel.pfr;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-14
 */
public class BranchPfResult implements Serializable {
    private String branchName;
    private double branchP;
    private double branchQ;
    private double branchPLoss;
    private double branchQLoss;

    public String getBranchName() {
        return branchName;
    }

    public void setBranchName(String branchName) {
        this.branchName = branchName;
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
}
