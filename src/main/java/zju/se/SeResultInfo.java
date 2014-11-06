package zju.se;

import zju.pf.PfResultInfo;

import java.io.Serializable;

/**
 * Result of state estimation calculation.
 * @author Dong Shufeng
 * Date: 11-8-21
 */
public class SeResultInfo implements Serializable {
    private boolean isConverged;
    private long startTime;
    private long endTime;
    private long timeUsed;
    private int analogNum;
    private double eligibleRate;
    private PfResultInfo pfResult;

    public boolean isConverged() {
        return isConverged;
    }

    public void setConverged(boolean converged) {
        isConverged = converged;
    }

    public long getStartTime() {
        return startTime;
    }

    public void setStartTime(long startTime) {
        this.startTime = startTime;
    }

    public long getEndTime() {
        return endTime;
    }

    public void setEndTime(long endTime) {
        this.endTime = endTime;
    }

    public long getTimeUsed() {
        return timeUsed;
    }

    public void setTimeUsed(long timeUsed) {
        this.timeUsed = timeUsed;
    }

    public int getAnalogNum() {
        return analogNum;
    }

    public void setAnalogNum(int analogNum) {
        this.analogNum = analogNum;
    }

    public PfResultInfo getPfResult() {
        return pfResult;
    }

    public void setPfResult(PfResultInfo pfResult) {
        this.pfResult = pfResult;
    }

    public double getEligibleRate() {
        return eligibleRate;
    }

    public void setEligibleRate(double eligibleRate) {
        this.eligibleRate = eligibleRate;
    }
}