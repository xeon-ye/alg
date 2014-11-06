package zju.se;

import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.AVector;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVector;
import zju.util.YMatrixGetter;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-19
 */
public abstract class AbstractSeAlg implements SeConstants, MeasTypeCons {
    protected int maxIter = 500;

    protected double tolerance = 1e-4;

    protected boolean isConverged = false, printPath = true;

    protected IEEEDataIsland island;

    protected MeasVector meas;

    protected YMatrixGetter Y;

    protected int[] unObserveBuses;

    protected int[] zeroPBuses = new int[0], zeroQBuses = new int[0];

    protected double[] variableState = null;

    protected int slackBusNum = 0;

    protected boolean isSlackBusVoltageFixed = false;

    protected double slackBusVoltage = 1.0, slackBusAngle = 0.0;

    protected double objective;

    protected int iterNum = -1;

    protected long timeUsed = 0;

    protected int[] measInObjFunc;

    public abstract AVector getFinalVTheta();

    public double getObjective() {
        return objective;
    }

    public double getSlackBusVoltage() {
        return slackBusVoltage;
    }

    public void setSlackBusVoltage(double slackBusVoltage) {
        this.slackBusVoltage = slackBusVoltage;
    }

    public double getSlackBusAngle() {
        return slackBusAngle;
    }

    public void setSlackBusAngle(double slackBusAngle) {
        this.slackBusAngle = slackBusAngle;
    }

    public boolean isSlackBusVoltageFixed() {
        return isSlackBusVoltageFixed;
    }

    public void setSlackBusVoltageFixed(boolean slackBusVoltageFixed) {
        isSlackBusVoltageFixed = slackBusVoltageFixed;
    }

    /**
     * @param busSize bus number
     * @param length  length of variables
     */
    protected void getInitialGuess(int busSize, int length) {
        if (variableState == null || variableState.length != length)
            variableState = new double[length];
        for (int i = 0; i < busSize; i++) {
            variableState[i] = 1.0;
            variableState[i + busSize] = 0.0;
        }
    }

    abstract public void doSeAnalyse();

    public double[] getVariableState() {
        return variableState;
    }

    public void setVariableState(double[] variableState) {
        this.variableState = variableState;
    }

    public IEEEDataIsland getIsland() {
        return island;
    }

    public void setIsland(IEEEDataIsland island) {
        this.island = island;
    }

    public int getMaxIter() {
        return maxIter;
    }

    public void setMaxIter(int maxIter) {
        this.maxIter = maxIter;
    }

    public double getTolerance() {
        return tolerance;
    }

    public void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }

    public boolean isConverged() {
        return isConverged;
    }

    public void setConverged(boolean converged) {
        isConverged = converged;
    }


    public boolean isPrintPath() {
        return printPath;
    }

    public void setPrintPath(boolean printPath) {
        this.printPath = printPath;
    }

    public MeasVector getMeas() {
        return meas;
    }

    public void setMeas(MeasVector meas) {
        this.meas = meas;
    }

    public int[] getUnObserveBuses() {
        return unObserveBuses;
    }

    public void setUnObserveBuses(int[] unObserveBuses) {
        this.unObserveBuses = unObserveBuses;
    }

    public YMatrixGetter getY() {
        return Y;
    }

    public void setY(YMatrixGetter y) {
        this.Y = y;
    }

    public int getSlackBusNum() {
        return slackBusNum;
    }

    public void setSlackBusNum(int slackBusNum) {
        this.slackBusNum = slackBusNum;
    }

    public int[] getZeroPBuses() {
        return zeroPBuses;
    }

    public void setZeroPBuses(int[] zeroPBuses) {
        this.zeroPBuses = zeroPBuses;
    }

    public int[] getZeroQBuses() {
        return zeroQBuses;
    }

    public void setZeroQBuses(int[] zeroQBuses) {
        this.zeroQBuses = zeroQBuses;
    }

    public int getIterNum() {
        return iterNum;
    }

    public void setIterNum(int iterNum) {
        this.iterNum = iterNum;
    }

    public int[] getMeasInObjFunc() {
        return measInObjFunc;
    }

    public void setMeasInObjFunc(int[] measInObjFunc) {
        this.measInObjFunc = measInObjFunc;
    }

    public long getTimeUsed() {
        return timeUsed;
    }

    public void setTimeUsed(long timeUsed) {
        this.timeUsed = timeUsed;
    }
}
