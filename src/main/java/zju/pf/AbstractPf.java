package zju.pf;

import cern.colt.matrix.DoubleMatrix2D;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.AVector;
import zju.measure.MeasVector;
import zju.util.NumberOptHelper;
import zju.util.YMatrixGetter;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-12-1
 */
public class AbstractPf implements PfConstants {

    protected int maxIter = 50, iterNum;

    protected double tol_v = .005, tol_p = .005, tol_q = .005;

    protected double tolerance = 1e-5;

    protected boolean isConverged = false;

    protected boolean printPath = true;

    protected boolean isDebug = false;

    protected boolean isHandleQLim = false;

    protected IEEEDataIsland oriIsland;

    protected IEEEDataIsland clonedIsland;

    protected YMatrixGetter Y;

    protected DoubleMatrix2D jacobian;

    protected double[] variableState;

    protected MeasVector meas;

    protected NumberOptHelper numberOpt = new NumberOptHelper();

    protected String pfMethod;

    protected int[] outageBranches;

    protected boolean isOutagePf = false;

    protected AVector origVTheta;  //计算开断潮流时，用于存储原潮流计算的结果

    public void setOriIsland(IEEEDataIsland oriIsland) {
        this.oriIsland = oriIsland;
        this.clonedIsland = oriIsland.clone();

        numberOpt.simple2(clonedIsland);
        numberOpt.trans(clonedIsland);

        //形成导纳矩阵
        Y = new YMatrixGetter(clonedIsland);
        Y.formYMatrix();
        //每次设置电气岛后都将置“开断潮流状态”为false
        setOutagePf(false);
        //设初值
        variableState = null;
        origVTheta = null;
        jacobian = null;
    }

    public String getPfMethod() {
        return pfMethod;
    }

    public void setConverged(boolean converged) {
        isConverged = converged;
    }

    public void setMeas(MeasVector meas) {
        this.meas = meas;
    }

    public void setMaxIter(int maxIter) {
        this.maxIter = maxIter;
    }

    public void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }

    public void setPrintPath(boolean printPath) {
        this.printPath = printPath;
    }

    public void setHandleQLim(boolean handleQLim) {
        isHandleQLim = handleQLim;
    }

    public void setVariableState(double[] variableState) {
        this.variableState = variableState;
    }

    public void setPfMethod(String pfMethod) {
        this.pfMethod = pfMethod;
    }

    public void setOutageBranches(int[] outageBranches) {
        this.outageBranches = outageBranches;
    }

    public void setOutagePf(boolean outagePf) {
        isOutagePf = outagePf;
    }

    public int getMaxIter() {
        return maxIter;
    }

    public double getTolerance() {
        return tolerance;
    }

    public boolean isConverged() {
        return isConverged;
    }

    public boolean isPrintPath() {
        return printPath;
    }

    public boolean isDebug() {
        return isDebug;
    }

    public boolean isHandleQLim() {
        return isHandleQLim;
    }

    public IEEEDataIsland getOriIsland() {
        return oriIsland;
    }

    public IEEEDataIsland getClonedIsland() {
        return clonedIsland;
    }

    public YMatrixGetter getY() {
        return Y;
    }

    public DoubleMatrix2D getJacobian() {
        return jacobian;
    }

    public double[] getVariableState() {
        return variableState;
    }

    public MeasVector getMeas() {
        return meas;
    }

    public NumberOptHelper getNumberOpt() {
        return numberOpt;
    }

    public int[] getOutageBranches() {
        return outageBranches;
    }

    public boolean isOutagePf() {
        return isOutagePf;
    }

    public AVector getOrigVTheta() {
        return origVTheta;
    }

    public int getIterNum() {
        return iterNum;
    }

    public void setIterNum(int iterNum) {
        this.iterNum = iterNum;
    }

    public double getTol_v() {
        return tol_v;
    }

    public void setTol_v(double tol_v) {
        this.tol_v = tol_v;
    }

    public double getTol_p() {
        return tol_p;
    }

    public void setTol_p(double tol_p) {
        this.tol_p = tol_p;
    }

    public double getTol_q() {
        return tol_q;
    }

    public void setTol_q(double tol_q) {
        this.tol_q = tol_q;
    }
}
