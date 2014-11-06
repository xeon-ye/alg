package zju.pf;

import cern.colt.matrix.DoubleMatrix2D;
import jpscpu.LinearSolver;
import org.apache.log4j.Logger;
import zju.common.NewtonModel;
import zju.common.NewtonSolver;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.measure.MeasTypeCons;
import zju.util.JacobianMakerPC;
import zju.util.PfUtil;
import zju.util.StateCalByPolar;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-8-9
 */
public class PolarPf extends AbstractPf implements PfConstants, NewtonModel, MeasTypeCons {
    private static Logger log = Logger.getLogger(PolarPf.class);

    protected int maxQLimWatchTime = 10; //最大Q越限处理次数
    //母线个数
    protected int busNumber;

    protected VqPfSens pfSens;

    protected ASparseMatrixLink2D jacStructure;
    //在newton法之前使用PQ分解法求一个初值，该参数设置PQ分解法的次数
    private int decoupledPqNum = 2;

    private ASparseMatrixLink2D bApos, bAposTwo;

    private double[] pvBusDeltaQ; //计算PV转PQ时用

    private double[] newtonResult;//牛顿法存储使用Delta用的

    private AVector newtonState; //牛顿法计算时状态量

    private NewtonSolver newtonSolver;

    private boolean isXB = true;

    //下面两个参数是PQ分解法使用，分别用于相角和幅值的迭代
    private double[] deltaQ;

    private double[] deltaP;

    public PolarPf() {
        pfMethod = ALG_NEWTON;
        newtonSolver = new NewtonSolver(this);
        newtonSolver.setLinearSolver(NewtonSolver.LINEAR_SOLVER_SUPERLU);
    }

    public PolarPf(IEEEDataIsland oriIsland) {
        this();
        setOriIsland(oriIsland);
    }

    public PolarPf(String pfMethod) {
        this();
        setPfMethod(pfMethod);
    }

    public void doPf() {
        if (clonedIsland == null) {
            log.warn("电气岛为NULL, 潮流计算中止.");
            return;
        }
        if (clonedIsland.getSlackBusSize() > 1) {
            log.warn("目前不支持平衡节点个数大于1的情况, 潮流计算中止.");
            return;
        }
        if (!ALG_NEWTON.equals(getPfMethod()) && !ALG_PQ_DECOUPLED.equals(getPfMethod())) {
            log.warn("目前只支持牛顿法和PQ分解法, 潮流计算中止.");
            return;
        }

        //是否计算开断潮流
        if (isOutagePf) {
            //设置Jacobian矩阵重用
            newtonSolver.setJacStrucReuse(true);
            beforeOutagePf();
        } else if (isHandleQLim) {
            //如果处理PV节点转PQ的机制，事先将要用到的矩阵形成
            pfSens.formSubMarix(Y.getAdmittance()[1]);
        }

        if (!isOutagePf)//形成量测
            setMeas(PfUtil.formPQMeasure(clonedIsland));
        if (pfMethod.equals(ALG_NEWTON)) {
            if (meas.getZ_estimate() == null || meas.getZ_estimate().getN() != meas.getZ().getN())
                meas.setZ_estimate(new AVector(meas.getZ().getN()));
            if (newtonResult == null || newtonResult.length != meas.getZ().getN())
                newtonResult = new double[meas.getZ().getN()];
            if (newtonState == null || newtonState.getN() != meas.getZ().getN())
                newtonState = new AVector(meas.getZ().getN());
            //形成Jacobian矩阵的结构
            if (jacStructure == null) {
                jacStructure = JacobianMakerPC.getJacStrucOfVTheta(meas, Y, clonedIsland.getPqBusSize(), clonedIsland.getPvBusSize());
                jacobian = new MySparseDoubleMatrix2D(jacStructure.getM(), jacStructure.getN(), jacStructure.getVA().size(), 0.2, 0.5);
            }
        }
        if (decoupledPqNum > 0 || pfMethod.equals(ALG_PQ_DECOUPLED)) {
            if (bApos == null)
                formBApostrophe();
            if (bAposTwo == null)
                formBApostropheTwo();

            //Delta向量，用于存储每次计算P,Q量测的偏差
            if (deltaQ == null || deltaQ.length != meas.getBus_q_pos().length)
                deltaQ = new double[meas.getBus_q_pos().length];
            if (deltaP == null || deltaP.length != meas.getBus_p_pos().length)
                deltaP = new double[meas.getBus_p_pos().length];
        }

        doPfOnce();

        if (!isConverged()) {
            log.warn("潮流计算不收敛.");
        } else if (isHandleQLim)
            handlePvToPq();

        if (isOutagePf) {
            newtonSolver.setJacStrucReuse(false);
            afterOutagePf();
        }
    }

    protected void beforeOutagePf() {
        //保存原始的潮流结果
        if (origVTheta == null || origVTheta.getN() != variableState.length)
            origVTheta = new AVector(variableState.length);
        origVTheta.assign(variableState);
        //修改Y的值，将开断线路对导纳矩阵的作用消除
        for (int branchId : outageBranches)
            Y.dealBranch(clonedIsland.getId2branch().get(branchId), -1);
        if (pfMethod.equals(ALG_PQ_DECOUPLED)) {
            int f, t, size = clonedIsland.getPqBusSize() + clonedIsland.getPvBusSize();
            BranchData branch;
            for (int branchId : outageBranches) {
                branch = clonedIsland.getId2branch().get(branchId);
                f = branch.getTapBusNumber() - 1;
                t = branch.getZBusNumber() - 1;
                if (f >= size && t < size) {
                    bApos.increase(t, t, 1.0 / branch.getBranchX());
                } else if (f < size && t >= size) {
                    bApos.increase(f, f, 1.0 / branch.getBranchX());
                } else if (f < size && t < size) {
                    bApos.increase(f, f, 1.0 / branch.getBranchX());
                    bApos.increase(t, t, 1.0 / branch.getBranchX());
                    bApos.increase(f, t, -1.0 / branch.getBranchX());
                    bApos.increase(t, f, -1.0 / branch.getBranchX());
                }
                //todo: 这个todo要完成之后才能用PQ分解法做开断潮流
                //Y.dealBranch(branch, -1, null, bAposTwo);
            }
        }
        if (isHandleQLim) {
            //如果处理PV节点转PQ的机制，相应的灵敏度矩阵中的元素也要处理
            for (int branchId : outageBranches)
                pfSens.dealBranch(clonedIsland.getId2branch().get(branchId), -1);
        }
    }

    protected void afterOutagePf() {
        //修改Y的值
        for (int branchId : outageBranches)
            Y.dealBranch(clonedIsland.getId2branch().get(branchId));
        if (pfMethod.equals(ALG_PQ_DECOUPLED)) {
            int f, t, size = clonedIsland.getPqBusSize() + clonedIsland.getPvBusSize();
            BranchData branch;
            for (int branchId : outageBranches) {
                branch = clonedIsland.getId2branch().get(branchId);
                f = branch.getTapBusNumber() - 1;
                t = branch.getZBusNumber() - 1;
                if (f >= size && t < size) {
                    bApos.increase(t, t, -1.0 / branch.getBranchX());
                } else if (f < size && t >= size) {
                    bApos.increase(f, f, -1.0 / branch.getBranchX());
                } else if (f < size && t < size) {
                    bApos.increase(f, f, -1.0 / branch.getBranchX());
                    bApos.increase(t, t, -1.0 / branch.getBranchX());
                    bApos.increase(f, t, 1.0 / branch.getBranchX());
                    bApos.increase(t, f, 1.0 / branch.getBranchX());
                }
                //todo: 这个todo要完成之后才能用PQ分解法做开断潮流
                //Y.dealBranch(branch, -1, null, bAposTwo);
            }
        }
        if (isHandleQLim) {
            for (int branchId : outageBranches)
                pfSens.dealBranch(clonedIsland.getId2branch().get(branchId));
        }
        System.arraycopy(origVTheta.getValues(), 0, variableState, 0, variableState.length);
    }

    public PfResultInfo createPfResult() {
        if (isConverged()) {
            return PfResultMaker.getResult(oriIsland, variableState, Y, numberOpt.getOld2new());
        } else
            return null;
    }

    public void fillOriIslandPfResult() {
        if (isConverged())
            PfResultMaker.fillResult(oriIsland, Y, variableState, numberOpt.getOld2new());
    }

    public void fillClonedIslandPfResult() {
        if (isConverged())
            PfResultMaker.fillResult(clonedIsland, Y, variableState);
    }

    protected void handlePvToPq() {
        int watchCount = 0;
        double mvaBase = clonedIsland.getTitle().getMvaBase();
        double errMvar = 1.0 / mvaBase; //todo: not right
        int qOverLimCount;
        while (watchCount < maxQLimWatchTime) {
            //long start = System.currentTimeMillis();
            watchCount++;
            qOverLimCount = 0;
            for (BusData bus : clonedIsland.getBuses()) {
                if (bus.getType() == BusData.BUS_TYPE_GEN_PV) {  //PV nodes
                    int busNum = bus.getBusNumber();
                    double measQ = StateCalByPolar.calBusQ(busNum, Y, variableState) + (bus.getLoadMVAR() / mvaBase);
                    double maxQ = bus.getMaximum() / mvaBase;
                    double minQ = bus.getMinimum() / mvaBase;
                    if (maxQ - minQ <= errMvar) {
                        pvBusDeltaQ[busNum - clonedIsland.getPqBusSize() - 1] = 0.0;
                        continue;
                    }
                    if (measQ - maxQ > errMvar) {
                        qOverLimCount++;
                        double v = pfSens.calPvDeltaV(maxQ - measQ, busNum);
                        variableState[busNum - 1] -= v;
                    } else if (minQ - measQ > errMvar) {
                        qOverLimCount++;
                        //pvBusDeltaQ[busNum - pqBusSize - 1] = minQ - measQ;
                        double v = pfSens.calPvDeltaV(minQ - measQ, busNum);
                        variableState[busNum - 1] -= v;
                    } else
                        pvBusDeltaQ[busNum - clonedIsland.getPqBusSize() - 1] = 0.0;
                }
            }
            log.info("第" + watchCount + "次观察共有" + qOverLimCount + "个PV节点无功越限.");
            if (qOverLimCount < 1)
                break;
            doPfOnce();
            if (!isConverged())
                break;
        }
    }

    protected void doPfOnce() {
        isConverged = false;
        if (pfMethod.equals(ALG_NEWTON)) {
            if (decoupledPqNum > 0)
                doPf_decoupledPq(decoupledPqNum);
            if (!isConverged()) {
                setConverged(newtonSolver.solve());
                setIterNum(newtonSolver.getIterNum());
            }
            log.info("潮流收敛，迭代次数:" + getIterNum());
        } else if (pfMethod.equals(ALG_PQ_DECOUPLED)) {
            doPf_decoupledPq(getMaxIter());
            if (!isConverged())
                log.info("达到最大迭代次数, 潮流计算仍不收敛.");
        }
    }

    //todo：如用该方法做开短潮流，存储空间还可以进一步优化
    protected void doPf_decoupledPq(int maxIter) {
        LinearSolver sluSolver1 = new LinearSolver();
        LinearSolver sluSolver2 = new LinearSolver();

        iterNum = 0;
        int busNo, bus_p_index = meas.getBus_p_index(), bus_q_index = meas.getBus_q_index();
        boolean isQConverged, isPConverged = false;
        boolean isQIterFirst = true, isPIterFirst = true;
        double maxDelta, delta;

        setConverged(false);
        //初始化状态量
        initialState();
        while (iterNum < maxIter) {
            iterNum++;
            maxDelta = 0.0;
            for (int i = 0; i < clonedIsland.getPqBusSize(); i++) {
                delta = meas.getZ().getValue(i + bus_q_index) - StateCalByPolar.calBusQ(i + 1, Y, variableState);
                deltaQ[i] = delta / variableState[i];
                double absDelta = Math.abs(delta);
                if (maxDelta < absDelta)
                    maxDelta = absDelta;
            }
            if (maxDelta > tol_q) {
                if (isQIterFirst) {
                    sluSolver1.solve3(bAposTwo, deltaQ);
                    isQIterFirst = false;
                } else {
                    sluSolver1.solve3(deltaQ);
                }
                for (int i = 0; i < deltaQ.length; i++)
                    variableState[i] -= deltaQ[i];
                isQConverged = false;
            } else if (isPConverged) {
                setConverged(true);
                log.info("潮流计算收敛，迭代次数:" + iterNum);
                break;
            } else
                isQConverged = true;

            maxDelta = 0.0;
            for (int i = 0; i < meas.getBus_p_pos().length; i++) {
                busNo = meas.getBus_p_pos()[i];
                delta = meas.getZ().getValue(i + bus_p_index) - StateCalByPolar.calBusP(busNo, Y, variableState);
                deltaP[i] = delta / variableState[i];
                double absDelta = Math.abs(delta);
                if (maxDelta < absDelta)
                    maxDelta = absDelta;
            }
            if (maxDelta > tol_p) {
                if (isPIterFirst) {
                    sluSolver2.solve3(bApos, deltaP);
                    isPIterFirst = false;
                } else
                    sluSolver2.solve3(deltaP);
                for (int i = 0; i < deltaP.length; i++)
                    variableState[busNumber + i] -= deltaP[i];
                isPConverged = false;
            } else if (isQConverged) {
                setConverged(true);
                log.info("潮流计算收敛，迭代次数:" + iterNum);
                break;
            } else
                isPConverged = true;
        }
    }

    private void formBApostrophe() {
        int size = clonedIsland.getPqBusSize() + clonedIsland.getPvBusSize();
        bApos = new ASparseMatrixLink2D(size);
        if (isXB) {
            int f, t;
            for (BranchData branch : clonedIsland.getBranches()) {
                f = branch.getTapBusNumber() - 1;
                t = branch.getZBusNumber() - 1;
                if (f >= size && t < size) {
                    bApos.increase(t, t, -1.0 / branch.getBranchX());
                } else if (f < size && t >= size) {
                    bApos.increase(f, f, -1.0 / branch.getBranchX());
                } else if (f < size && t < size) {
                    bApos.increase(f, f, -1.0 / branch.getBranchX());
                    bApos.increase(t, t, -1.0 / branch.getBranchX());
                    bApos.increase(f, t, 1.0 / branch.getBranchX());
                    bApos.increase(t, f, 1.0 / branch.getBranchX());
                }
            }
        } else {
            for (BranchData branch : clonedIsland.getBranches()) {
                int f = branch.getTapBusNumber() - 1;
                int t = branch.getZBusNumber() - 1;
                double r = branch.getBranchR();
                double x = branch.getBranchX();
                double b = -x / (r * r + x * x);
                if (f >= size && t < size) {
                    bApos.increase(t, t, b);
                } else if (f < size && t >= size) {
                    bApos.increase(f, f, b);
                } else if (f < size && t < size) {
                    bApos.increase(f, f, b);
                    bApos.increase(t, t, b);
                    bApos.increase(f, t, -b);
                    bApos.increase(t, f, -b);
                }
            }
        }
        //for (int i = 0; i < size; i++) {
        //    int k = Y.getAdmittance()[1].getIA()[i];
        //    while (k != -1) {
        //        int j = Y.getAdmittance()[1].getJA().get(k);
        //        if (j >= size)
        //            break;
        //        bApos.setValue(i, j, Y.getAdmittance()[1].getVA().get(k));
        //        k = Y.getAdmittance()[1].getLINK().get(k);
        //    }
        //}
        //for (BusData bus : clonedIsland.getBuses()) {
        //    int i = bus.getBusNumber() - 1;
        //    if (i >= size)
        //        continue;
        //    bApos.increase(i, i, -bus.getShuntSusceptance());
        //}
    }

    public void formBApostropheTwo() {
        int size = clonedIsland.getPqBusSize();
        bAposTwo = new ASparseMatrixLink2D(size);
        if (isXB) {
            for (int i = 0; i < clonedIsland.getPqBusSize(); i++) {
                int k = Y.getAdmittance()[1].getIA()[i];
                while (k != -1) {
                    int j = Y.getAdmittance()[1].getJA().get(k);
                    if (j >= size)
                        break;
                    bAposTwo.setValue(i, j, Y.getAdmittance()[1].getVA().get(k));
                    k = Y.getAdmittance()[1].getLINK().get(k);
                }
            }
        } else {
            int f, t;
            for (BranchData branch : clonedIsland.getBranches()) {
                f = branch.getTapBusNumber() - 1;
                t = branch.getZBusNumber() - 1;
                if (f >= size && t < size) {
                    bAposTwo.increase(t, t, -1.0 / branch.getBranchX());
                    bAposTwo.increase(t, t, branch.getLineB() / 2.0);
                } else if (f < size && t >= size) {
                    bAposTwo.increase(f, f, -1.0 / branch.getBranchX());
                    bAposTwo.increase(f, f, branch.getLineB() / 2.0);
                } else if (f < size && t < size) {
                    bAposTwo.increase(f, f, branch.getLineB() / 2.0);
                    bAposTwo.increase(t, t, branch.getLineB() / 2.0);
                    bAposTwo.increase(f, f, -1.0 / branch.getBranchX());
                    bAposTwo.increase(t, t, -1.0 / branch.getBranchX());
                    bAposTwo.increase(f, t, 1.0 / branch.getBranchX());
                    bAposTwo.increase(t, f, 1.0 / branch.getBranchX());
                }
            }
            for (BusData bus : clonedIsland.getBuses()) {
                int i = bus.getBusNumber() - 1;
                if (i >= clonedIsland.getPqBusSize())
                    continue;
                bAposTwo.increase(i, i, bus.getShuntSusceptance());
            }
        }
    }

    public void initialState() {
        if (variableState == null) {
            variableState = new double[2 * busNumber];
            for (int i = 0; i < clonedIsland.getPqBusSize(); i++) {
                variableState[i] = 1.0;
                variableState[i + busNumber] = 0.0;
            }
            for (BusData bus : clonedIsland.getBuses()) {
                switch (bus.getType()) {
                    case BusData.BUS_TYPE_GEN_PV:
                        variableState[bus.getBusNumber() - 1] = bus.getFinalVoltage();
                        break;
                    case BusData.BUS_TYPE_SLACK:
                        variableState[bus.getBusNumber() - 1] = bus.getFinalVoltage();
                        variableState[bus.getBusNumber() - 1 + busNumber] = bus.getFinalAngle() * Math.PI / 180.0;
                        break;
                    default:
                        break;
                }
            }
        }
    }

    public void setOriIsland(IEEEDataIsland oriIsland) {
        super.setOriIsland(oriIsland);
        busNumber = clonedIsland.getBuses().size();
        pvBusDeltaQ = new double[clonedIsland.getPvBusSize()];
        pfSens = new VqPfSens(clonedIsland.getPqBusSize(), clonedIsland.getPvBusSize());
        jacStructure = null;
        bApos = null;
        bAposTwo = null;
    }

    protected void updateShortState(double[] shortState) {
        System.arraycopy(variableState, 0, shortState, 0, clonedIsland.getPqBusSize());
        int thetaSize = clonedIsland.getPqBusSize() + clonedIsland.getPvBusSize();
        System.arraycopy(variableState, busNumber, shortState, clonedIsland.getPqBusSize(), thetaSize);
    }

    protected void updateState(double[] shortState) {
        System.arraycopy(shortState, 0, variableState, 0, clonedIsland.getPqBusSize());
        int thetaSize = clonedIsland.getPqBusSize() + clonedIsland.getPvBusSize();
        System.arraycopy(shortState, clonedIsland.getPqBusSize(), variableState, busNumber, thetaSize);
    }

    @Override
    public boolean isTolSatisfied(double[] delta) {
        if (tol_p < tolerance)
            return false;
        if (tol_q < tolerance)
            return false;
        int index = 0;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_a_pos().length; i++, index++)
                        if (Math.abs(delta[index]) > tol_p)
                            return false;
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        if (Math.abs(delta[index]) > tol_q)
                            return false;
                    }
                    break;
                default:
                    log.warn("潮流方程不支持的类型:" + type);
                    break;
            }
        }
        return true;
    }

    @Override
    public AVector getInitial() {
        initialState();
        updateShortState(newtonState.getValues());
        return newtonState;
    }

    public DoubleMatrix2D getJocobian(AVector state) {
        updateState(state.getValues());
        JacobianMakerPC.getJacobianOfVTheta(meas, Y, variableState, clonedIsland.getPqBusSize(), clonedIsland.getPvBusSize(), jacobian);
        return jacobian;
    }

    @Override
    public ASparseMatrixLink2D getJacobianStruc() {
        return jacStructure;
    }

    public AVector getZ() {
        return meas.getZ();
    }

    @Override
    public double[] getDeltaArray() {
        return newtonResult;
    }

    public AVector calZ(AVector state) {
        updateState(state.getValues());
        return StateCalByPolar.getEstimatedZ(meas, Y, variableState, meas.getZ_estimate());
    }

    public boolean isJacStrucChange() {
        return false;
    }

    public VqPfSens getPfSens() {
        return pfSens;
    }

    public int getMaxQLimWatchTime() {
        return maxQLimWatchTime;
    }

    public void setMaxQLimWatchTime(int maxQLimWatchTime) {
        this.maxQLimWatchTime = maxQLimWatchTime;
    }

    public int getDecoupledPqNum() {
        return decoupledPqNum;
    }

    public void setDecoupledPqNum(int decoupledPqNum) {
        this.decoupledPqNum = decoupledPqNum;
    }

    public boolean isXB() {
        return isXB;
    }

    public void setXB(boolean XB) {
        isXB = XB;
    }
}
