package zju.se;

import org.apache.log4j.Logger;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.AVector;
import zju.measure.*;
import zju.pf.PfResultMaker;
import zju.util.NumberOptHelper;
import zju.util.YMatrixGetter;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2009-3-28
 */
public class StateEstimator implements SeConstants, MeasTypeCons {
    private static Logger log = Logger.getLogger(StateEstimator.class);

    private double k = 3, lambda = 4, alpha1 = 3, alpha2 = alpha1 * (1 + lambda);

    private IEEEDataIsland oriIsland;

    private IEEEDataIsland clonedIsland;

    private YMatrixGetter Y = new YMatrixGetter();

    private SystemMeasure sm;

    private NumberOptHelper numOpt = new NumberOptHelper();

    private AbstractSeAlg alg;

    private double[] initial;

    private boolean isFlatStart = true;

    private long startTime, endTime;

    public StateEstimator() {
    }

    public StateEstimator(IEEEDataIsland island, SystemMeasure sm) {
        setOriIsland(island);
        setSm(sm);
    }

    public void setAlg(AbstractSeAlg alg) {
        this.alg = alg;
        //for (String s : sm.getLinkBusList()) //todo:
        //    alg.getLinkBuses().add(Integer.parseInt(s));
    }

    //给算法程序赋初值
    private void setAlgInitial() {
        if (initial != null) {
            alg.setVariableState(initial);
            return;
        }
        int n = clonedIsland.getBuses().size();
        if (alg instanceof IpoptSeAlg) {
            int variable_type = ((IpoptSeAlg) alg).getVariable_type();
            switch (variable_type) {
                case IpoptSeAlg.VARIABLE_UI:
                case IpoptSeAlg.VARIABLE_U:
                    double[] initialU;
                    if (variable_type == IpoptSeAlg.VARIABLE_U)
                        initialU = new double[2 * n];
                    else
                        initialU = new double[4 * n];
                    if (isFlatStart) {
                        for (int i = 0; i < n; i++) {
                            initialU[i] = 1.0;
                            initialU[i + n] = 0.0;
                        }
                    } else {
                        double v, theta;
                        for (int i = 0; i < n; i++) {
                            BusData bus = clonedIsland.getBuses().get(i);
                            v = bus.getFinalVoltage();
                            theta = bus.getFinalAngle() * Math.PI / 180;
                            initialU[bus.getBusNumber() - 1] = v * Math.cos(theta);
                            initialU[bus.getBusNumber() + n - 1] = v * Math.sin(theta);
                        }
                    }
                    alg.setVariableState(initialU);
                    break;
                case IpoptSeAlg.VARIABLE_VTHETA:
                case IpoptSeAlg.VARIABLE_VTHETA_PQ:
                    double[] vTheta;
                    if (variable_type == IpoptSeAlg.VARIABLE_VTHETA)
                        vTheta = new double[n * 2];
                    else
                        vTheta = new double[n * 4];
                    if (isFlatStart)
                        for (int i = 0; i < n; i++) {
                            vTheta[i] = 1.0;
                            vTheta[i + n] = 0.0;
                        }
                    else {
                        for (int i = 0; i < n; i++) {
                            BusData bus = clonedIsland.getBuses().get(i);
                            vTheta[bus.getBusNumber() - 1] = bus.getFinalVoltage();
                            vTheta[bus.getBusNumber() - 1 + n] = bus.getFinalAngle() * Math.PI / 180.0;
                        }
                    }
                    alg.setVariableState(vTheta);
                default:
                    break;
            }
        }
    }

    public void doSe() {
        startTime = System.currentTimeMillis();

        MeasureUtil.trans(sm, numOpt.getOld2new());
        alg.setIsland(clonedIsland);
        alg.setY(Y);
        alg.setSlackBusNum(clonedIsland.getSlackBusNum());

        setAlgInitial();
        Map<String, MeasureInfo> origVMeas = sm.getContainer(TYPE_BUS_VOLOTAGE);
        Map<String, MeasureInfo> origPMeas = sm.getContainer(TYPE_BUS_ACTIVE_POWER);
        Map<String, MeasureInfo> origQMeas = sm.getContainer(TYPE_BUS_REACTIVE_POWER);
        if (alg instanceof IpoptSeAlg) {
            int variable_type = ((IpoptSeAlg) alg).getVariable_type();
            SeObjective objFunc = ((IpoptSeAlg) alg).getObjFunc();
            objFunc.setVMeas(new MeasureInfo[0]);
            objFunc.setVMeasPos(new int[0]);
            objFunc.setPMeas(new MeasureInfo[0]);
            objFunc.setPMeasPos(new int[0]);
            objFunc.setQMeas(new MeasureInfo[0]);
            objFunc.setQMeasPos(new int[0]);
            if (variable_type == IpoptSeAlg.VARIABLE_VTHETA
                    || variable_type == IpoptSeAlg.VARIABLE_VTHETA_PQ) {
                if (objFunc.getObjType() == SeObjective.OBJ_TYPE_WLS) {
                    //todo; system measure is changed, must pay attention to it.
                    MeasureInfo[] vMeas = new MeasureInfo[origVMeas.size()];
                    origVMeas.values().toArray(vMeas);
                    int[] vMeasPos = new int[vMeas.length];
                    for (int i = 0; i < vMeasPos.length; i++)
                        vMeasPos[i] = Integer.parseInt(vMeas[i].getPositionId()) - 1;
                    sm.setBus_v(new HashMap<String, MeasureInfo>(0));
                    objFunc.setVMeas(vMeas);
                    objFunc.setVMeasPos(vMeasPos);

                    if (variable_type == IpoptSeAlg.VARIABLE_VTHETA_PQ) {
                        MeasureInfo[] pMeas = new MeasureInfo[origPMeas.size()];
                        origPMeas.values().toArray(pMeas);
                        int[] pMeasPos = new int[pMeas.length];
                        for (int i = 0; i < pMeasPos.length; i++)
                            pMeasPos[i] = Integer.parseInt(pMeas[i].getPositionId()) - 1 + 2 * clonedIsland.getBuses().size();
                        sm.setBus_p(new HashMap<String, MeasureInfo>(0));
                        objFunc.setPMeas(pMeas);
                        objFunc.setPMeasPos(pMeasPos);

                        MeasureInfo[] qMeas = new MeasureInfo[origQMeas.size()];
                        origQMeas.values().toArray(qMeas);
                        int[] qMeasPos = new int[qMeas.length];
                        for (int i = 0; i < qMeasPos.length; i++)
                            qMeasPos[i] = Integer.parseInt(qMeas[i].getPositionId()) - 1 + 3 * clonedIsland.getBuses().size();
                        sm.setBus_q(new HashMap<String, MeasureInfo>(0));
                        objFunc.setQMeas(qMeas);
                        objFunc.setQMeasPos(qMeasPos);
                    }
                }
            }
            if (variable_type == IpoptSeAlg.VARIABLE_U
                    || variable_type == IpoptSeAlg.VARIABLE_UI) {
                for (MeasureInfo info : origVMeas.values()) {
                    double v = info.getValue();
                    info.setValue(v * v);
                }
            }
        }
        MeasVector meas = new MeasVectorCreator().getMeasureVector(sm);
        alg.setMeas(meas);

        //alg.setPrintPath(false);//show computation details
        if (alg instanceof IpoptSeAlg)
            initialObjFunc(meas, ((IpoptSeAlg) alg).getObjFunc());

        long start = System.currentTimeMillis();
        alg.doSeAnalyse();
        log.debug("状态估计迭代过程用时: " + (System.currentTimeMillis() - start) + "ms");

        //MeasureUtil.setEstValue(sm, r, Y);
        //showInfo(initial, r, sm);
        sm.setBus_v(origVMeas);
        sm.setBus_p(origPMeas);
        sm.setBus_q(origQMeas);
        if (alg instanceof IpoptSeAlg) {
            int variable_type = ((IpoptSeAlg) alg).getVariable_type();
            switch (variable_type) {
                case IpoptSeAlg.VARIABLE_UI:
                case IpoptSeAlg.VARIABLE_U:
                    for (MeasureInfo info : origVMeas.values())
                        info.setValue(Math.sqrt(info.getValue()));
                    break;
                default:
                    break;
            }
        }
        if (alg.isConverged()) {
            log.debug("状态估计迭代收敛.");
            if (alg instanceof IpoptSeAlg) {
                int variable_type = ((IpoptSeAlg) alg).getVariable_type();
                switch (variable_type) {
                    case IpoptSeAlg.VARIABLE_UI:
                    case IpoptSeAlg.VARIABLE_U:
                        int idx = meas.getBus_v_index();
                        for (int i = 0; i < meas.getBus_v_pos().length; i++)
                            meas.getZ_estimate().setValue(idx + i, Math.sqrt(meas.getZ_estimate().getValue(idx + i)));
                        break;
                    case IpoptSeAlg.VARIABLE_VTHETA:
                    case IpoptSeAlg.VARIABLE_VTHETA_PQ:
                        SeObjective objFunc = ((IpoptSeAlg) alg).getObjFunc();
                        if (objFunc.getObjType() == SeObjective.OBJ_TYPE_WLS) {
                            if (variable_type == IpoptSeAlg.VARIABLE_VTHETA)
                                MeasureUtil.setVTheta(alg.getVariableState(), sm, clonedIsland.getBuses().size());
                            else
                                MeasureUtil.setVThetaPq(alg.getVariableState(), sm, clonedIsland.getBuses().size());
                        }
                        break;
                }
            }
            MeasureUtil.setEstValue(meas, sm);
        } else
            log.warn("状态估计迭代不收敛.");

        MeasureUtil.trans(sm, numOpt.getNew2old());
        endTime = System.currentTimeMillis();
    }

    //todo: should be more sophisticated
    public double[] initialObjFunc(MeasVector meas, SeObjective objFunc) {
        double[] para1 = new double[meas.getZ().getN()];
        double[] para2 = new double[meas.getZ().getN()];
        double[] badData_threshhold = null;
        if (objFunc.getObjType() == SeObjective.OBJ_TYPE_WLS) {
            objFunc.setMeas(meas);
        } else if (objFunc.getObjType() == SeObjective.OBJ_TYPE_SIGMOID) {
            System.out.println("k=" + k + ", lambda = " + ((alpha2 - alpha1) / alpha1));
            badData_threshhold = new double[meas.getZ().getN()];
            for (int i = 0; i < meas.getZ().getN(); i++) {
                double a0 = meas.getSigma().getValue(i) * alpha1;
                double a1 = meas.getSigma().getValue(i) * alpha2; // lamda=(a1-a0)/a0
                //if (a1 < 1e-3)
                //    a1 = 0.01;
                double a = (2 * k / (a0 - a1));
                double b = (k * (a1 + a0) / (a1 - a0));
                para1[i] = b / a;
                para2[i] = a;
                badData_threshhold[i] = a1;
            }
            objFunc.setA(para1);
            objFunc.setB(para2);
        } else if (objFunc.getObjType() == SeObjective.OBJ_TYPE_QC) {
            badData_threshhold = new double[meas.getZ().getN()];
            for (int i = 0; i < meas.getZ().getN(); i++) {
                para1[i] = meas.getSigma().getValue(i) * 6;
                badData_threshhold[i] = meas.getSigma().getValue(i) * 6;
            }
            objFunc.setThresholds(para1);
            objFunc.setMeas(meas);
        } else if (objFunc.getObjType() == SeObjective.OBJ_TYPE_QL) {
            badData_threshhold = new double[meas.getZ().getN()];
            for (int i = 0; i < meas.getZ().getN(); i++) {
                para1[i] = meas.getSigma().getValue(i) * 6;
                badData_threshhold[i] = meas.getSigma().getValue(i) * 6;
            }
            objFunc.setThresholds(para1);
            objFunc.setMeas(meas);
        } else if (objFunc.getObjType() == SeObjective.OBJ_TYPE_PARTION) {
            //badData_threshhold = new double[meas.getZ().getN()];
            //for (int i = 0; i < meas.getZ().getN(); i++) {
            //    para1[i] = meas.getSigma().getValue(i) * 0.0;
            //    para2[i] = meas.getSigma().getValue(i) * 6;
            //    badData_threshhold[i] = meas.getSigma().getValue(i) * 6;
            //}
            //objFunc.setThresholds1(para1);
            //objFunc.setThresholds2(para2);
            //objFunc.setMeas(meas);
        }
        return badData_threshhold;
    }

    public SeResultInfo createPfResult() {
        SeResultInfo r = new SeResultInfo();
        r.setStartTime(startTime);
        r.setEndTime(endTime);
        r.setConverged(alg.isConverged());
        r.setTimeUsed(endTime - startTime);
        if (r.isConverged()) {
            AVector finalVTheta = alg.getFinalVTheta();
            //todo:
            r.setPfResult(PfResultMaker.getResult(oriIsland, finalVTheta, Y, numOpt.getOld2new()));
            MeasureUtil.trans(sm, numOpt.getOld2new());
            SeResultFiller filler = new SeResultFiller(clonedIsland, Y, sm);
            filler.fillSeResult(finalVTheta, r);
            MeasureUtil.trans(sm, numOpt.getNew2old());
        }
        return r;
    }

    public void setK(double k) {
        this.k = k;
    }

    public void setAlpha(double alpha1, double alpha2) {
        this.alpha1 = alpha1;
        this.alpha2 = alpha2;
    }

    public void setFlatStart(boolean flatStart) {
        isFlatStart = flatStart;
    }

    public void setInitial(double[] initial) {
        this.initial = initial;
    }

    public IEEEDataIsland getOriIsland() {
        return oriIsland;
    }

    public SystemMeasure getSm() {
        return sm;
    }

    public void setSm(SystemMeasure sm) {
        this.sm = sm;
        //showMeasureInfo(sm);
    }

    public NumberOptHelper getNumOpt() {
        return numOpt;
    }

    public IEEEDataIsland getClonedIsland() {
        return clonedIsland;
    }

    public YMatrixGetter getY() {
        return Y;
    }

    public void setOriIsland(IEEEDataIsland oriIsland) {
        this.oriIsland = oriIsland;
        clonedIsland = oriIsland.clone();

        numOpt.simple(clonedIsland);
        numOpt.trans(clonedIsland);

        Y.setIsland(clonedIsland);
        Y.formYMatrix();
        Y.formConnectedBusCount();
    }
}

