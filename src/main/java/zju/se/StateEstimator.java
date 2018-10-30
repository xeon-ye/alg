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
 * Date: 2009-3-28
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
        // 给状态变量赋初值
        int variable_type = alg.getVariable_type();
        switch (variable_type) {
            case IpoptSeAlg.VARIABLE_UI:
            case IpoptSeAlg.VARIABLE_U:
                double[] initialU;
                if (variable_type == IpoptSeAlg.VARIABLE_U) // 直角坐标的电压
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
                if (variable_type == IpoptSeAlg.VARIABLE_VTHETA) // 极坐标的电压
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

    public void doSe() {
        startTime = System.currentTimeMillis();

        MeasureUtil.trans(sm, numOpt.getOld2new()); // 把量测又重新编号为新的编号
        alg.setIsland(clonedIsland);
        alg.setY(Y);
        alg.setSlackBusNum(clonedIsland.getSlackBusNum()); // 已经重新编号过

        setAlgInitial(); // 给状态变量赋初值
        Map<String, MeasureInfo> origVMeas = sm.getContainer(TYPE_BUS_VOLOTAGE);
        Map<String, MeasureInfo> origPMeas = sm.getContainer(TYPE_BUS_ACTIVE_POWER);
        Map<String, MeasureInfo> origQMeas = sm.getContainer(TYPE_BUS_REACTIVE_POWER);

        SeObjective objFunc = null;
        if (alg instanceof IpoptSeAlg) {
            objFunc = ((IpoptSeAlg) alg).getObjFunc();
        } else if (alg instanceof PsoSeAlg) {
            objFunc = ((PsoSeAlg) alg).getObjFunc();
        }

        if (alg instanceof IpoptSeAlg || alg instanceof PsoSeAlg) {
            int variable_type = alg.getVariable_type();
            assert objFunc != null;
            objFunc.setVMeas(new MeasureInfo[0]);
            objFunc.setVMeasPos(new int[0]);
            objFunc.setPMeas(new MeasureInfo[0]);
            objFunc.setPMeasPos(new int[0]);
            objFunc.setQMeas(new MeasureInfo[0]);
            objFunc.setQMeasPos(new int[0]);
            // 采用VTHETA作为状态变量时，会把sm中对应的量测给抹去
            if (variable_type == AbstractSeAlg.VARIABLE_VTHETA
                    || variable_type == AbstractSeAlg.VARIABLE_VTHETA_PQ) {
                if (objFunc.getObjType() == SeObjective.OBJ_TYPE_WLS) {
                    //todo; system measure is changed, must pay attention to it.
                    MeasureInfo[] vMeas = new MeasureInfo[origVMeas.size()];
                    origVMeas.values().toArray(vMeas); // 把origVMeas中的量测排成数组存入vMeas
                    int[] vMeasPos = new int[vMeas.length];
                    for (int i = 0; i < vMeasPos.length; i++)
                        vMeasPos[i] = Integer.parseInt(vMeas[i].getPositionId()) - 1; // 在状态变量中的位置
                    sm.setBus_v(new HashMap<>(0)); // 把节点电压量测抹除
                    objFunc.setVMeas(vMeas);
                    objFunc.setVMeasPos(vMeasPos);

                    if (variable_type == IpoptSeAlg.VARIABLE_VTHETA_PQ) {
                        MeasureInfo[] pMeas = new MeasureInfo[origPMeas.size()];
                        origPMeas.values().toArray(pMeas);
                        int[] pMeasPos = new int[pMeas.length];
                        for (int i = 0; i < pMeasPos.length; i++)
                            pMeasPos[i] = Integer.parseInt(pMeas[i].getPositionId()) - 1 + 2 * clonedIsland.getBuses().size();
                        sm.setBus_p(new HashMap<>(0));
                        objFunc.setPMeas(pMeas);
                        objFunc.setPMeasPos(pMeasPos);

                        MeasureInfo[] qMeas = new MeasureInfo[origQMeas.size()];
                        origQMeas.values().toArray(qMeas);
                        int[] qMeasPos = new int[qMeas.length];
                        for (int i = 0; i < qMeasPos.length; i++)
                            qMeasPos[i] = Integer.parseInt(qMeas[i].getPositionId()) - 1 + 3 * clonedIsland.getBuses().size();
                        sm.setBus_q(new HashMap<>(0));
                        objFunc.setQMeas(qMeas);
                        objFunc.setQMeasPos(qMeasPos);
                    }
                }
            }

            if (variable_type == AbstractSeAlg.VARIABLE_U
                    || variable_type == AbstractSeAlg.VARIABLE_UI) { // 采用直角坐标
                for (MeasureInfo info : origVMeas.values()) {
                    double v = info.getValue();
                    info.setValue(v * v); // 对于PV节点不平衡量是v * v
                }
            }
        }

        // 通过sm获取量测向量
        MeasVector meas = new MeasVectorCreator().getMeasureVector(sm);
        alg.setMeas(meas);

        //alg.setPrintPath(false);//show computation details
        // 初始化目标函数
        if (alg instanceof IpoptSeAlg)
            initialObjFunc(meas, ((IpoptSeAlg) alg).getObjFunc());
        else if (alg instanceof PsoSeAlg)
            initialObjFunc(meas, ((PsoSeAlg) alg).getObjFunc());

        long start = System.currentTimeMillis();
        alg.doSeAnalyse();
        log.debug("状态估计迭代过程用时: " + (System.currentTimeMillis() - start) + "ms");

        //MeasureUtil.setEstValue(sm, r, Y);
        //showInfo(initial, r, sm);
        sm.setBus_v(origVMeas);
        sm.setBus_p(origPMeas);
        sm.setBus_q(origQMeas);
        if (alg instanceof IpoptSeAlg || alg instanceof PsoSeAlg) {
            int variable_type = alg.getVariable_type();
            switch (variable_type) {
                case AbstractSeAlg.VARIABLE_UI:
                case AbstractSeAlg.VARIABLE_U:
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
                int variable_type = alg.getVariable_type();
                switch (variable_type) {
                    case IpoptSeAlg.VARIABLE_UI:
                    case IpoptSeAlg.VARIABLE_U:
                        int idx = meas.getBus_v_index();
                        for (int i = 0; i < meas.getBus_v_pos().length; i++)
                            meas.getZ_estimate().setValue(idx + i, Math.sqrt(meas.getZ_estimate().getValue(idx + i)));
                        break;
                    case IpoptSeAlg.VARIABLE_VTHETA:
                    case IpoptSeAlg.VARIABLE_VTHETA_PQ:
                        objFunc = ((IpoptSeAlg) alg).getObjFunc();
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
            System.out.println("k = " + k + ", lambda = " + ((alpha2 - alpha1) / alpha1));
            badData_threshhold = new double[meas.getZ().getN()];
            for (int i = 0; i < meas.getZ().getN(); i++) {
                double a0 = meas.getSigma().getValue(i) * alpha1; // 代表与置信区间p对应的扩展不确定度，偏移在此区域内代表为合格测点
                double a1 = meas.getSigma().getValue(i) * alpha2; // 1 + lamda = a1/a0 偏移在此区域外为不合格测点
                //if (a1 < 1e-3)
                //    a1 = 0.01;
                // a和b代表的是指数的ax+b形式
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
        } else if (objFunc.getObjType() == SeObjective.OBJ_TYPE_MNMR) {
            System.out.println("k = " + k + ", lambda = " + ((alpha2 - alpha1) / alpha1));
            badData_threshhold = new double[meas.getZ().getN()];
            for (int i = 0; i < meas.getZ().getN(); i++) {
                double a0 = meas.getSigma().getValue(i) * alpha1; // 代表与置信区间p对应的扩展不确定度，偏移在此区域内代表为合格测点
                double a1 = meas.getSigma().getValue(i) * alpha2; // 1 + lamda = a1/a0 偏移在此区域外为不合格测点
                //if (a1 < 1e-3)
                //    a1 = 0.01;
                // a和b代表的是指数的ax+b形式
                double a = (2 * k / (a0 - a1));
                double b = (k * (a1 + a0) / (a1 - a0));
                para1[i] = b / a;
                para2[i] = a;
                badData_threshhold[i] = a0;
            }
            objFunc.setA(para1);
            objFunc.setB(para2);
            objFunc.setThresholds(badData_threshhold);
            objFunc.setMeas(meas);
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
        numOpt.trans(clonedIsland); // 编号

        Y.setIsland(clonedIsland);
        Y.formYMatrix(); // 形成导纳矩阵
        Y.formConnectedBusCount(); // 形成存储各节点所相连节点数量的数组
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }
}

