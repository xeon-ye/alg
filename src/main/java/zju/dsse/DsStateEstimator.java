package zju.dsse;

import org.apache.log4j.Logger;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsModelCons;
import zju.dsmodel.DsTopoIsland;
import zju.dsmodel.DsTopoNode;
import zju.measure.MeasVector;
import zju.measure.MeasVectorCreator;
import zju.measure.SystemMeasure;
import zju.se.IpoptSeAlg;
import zju.se.SeObjective;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-11-11
 */
public class DsStateEstimator implements DsModelCons {

    private static Logger log = Logger.getLogger(DsStateEstimator.class);

    private double k = 3;

    private double lambda = 4;

    private double alpha1 = 3, alpha2 = alpha1 * (1 + lambda);

    private DistriSys distriSys;

    private double[] initial;

    private boolean isFlatStart = true;

    private IpoptSeAlg alg;

    private DsSeResult result;

    private SystemMeasure sm;

    public DsStateEstimator(DistriSys distriSys, SystemMeasure sm) {
        this.distriSys = distriSys;
        this.sm = sm;
        alg = new IpoptDsSe();
    }

    public DsStateEstimator(DistriSys distriSys) {
        this.distriSys = distriSys;
        alg = new IpoptDsSe();
    }

    public void doSe() {
        int convergeNum = 0;
        for (DsTopoIsland island : distriSys.getActiveIslands()) {
            if (island.getBusV() == null)
                island.initialVariables();
            int count = 0;
            for (DsTopoNode tn : island.getTns())
                if (tn.getType() == DsTopoNode.TYPE_LINK)
                    count++;
            int[] linkBuses = new int[count];
            count = 0;
            for (DsTopoNode tn : island.getTns())
                if (tn.getType() == DsTopoNode.TYPE_LINK)
                    linkBuses[count++] = tn.getBusNo();
            MeasVector meas = new MeasVectorCreator().getMeasureVector(sm, true);//todo: check it
            alg.setSlackBusNum(1);//todo:
            if(alg instanceof IpoptDsSe) {
                ((IpoptDsSe)alg).setDsIsland(island);
            } else if(alg instanceof IpoptLcbSe) {
                ((IpoptLcbSe)alg).setDsIsland(island);
            }
            alg.setMeas(meas);
            alg.setZeroPBuses(linkBuses);
            initialObjFunc(meas, alg.getObjFunc());
            alg.doSeAnalyse();
            if (alg.isConverged())
                convergeNum++;
        }
        result = new DsSeResult();
        result.setActiveIslandNum(distriSys.getActiveIslands().length);
        result.setConvergedNum(convergeNum);
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

    public IpoptSeAlg getAlg() {
        return alg;
    }

    public double getK() {
        return k;
    }

    public void setK(double k) {
        this.k = k;
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    public double getAlpha1() {
        return alpha1;
    }

    public void setAlpha1(double alpha1) {
        this.alpha1 = alpha1;
    }

    public double getAlpha2() {
        return alpha2;
    }

    public void setAlpha2(double alpha2) {
        this.alpha2 = alpha2;
    }

    public double[] getInitial() {
        return initial;
    }

    public void setInitial(double[] initial) {
        this.initial = initial;
    }

    public boolean isFlatStart() {
        return isFlatStart;
    }

    public void setFlatStart(boolean flatStart) {
        isFlatStart = flatStart;
    }

    public DsSeResult getResult() {
        return result;
    }

    public void setAlg(IpoptSeAlg alg) {
        this.alg = alg;
    }
}
