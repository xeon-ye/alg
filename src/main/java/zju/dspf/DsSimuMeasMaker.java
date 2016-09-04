package zju.dspf;

import zju.devmodel.MapObject;
import zju.dsmodel.DsTopoIsland;
import zju.dsmodel.DsTopoNode;
import zju.measure.MeasTypeCons;
import zju.measure.MeasureInfo;
import zju.measure.SystemMeasure;
import zju.pf.SimuMeasMaker;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2011-5-10
 */
public class DsSimuMeasMaker implements MeasTypeCons {

    public SystemMeasure createFullMeasure_withBadData(DsTopoIsland island, int errorDistribution, double rate) {
        SystemMeasure sm = createFullMeasure(island, errorDistribution);
        SimuMeasMaker.addBadData(sm, rate);
        return sm;
    }

    public SystemMeasure createFullMeasure(DsTopoIsland island, int errorDistribution) {
        return createFullMeasure(island, errorDistribution, 0.02);
    }

    /**
     * this method using power flow result as true value and add random noise as measurement value
     *
     * @param errorDistribution 0: equality 1:Gauss
     * @param ratio             error limit
     * @return every bus has injection p, q  and thresholds1 measurement, every branch has power measurement
     */
    public SystemMeasure createFullMeasure(DsTopoIsland island, int errorDistribution, double ratio) {
        return createMeasureOfTypes(island, new int[]{
                TYPE_BUS_ACTIVE_POWER,
                TYPE_BUS_REACTIVE_POWER,
                TYPE_BUS_VOLOTAGE,
                TYPE_LINE_FROM_ACTIVE,
                TYPE_LINE_FROM_REACTIVE,
                TYPE_LINE_TO_ACTIVE,
                TYPE_LINE_TO_REACTIVE,
                TYPE_LINE_FROM_CURRENT,
                TYPE_LINE_TO_CURRENT,
        }, errorDistribution, ratio);
    }

    /**
     * this method using powerflow result as true value and add random noise as measurement value
     *
     * @param types             measure types you want to return
     * @param errorDistribution 0: equality 1:Gauss
     * @param ratio             error limit
     * @return return system measure of the types you set
     */
    public SystemMeasure createMeasureOfTypes(DsTopoIsland island, int[] types, int errorDistribution, double ratio) {
        DsStateCal cal = new DsStateCal(island);
        SystemMeasure sm = new SystemMeasure();
        for (int type : types) {
            switch (type) {
                case TYPE_BUS_ACTIVE_POWER:
                    for (DsTopoNode tn : island.getTns()) {
                        if (tn.getType() == DsTopoNode.TYPE_PQ) {
                            for (int i : tn.getPhases()) {
                                double trueValue = cal.calBusPQ(tn, i)[0];
                                String positionId = tn.getTnNo() + "_" + (i);
                                MeasureInfo injectionP = new MeasureInfo(positionId, TYPE_BUS_ACTIVE_POWER, 0);
                                SimuMeasMaker.formMeasure(trueValue, errorDistribution, ratio, injectionP);
                                sm.addEfficientMeasure(injectionP);
                            }
                        }
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (DsTopoNode tn : island.getTns()) {
                        if (tn.getType() == DsTopoNode.TYPE_PQ) {
                            for (int i : tn.getPhases()) {
                                double trueValue = cal.calBusPQ(tn, i)[1];
                                String positionId = tn.getTnNo() + "_" + (i);
                                MeasureInfo injectionQ = new MeasureInfo(positionId, TYPE_BUS_REACTIVE_POWER, 0);
                                SimuMeasMaker.formMeasure(trueValue, errorDistribution, ratio, injectionQ);
                                sm.addEfficientMeasure(injectionQ);
                            }
                        }
                    }
                    break;
                case TYPE_BUS_VOLOTAGE:
                    for (DsTopoNode tn : island.getTns()) {
                        if(!island.getBusV().containsKey(tn))
                            continue;
                        for (int i : tn.getPhases()) {
                            double[] v = island.getBusV().get(tn)[i];
                            if (island.isVCartesian()) {
                                //double trueValue = Math.sqrt(v[0] * v[0] + v[1] * v[1]);
                                double trueValue = v[0] * v[0] + v[1] * v[1]; //todo;
                                String positionId = tn.getTnNo() + "_" + (i);
                                MeasureInfo busV = new MeasureInfo(positionId, TYPE_BUS_VOLOTAGE, 0);
                                SimuMeasMaker.formMeasure(trueValue, errorDistribution, ratio, busV);
                                sm.addEfficientMeasure(busV);
                            } else {
                                double trueValue = v[0];
                                String positionId = tn.getTnNo() + "_" + (i);
                                MeasureInfo busV = new MeasureInfo(positionId, TYPE_BUS_VOLOTAGE, 0);
                                SimuMeasMaker.formMeasure(trueValue, errorDistribution, ratio, busV);
                                sm.addEfficientMeasure(busV);
                            }
                        }
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (Integer branchId : island.getIdToBranch().keySet()) {
                        MapObject obj = island.getIdToBranch().get(branchId);
                        for (int i = 0; i < 3; i++) {
                            if (!island.getBranches().get(obj).containsPhase(i))
                                continue;
                            double trueValue = cal.calLinePQFrom(obj, i)[0];
                            String positionId = branchId + "_" + (i);
                            MeasureInfo lineFromP = new MeasureInfo(positionId, TYPE_LINE_FROM_ACTIVE, 0);
                            SimuMeasMaker.formMeasure(trueValue, errorDistribution, ratio, lineFromP);
                            sm.addEfficientMeasure(lineFromP);
                        }
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (Integer branchId : island.getIdToBranch().keySet()) {
                        MapObject obj = island.getIdToBranch().get(branchId);
                        for (int i = 0; i < 3; i++) {
                            if (!island.getBranches().get(obj).containsPhase(i))
                                continue;
                            double trueValue = cal.calLinePQFrom(obj, i)[1];
                            String positionId = branchId + "_" + (i);
                            MeasureInfo lineFromQ = new MeasureInfo(positionId, TYPE_LINE_FROM_REACTIVE, 0);
                            SimuMeasMaker.formMeasure(trueValue, errorDistribution, ratio, lineFromQ);
                            sm.addEfficientMeasure(lineFromQ);
                        }
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (Integer branchId : island.getIdToBranch().keySet()) {
                        MapObject obj = island.getIdToBranch().get(branchId);
                        DsTopoNode tn1 = island.getGraph().getEdgeSource(obj);
                        DsTopoNode tn2 = island.getGraph().getEdgeTarget(obj);
                        DsTopoNode tn = tn1.getTnNo() > tn2.getTnNo() ? tn1 : tn2;
                        if(!island.getBusV().containsKey(tn))
                            continue;
                        for (int i = 0; i < 3; i++) {
                            if (!island.getBranches().get(obj).containsPhase(i))
                                continue;
                            double trueValue = cal.calLinePQTo(obj, i)[0];
                            String positionId = branchId + "_" + (i);
                            MeasureInfo lineToP = new MeasureInfo(positionId, TYPE_LINE_TO_ACTIVE, 0);
                            SimuMeasMaker.formMeasure(trueValue, errorDistribution, ratio, lineToP);
                            sm.addEfficientMeasure(lineToP);
                        }
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (Integer branchId : island.getIdToBranch().keySet()) {
                        MapObject obj = island.getIdToBranch().get(branchId);
                        DsTopoNode tn1 = island.getGraph().getEdgeSource(obj);
                        DsTopoNode tn2 = island.getGraph().getEdgeTarget(obj);
                        DsTopoNode tn = tn1.getTnNo() > tn2.getTnNo() ? tn1 : tn2;
                        if(!island.getBusV().containsKey(tn))
                            continue;
                        for (int i = 0; i < 3; i++) {
                            if (!island.getBranches().get(obj).containsPhase(i))
                                continue;
                            double trueValue = cal.calLinePQTo(obj, i)[1];
                            String positionId = branchId + "_" + (i);
                            MeasureInfo lineToQ = new MeasureInfo(positionId, TYPE_LINE_TO_REACTIVE, 0);
                            SimuMeasMaker.formMeasure(trueValue, errorDistribution, ratio, lineToQ);
                            sm.addEfficientMeasure(lineToQ);
                        }
                    }
                    break;
                case TYPE_LINE_FROM_CURRENT:
                    for (Integer branchId : island.getIdToBranch().keySet()) {
                        MapObject obj = island.getIdToBranch().get(branchId);
                        for (int i = 0; i < 3; i++) {
                            if (!island.getBranches().get(obj).containsPhase(i))
                                continue;
                            double[] c = island.getBranchHeadI().get(obj)[i];
                            double trueValue;
                            if (island.isICartesian()) {
                                //trueValue = Math.sqrt(c[0] * c[0] + c[1] * c[1]);
                                trueValue = c[0] * c[0] + c[1] * c[1]; //todo
                            } else
                                trueValue = c[0];
                            String positionId = branchId + "_" + (i);
                            MeasureInfo branchHeadI = new MeasureInfo(positionId, TYPE_LINE_FROM_CURRENT, 0);
                            SimuMeasMaker.formMeasure(trueValue, errorDistribution, ratio, branchHeadI);
                            sm.addEfficientMeasure(branchHeadI);
                        }
                    }
                    break;
                case TYPE_LINE_TO_CURRENT:
                    for (Integer branchId : island.getIdToBranch().keySet()) {
                        MapObject obj = island.getIdToBranch().get(branchId);
                        double[][] c = island.getBranchTailI().get(obj);
                        if (island.getBranchHeadI().get(obj) == c)
                            continue;
                        double trueValue;
                        for (int i = 0; i < 3; i++) {
                            if (!island.getBranches().get(obj).containsPhase(i))
                                continue;
                            if (island.isICartesian()) {
                                //trueValue = Math.sqrt(c[0] * c[0] + c[1] * c[1]);
                                trueValue = c[i][0] * c[i][0] + c[i][1] * c[i][1]; //todo
                            } else
                                trueValue = c[i][0];
                            String positionId = branchId + "_" + (i);
                            MeasureInfo branchTailI = new MeasureInfo(positionId, TYPE_LINE_TO_CURRENT, 0);
                            SimuMeasMaker.formMeasure(trueValue, errorDistribution, ratio, branchTailI);
                            sm.addEfficientMeasure(branchTailI);
                        }
                    }
                    break;
                default:
                    break;
            }
        }
        return sm;
    }
}
