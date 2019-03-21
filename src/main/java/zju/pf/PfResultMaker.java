package zju.pf;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.AVector;
import zju.util.StateCalByPolar;
import zju.util.YMatrixGetter;

import java.util.HashMap;
import java.util.Map;

/**
 * This class provides static methods generating complete power flow results.
 * <br>V theta or real and image parts of voltage in rectangular coordinate, number optimization result are needed by these methods.</br>
 *
 * @author Dong Shufeng
 *         Date: 2008-5-25
 */
public class PfResultMaker {
    private static Logger log = LogManager.getLogger(PfResultMaker.class);

    /**
     * @param island IEEE format data
     * @param vTheta voltage in polar coordinate
     * @param Y      admittance matrix
     * @return power flow result
     */
    public static PfResultInfo getResult(IEEEDataIsland island, AVector vTheta, YMatrixGetter Y) {
        return getResult(island, vTheta, Y, null);
    }

    public static PfResultInfo getResult(IEEEDataIsland island, AVector vtheta, YMatrixGetter Y, Map<Integer, Integer> oldToNew) {
        return getResult(island, vtheta.getValues(), Y, oldToNew);
    }

    public static PfResultInfo getResult(IEEEDataIsland island, double[] vtheta, YMatrixGetter Y, Map<Integer, Integer> oldToNew) {
        PfResultInfo result = new PfResultInfo();
        int n = island.getBuses().size();
        //system state data
        double genPCapacity = 0.0;//todo: generator capacity has not been filled yet
        double genQCapacity = 0.0;
        double linePLossTotal = 0.0;
        double lineQLossTotal = 0.0;
        double transformerPLossTotal = 0.0;
        double transformerQLossTotal = 0.0;
        double genPTotal = 0.0;
        double genQTotal = 0.0;
        double loadPTotal = 0.0;
        double loadQTotal = 0.0;

        //bus data
        Map<Integer, Double> busV = new HashMap<Integer, Double>(n);
        Map<Integer, Double> busTheta = new HashMap<Integer, Double>(n);
        Map<Integer, Double> busP = new HashMap<Integer, Double>(n);
        Map<Integer, Double> busQ = new HashMap<Integer, Double>(n);
        Map<Integer, Double> busPGen = new HashMap<Integer, Double>(n);
        Map<Integer, Double> busQGen = new HashMap<Integer, Double>(n);
        Map<Integer, Double> busPLoad = new HashMap<Integer, Double>(n);
        Map<Integer, Double> busQLoad = new HashMap<Integer, Double>(n);

        //branch data
        Map<Integer, Double> branchPLoss = new HashMap<Integer, Double>(island.getBranches().size());
        Map<Integer, Double> branchQLoss = new HashMap<Integer, Double>(island.getBranches().size());
        double baseMVA = island.getTitle().getMvaBase();
        for (BusData bus : island.getBuses()) {
            int busNum = bus.getBusNumber();
            Integer newBusNum = oldToNew != null ? oldToNew.get(busNum) : busNum;

            bus.setFinalVoltage(vtheta[newBusNum - 1]);
            bus.setFinalAngle((vtheta[newBusNum - 1 + n]) * 180.0 / Math.PI);

            busV.put(busNum, vtheta[newBusNum - 1]);
            busTheta.put(busNum, vtheta[newBusNum - 1 + n]);
            busP.put(busNum, StateCalByPolar.calBusP(newBusNum, Y, vtheta) * baseMVA);
            busQ.put(busNum, StateCalByPolar.calBusQ(newBusNum, Y, vtheta) * baseMVA);
            switch (bus.getType()) {
                case BusData.BUS_TYPE_GEN_PQ://Gen PQ
                    busPGen.put(busNum, bus.getGenerationMW());
                    busQGen.put(busNum, bus.getGenerationMVAR());
                    busPLoad.put(busNum, bus.getGenerationMW() - busP.get(busNum));
                    busQLoad.put(busNum, bus.getGenerationMVAR() - busQ.get(busNum));
                    break;
                case BusData.BUS_TYPE_LOAD_PQ: //Load PQ
                case BusData.BUS_TYPE_GEN_PV://Gen PV
                case BusData.BUS_TYPE_SLACK://Slack bus
                    busPGen.put(busNum, busP.get(busNum) + bus.getLoadMW());
                    busQGen.put(busNum, busQ.get(busNum) + bus.getLoadMVAR());
                    busPLoad.put(busNum, bus.getLoadMW());
                    busQLoad.put(busNum, bus.getLoadMVAR());
                    break;
                default:
                    log.error("no such type of bus, type = " + bus.getType());
                    break;
            }
            bus.setGenerationMW(busPGen.get(busNum));
            bus.setGenerationMVAR(busQGen.get(busNum));
            bus.setLoadMW(busPLoad.get(busNum));
            bus.setLoadMVAR(busQLoad.get(busNum));
            genPTotal += bus.getGenerationMW();
            genQTotal += bus.getGenerationMVAR();
            loadPTotal += bus.getLoadMW();
            loadQTotal += bus.getLoadMVAR();
        }

        for (BranchData branch : island.getBranches()) {
            double pFrom = StateCalByPolar.calLinePFrom(branch.getId(), Y, vtheta) * baseMVA;
            double pTo = StateCalByPolar.calLinePTo(branch.getId(), Y, vtheta) * baseMVA;
            double qFrom = StateCalByPolar.calLineQFrom(branch.getId(), Y, vtheta) * baseMVA;
            double qTo = StateCalByPolar.calLineQTo(branch.getId(), Y, vtheta) * baseMVA;
            double v1 = pFrom + pTo;
            branchPLoss.put(branch.getId(), v1);
            double v2 = qFrom + qTo;
            branchQLoss.put(branch.getId(), v2);
            if (branch.getType() == BranchData.BRANCH_TYPE_ACLINE) {
                linePLossTotal += v1;
                lineQLossTotal += v2;
            } else {
                transformerPLossTotal += v1;
                transformerQLossTotal += v2;
            }
        }
        //fill system state data
        result.setIsland(island);

        result.setGenPCapacity(genPCapacity);
        result.setGenQCapacity(genQCapacity);
        result.setLinePLossTotal(linePLossTotal);
        result.setLineQLossTotal(lineQLossTotal);
        result.setTransformerPLossTotal(transformerPLossTotal);
        result.setTransformerQLossTotal(transformerQLossTotal);
        result.setGenPTotal(genPTotal);
        result.setGenQTotal(genQTotal);
        result.setLoadPTotal(loadPTotal);
        result.setLoadQTotal(loadQTotal);
        result.setPLossTotal(linePLossTotal + transformerPLossTotal);
        result.setQLossTotal(lineQLossTotal + transformerQLossTotal);

        //fill bus data
        result.setBusV(busV);
        result.setBusTheta(busTheta);
        result.setBusP(busP);
        result.setBusQ(busQ);
        result.setBusPGen(busPGen);
        result.setBusQGen(busQGen);
        result.setBusPLoad(busPLoad);
        result.setBusQLoad(busQLoad);

        //fill branch data
        result.setBranchPLoss(branchPLoss);
        result.setBranchQLoss(branchQLoss);
        return result;
    }

    public static void setVTheta(IEEEDataIsland island, AVector vtheta, Map<Integer, Integer> oldToNew) {
        int n = island.getBuses().size();
        for (BusData bus : island.getBuses()) {
            int busNum = bus.getBusNumber();
            Integer newBusNum = oldToNew != null ? oldToNew.get(busNum) : busNum;

            bus.setFinalVoltage(vtheta.getValue(newBusNum - 1));
            bus.setFinalAngle((vtheta.getValue(newBusNum - 1 + n)) * 180.0 / Math.PI);
        }
    }

    public static void fillResult(IEEEDataIsland island, YMatrixGetter Y, AVector vtheta, Map<Integer, Integer> oldToNew) {
        fillResult(island, Y, vtheta.getValues(), oldToNew);
    }

    public static void fillResult(IEEEDataIsland island, YMatrixGetter Y, double[] vtheta, Map<Integer, Integer> oldToNew) {
        int n = island.getBuses().size();
        double baseMVA = island.getTitle().getMvaBase();
        for (BusData bus : island.getBuses()) {
            int busNum = bus.getBusNumber();
            busNum = oldToNew != null ? oldToNew.get(busNum) : busNum;

            bus.setFinalVoltage(vtheta[busNum - 1]);
            bus.setFinalAngle((vtheta[busNum - 1 + n]) * 180.0 / Math.PI);

            double busP = StateCalByPolar.calBusP(busNum, Y, vtheta) * baseMVA;
            double busQ = StateCalByPolar.calBusQ(busNum, Y, vtheta) * baseMVA;
            double busPGen = 0, busQGen = 0, busPLoad = 0, busQLoad = 0;
            switch (bus.getType()) {
                case BusData.BUS_TYPE_GEN_PQ://Gen PQ
                    busPGen = bus.getGenerationMW();
                    busQGen = bus.getGenerationMVAR();
                    busPLoad = bus.getGenerationMW() - busP;
                    busQLoad = bus.getGenerationMVAR() - busQ;
                    break;
                case BusData.BUS_TYPE_LOAD_PQ: //Load PQ
                case BusData.BUS_TYPE_GEN_PV://Gen PV
                case BusData.BUS_TYPE_SLACK://Slack bus
                    busPGen = busP + bus.getLoadMW();
                    busQGen = busQ + bus.getLoadMVAR();
                    busPLoad = bus.getLoadMW();
                    busQLoad = bus.getLoadMVAR();
                    break;
                default:
                    log.error("no such type of bus, type = " + bus.getType());
                    break;
            }
            bus.setGenerationMW(busPGen);
            bus.setGenerationMVAR(busQGen);
            bus.setLoadMW(busPLoad);
            bus.setLoadMVAR(busQLoad);
        }
    }

    public static void fillResult(IEEEDataIsland clonedIsland, YMatrixGetter Y, double[] variableState) {
        fillResult(clonedIsland, Y, variableState, null);
    }
}
