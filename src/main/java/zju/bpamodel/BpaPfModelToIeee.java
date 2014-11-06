package zju.bpamodel;

import org.apache.log4j.Logger;
import zju.bpamodel.pf.AcLine;
import zju.bpamodel.pf.Bus;
import zju.bpamodel.pf.ElectricIsland;
import zju.bpamodel.pf.Transformer;
import zju.ieeeformat.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-7-25
 */
public class BpaPfModelToIeee {
    private static Logger log = Logger.getLogger(BpaPfModelToIeee.class);
    private Map<BusData, Bus> ieeeBusToBpaBus;
    private Map<BranchData, Object> ieeeBranchToBpaBranch;

    public IEEEDataIsland createIeeeIsland(ElectricIsland bpaIsland) {
        IEEEDataIsland ieeeIsland = new IEEEDataIsland();
        double baseMva = (Double) bpaIsland.getPfCtrlInfo().getCtrlProperties().get("MVA_BASE");

        Map<String, Integer> busNameToNumber = new HashMap<String, Integer>(bpaIsland.getBuses().size());
        int busNumber = 1;
        String busName;
        for (Bus bpaBus : bpaIsland.getBuses()) {
            busName = bpaBus.getName() + ((int) bpaBus.getBaseKv());
            if (!busNameToNumber.containsKey(busName))
                busNameToNumber.put(busName, busNumber++);
        }
        ieeeBusToBpaBus = new HashMap<BusData, Bus>(busNameToNumber.size());
        ArrayList<BusData> ieeeBuses = new ArrayList<BusData>(busNameToNumber.size());
        for (Bus bpaBus : bpaIsland.getBuses()) {
            BusData ieeeBus = new BusData();
            ieeeBusToBpaBus.put(ieeeBus, bpaBus);
            ieeeBus.setBusNumber(busNameToNumber.get(bpaBus.getName() + ((int) bpaBus.getBaseKv())));
            ieeeBus.setName(bpaBus.getName() + ((int) bpaBus.getBaseKv()));
            ieeeBus.setArea(1);
            ieeeBus.setLossZone(1); //暂不需要
            switch (bpaBus.getSubType()) {
                case 'E':
                case 'Q':
                case 'G':
                case 'F':
                case 'K':
                case 'L':
                case 'X':
                    ieeeBus.setType(BusData.BUS_TYPE_GEN_PV);
                    ieeeBus.setMaximum(bpaBus.getGenMvarMax());
                    ieeeBus.setMinimum(bpaBus.getGenMvarMin());
                    break;
                case 'S':
                    ieeeBus.setType(BusData.BUS_TYPE_SLACK);
                    ieeeBus.setFinalAngle(bpaBus.getSlackBusVAngle());
                    System.out.println("BUS_TYPE_SLACK:" + ieeeBus.getName());
                    break;
                case 'T':
                case 'C':
                case 'V':
                    ieeeBus.setType(BusData.BUS_TYPE_GEN_PQ);
                    ieeeBus.setMaximum(bpaBus.getvAmplMax() == 0 ? 1.15 : bpaBus.getvAmplMax());
                    ieeeBus.setMinimum(bpaBus.getvAmplMin() == 0 ? 0.85 : bpaBus.getvAmplMin());
                    ieeeBus.setGenerationMVAR(bpaBus.getGenMvar());
                    System.out.println("BUS_TYPE_GEN_PQ:" + ieeeBus.getName());
                    break;
                case 'J':
                    ieeeBus.setType(BusData.BUS_TYPE_LOAD_PQ);
                    break;
                case ' ':
                    ieeeBus.setType(BusData.BUS_TYPE_LOAD_PQ);
                    ieeeBus.setGenerationMVAR(bpaBus.getGenMvar());
                    break;
                default:
                    System.out.println("Error!bpaBus.getSubType():" + bpaBus.getSubType());
                    break;
            }
            ieeeBus.setFinalVoltage(bpaBus.getvAmplDesired() == 0 ? 1 : bpaBus.getvAmplDesired());
            ieeeBus.setDesiredVolt(bpaBus.getvAmplDesired() == 0 ? 1 : bpaBus.getvAmplDesired());
            ieeeBus.setLoadMW(bpaBus.getLoadMw());
            ieeeBus.setLoadMVAR(bpaBus.getLoadMvar());
            ieeeBus.setGenerationMW(bpaBus.getGenMw());
            ieeeBus.setBaseVoltage(bpaBus.getBaseKv());
            ieeeBus.setShuntConductance(bpaBus.getShuntMw() / baseMva);
            ieeeBus.setShuntSusceptance(bpaBus.getShuntMvar() / baseMva);
            if ((bpaBus.getSubType() == 'G' || bpaBus.getSubType() == 'X') && !bpaBus.getRemoteCtrlBusName().trim().equals(""))
                ieeeBus.setRemoteControlBusNumber(busNameToNumber.get(bpaBus.getRemoteCtrlBusName() + ((int) bpaBus.getRemoteCtrlBusBaseKv())));
            ieeeBuses.add(ieeeBus);
        }
        ieeeIsland.setBuses(ieeeBuses);

        int branchSize = bpaIsland.getAclines().size() + bpaIsland.getTransformers().size();
        ieeeBranchToBpaBranch = new HashMap<BranchData, Object>(branchSize);
        ArrayList<BranchData> ieeeBranches = new ArrayList<BranchData>(branchSize);
        String busName1, busName2;
        for (AcLine bpaAcLine : bpaIsland.getAclines()) {
            BranchData ieeeBranch = new BranchData();
            ieeeBranchToBpaBranch.put(ieeeBranch, bpaAcLine);
            busName1 = bpaAcLine.getBusName1() + ((int) bpaAcLine.getBaseKv1());
            busName2 = bpaAcLine.getBusName2() + ((int) bpaAcLine.getBaseKv2());
            ieeeBranch.setTapBusNumber(busNameToNumber.get(busName1));
            ieeeBranch.setZBusNumber(busNameToNumber.get(busName2));
            ieeeBranch.setLossZone(1); //暂不需要
            if (bpaAcLine.getCircuit() == ' ' || bpaAcLine.getCircuit() == '0') {
                ieeeBranch.setCircuit(1);
            } else if (bpaAcLine.getCircuit() >= '1' && bpaAcLine.getCircuit() <= '9') {
                ieeeBranch.setCircuit(bpaAcLine.getCircuit() - '1' + 1);
            } else if (bpaAcLine.getCircuit() >= 'A' && bpaAcLine.getCircuit() <= 'I') {
                ieeeBranch.setCircuit(bpaAcLine.getCircuit() - 'A' + 1);
            } else if (bpaAcLine.getCircuit() >= 'a' && bpaAcLine.getCircuit() <= 'i') {
                ieeeBranch.setCircuit(bpaAcLine.getCircuit() - 'a' + 1);
            } else {
                log.debug("错误：不能识别的Circuit符号，bpaAcline:" + busName1 + "," + busName2 + ",Circuit:" + bpaAcLine.getCircuit());
            }
            ieeeBranch.setType(BranchData.BRANCH_TYPE_ACLINE);  //输电线路为0
            ieeeBranch.setBranchR(bpaAcLine.getR());
            ieeeBranch.setBranchX(bpaAcLine.getX());
            ieeeBranch.setLineB(bpaAcLine.getHalfB() * 2);
            int mvaRating = (int) ((Math.sqrt(3) * bpaAcLine.getBaseI() / 1000) * (bpaAcLine.getBaseKv1() + bpaAcLine.getBaseKv2()) / 2);
            ieeeBranch.setMvaRating1(mvaRating);
            ieeeBranch.setMvaRating2(mvaRating);
            ieeeBranch.setMvaRating3(mvaRating);
            ieeeBranches.add(ieeeBranch);
        }
        for (Transformer bpaTransformer : bpaIsland.getTransformers()) {
            BranchData ieeeBranch = new BranchData();
            ieeeBranchToBpaBranch.put(ieeeBranch, bpaTransformer);
            busName1 = bpaTransformer.getBusName1() + ((int) bpaTransformer.getBaseKv1());
            busName2 = bpaTransformer.getBusName2() + ((int) bpaTransformer.getBaseKv2());
            switch (bpaTransformer.getSubType()) {
                case 'P':
                    ieeeBranch.setType(BranchData.BRANCH_TYPE_PHASE_SHIFTER);
                    ieeeBranch.setTapBusNumber(busNameToNumber.get(busName1));
                    ieeeBranch.setZBusNumber(busNameToNumber.get(busName2));
                    ieeeBranch.setBranchR(bpaTransformer.getR());
                    ieeeBranch.setBranchX(bpaTransformer.getX());
                    ieeeBranch.setLineB(bpaTransformer.getB());
                    ieeeBranch.setTransformerRatio(1);
                    ieeeBranch.setTransformerAngle(bpaTransformer.getPhaseAngle());
                    break;
                case ' ':
                    ieeeBranch.setType(BranchData.BRANCH_TYPE_TF_FIXED_TAP);
                    if (bpaTransformer.getTapKv1() != bpaTransformer.getBaseKv1()
                            && bpaTransformer.getTapKv2() == bpaTransformer.getBaseKv2()) {
                        ieeeBranch.setTapBusNumber(busNameToNumber.get(busName1));
                        ieeeBranch.setZBusNumber(busNameToNumber.get(busName2));
                        ieeeBranch.setTransformerRatio((bpaTransformer.getTapKv1() / bpaTransformer.getBaseKv1()));
                        ieeeBranch.setBranchR(bpaTransformer.getR());
                        ieeeBranch.setBranchX(bpaTransformer.getX());
                        ieeeBranch.setLineB(bpaTransformer.getB());
                    } else if (bpaTransformer.getTapKv1() == bpaTransformer.getBaseKv1()
                            && bpaTransformer.getTapKv2() != bpaTransformer.getBaseKv2()) {
                        ieeeBranch.setZBusNumber(busNameToNumber.get(busName1));
                        ieeeBranch.setTapBusNumber(busNameToNumber.get(busName2));
                        ieeeBranch.setTransformerRatio((bpaTransformer.getTapKv2() / bpaTransformer.getBaseKv2()));
                        ieeeBranch.setBranchR(bpaTransformer.getR());
                        ieeeBranch.setBranchX(bpaTransformer.getX());
                        ieeeBranch.setLineB(bpaTransformer.getB());
                    } else if (bpaTransformer.getTapKv1() == bpaTransformer.getBaseKv1()
                            && bpaTransformer.getTapKv2() == bpaTransformer.getBaseKv2()) {
                        ieeeBranch.setTapBusNumber(busNameToNumber.get(busName1));
                        ieeeBranch.setZBusNumber(busNameToNumber.get(busName2));
                        ieeeBranch.setTransformerRatio(1.0);
                        ieeeBranch.setBranchR(bpaTransformer.getR());
                        ieeeBranch.setBranchX(bpaTransformer.getX());
                        ieeeBranch.setLineB(bpaTransformer.getB());
                    } else {
                        ieeeBranch.setTapBusNumber(busNameToNumber.get(busName1));
                        ieeeBranch.setZBusNumber(busNameToNumber.get(busName2));
                        ieeeBranch.setTransformerRatio((bpaTransformer.getTapKv1() / bpaTransformer.getBaseKv1())
                                / (bpaTransformer.getTapKv2() / bpaTransformer.getBaseKv2()));
                        double ratio2 = (bpaTransformer.getTapKv2() / bpaTransformer.getBaseKv2());
                        double tmp = ratio2 * ratio2;
                        ieeeBranch.setBranchR(bpaTransformer.getR() * tmp);
                        ieeeBranch.setBranchX(bpaTransformer.getX() * tmp);
                        ieeeBranch.setLineB(bpaTransformer.getB() / tmp);
                    }
                    ieeeBranch.setTransformerAngle(0);
                    break;
                default:
                    log.debug("错误!bpaTransformer.getSubType():" + bpaTransformer.getSubType());
                    break;
            }
            if (bpaTransformer.getCircuit() == ' ' || bpaTransformer.getCircuit() == '0') {
                ieeeBranch.setCircuit(1);
            } else if (bpaTransformer.getCircuit() >= '1' && bpaTransformer.getCircuit() <= '9') {
                ieeeBranch.setCircuit(bpaTransformer.getCircuit() - '1' + 1);
            } else if (bpaTransformer.getCircuit() >= 'A' && bpaTransformer.getCircuit() <= 'I') {
                ieeeBranch.setCircuit(bpaTransformer.getCircuit() - 'A' + 1);
            } else if (bpaTransformer.getCircuit() >= 'a' && bpaTransformer.getCircuit() <= 'i') {
                ieeeBranch.setCircuit(bpaTransformer.getCircuit() - 'a' + 1);
            } else {
                log.debug("错误：不能识别的Circuit符号，bpaTransformer:" + busName1 + "," + busName2 + ",Circuit:" + bpaTransformer.getCircuit());
            }
            int mvaRating = (int) (bpaTransformer.getBaseMva());
            ieeeBranch.setMvaRating1(mvaRating);
            ieeeBranch.setMvaRating2(mvaRating);
            ieeeBranch.setMvaRating3(mvaRating);
            ieeeBranches.add(ieeeBranch);
        }
        ieeeIsland.setBranches(ieeeBranches);

        TitleData titleData = new TitleData();
        titleData.setDate("27/07/13");
        titleData.setOriginatorName("ZJU RDS");
        titleData.setMvaBase(baseMva);
        titleData.setYear(2013);
        titleData.setSeason('S');
        titleData.setCaseIdentification("Anhuisheng Test Case");
        ieeeIsland.setTitle(titleData);

        ArrayList<LossZoneData> lossZones = new ArrayList<LossZoneData>();
        ieeeIsland.setLossZones(lossZones);
        ArrayList<InterchangeData> interchangeDatas = new ArrayList<InterchangeData>();
        ieeeIsland.setInterchanges(interchangeDatas);
        ArrayList<TieLineData> tieLineDatas = new ArrayList<TieLineData>();
        ieeeIsland.setTieLines(tieLineDatas);

        return ieeeIsland;
    }

    public Map<BusData, Bus> getIeeeBusToBpaBus() {
        return ieeeBusToBpaBus;
    }

    public Map<BranchData, Object> getIeeeBranchToBpaBranch() {
        return ieeeBranchToBpaBranch;
    }
}

