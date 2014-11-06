package zju.ta;

import zju.bpamodel.BpaSwiModel;
import zju.bpamodel.pf.*;
import zju.bpamodel.pfr.PfResult;
import zju.bpamodel.swi.*;
import zju.util.MathUtil;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-12-31
 */
public class HpsBuilder {

    public static HeffronPhilipsSystem createHpSys(BpaSwiModel swiModel, ElectricIsland island, PfResult pfResult, Generator gen) {
        //List<Generator> toOptGen = ;
        //List<GeneratorDW> toOptGenDW = ;
        //List<Exciter> toOptExciter = ;
        //int i = ;
        //Generator gen = toOptGen.get(i);
        //Exciter exciter = toOptExciter.get(i);
        HeffronPhilipsSystem hpSys = new HeffronPhilipsSystem();
        hpSys.setSysBaseMva((Double) island.getPfCtrlInfo().getProperty(PfCtrlInfo.MVA_BASE));
        hpSys.setGen(gen);
        hpSys.setGenDw(swiModel.getGeneratorDwMap().get(gen));
        Exciter exciter = swiModel.getExciterMap().get(gen);
        hpSys.setExciter(exciter);
        if (exciter != null)
            hpSys.setExciterExtraInfo(swiModel.getExciterExtraInfoMap().get(exciter));
        hpSys.setGenBusPf(pfResult.getBusData().get(gen.getBusName()));

        boolean isParaOk = fillPara(island, pfResult, hpSys);//fill xe, high kv level bus
        if (!isParaOk)
            return null;
        return hpSys;
    }

    public static boolean fillPara(ElectricIsland island, PfResult pfResult, HeffronPhilipsSystem hpSys) {
        Generator gen = hpSys.getGen();
        List<Transformer> l;
        String busName1OfT = gen.getBusName();
        if (!island.getBusToTransformers().containsKey(gen.getBusName())) {
            if (!island.getBusToAclines().containsKey(gen.getBusName())) {
                //System.out.println("Suspicious generator bus connected to neither a transformer nor acline: " + gen.getBusName());
                return false;
            }
            //assertEquals(1, island.getBusToAclines().get(gen.getBusName()).size());
            AcLine acline = island.getBusToAclines().get(gen.getBusName()).get(0);
            busName1OfT = acline.getBusName1().equals(gen.getBusName()) ? acline.getBusName2() : acline.getBusName1();
            //assertTrue(island.getBusToTransformers().containsKey(anotherBusName));
            l = island.getBusToTransformers().get(busName1OfT);
            if (acline.getX() > 1e-4)
                System.out.println("Suspicious acline: " + acline.getBusName1() + " --> " + acline.getBusName2());
        } else
            l = island.getBusToTransformers().get(gen.getBusName());
        //assertEquals(1, l.size());
        Transformer t = l.get(0);
        String anotherBus = t.getBusName1().equals(busName1OfT) ? t.getBusName2() : t.getBusName1();
        l = island.getBusToTransformers().get(anotherBus);
        double maxKvLevel = t.getBaseKv1() < t.getBaseKv2() ? t.getBaseKv2() : t.getBaseKv1();
        String highBusName = t.getBaseKv1() < t.getBaseKv2() ? t.getBusName2() : t.getBusName1();
        if (l.size() == 3 && pfResult.getBusData().get(anotherBus).getBaseKv() < 2.0) {//its a virtual bus of three windings transformer
            Transformer highKvTransformer = t;
            Transformer[] transformers = new Transformer[2];
            transformers[0] = t;
            for (Transformer tr : l) {
                double kvLevel = tr.getBaseKv1() < tr.getBaseKv2() ? tr.getBaseKv2() : tr.getBaseKv1();
                String name = tr.getBaseKv1() < tr.getBaseKv2() ? tr.getBusName2() : tr.getBusName1();
                if (kvLevel > maxKvLevel) {
                    maxKvLevel = kvLevel;
                    highBusName = name;
                    highKvTransformer = tr;
                }
            }
            transformers[1] = highKvTransformer;
            hpSys.setTransformers(transformers);
            hpSys.setVirtualBus(island.getNameToBus().get(anotherBus));
        } else
            hpSys.setTransformers(new Transformer[]{t});
        hpSys.setGenBus(island.getNameToBus().get(gen.getBusName()));
        hpSys.setHighVBus(island.getNameToBus().get(highBusName));
        hpSys.setHighVBusPf(pfResult.getBusData().get(highBusName));
        hpSys.initialPara();
        return true;
    }


    public static boolean fillInfiniteBusInfo(TransferImpedanceC tic, HeffronPhilipsSystem hpSys) {
        double[] infiniteV = {tic.getInfiniteBusVx(), tic.getInfiniteBusVy()};
        MathUtil.trans_rect2polar(infiniteV);
        //System.out.println("X of high voltage bus to infinite bus:" + tic.getBusToInfiniteBusX());
        //System.out.println("Voltage of infinite bus:" + infiniteV[0] + " " + infiniteV[1] * 180 / Math.PI);

        Bus infiniteBus = new Bus();
        infiniteBus.setSubType('S');
        infiniteBus.setName("InfiniteBus");
        infiniteBus.setBaseKv(hpSys.getHighVBusPf().getBaseKv());
        infiniteBus.setvAmplDesired(infiniteV[0]);
        infiniteBus.setSlackBusVAngle(infiniteV[1]);
        hpSys.setInfiniteBus(infiniteBus);

        AcLine acLine1 = new AcLine();
        acLine1.setBusName1(hpSys.getHighVBus().getName());
        acLine1.setBusName2(hpSys.getInfiniteBus().getName());
        acLine1.setBaseKv1(hpSys.getHighVBus().getBaseKv());
        acLine1.setBaseKv2(hpSys.getInfiniteBus().getBaseKv());
        acLine1.setCircuit('1');
        acLine1.setX(2.0 * tic.getBusToInfiniteBusX());
        hpSys.setAclines(new AcLine[]{acLine1});

        AcLine acLine2 = new AcLine();
        acLine2.setBusName1(hpSys.getHighVBus().getName());
        acLine2.setBusName2(hpSys.getInfiniteBus().getName());
        acLine2.setBaseKv1(hpSys.getHighVBus().getBaseKv());
        acLine2.setBaseKv2(hpSys.getInfiniteBus().getBaseKv());
        acLine2.setCircuit('2');
        acLine2.setX(2.0 * tic.getBusToInfiniteBusX());
        hpSys.setAclines(new AcLine[]{acLine1, acLine2});
        return false;
    }

    public static void writeSwiFile(HeffronPhilipsSystem hpSys, String fileName, boolean printOnScreen) throws IOException {
        writeSwiFile(hpSys, fileName, "GBK", printOnScreen);
    }
    public static void writeSwiFile(HeffronPhilipsSystem hpSys, String fileName, String charset, boolean printOnScreen) throws IOException {
        String caseCard = "CASE 2012nd           1                     0.65 0.65 0.03 0.05 0.04 0.3  0.36";
        String ffCard = "FF      0.5 600.                                         1         1     1";
        String mhCard = "MH 15";
        String mhcCard = "MHC 50. 1000";
        StringBuilder swiStr = new StringBuilder();
        swiStr.append(".#define bsefile \"2012smib.bse\"").append("\n");
        swiStr.append(caseCard).append("\n");
        //fault information
        AcLine acLine = hpSys.getAclines()[0];
        ShortCircuitFault fault = new ShortCircuitFault();
        fault.setBusAName(acLine.getBusName1());
        fault.setBusBName(acLine.getBusName2());
        fault.setBusABaseKv(acLine.getBaseKv1());
        fault.setBusBBaseKv(acLine.getBaseKv2());
        fault.setParallelBranchCode(acLine.getCircuit());
        fault.setMode(3);
        fault.setStartCycle(0.0);
        fault.setPosPercent(0.5);
        swiStr.append(fault.toString()).append("\n");
        fault.setMode(-3);
        fault.setBusASign('-');
        fault.setBusBSign('-');
        fault.setStartCycle(5.0);
        swiStr.append(fault.toString()).append("\n");

        //generator, exciter and other devices parameters
        swiStr.append(hpSys.toBpaSwiData());

        BusOutput busOutput = new BusOutput();
        busOutput.setBusName(hpSys.getGenBus().getName());
        busOutput.setBaseKv(hpSys.getGenBus().getBaseKv());
        busOutput.setVoltageAmpl(2);

        //output control information
        GenOutput genOutput = new GenOutput();
        genOutput.setGenBusName(hpSys.getGen().getBusName());
        genOutput.setBaseKv(hpSys.getGen().getBaseKv());
        genOutput.setAngle(2);
        genOutput.setGenMvar(2);

        swiStr.append(ffCard).append("\n");
        swiStr.append("90").append("\n");
        swiStr.append(mhCard).append("\n");
        swiStr.append(mhcCard).append("\n");
        swiStr.append(busOutput.toString()).append("\n");
        swiStr.append(genOutput.toString()).append("\n");
        swiStr.append("99").append("\n");
        if (printOnScreen)
            System.out.println(swiStr);

        FileOutputStream stream = new FileOutputStream(fileName + ".swi");
        OutputStreamWriter writer = new OutputStreamWriter(stream, charset);

        writer.write(swiStr.toString());
        writer.close();
    }

    public static void writePfFile(HeffronPhilipsSystem hpSys, String fileName, boolean printOnScreen) throws IOException {
        writePfFile(hpSys, fileName, "GBK", printOnScreen);
    }

    public static void writePfFile(HeffronPhilipsSystem hpSys, String fileName, String charset, boolean printOnScreen) throws IOException {
        String pfContrlCard = "(POWERFLOW,CASEID=2012nd,PROJECT=2012nd)\n" +
                "/SOL_ITER,DECOUPLED=4,NEWTON=20,OPITM=0\\\n" +
                "/P_OUTPUT_LIST,ZONES=ALL\\\n" +
                "/RPT_SORT=ZONE\\  \n" +
                "/OVERLOAD_RPT,TX=80,LINE=80\\\n" +
                "/P_ANALYSIS_RPT,LEVEL=1\\\n" +
                "/ANALYSIS_SELECT\\\n" +
                ">PAPER,ZONES=0,1,2,3,4,5,6,7,8,9<\n" +
                ">UVOV<\n" +
                ">LINELOAD<\n" +
                ">TRANLOAD<\n" +
                "/NEW_BASE,FILE=2012smib.BSE\\\n" +
                "/PF_MAP,FILE=2012smib.MAP\\\n" +
                "/NETWORK_DATA\\";

        StringBuilder pfStr = new StringBuilder();
        pfStr.append(pfContrlCard).append("\n");
        pfStr.append(hpSys.toBpaPfData());
        if (printOnScreen)
            System.out.println(pfStr);

        FileOutputStream stream = new FileOutputStream(fileName + ".dat");
        OutputStreamWriter writer = new OutputStreamWriter(stream, charset);
        writer.write(pfStr.toString());
        writer.close();
    }
}
