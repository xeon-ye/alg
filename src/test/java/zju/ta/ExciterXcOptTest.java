package zju.ta;

import junit.framework.TestCase;
import zju.bpamodel.BpaPfModelParser;
import zju.bpamodel.BpaPfResultParser;
import zju.bpamodel.BpaSwiModel;
import zju.bpamodel.BpaSwiModelParser;
import zju.bpamodel.pf.ElectricIsland;
import zju.bpamodel.pfr.PfResult;
import zju.bpamodel.sccpc.SccResult;
import zju.bpamodel.swi.Generator;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * ExciterXcOpt Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>07/22/2012</pre>
 */
public class ExciterXcOptTest extends TestCase {

    //public List<String> toFound;
    public List<String> genNames = new ArrayList<String>();

    public ExciterXcOptTest(String name) throws IOException {
        super(name);
        InputStream stream = this.getClass().getResourceAsStream("/bpafiles/fj/gennames.txt");
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream, "UTF-8"));
        String s;
        while ((s = reader.readLine()) != null)
            if (!s.startsWith("."))
                genNames.add(s.trim());
        reader.close();
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void testCaseFj_2012() throws IOException {
        long start = System.currentTimeMillis();
        InputStream stream = this.getClass().getResourceAsStream("/bpafiles/fj/2012年稳定20111103.swi");
        BpaSwiModel swiModel = BpaSwiModelParser.parse(stream, "GBK");
        System.out.println("Time used for parsing swi model: " + (System.currentTimeMillis() - start) + "ms");

        start = System.currentTimeMillis();
        stream = this.getClass().getResourceAsStream("/bpafiles/fj/2012年年大方式20111103.dat");
        ElectricIsland island = BpaPfModelParser.parse(stream, "GBK");
        stream = this.getClass().getResourceAsStream("/bpafiles/fj/2012年年大方式20111103.pfo");
        PfResult pfResult = BpaPfResultParser.parse(stream, "GBK");
        System.out.println("Time used for parsing pf model: " + (System.currentTimeMillis() - start) + "ms");

        start = System.currentTimeMillis();
        stream = this.getClass().getResourceAsStream("/bpafiles/fj/2012年年大方式20111103.lis");
        SccResult sccResult = SccResult.parse(stream, "GBK");
        System.out.println("Time used for parsing scc result: " + (System.currentTimeMillis() - start) + "ms");

        testCaseFj(island, pfResult, swiModel, sccResult);
        //FileOutputStream out = new FileOutputStream("2012年稳定20111103-opted.swi");
        //InputStream in = this.getClass().getResourceAsStream("/tacase/fj/2012年稳定20111103.swi");
        //BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", swiModel);
    }

    public void testCaseFj_2013() throws IOException {
        long start = System.currentTimeMillis();
        InputStream stream = this.getClass().getResourceAsStream("/bpafiles/fj/2013年年大方式.swi");
        BpaSwiModel swiModel = BpaSwiModelParser.parse(stream, "GBK");
        System.out.println("Time used for parsing swi model: " + (System.currentTimeMillis() - start) + "ms");

        start = System.currentTimeMillis();
        stream = this.getClass().getResourceAsStream("/bpafiles/fj/2013年年大方式.dat");
        ElectricIsland island = BpaPfModelParser.parse(stream, "GBK");
        stream = this.getClass().getResourceAsStream("/bpafiles/fj/2013年年大方式.pfo");
        PfResult pfResult = BpaPfResultParser.parse(stream, "GBK");
        System.out.println("Time used for parsing pf model: " + (System.currentTimeMillis() - start) + "ms");

        start = System.currentTimeMillis();
        stream = this.getClass().getResourceAsStream("/bpafiles/fj/2013年年大方式.lis");
        SccResult sccResult = SccResult.parse(stream, "GBK");
        System.out.println("Time used for parsing scc result: " + (System.currentTimeMillis() - start) + "ms");

        testCaseFj(island, pfResult, swiModel, sccResult);
        //FileOutputStream out = new FileOutputStream("2013年年大方式-opted.swi");
        //InputStream in = this.getClass().getResourceAsStream("/tacase/fj/2013年年大方式.swi");
        //BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", swiModel);
    }

    public void testCaseFj_2014() throws IOException {
        long start = System.currentTimeMillis();
        InputStream stream = this.getClass().getResourceAsStream("/bpafiles/fj/2014年稳定20120218.swi");
        BpaSwiModel swiModel = BpaSwiModelParser.parse(stream, "GBK");
        System.out.println("Time used for parsing swi model: " + (System.currentTimeMillis() - start) + "ms");

        start = System.currentTimeMillis();
        stream = this.getClass().getResourceAsStream("/bpafiles/fj/2014年年大方式20120326.dat");
        ElectricIsland island = BpaPfModelParser.parse(stream, "GBK");
        stream = this.getClass().getResourceAsStream("/bpafiles/fj/2014年年大方式20120326.pfo");
        PfResult pfResult = BpaPfResultParser.parse(stream, "GBK");
        System.out.println("Time used for parsing pf model: " + (System.currentTimeMillis() - start) + "ms");

        start = System.currentTimeMillis();
        stream = this.getClass().getResourceAsStream("/bpafiles/fj/2013年年大方式.lis");
        SccResult sccResult = SccResult.parse(stream, "GBK");
        System.out.println("Time used for parsing scc result: " + (System.currentTimeMillis() - start) + "ms");

        testCaseFj(island, pfResult, swiModel, sccResult);
        //FileOutputStream out = new FileOutputStream("2014年稳定20120218-opted.swi");
        //InputStream in = this.getClass().getResourceAsStream("/tacase/fj/2014年稳定20120218.swi");
        //BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", swiModel);
    }

    public void testCaseFj(ElectricIsland island, PfResult pfResult, BpaSwiModel swiModel, SccResult sccResult) throws IOException {
        Map<String, Generator> toOptGen = new HashMap<String, Generator>();
        for (String name : genNames) {
            for (Generator gen : swiModel.getGenerators()) {
                if (gen.getBusName().equals(name)) {
                    toOptGen.put(name, gen);
                }
            }
        }

        ExciterXcOpt exciterXcOpt = new ExciterXcOpt();
        long start = System.currentTimeMillis();
        Map<Generator, String> generatorToHighVBus = new HashMap<Generator, String>();
        Map<String, List<Generator>> shuntGenerators = new HashMap<String, List<Generator>>();
        Map<Generator, HeffronPhilipsSystem> generatorToHps = new HashMap<Generator, HeffronPhilipsSystem>();
        for (Generator gen : toOptGen.values()) {
            if (gen == null)
                continue;
            HeffronPhilipsSystem hpSys = HpsBuilder.createHpSys(swiModel, island, pfResult, gen);
            if (hpSys == null)
                continue;
            generatorToHps.put(gen, hpSys);
            generatorToHighVBus.put(gen, hpSys.getHighVBus().getName());
            String highVbusName = hpSys.getHighVBus().getName();
            if (!shuntGenerators.containsKey(highVbusName))
                shuntGenerators.put(highVbusName, new ArrayList<Generator>());
            shuntGenerators.get(highVbusName).add(gen);
        }
        for (String genName : genNames) {
            Generator gen = toOptGen.get(genName);
            if (gen == null) {
                System.out.println(genName);
                continue;
            }
            HeffronPhilipsSystem hpSys = generatorToHps.get(gen);
            if (hpSys == null) {
                System.out.println(gen.getBusName());
                continue;
            }
            String highVbusName = hpSys.getHighVBus().getName();
            if (!shuntGenerators.containsKey(highVbusName)) {
                System.out.println(gen.getBusName());
                continue;
            }
            if (!sccResult.getBusData().containsKey(hpSys.getHighVBusPf().getName())) {
                System.out.println(gen.getBusName());
                continue;
            }
            TransferImpedanceC tic = new TransferImpedanceC();
            tic.setHpSys(hpSys);
            tic.setShuntResult(sccResult.getBusData().get(hpSys.getHighVBusPf().getName()));
            tic.cal2();
            tic.setBusToInfiniteBusX(tic.getBusToInfiniteBusX() * shuntGenerators.get(highVbusName).size());
            if (HpsBuilder.fillInfiniteBusInfo(tic, hpSys))
                continue;
            generatorToHighVBus.put(gen, hpSys.getHighVBus().getName());

            double xt = hpSys.getTransformerX();//xt is not equal to xe for every cases
            if (!island.getBusToTransformers().containsKey(gen.getBusName())) {
                System.out.println("=========== 机端并联 " + gen.getBusName() + "====== ");
                xt = 0;
            }
            //double ratedPFactor = Math.cos(hpSys.getGenBusPf().getAngleInArc() - hpSys.getiAngleInArc());
            double ratedPFactor = 0.85;
            double k = 1 / Math.sqrt(1 - ratedPFactor * ratedPFactor);
            double toReached1 = 0.03 * k;
            double toReached2 = 10.04 * k;
            //from Zhu Yongming's paper
            //xt = xt * hpSys.getiAmpl() / hpSys.getGenBusPf().getvInPu();
            //double deltaT1 = toReached1 - (1.0 - Math.sqrt(1.0 + xt * xt - 2.0 * xt * Math.sqrt(1.0 - ratedPFactor * ratedPFactor)));
            //double deltaT2 = toReached2 - (1.0 - Math.sqrt(1.0 + xt * xt - 2.0 * xt * Math.sqrt(1.0 - ratedPFactor * ratedPFactor)));
            //from Fang sili's paper
            //double deltaT1 = toReached1 / Math.sqrt(1 - ratedPFactor * ratedPFactor) - xt;
            //double deltaT2 = toReached2 / Math.sqrt(1 - ratedPFactor * ratedPFactor) - xt;
            //
            double deltaT1 = Math.max(-0.05 * k, xt - toReached2);
            double deltaT2 = Math.min(0.05 * k, xt - toReached1);
            if (xt - toReached2 > 0.05 * k) {
                deltaT1 = 0.0;
                deltaT2 = 0.05 * k;
            } else if (xt - toReached1 < -0.05 * k) {
                deltaT1 = -0.05 * k;
                deltaT2 = 0.0;
            }

            exciterXcOpt.setXcMin(deltaT1);
            exciterXcOpt.setXcMax(deltaT2);
            //============================== out put =================================================================
            StringBuilder str = new StringBuilder();
            //generator parameters
            str.append(gen.getBusName()).append("\t");
            //if(island.getBusToTransformers().containsKey(gen.getBusName()))
            //    str.append("高压侧并联").append("\t");
            //else
            //    str.append("机端并联").append("\t");
            //str.append(gen.getBaseKv()).append("\t");
            //str.append(gen.getXdp()).append("\t").append(gen.getXqp()).append("\t").append(gen.getXd()).append("\t").append(gen.getXq()).append("\t");
            //str.append(gen.getTdop()).append("\t").append(gen.getTqop()).append("\t");

            //exciter parameters
            //str.append(exciter.getType()).append(exciter.getSubType()).append("\t").append(exciter.getKa()).append("\t").append(exciter.getTa()).append("\t").append(exciter.getXc()).append("\t");
            //str.append(hpSys.getExciterK()).append("\t");
            //str.append(hpSys.getExciterK()).append("\t").append(hpSys.getExciter().getXc()).append("\t");

            //transformer parameters
            //str.append(xt).append("\t");

            //traditional method result
            //str.append(deltaT1).append("\t").append(deltaT2).append("\t");

            //system parameters
            //str.append(hpSys.getHighVBusPf().getvInPu()).append("\t");
            //str.append(hpSys.getHighVBusPf().getAngleInArc()).append("\t");
            //str.append(hpSys.getGenBusPf().getvInPu()).append("\t");
            //str.append(hpSys.getGenBusPf().getAngleInArc()).append("\t");

            //stability indexes when no opt xc is foun
            //double[] deltaTsTd = hpSys.setExciterXc(0.0);
            //str.append(deltaTsTd[0]).append("\t").append(deltaTsTd[1]).append("\t");
            //deltaTsTd = hpSys.setExciterXc(exciter.getXc());
            //str.append(deltaTsTd[0]).append("\t").append(deltaTsTd[1]).append("\t");

            //start to solve the optimum problem
            //exciterXcOpt.setPrintPath(true);
            exciterXcOpt.setHpSys(hpSys);
            //exciterXcOpt.doOpt((deltaT1 + deltaT2) / 2.0, ExciterXcOpt.OBJ_TS_MAX, ExciterXcOpt.CONSTRAINT_TD);
            exciterXcOpt.doOpt((deltaT1 + deltaT2) / 2.0, ExciterXcOpt.OBJ_TS_MAX);
            assertTrue(exciterXcOpt.isConverged());

            //deltaTsTd = hpSys.setExciterXc(exciterXcOpt.getOptXc());
            //str.append(deltaTsTd[0]).append("\t").append(deltaTsTd[1]).append("\t");
            //str.append(exciterXcOpt.getOptXc()).append("\t").append(hpSys.getAclineX());
            str.append(exciterXcOpt.getOptXc()).append("\t").append(-exciterXcOpt.getOptXc() / k);
            System.out.println(str);
            //============================== out put end ==============================================================
            //writeSwiFile(hpSys, hpSys.getGen().getBusName() + "_ori", false);
            //writeSwiFile(hpSys, "swi_ori", false);
            //hpSys.getExciter().setXc(0.0);
            //writeSwiFile(hpSys, hpSys.getGen().getBusName() + "_zero");
            hpSys.getExciter().setXc(exciterXcOpt.getOptXc());
            //writePfFile(hpSys, hpSys.getGen().getBusName(), false);
            //writeSwiFile(hpSys, hpSys.getGen().getBusName() + "_opt", false);
            //writePfFile(hpSys, "pf", false);
            //writeSwiFile(hpSys, "swi_opt", false);
        }

        //printing generators shunt to a same bus
        //List<String> highVBusNames = new ArrayList<String>();
        //for(String plantName : toFound) {
        //    for(Generator gen : generatorsInPlant.get(plantName)) {
        //        String highVbusName = generatorToHighVBus.get(gen);
        //        if(!highVBusNames.contains(highVbusName))
        //            highVBusNames.add(highVbusName);
        //    }
        //}
        //for(String highVBusName : highVBusNames) {
        //    System.out.println("========================== bus name " + highVBusName + " and shunt generators: ");
        //    System.out.print("*******");
        //    for(Generator gen : shuntGenerators.get(highVBusName))
        //        System.out.print(gen.getBusName() + "\t");
        //    System.out.println();
        //}
    }
}
