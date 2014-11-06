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

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * TransferImpedanceC Tester.
 *
 * @author Dong Shufeng
 * @version 1.0
 * @since <pre>11/04/2012</pre>
 */
public class TransferImpedanceCTest extends TestCase {
    public TransferImpedanceCTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testCaseFj_2012() {
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
        assertNotNull(swiModel);
        assertNotNull(island);
        assertNotNull(pfResult);
        assertNotNull(sccResult);
        testCaseFj(island, pfResult, swiModel, sccResult);
    }

    public void testCaseFj_2013() {
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

        assertNotNull(swiModel);
        assertNotNull(island);
        assertNotNull(pfResult);
        assertNotNull(sccResult);
        testCaseFj(island, pfResult, swiModel, sccResult);
    }

    public void testCaseFj_2014() {
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

        assertNotNull(swiModel);
        assertNotNull(island);
        assertNotNull(pfResult);
        assertNotNull(sccResult);
        testCaseFj(island, pfResult, swiModel, sccResult);
    }

    public void testCaseFj(ElectricIsland island, PfResult pfResult, BpaSwiModel swiModel, SccResult sccResult) {
        List<String> toFound = new ArrayList<String>();
        toFound.add("后石");
        toFound.add("可门");
        toFound.add("江阴");
        toFound.add("华能");
        toFound.add("鸿山");
        toFound.add("嵩屿");
        toFound.add("湄电"); //湄洲湾
        toFound.add("坑口");
        toFound.add("漳平");
        toFound.add("前云");
        toFound.add("石圳");
        toFound.add("新店");
        toFound.add("水口");
        toFound.add("安砂");
        toFound.add("周宁");
        toFound.add("池潭");
        toFound.add("南埔");
        toFound.add("宁核");//宁德核电
        toFound.add("西抽");//西苑抽蓄
        toFound.add("福核");//福清核电
        toFound.add("大唐");
        toFound.add("永安");
        toFound.add("邵武");
        toFound.add("一级");
        toFound.add("二级");
        toFound.add("三级");
        toFound.add("四级");
        toFound.add("芹山");
        toFound.add("丰源");
        toFound.add("洪口");
        toFound.add("街面");
        toFound.add("晴川");
        //The following generator with shunt generator
        //toFound.add("棉电");//棉滩
        //toFound.add("沙电");//沙溪口

        List<Generator> toOptGen = new ArrayList<Generator>();
        for (String name : toFound) {
            for (Generator gen : swiModel.getGenerators()) {
                if (gen.getBusName().contains(name)) {
                    toOptGen.add(gen);
                }
            }
        }

        for (Generator aToOptGen : toOptGen) {
            HeffronPhilipsSystem hpSys = HpsBuilder.createHpSys(swiModel, island, pfResult, aToOptGen);
            if (hpSys == null)
                continue;
            if (!sccResult.getBusData().containsKey(hpSys.getHighVBusPf().getName())) {
                System.out.println("No shunt current calculation result if found for bus: " + hpSys.getHighVBusPf().getName());
                continue;
            }
            TransferImpedanceC tic = new TransferImpedanceC();
            tic.setHpSys(hpSys);
            tic.setShuntResult(sccResult.getBusData().get(hpSys.getHighVBusPf().getName()));
            tic.cal2();
            if (HpsBuilder.fillInfiniteBusInfo(tic, hpSys))
                continue;
            assertNotNull(hpSys.getGen());
            assertNotNull(hpSys.getExciter());
            assertNotNull(hpSys.getHighVBus());
            assertNotNull(hpSys.getHighVBusPf());
        }
    }
}
