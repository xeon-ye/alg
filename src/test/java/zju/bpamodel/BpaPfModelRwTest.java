package zju.bpamodel;

import junit.framework.TestCase;
import zju.bpamodel.pf.AcLine;
import zju.bpamodel.pf.Bus;
import zju.bpamodel.pf.ElectricIsland;
import zju.bpamodel.pf.Transformer;
import zju.bpamodel.pfr.PfResult;

import java.util.List;

/**
 * BpaPfModelParser Tester.
 *
 * @author <Authors name>
 * @since <pre>07/12/2012</pre>
 * @version 1.0
 */
public class BpaPfModelRwTest extends TestCase {
    public BpaPfModelRwTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testParse() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/fj/2012年年大方式20111103.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("浙北三_127");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
    }

    public void testParse_caseAnhui() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/anhui/sdxx201307081415.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("浙北三_127");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
    }

    public void testParseResult() {
        PfResult r = BpaPfResultParser.parse(this.getClass().getResourceAsStream("/bpafiles/fj/2012年年大方式20111103.pfo"), "GBK");
        assertNotNull(r);
        assertTrue(r.getBusData().size() > 1000);
    }

    public void testBus003() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/003_bus/bpa/003bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus-1100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("bus-1100");
        assertNotNull(transformers);
        assertEquals(1, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("bus-2100");
        assertNotNull(acLines);
        assertEquals(2, acLines.size());
    }

    public void testBus004() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/004_bus/bpa/004bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus_3100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("bus_1100");
        assertNull(transformers);
        List<AcLine> acLines = island.getBusToAclines().get("bus_2100");
        assertNotNull(acLines);
        assertEquals(3, acLines.size());
    }

    public void testBus005() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/005_bus/bpa/005bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus_4100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("bus_5100");
        assertNotNull(transformers);
        assertEquals(1, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("bus_2100");
        assertNotNull(acLines);
        assertEquals(2, acLines.size());
    }

    public void testBus009() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/009_bus/bpa/009bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus-1100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("bus-2100");
        assertNotNull(transformers);
        assertEquals(1, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("bus-6100");
        assertNotNull(acLines);
        assertEquals(2, acLines.size());
    }

    public void testBus010() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/010_bus/bpa/010bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus-10100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("bus-5100");
        assertNotNull(transformers);
        assertEquals(3, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("bus-4100");
        assertNotNull(acLines);
        assertEquals(6, acLines.size());
    }

    public void testBus011() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/011_bus/bpa/011bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus_11100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<AcLine> acLines = island.getBusToAclines().get("bus_11100");
        assertNotNull(acLines);
        assertEquals(1, acLines.size());
    }

    public void testBus013() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/013_bus/bpa/013bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("BUS-13100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("BUS-13100");
        assertNotNull(transformers);
        assertEquals(2, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("BUS-12100");
        assertNotNull(acLines);
        assertEquals(2, acLines.size());
    }

    public void testBus014() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/014_bus/bpa/014bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("BUS-14100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("BUS-9100");
        assertNotNull(transformers);
        assertEquals(1, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("BUS-13100");
        assertNotNull(acLines);
        assertEquals(3, acLines.size());
    }

    public void testBus030() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/030_bus/bpa/030bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus-24100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("bus-28100");
        assertNotNull(transformers);
        assertEquals(1, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("bus-1100");
        assertNotNull(acLines);
        assertEquals(2, acLines.size());
    }

    public void testBus039() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/039_bus/bpa/039bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("BUS-39100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("BUS-12100");
        assertNotNull(transformers);
        assertEquals(2, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("BUS-29100");
        assertNotNull(acLines);
        assertEquals(2, acLines.size());
    }

    public void testBus043() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/043_bus/bpa/043bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("BUS-42100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<AcLine> acLines = island.getBusToAclines().get("BUS-2100");
        assertNotNull(acLines);
        assertEquals(4, acLines.size());
    }

    public void testBus057() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/057_bus/bpa/057bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus-53100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("bus-24100");
        assertNotNull(transformers);
        assertEquals(3, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("bus-56100");
        assertNotNull(acLines);
        assertEquals(3, acLines.size());
    }

    public void testBus118() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/118_bus/bpa/118bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus-116100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("bus-81100");
        assertNotNull(transformers);
        assertEquals(1, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("bus-118100");
        assertNotNull(acLines);
        assertEquals(2, acLines.size());
    }

    public void testBus145() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/145_bus/bpa/145bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus-146100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("bus-86100");
        assertNotNull(transformers);
        assertEquals(3, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("bus-142100");
        assertNotNull(acLines);
        assertEquals(11, acLines.size());
    }

    public void testBus162() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/162_bus/bpa/162bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus-162100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("bus-153100");
        assertNotNull(transformers);
        assertEquals(2, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("bus-162100");
        assertNotNull(acLines);
        assertEquals(2, acLines.size());
    }

    public void testBus300() {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/300_bus/bpa/300bpa.dat"), "GBK");
        assertNotNull(island);
        Bus slackBus = island.getNameToBus().get("bus-9533100");
        assertNotNull(slackBus);
        assertEquals(0.0, slackBus.getSlackBusVAngle());
        List<Transformer> transformers = island.getBusToTransformers().get("bus-9052100");
        assertNotNull(transformers);
        assertEquals(1, transformers.size());
        List<AcLine> acLines = island.getBusToAclines().get("bus-9001100");
        assertNotNull(acLines);
        assertEquals(1, acLines.size());
    }

    public void testXJ() {
        BpaPfModelRw.CreateTables("D:/rsa.db");
        BpaPfModelRw.parseAndSave(this.getClass().getResourceAsStream("/bpafiles/示范区BPA运行方式/XIAOJIN.DAT"), "d:/rsa.db");
        BpaPfModelRw.write("d:/rsa.db", "D:/IdeaProjects/alg/src/test/resources/bpafiles/示范区BPA运行方式/XIAOJIN.DAT",
                "D:/IdeaProjects/alg/src/test/resources/bpafiles/示范区BPA运行方式/XIAOJIN_modify.DAT");
    }

    public void testRead() {
        ElectricIsland model = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/示范区BPA运行方式/XIAOJIN.DAT"), "GBK");
        assertNotNull(model);
        ElectricIsland modifiedModel = BpaPfModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/示范区BPA运行方式/XIAOJIN_modify.DAT"), "GBK");
        assertNotNull(modifiedModel);
    }

    public void testParsePfResult() {
        BpaPfResultRw.CreateTables("D:/rsa.db");
        BpaPfResultRw.parseAndSave("C:/Users/bingtekeji/Desktop/写结果/XIAOJIN.pfo", "d:/rsa.db");
    }

    public void testDeleteTable() {
        ChangeDbData.truncateTable("d:/rsa.db", "Bc");
    }

    public void testUpdataGenPlan() {
        ChangeDbData.updataGenPlan("d:/rsa.db", "木坡101", 15);
    }

    public void testSetFault() {
        ChangeDbData.setShortCircuitFault("d:/rsa.db", 'A', "name1" ,
                10, 'B', "name2", 10, '1', 0, 10, 1, 1, 5);
    }
}
