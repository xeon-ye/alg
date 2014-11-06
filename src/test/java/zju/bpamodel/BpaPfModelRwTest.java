package zju.bpamodel;

import junit.framework.TestCase;
import zju.bpamodel.pf.Bus;
import zju.bpamodel.pf.ElectricIsland;
import zju.bpamodel.pfr.PfResult;

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
}
