package zju.bpamodel.pf;

import junit.framework.TestCase;

/**
 * Transformer Tester.
 *
 * @author <Authors name>
 * @since <pre>07/11/2012</pre>
 * @version 1.0
 */
public class TransformerTest extends TestCase {
    public TransformerTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testParse() {
        Transformer t = Transformer.createTransformer("T     闽可门__525.1闽可门_120. 1 720.1.00030.02286            53625 2000                  1.4  可门#1主变                         1   0.00000 0.02083");
        assertNotNull(t);
        assertEquals(t.getBusName1(), "闽可门__");
        assertEquals(t.getBaseKv1(), 525.0);
        assertEquals(t.getLinkMeterCode(), 1);
        assertEquals(t.getBusName2(), "闽可门_1");
        assertEquals(t.getBaseKv2(), 20.0);
        assertEquals(t.getCircuit(), '1');
        assertEquals(t.getBaseMva(), 720.0);
        assertEquals(t.getShuntTransformerNum(), 1);
        assertEquals(t.getR(), 0.0003);
        assertEquals(t.getX(), 0.02286);
        assertEquals(t.getTapKv1(), 536.25);
        assertEquals(t.getTapKv2(), 20.0);
        t = Transformer.createTransformer(t.toString());
        assertNotNull(t);
        assertEquals(t.getBusName1(), "闽可门__");
        assertEquals(t.getBaseKv1(), 525.0);
        assertEquals(t.getLinkMeterCode(), 1);
        assertEquals(t.getBusName2(), "闽可门_1");
        assertEquals(t.getBaseKv2(), 20.0);
        assertEquals(t.getCircuit(), '1');
        assertEquals(t.getBaseMva(), 720.0);
        assertEquals(t.getShuntTransformerNum(), 1);
        assertEquals(t.getR(), 0.0003);
        assertEquals(t.getX(), 0.02286);
        assertEquals(t.getTapKv1(), 536.250);
        assertEquals(t.getTapKv2(), 20.000);

        t = Transformer.createTransformer("T     闽江阴__525. 闽江阴_1 20.1 720.1.00032.02032            52500 2000                  1.4  江阴#1主变                         1   0.00000 0.01944");
        assertNotNull(t);
        assertEquals(t.getBusName1(), "闽江阴__");
        assertEquals(t.getBaseKv1(), 525.0);
        assertEquals(t.getBusName2(), "闽江阴_1");
        assertEquals(t.getBaseKv2(), 20.0);
        assertEquals(t.getCircuit(), '1');
        assertEquals(t.getBaseMva(), 720.0);
        assertEquals(t.getShuntTransformerNum(), 1);
        assertEquals(t.getR(), 0.00032);
        assertEquals(t.getX(), 0.02032);
        assertEquals(t.getTapKv1(), 525.000);
        assertEquals(t.getTapKv2(), 20.000);
        t = Transformer.createTransformer(t.toString());
        assertNotNull(t);
        assertEquals(t.getBusName1(), "闽江阴__");
        assertEquals(t.getBaseKv1(), 525.0);
        assertEquals(t.getBusName2(), "闽江阴_1");
        assertEquals(t.getBaseKv2(), 20.0);
        assertEquals(t.getCircuit(), '1');
        assertEquals(t.getBaseMva(), 720.0);
        assertEquals(t.getShuntTransformerNum(), 1);
        assertEquals(t.getR(), 0.00032);
        assertEquals(t.getX(), 0.02032);
        assertEquals(t.getTapKv1(), 525.000);
        assertEquals(t.getTapKv2(), 20.000);
    }
}
