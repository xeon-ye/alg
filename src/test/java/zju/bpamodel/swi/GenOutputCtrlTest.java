package zju.bpamodel.swi;

import junit.framework.TestCase;

/**
 * GenOutputCtrl Tester.
 *
 * @author <Authors name>
 * @since <pre>07/29/2012</pre>
 * @version 1.0
 */
public class GenOutputCtrlTest extends TestCase {
    public GenOutputCtrlTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testParse() {
        GenOutputCtrl genOutCtrl = GenOutputCtrl.createOutput("GH 1  华新厂_113.8   1    200.   -200. ");
        assertNotNull(genOutCtrl);
        //assertEquals(genOutCtrl.getnDot(), 1);
        assertEquals(genOutCtrl.getRefBusName(), "华新厂_1");
        assertEquals(genOutCtrl.getRefBusBaseKv(), 13.8);
        assertEquals(genOutCtrl.getClassification()[0], 1);
        assertEquals(genOutCtrl.getMax()[0], 200.0);
        assertEquals(genOutCtrl.getMin()[0], -200.0);

        genOutCtrl = GenOutputCtrl.createOutput(genOutCtrl.toString());
        assertNotNull(genOutCtrl);
        assertEquals(genOutCtrl.getRefBusName(), "华新厂_1");
        assertEquals(genOutCtrl.getRefBusBaseKv(), 13.8);
        assertEquals(genOutCtrl.getClassification()[0], 1);
        assertEquals(genOutCtrl.getMax()[0], 200.0);
        assertEquals(genOutCtrl.getMin()[0], -200.0);
    }

    public void testToString() {
        GenOutputCtrl genOutCtrl = new GenOutputCtrl();
    }
}
