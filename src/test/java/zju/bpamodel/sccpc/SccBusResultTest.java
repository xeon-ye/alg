package zju.bpamodel.sccpc;

import junit.framework.TestCase;

/**
 * SccBusResult Tester.
 *
 * @author <Authors name>
 * @since <pre>11/04/2012</pre>
 * @version 1.0
 */
public class SccBusResultTest extends TestCase {
    public SccBusResultTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testParse() {
        SccBusResult r = SccBusResult.createBusResult("\"国东明__\"  525.0     B      0.0     2     63.000     19.161  17423.981      2.025     15.294      1.962     14.910      9.575     36.278 \"GD\" ");
        assertNotNull(r);
        assertEquals("国东明__", r.getBusName());
        assertEquals(525.0, r.getBaseKv());
        assertEquals(2, r.getFaultType());
        assertEquals(63.0, r.getBreakCurrent());
        assertEquals(19.161, r.getShuntCurrent());
        assertEquals(17423.981, r.getShuntCapacity());
        assertEquals(2.025, r.getPositiveSequenceR());
        assertEquals(15.294, r.getPositiveSequenceX());
        assertEquals(1.962, r.getNegativeSequenceR());
        assertEquals(14.910, r.getNegativeSequenceX());
        assertEquals(9.575, r.getZeroSequenceR());
        assertEquals(36.278, r.getZeroSequenceX());
        assertEquals("GD", r.getArea());
    }
}
