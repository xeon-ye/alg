package zju.bpamodel.swi;

import junit.framework.TestCase;

/**
 * ShortCircuitFault Tester.
 *
 * @author <Authors name>
 * @since <pre>07/29/2012</pre>
 * @version 1.0
 */
public class ShortCircuitFaultTest extends TestCase {
    public ShortCircuitFaultTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testParse() {
        ShortCircuitFault fault = ShortCircuitFault.createFault("LS  闽宁德__525.  浙双龙__525. 1    1    0.00");
        assertNotNull(fault);
        assertEquals(fault.getBusASign(), ' ');
        assertEquals(fault.getBusAName(), "闽宁德__");
        assertEquals(fault.getBusABaseKv(), 525.0);
        assertEquals(fault.getBusBSign(), ' ');
        assertEquals(fault.getBusBName(), "浙双龙__");
        assertEquals(fault.getBusBBaseKv(), 525.);
        assertEquals(fault.getParallelBranchCode(), '1');
        assertEquals(fault.getMode(), 1);
        assertEquals(fault.getStartCycle(), 0.0);

        fault = ShortCircuitFault.createFault(fault.toString());
        assertNotNull(fault);
        assertEquals(fault.getBusASign(), ' ');
        assertEquals(fault.getBusAName(), "闽宁德__");
        assertEquals(fault.getBusABaseKv(), 525.0);
        assertEquals(fault.getBusBSign(), ' ');
        assertEquals(fault.getBusBName(), "浙双龙__");
        assertEquals(fault.getBusBBaseKv(), 525.);
        assertEquals(fault.getParallelBranchCode(), '1');
        assertEquals(fault.getMode(), 1);
        assertEquals(fault.getStartCycle(), 0.0);

        fault = ShortCircuitFault.createFault("LS -闽宁德__525. -浙双龙__525. 1   -1    5.00");
        assertNotNull(fault);
        assertEquals(fault.getBusASign(), '-');
        assertEquals(fault.getBusAName(), "闽宁德__");
        assertEquals(fault.getBusABaseKv(), 525.0);
        assertEquals(fault.getBusBSign(), '-');
        assertEquals(fault.getBusBName(), "浙双龙__");
        assertEquals(fault.getBusBBaseKv(), 525.);
        assertEquals(fault.getParallelBranchCode(), '1');
        assertEquals(fault.getMode(), -1);
        assertEquals(fault.getStartCycle(), 5.0);

        fault = ShortCircuitFault.createFault(fault.toString());
        assertNotNull(fault);
        assertEquals(fault.getBusASign(), '-');
        assertEquals(fault.getBusAName(), "闽宁德__");
        assertEquals(fault.getBusABaseKv(), 525.0);
        assertEquals(fault.getBusBSign(), '-');
        assertEquals(fault.getBusBName(), "浙双龙__");
        assertEquals(fault.getBusBBaseKv(), 525.);
        assertEquals(fault.getParallelBranchCode(), '1');
        assertEquals(fault.getMode(), -1);
        assertEquals(fault.getStartCycle(), 5.0);

        fault = ShortCircuitFault.createFault("LS -闽宁德__525. -浙双龙__525. 2   -1    5.00 ");
        assertNotNull(fault);
        assertEquals(fault.getBusASign(), '-');
        assertEquals(fault.getBusAName(), "闽宁德__");
        assertEquals(fault.getBusABaseKv(), 525.0);
        assertEquals(fault.getBusBSign(), '-');
        assertEquals(fault.getBusBName(), "浙双龙__");
        assertEquals(fault.getBusBBaseKv(), 525.);
        assertEquals(fault.getParallelBranchCode(), '2');
        assertEquals(fault.getMode(), -1);
        assertEquals(fault.getStartCycle(), 5.0);

        fault = ShortCircuitFault.createFault(fault.toString());
        assertNotNull(fault);
        assertEquals(fault.getBusASign(), '-');
        assertEquals(fault.getBusAName(), "闽宁德__");
        assertEquals(fault.getBusABaseKv(), 525.0);
        assertEquals(fault.getBusBSign(), '-');
        assertEquals(fault.getBusBName(), "浙双龙__");
        assertEquals(fault.getBusBBaseKv(), 525.);
        assertEquals(fault.getParallelBranchCode(), '2');
        assertEquals(fault.getMode(), -1);
        assertEquals(fault.getStartCycle(), 5.0);
    }

    public void testToString() {
        ShortCircuitFault fault = new ShortCircuitFault();
        fault.setBusAName("闽江阴_1");
        fault.setBusABaseKv(20.0);
        fault.setBusBName("闽江阴__");
        fault.setBusBBaseKv(525.0);
        fault.setParallelBranchCode('1');
        fault.setMode(2);
        fault.setStartCycle(0.0);
        System.out.println(fault);
        fault.setMode(-2);
        fault.setBusASign('-');
        fault.setBusBSign('-');
        fault.setStartCycle(5.0);
        System.out.println(fault);
    }
}
