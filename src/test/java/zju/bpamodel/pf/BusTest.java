package zju.bpamodel.pf;

import junit.framework.TestCase;

/**
 * Bus Tester.
 *
 * @author <Authors name>
 * @since <pre>07/25/2012</pre>
 * @version 1.0
 */
public class BusTest extends TestCase {
    public BusTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void testParse() {
        Bus bus = Bus.createBus("BQ FND闽大唐_122.00 30.  10.          660.660. 300. -50. 1.00    ");
        assertNotNull(bus);
        assertEquals(bus.getSubType(), 'Q');
        assertEquals(bus.getOwner(), "FND");
        assertEquals(bus.getName(), "闽大唐_1");
        assertEquals(bus.getBaseKv(), 22.0);
        assertEquals(bus.getZoneName(), "0");
        assertEquals(bus.getLoadMw(), 30.0);
        assertEquals(bus.getLoadMvar(), 10.0);
        assertEquals(bus.getGenMwMax(), 660.0);
        assertEquals(bus.getGenMw(), 660.0);
        assertEquals(bus.getGenMvarMax(), 300.0);
        assertEquals(bus.getGenMvarMin(), -50.0);
        assertEquals(bus.getvAmplDesired(), 1.0);

        Bus copyBus = Bus.createBus(bus.toString());
        assertNotNull(copyBus);
        assertEquals(copyBus.getSubType(), 'Q');
        assertEquals(copyBus.getOwner(), "FND");
        assertEquals(copyBus.getName(), "闽大唐_1");
        assertEquals(copyBus.getBaseKv(), 22.0);
        assertEquals(copyBus.getZoneName(), "0");
        assertEquals(copyBus.getLoadMw(), 30.0);
        assertEquals(copyBus.getLoadMvar(), 10.0);
        assertEquals(copyBus.getGenMwMax(), 660.0);
        assertEquals(copyBus.getGenMw(), 660.0);
        assertEquals(copyBus.getGenMvarMax(), 300.0);
        assertEquals(copyBus.getGenMvarMin(), -50.0);
        assertEquals(copyBus.getvAmplDesired(), 1.0);
    }
}
