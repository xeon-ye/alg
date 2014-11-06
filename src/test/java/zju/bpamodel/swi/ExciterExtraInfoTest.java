package zju.bpamodel.swi;

import junit.framework.TestCase;

/**
 * ExciterExtraInfo Tester.
 *
 * @author <Authors name>
 * @since <pre>11/06/2012</pre>
 * @version 1.0
 */
public class ExciterExtraInfoTest extends TestCase {
    public ExciterExtraInfoTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }
    public void testParse() {
        ExciterExtraInfo exciterExtra = ExciterExtraInfo.createExciterExtraInfo("F+ 闽可门_1 20. 9.93 -8.44                          9.93-8.4.069");
        assertNotNull(exciterExtra);
        assertEquals(exciterExtra.getBusName(), "闽可门_1");
        assertEquals(exciterExtra.getBaseKv(), 20.0);
        assertEquals(exciterExtra.getVamax(), 9.93);
        assertEquals(exciterExtra.getVamin(), -8.44);
        assertEquals(exciterExtra.getVrmax(), 9.93);
        assertEquals(exciterExtra.getVrmin(), -8.4);
        assertEquals(exciterExtra.getKc(), 0.069);

        exciterExtra = ExciterExtraInfo.createExciterExtraInfo("F+ 闽后石_123.  999. -999.304..0031.0 2.14.0165.0115150.-132.244.6711.0 7.297.29");
        assertNotNull(exciterExtra);
        assertEquals(exciterExtra.getBusName(), "闽后石_1");
        assertEquals(exciterExtra.getBaseKv(), 23.);
        assertEquals(exciterExtra.getVamax(), 999.);
        assertEquals(exciterExtra.getVamin(), -999.);
        assertEquals(exciterExtra.getKb(), 304.);
        assertEquals(exciterExtra.getT5(), .003);
        assertEquals(exciterExtra.getKe(), 1.0);
        assertEquals(exciterExtra.getTe(), 2.14);
        assertEquals(exciterExtra.getSe1(), .0165);
        assertEquals(exciterExtra.getSe2(), .0115);
        assertEquals(exciterExtra.getVrmax(), 150.);
        assertEquals(exciterExtra.getVrmin(), -1.32);
        assertEquals(exciterExtra.getKc(), .244);
        assertEquals(exciterExtra.getKd(), .671);
        assertEquals(exciterExtra.getKli(), 1.0);
        assertEquals(exciterExtra.getVlir(), 7.29);
        assertEquals(exciterExtra.getEfdmax(), 7.29);
    }

    public void testWrite() {
        ExciterExtraInfo exciterExtra = ExciterExtraInfo.createExciterExtraInfo("F+ 闽可门_1 20. 9.93 -8.44                          9.93-8.4.069");
        String anotherStr = exciterExtra.toString();
        exciterExtra = ExciterExtraInfo.createExciterExtraInfo(anotherStr);
        assertEquals(exciterExtra.getBusName(), "闽可门_1");
        assertEquals(exciterExtra.getBaseKv(), 20.0);
        assertEquals(exciterExtra.getVamax(), 9.93);
        assertEquals(exciterExtra.getVamin(), -8.44);
        assertEquals(exciterExtra.getVrmax(), 9.93);
        assertEquals(exciterExtra.getVrmin(), -8.4);
        assertEquals(exciterExtra.getKc(), 0.069);

        exciterExtra = ExciterExtraInfo.createExciterExtraInfo("F+ 闽后石_123.  999. -999.304..0031.0 2.14.0165.0115150.-132.244.6711.0 7.297.29");
        anotherStr = exciterExtra.toString();
        exciterExtra = ExciterExtraInfo.createExciterExtraInfo(anotherStr);
        assertEquals(exciterExtra.getBaseKv(), 23.);
        assertEquals(exciterExtra.getVamax(), 999.);
        assertEquals(exciterExtra.getVamin(), -999.);
        assertEquals(exciterExtra.getKb(), 304.);
        assertEquals(exciterExtra.getT5(), .003);
        assertEquals(exciterExtra.getKe(), 1.0);
        assertEquals(exciterExtra.getTe(), 2.14);
        assertEquals(exciterExtra.getSe1(), .0165);
        assertEquals(exciterExtra.getSe2(), .0115);
        assertEquals(exciterExtra.getVrmax(), 150.);
        assertEquals(exciterExtra.getVrmin(), -1.32);
        assertEquals(exciterExtra.getKc(), .244);
        assertEquals(exciterExtra.getKd(), .67);//the original one is .671
        assertEquals(exciterExtra.getKli(), 1.0);
        assertEquals(exciterExtra.getVlir(), 7.29);
        assertEquals(exciterExtra.getEfdmax(), 7.29);
    }
}
