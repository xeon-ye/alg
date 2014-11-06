package zju.bpamodel.swi;

import junit.framework.TestCase;

/**
 * Exciter Tester.
 *
 * @author <Authors name>
 * @since <pre>07/12/2012</pre>
 * @version 1.0
 */
public class ExciterTest extends TestCase {
    public ExciterTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testParse() {
        Exciter exciter = Exciter.createExciter("FV 闽可门_1 20.         .02  500. 1  .025 .025 1.   10.  1.   .003 0    .001  ");
        assertNotNull(exciter);
        assertEquals(exciter.getSubType(), 'V');
        assertEquals(exciter.getBusName(), "闽可门_1");
        assertEquals(exciter.getBaseKv(), 20.0);
        assertEquals(exciter.getTr(), 0.02);
        assertEquals(exciter.getK(), 500.0);
        assertEquals(exciter.getKv(), 1.0);
        assertEquals(exciter.getT1(), 0.025);
        assertEquals(exciter.getT2(), 0.025);
        assertEquals(exciter.getT3(), 1.0);
        assertEquals(exciter.getT4(), 10.0);
        assertEquals(exciter.getKa(), 1.0);
        assertEquals(exciter.getTa(), 0.003);
        assertEquals(exciter.getKf(), 0.0);
        assertEquals(exciter.getTf(), 0.001);
        exciter = Exciter.createExciter("EJ 闽水口_113.8 .012 500..001.001-8.6                -6.848.55.021.88 ");
        assertNotNull(exciter);
        assertEquals(exciter.getSubType(), 'J');
        assertEquals(exciter.getBusName(), "闽水口_1");
        assertEquals(exciter.getBaseKv(), 13.8);
        assertEquals(exciter.getTr(), 0.012);
        assertEquals(exciter.getKa(), 500.0);
        assertEquals(exciter.getTa(), 0.001);
        assertEquals(exciter.getTa1(), 0.001);
        assertEquals(exciter.getVrmin(), -8.6);
        assertEquals(exciter.getEfdmin(), -6.84);
        assertEquals(exciter.getEfdmax(), 8.55);
        assertEquals(exciter.getKf(), 0.021);
        assertEquals(exciter.getTf(), 0.88);

        exciter = Exciter.createExciter("FV 闽华能_520.  0.  -025.034 1.   1. .1   .1   1.   10.  484.8.003 0    0");
        assertNotNull(exciter);
        assertEquals(exciter.getSubType(), 'V');
        assertEquals(exciter.getBusName(), "闽华能_5");
        assertEquals(exciter.getBaseKv(), 20.0);
        assertEquals(exciter.getRc(), 0.0);
        assertEquals(exciter.getXc(), -0.025);
        assertEquals(exciter.getTr(), 0.034);
        assertEquals(exciter.getK(), 1.0);
        assertEquals(exciter.getKv(), 1.0);
        assertEquals(exciter.getT1(), .1);
        assertEquals(exciter.getT2(), .1);
        assertEquals(exciter.getT3(), 1.0);
        assertEquals(exciter.getT4(), 10.0);
        assertEquals(exciter.getKa(), 484.8);
        assertEquals(exciter.getTa(), 0.003);
        assertEquals(exciter.getKf(), 0.);
        assertEquals(exciter.getTf(), 0.);
    }

    public void testWrite() {
        Exciter exciter = Exciter.createExciter("FV 闽可门_1 20.         .02  500. 1  .025 .025 1.   10.  1.   .003 0    .001  ");
        String anotherStr = exciter.toString();
        exciter = Exciter.createExciter(anotherStr);
        assertEquals(exciter.getSubType(), 'V');
        assertEquals(exciter.getBusName(), "闽可门_1");
        assertEquals(exciter.getBaseKv(), 20.0);
        assertEquals(exciter.getTr(), 0.02);
        assertEquals(exciter.getK(), 500.0);
        assertEquals(exciter.getKv(), 1.0);
        assertEquals(exciter.getT1(), 0.025);
        assertEquals(exciter.getT2(), 0.025);
        assertEquals(exciter.getT3(), 1.0);
        assertEquals(exciter.getT4(), 10.0);
        assertEquals(exciter.getKa(), 1.0);
        assertEquals(exciter.getTa(), 0.003);
        assertEquals(exciter.getKf(), 0.0);
        assertEquals(exciter.getTf(), 0.001);
        exciter = Exciter.createExciter("EJ 闽水口_113.8 .012 500..001.001-8.6                -6.848.55.021.88 ");
        anotherStr = exciter.toString();
        exciter = Exciter.createExciter(anotherStr);
        assertEquals(exciter.getSubType(), 'J');
        assertEquals(exciter.getBusName(), "闽水口_1");
        assertEquals(exciter.getBaseKv(), 13.8);
        assertEquals(exciter.getTr(), 0.012);
        assertEquals(exciter.getKa(), 500.0);
        assertEquals(exciter.getTa(), 0.001);
        assertEquals(exciter.getTa1(), 0.001);
        assertEquals(exciter.getVrmin(), -8.6);
        assertEquals(exciter.getEfdmin(), -6.84);
        assertEquals(exciter.getEfdmax(), 8.55);
        assertEquals(exciter.getKf(), 0.021);
        assertEquals(exciter.getTf(), 0.88);

        exciter = Exciter.createExciter("FV 闽华能_520.  0.  -025.034 1.   1. .1   .1   1.   10.  484.8.003 0    0");
        anotherStr = exciter.toString();
        exciter = Exciter.createExciter(anotherStr);
        assertEquals(exciter.getSubType(), 'V');
        assertEquals(exciter.getBusName(), "闽华能_5");
        assertEquals(exciter.getBaseKv(), 20.0);
        assertEquals(exciter.getRc(), 0.0);
        assertEquals(exciter.getXc(), -0.025);
        assertEquals(exciter.getTr(), 0.034);
        assertEquals(exciter.getK(), 1.0);
        assertEquals(exciter.getKv(), 1.0);
        assertEquals(exciter.getT1(), .1);
        assertEquals(exciter.getT2(), .1);
        assertEquals(exciter.getT3(), 1.0);
        assertEquals(exciter.getT4(), 10.0);
        assertEquals(exciter.getKa(), 484.8);
        assertEquals(exciter.getTa(), 0.003);
        assertEquals(exciter.getKf(), 0.);
        assertEquals(exciter.getTf(), 0.);

        exciter = Exciter.createExciter("FM 闽华能_122.          .02744.235 0.7.6651.   0.3340.0  1.961.00010.0  0.0 ");
        anotherStr = exciter.toString();
        exciter = Exciter.createExciter(anotherStr);
        assertEquals(exciter.getSubType(), 'M');
        assertEquals(exciter.getBusName(), "闽华能_1");
        assertEquals(exciter.getBaseKv(), 22.0);
        assertEquals(exciter.getTr(), 0.0274);
        assertEquals(exciter.getK(), 4.235);
        assertEquals(exciter.getKv(), .0);
        assertEquals(exciter.getT1(), 7.665);
        assertEquals(exciter.getT2(), 1.0);
        assertEquals(exciter.getT3(), 0.334);
        assertEquals(exciter.getT4(), .0);
        assertEquals(exciter.getKa(), 1.961);
        assertEquals(exciter.getTa(), 0.0001);
        assertEquals(exciter.getKf(), 0.);
        assertEquals(exciter.getTf(), 0.);

        exciter = Exciter.createExciter("FJ 闽池潭_110.5                          1.2  10.  50.  .36  999. -999.");
        anotherStr = exciter.toString();
        exciter = Exciter.createExciter(anotherStr);
        assertEquals(exciter.getSubType(), 'J');
        assertEquals(exciter.getBusName(), "闽池潭_1");
        assertEquals(exciter.getBaseKv(), 10.5);
        assertEquals(exciter.getTb(), 1.2);
        assertEquals(exciter.getTc(), 10.0);
        assertEquals(exciter.getKa(), 50.0);
        assertEquals(exciter.getTa(), 0.36);
        assertEquals(exciter.getVrmax(), 999.0);
        assertEquals(exciter.getVrmin(), -999.);
    }
}
