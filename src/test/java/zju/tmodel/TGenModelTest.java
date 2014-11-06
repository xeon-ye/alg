package zju.tmodel;

import junit.framework.TestCase;
import zju.bpamodel.swi.Generator;

/**
 * TGenModel Tester.
 *
 * @author <Authors name>
 * @since <pre>07/16/2012</pre>
 * @version 1.0
 */
public class TGenModelTest extends TestCase {
    public TGenModelTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testCal() {
        //the example of p38
        Generator generator = new Generator();
        generator.setXd(1.903);
        generator.setXq(1.801);
        TGenModel genModel = new TGenModel(generator);
        genModel.cal(1.0, 0.0, 1.0, -Math.PI * 31.79 / 180.0, 100);
        assertTrue(genModel.getDelta() * 180.0 / Math.PI - 38.15 < 1e-3);
        assertTrue(genModel.getEqd() - 2.478 < 1e-3);
        assertTrue(genModel.getUd() - 0.618 < 1e-3);
        assertTrue(genModel.getUq() - 0.786 < 1e-3);
        assertTrue(genModel.getId() - 0.939 < 1e-3);
        assertTrue(genModel.getIq() - 0.343 < 1e-3);
        assertTrue(genModel.getEq() - 2.573 < 1e-3);
    }
}
