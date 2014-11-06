package zju.bpamodel.swi;

import junit.framework.TestCase;

/**
 * GenOutput Tester.
 *
 * @author <Authors name>
 * @since <pre>08/01/2012</pre>
 * @version 1.0
 */
public class GenOutputTest extends TestCase {
    public GenOutputTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testParse() {
        GenOutput genOutput = GenOutput.createOutput("G  闽可门_4 20.    3                                                           1");
        assertNotNull(genOutput);
        assertEquals(genOutput.getGenBusName(), "闽可门_4");
        assertEquals(genOutput.getBaseKv(), 20.);
        assertEquals(genOutput.getAngle(), 3);
        assertEquals(genOutput.getOtherVar(), 1);

        genOutput = GenOutput.createOutput(genOutput.toString());
        assertNotNull(genOutput);
        assertEquals(genOutput.getGenBusName(), "闽可门_4");
        assertEquals(genOutput.getBaseKv(), 20.);
        assertEquals(genOutput.getAngle(), 3);
        assertEquals(genOutput.getOtherVar(), 1);
    }
}
