package zju.bpamodel.swi;

import junit.framework.TestCase;

/**
 * GeneratorDW Tester.
 *
 * @author <Authors name>
 * @since <pre>07/28/2012</pre>
 * @version 1.0
 */
public class GeneratorDWTest extends TestCase {
    public GeneratorDWTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void testParse() {
        GeneratorDW genDw = GeneratorDW.createGenDampingWinding("M  闽周宁_115.8 138.9 0.9     H      .197 .19  .13 .090");
        assertNotNull(genDw);
        assertEquals(genDw.getType(), "H");
        assertEquals(genDw.getBusName(), "闽周宁_1");
        assertEquals(genDw.getBaseKv(), 15.8);
        assertEquals(genDw.getBaseMva(), 138.9);
        assertEquals(genDw.getPowerFactor(), .9);
        assertEquals(genDw.getXdpp(), .197);
        assertEquals(genDw.getXqpp(), .19);
        assertEquals(genDw.getXdopp(), .13);
        assertEquals(genDw.getXqopp(), .09);

        genDw = GeneratorDW.createGenDampingWinding(genDw.toString());
        assertNotNull(genDw);
        assertEquals(genDw.getType(), "H");
        assertEquals(genDw.getBusName(), "闽周宁_1");
        assertEquals(genDw.getBaseKv(), 15.8);
        assertEquals(genDw.getBaseMva(), 138.9);
        assertEquals(genDw.getPowerFactor(), .9);
        assertEquals(genDw.getXdpp(), .197);
        assertEquals(genDw.getXqpp(), .19);
        assertEquals(genDw.getXdopp(), .13);
        assertEquals(genDw.getXqopp(), .09);
    }
}
