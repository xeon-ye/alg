package zju.bpamodel.pf;

import junit.framework.TestCase;

/**
 * AcLine Tester.
 *
 * @author <Authors name>
 * @since <pre>07/25/2012</pre>
 * @version 1.0
 */
public class AcLineTest extends TestCase {
    public AcLineTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void testParse() {
        AcLine acline = AcLine.createAcLine("L     闽芹山_110.5 闽芹山__10.5       .00000.00010                零支路 ");
        assertNotNull(acline);
        assertEquals(acline.getBusName1(), "闽芹山_1");
        assertEquals(acline.getBaseKv1(), 10.5);
        assertEquals(acline.getBusName2(), "闽芹山__");
        assertEquals(acline.getBaseKv2(), 10.5);
        assertEquals(acline.getR(), 0.0000);
        assertEquals(acline.getX(), 0.0001);
        assertEquals(acline.getDesc(),"零支路");

        acline = AcLine.createAcLine(acline.toString());
        assertNotNull(acline);
        assertEquals(acline.getBusName1(), "闽芹山_1");
        assertEquals(acline.getBaseKv1(), 10.5);
        assertEquals(acline.getBusName2(), "闽芹山__");
        assertEquals(acline.getBaseKv2(), 10.5);
        assertEquals(acline.getR(), 0.0000);
        assertEquals(acline.getX(), 0.0001);
        assertEquals(acline.getDesc(),"零支路");

        acline = AcLine.createAcLine("L     闽大唐__525. 闽宁德__525.1 2335 .00026.00320      .25735  32唐宁I路               4×LGJ-400                                          2802      实测       0.00226 0.00851 0.00000 0.13788");
        assertNotNull(acline);
        assertEquals(acline.getBusName1(), "闽大唐__");
        assertEquals(acline.getBaseKv1(), 525.0);
        assertEquals(acline.getBusName2(), "闽宁德__");
        assertEquals(acline.getBaseKv2(), 525.0);
        assertEquals(acline.getCircuit(), '1');
        assertEquals(acline.getBaseI(), 2335.0);
        assertEquals(acline.getR(), 0.00026);
        assertEquals(acline.getX(), 0.0032);
        assertEquals(acline.getHalfG(), 0.0);
        assertEquals(acline.getHalfB(), 0.25735);
        assertEquals(acline.getDesc(), "唐宁I路");

        acline = AcLine.createAcLine(acline.toString());
        assertNotNull(acline);
        assertEquals(acline.getBusName1(), "闽大唐__");
        assertEquals(acline.getBaseKv1(), 525.0);
        assertEquals(acline.getBusName2(), "闽宁德__");
        assertEquals(acline.getBaseKv2(), 525.0);
        assertEquals(acline.getCircuit(), '1');
        assertEquals(acline.getBaseI(), 2335.0);
        assertEquals(acline.getR(), 0.00026);
        assertEquals(acline.getX(), 0.0032);
        assertEquals(acline.getHalfG(), 0.0);
        assertEquals(acline.getHalfB(), 0.25735);
        assertEquals(acline.getDesc(), "唐宁I路");
    }
}
