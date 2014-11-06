package zju.ieeeformat;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * DataOutputFormat Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>07/19/2006</pre>
 */
public class DataOutputFormatTest extends TestCase {
    DataOutputFormat myformata;
    Double a = 10.110900;
    Double b = 0.009201;
    Double c = 9999.999999;
    Double d = 7.0912301000000000E-4;
    String s = "35445.2331";

    {
        myformata = new DataOutputFormat();
    }

    public DataOutputFormatTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testGetFormatStr() throws Exception {
        assertEquals("10.111", DataOutputFormat.format.getFormatStr(a, "6.3"));
        assertEquals("10.1", DataOutputFormat.format.getFormatStr(a, "4"));
    }

    public void testGetFormatStr1() throws Exception {
        assertEquals("10000", DataOutputFormat.format.getFormatStr(c, "5"));
        assertEquals("10000.000", DataOutputFormat.format.getFormatStr(c, "9.3"));
    }

    public void testGetFormatStr2() throws Exception {

        assertEquals(".0092", DataOutputFormat.format.getFormatStr(b, "5.5"));
    }

    public void testGetFormatStr3() throws Exception {
        assertEquals("35445", DataOutputFormat.format.getFormatStr(s, "5"));
    }

    public void testGetFormatStr4() throws Exception {
        assertEquals(".000709123", DataOutputFormat.format.getFormatStr(d, "10.10"));
    }

    public static Test suite() {
        return new TestSuite(DataOutputFormatTest.class);
    }
}
