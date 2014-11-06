package zju.bpamodel;

import junit.framework.TestCase;
import sun.nio.cs.ext.GBK;

/**
 * BpaFileRwUtil Tester.
 *
 * @author <Authors name>
 * @since <pre>08/08/2012</pre>
 * @version 1.0
 */
public class BpaFileRwUtilTest extends TestCase {
    public BpaFileRwUtilTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testGetTarget() throws Exception {
        GBK gbk = new GBK();
        byte[] s = BpaFileRwUtil.getTarget("闽芹山_110.5".getBytes(gbk), 0, 8);
        assertEquals("闽芹山_1", new String(s, gbk));
    }

    public void testGetFormatStr() throws Exception {
        String s = BpaFileRwUtil.getFormatStr(-0.025, "4.3");
        assertEquals("-025", s);
        s = BpaFileRwUtil.getFormatStr(-0.025, "5.3");
        assertEquals("-.025", s);
        s = BpaFileRwUtil.getFormatStr(0.12, "4.2");
        assertEquals(" .12", s);
        s = BpaFileRwUtil.getFormatStr(525.01, "5.2");
        assertEquals("52501", s);
        s = BpaFileRwUtil.getFormatStr(-525.1, "5.1");
        assertEquals("-5251", s);
        s = BpaFileRwUtil.getFormatStr(10.5, "4.3");
        assertEquals("10.5", s);
    }

    public void testGetDecimals() throws Exception {
        int[] a = BpaFileRwUtil.getDecimals("4.3");
        assertEquals(4, a[0]);
        assertEquals(3, a[1]);
    }

    public void testParseDouble() {
        double d = BpaFileRwUtil.parseDouble("-025", "4.3");
        assertEquals(-0.025, d);
        d = BpaFileRwUtil.parseDouble("52501", "5.2");
        assertEquals(525.01, d);
        d = BpaFileRwUtil.parseDouble("-5251", "5.1");
        assertEquals(-525.1, d);
        d = BpaFileRwUtil.parseDouble("2000", "5.2");
        assertEquals(20.0, d);
    }
}
