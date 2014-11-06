package zju.common;

import junit.framework.TestCase;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-24
 */
public class SigmoidFunctionTest extends TestCase {
    /**
     * a * a0 + b = Z
     * a * a1 + b = -Z
     * => a = 2 * Z/(a0 - a1)
     * b = z(a0 + a1)/(a1 - a0)
     */
    public void testPara() {
        double Z = 5;
        double a0 = 2;
        double a1 = 3;
        double a = (2 * Z / (a0 - a1));
        double b = (Z * (a1 + a0) / (a1 - a0));
        assertEquals(5.0, a * a0 + b);
        assertEquals(-5.0, a * a1 + b);
        double tmp1 = 1.0 / (1.0 + Math.exp(a * a1 + b));
        double tmp2 = 1.0 / (1.0 + Math.exp(-a * a1 + b));
        assertTrue(Math.abs(0.993307 - tmp1 + tmp2) < 1e-6);
    }
}
