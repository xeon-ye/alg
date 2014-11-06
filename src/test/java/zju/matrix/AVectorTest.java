package zju.matrix;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

/**
 * AVector Tester.
 *
 * @author <Dong Shufeng>
 * @version 1.0
 * @since <pre>12/07/2007</pre>
 */
public class AVectorTest extends TestCase {
    public AVectorTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testSetValue() throws Exception {
        AVector test = new AVector(10);
        test.setValue(3, 1.0);
        assertEquals(test.getValue(3), 1.0);
        assertEquals(test.getValue(0), 0.0);
        assertEquals(test.getValue(1), 0.0);
        assertEquals(test.getValue(2), 0.0);
        assertEquals(test.getValue(4), 0.0);
    }


    public void testGetN() throws Exception {
        AVector test = new AVector(10);
        assertEquals(test.getN(), 10);
    }

    public static Test suite() {
        return new TestSuite(AVectorTest.class);
    }
}
