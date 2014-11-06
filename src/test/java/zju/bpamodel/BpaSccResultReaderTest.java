package zju.bpamodel;

import junit.framework.TestCase;
import zju.bpamodel.sccpc.SccResult;

/**
 * BpaSccResultReader Tester.
 *
 * @author <Authors name>
 * @since <pre>11/04/2012</pre>
 * @version 1.0
 */
public class BpaSccResultReaderTest extends TestCase {
    public BpaSccResultReaderTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testParse() {
        SccResult r = SccResult.parse(this.getClass().getResourceAsStream("/bpafiles/fj/2013年年大方式.lis"), "GBK");
        assertNotNull(r);
    }
}
