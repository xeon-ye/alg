package zju.util;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import java.util.Map;

/**
 * SerializableUtil Tester.
 *
 * @author <Authors name>
 * @since <pre>09/06/2011</pre>
 * @version 1.0
 */
public class SerializableUtilTest extends TestCase {
    public SerializableUtilTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public static Test suite() {
        return new TestSuite(SerializableUtilTest.class);
    }

    public void test1() {
        String s2 = "id-=-210000099-@-type-=-Substation-@-class-=-Substation-@-name-=-中双港变电站-@-aliasName-=-中双港变电站-@-";
        Map m = SerializableUtil.createMap(s2, SerializableUtil.TYPE_STRING, SerializableUtil.TYPE_STRING);
        assertEquals(m.get("id"),"210000099");
        assertEquals(m.get("type"),"Substation");
        assertEquals(m.get("class"),"Substation");
        assertEquals(m.get("name"),"中双港变电站");
    }
}
