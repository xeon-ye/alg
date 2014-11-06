package zju.ieeeformat;

import junit.framework.TestCase;

import java.io.File;

/**
 * DefaultIcfParser Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>11/24/2006</pre>
 */
public class DefaultIcfParserTest extends TestCase {

    public DefaultIcfParserTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
    }

    public void testAll() {
        long start = System.currentTimeMillis();
        File dir = new File(this.getClass().getResource("/ieeefiles").getFile());
        DefaultIcfParser parser = new DefaultIcfParser();
        for (File f : dir.listFiles()) {
            if (f.getName().startsWith("sdxx2013"))
                assertNotNull(parser.parse(f, "UTF8"));
            else if (f.getName().startsWith("ahxx2013"))
                assertNotNull(parser.parse(f, "UTF8"));
            else
                assertNotNull(parser.parse(f, "GBK"));
        }
        System.out.println("读取ieeefiles文件夹中所有IEEE文件用时：" + (System.currentTimeMillis() - start) + "ms");
    }
}
