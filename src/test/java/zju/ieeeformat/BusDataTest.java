package zju.ieeeformat;

import junit.framework.TestCase;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2009-6-25
 */
public class BusDataTest extends TestCase {
    public void testToString() {
        BusData bus = new BusData();
        bus.setBusNumber(1);
        bus.setName("hi");
        String str = bus.toString();
        System.out.println(str);
        assertNotNull(str);
    }

    public void testToString2() {
        BusData bus = new BusData();
        bus.setBusNumber(1);
        bus.setName("中文名");
        DataOutputFormat.format.setCharset("GBK");
        String str = bus.toString();
        System.out.println(str);
        assertNotNull(str);
    }
}
