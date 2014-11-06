package zju.bpamodel.swi;

import junit.framework.TestCase;

/**
 * Generator Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>07/11/2012</pre>
 */
public class GeneratorTest extends TestCase {
    public GeneratorTest(String name) {
        super(name);
    }

    public void testParse() {
        Generator gen = Generator.createGen("MF 闽可门_1 20. 2104.       667.    .301 .448 2.1552.10 8.61.96.1583.164 .6630.1可门火电厂#1机");
        assertNotNull(gen);
        assertEquals(gen.getSubType(), 'F');
        assertEquals(gen.getBusName(), "闽可门_1");
        assertEquals(gen.getBaseKv(), 20.0);
        assertEquals(gen.geteMWS(), 2104.0);
        assertEquals(gen.getBaseMva(), 667.0);
        assertEquals(gen.getXdp(), 0.301);
        assertEquals(gen.getXqp(), 0.448);
        assertEquals(gen.getXd(), 2.155);
        assertEquals(gen.getXq(), 2.1);
        assertEquals(gen.getTdop(), 8.61);
        assertEquals(gen.getTqop(), 0.96);
        assertEquals(gen.getXl(), 0.1583);
        assertEquals(gen.getSg10(), 0.164);
        assertEquals(gen.getSg12(), 0.663);
        assertEquals(gen.getD(), 0.1);

        gen = Generator.createGen(gen.toString());
        assertNotNull(gen);
        assertEquals(gen.getSubType(), 'F');
        assertEquals(gen.getBusName(), "闽可门_1");
        assertEquals(gen.getBaseKv(), 20.0);
        assertEquals(gen.geteMWS(), 2104.0);
        assertEquals(gen.getBaseMva(), 667.0);
        assertEquals(gen.getXdp(), 0.301);
        assertEquals(gen.getXqp(), 0.448);
        assertEquals(gen.getXd(), 2.155);
        assertEquals(gen.getXq(), 2.1);
        assertEquals(gen.getTdop(), 8.61);
        assertEquals(gen.getTqop(), 0.96);
        assertEquals(gen.getXl(), 0.1583);
        assertEquals(gen.getSg10(), 0.164);
        assertEquals(gen.getSg12(), 0.663);
        assertEquals(gen.getD(), 0.1);

        gen = Generator.createGen("MF 闽江阴_120.  3335.       667.    .305 .395 2.1552.1008.610.4 .10 .164 .645 .1江阴火电厂#1机 ");
        assertNotNull(gen);
        assertEquals(gen.getBaseKv(), 20.0);
        assertEquals(gen.geteMWS(), 3335.0);
        assertEquals(gen.getBaseMva(), 667.0);
        assertEquals(gen.getXdp(), 0.305);
        assertEquals(gen.getXqp(), 0.395);
        assertEquals(gen.getXd(), 2.155);
        assertEquals(gen.getXq(), 2.1);
        assertEquals(gen.getTdop(), 8.61);
        assertEquals(gen.getTqop(), 0.4);
        assertEquals(gen.getXl(), 0.1);
        assertEquals(gen.getSg10(), 0.164);
        assertEquals(gen.getSg12(), 0.645);
        assertEquals(gen.getD(), 0.1);

        gen = Generator.createGen(gen.toString());
        assertNotNull(gen);
        assertEquals(gen.getBaseKv(), 20.0);
        assertEquals(gen.geteMWS(), 3335.0);
        assertEquals(gen.getBaseMva(), 667.0);
        assertEquals(gen.getXdp(), 0.305);
        assertEquals(gen.getXqp(), 0.395);
        assertEquals(gen.getXd(), 2.155);
        assertEquals(gen.getXq(), 2.1);
        assertEquals(gen.getTdop(), 8.61);
        assertEquals(gen.getTqop(), 0.4);
        assertEquals(gen.getXl(), 0.1);
        assertEquals(gen.getSg10(), 0.164);
        assertEquals(gen.getSg12(), 0.645);
        assertEquals(gen.getD(), 0.1);
    }

    public void testToString() {
        Generator gen = new Generator();
        gen.setType('M');
        gen.setSubType('C');
        gen.setBusName("闽江阴__");
        gen.setBaseKv(525.0);
        gen.setXdp(0.01);
        gen.seteMWS(999999.0);
        gen.setD(0.0);

        System.out.println(gen);
    }
}
