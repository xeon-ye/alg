package zju.util.expression;

import junit.framework.TestCase;

import java.util.HashMap;
import java.util.Map;

/**
 * Calculator Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>06/12/2010</pre>
 */
public class SimpleCalculatorTest extends TestCase {
    public SimpleCalculatorTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void test1() {
        Number r = doCal("3+5*(6-1)-18/3^2");
        assertNotNull(r);
        assertEquals(26, r.intValue());
        r = doCal("3+5*(6-1)-16%3");
        assertEquals(27, r.intValue());

        r = doCal("1/(2*2+3*3)^0.5");
        assertNotNull(r);

        r = doCal("(0 - 1.5625859498977661)/((0 - 1.5625859498977661)*(0 - 1.5625859498977661)+(0 - 0.444685697555542)*(0 - 0.444685697555542))^0.5");
        assertNotNull(r);

        r = doCal("(0 - 1E-2)/((0 - 2e-1)*(0 - 1)+ 2E-2^0.5)");
        assertNotNull(r);

        r = doCal("1.0 - 3.0 * sin(10 - 3) / 2.0");
        assertEquals(r, 1.0 - 3.0 * Math.sin(10 - 3) / 2.0);

        r = doCal("1.0 - 3.0^2 * cos(10 - 3) / 2.0 - sqrt(4)");
        assertEquals(r, 1.0 - 9.0 * Math.cos(10 - 3) / 2.0 - Math.sqrt(4.0));

        r = doCal("1.0 - exp(2.2)^2 * tan(10 - 3) / 2.0 - sqrt(4)");
        assertTrue(Math.abs(r.doubleValue() - (1.0 - Math.exp(2.2) * Math.exp(2.2) * Math.tan(10 - 3) / 2.0 - Math.sqrt(4.0))) < 1e-5);
    }

    public void test2() {
        Number r = doCal("3+5.5");
        assertNotNull(r);
        assertEquals(8.5, r.doubleValue());

        String exp = "10.0/(10.0*10.0 + 2.0 * 2.0)^0.5";
        r = doCal(exp);
        assertNotNull(r);
        assertTrue(Math.abs(r.doubleValue() - 0.98) < 0.01);
    }

    public void test3() throws IncalculableExpressionException, UnknownOperatorException, IllegalExpressionException {
        Calculator cal = new Calculator();
        Map<String, Number> varValues = new HashMap<String, Number>();
        varValues.put("var1", 10.0);
        Number r = cal.eval("10.0/(var(var1)*10.0 + 2.0 * 2.0)^0.5", varValues);
        assertTrue(Math.abs(r.doubleValue() - 0.98) < 0.01);
    }

    public Number doCal(String expression) {
        System.out.println("A infix expression to parse:" + expression);
        System.out.println();

        try {
            Calculator calculator = new Calculator();
            System.out.print("Calculating......");
            Number result = calculator.eval(expression);
            System.out.print("the result is: " + result);
            return result;
            //System.out.println(result.doubleValue());
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
