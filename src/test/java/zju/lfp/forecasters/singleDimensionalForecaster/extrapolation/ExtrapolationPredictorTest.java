package zju.lfp.forecasters.singleDimensionalForecaster.extrapolation;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import zju.lfp.utils.MultiTimeSeries;

/**
 * ExtrapolationPredictor Tester.
 *
 * @author <Authors name>
 * @since <pre>12/21/2007</pre>
 * @version 1.0
 */
public class ExtrapolationPredictorTest extends TestCase {
    private ExtrapolationPredictor predictor;
    private MultiTimeSeries multiTimeSeries;
    private int shortPeriod = 10;
    private int longPeriod = shortPeriod * 5;
    private int usefulPointsNum = 3;
    private int lenUnKnown = 10;
    private int numShortPeriod = 6;

    public ExtrapolationPredictorTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
        predictor = new ExtrapolationPredictor(longPeriod, shortPeriod, usefulPointsNum);
        multiTimeSeries = MultiTimeSeries.doublePeriodInstance(longPeriod, shortPeriod,
                numShortPeriod, lenUnKnown);
    }

    public void testPredict() throws Exception {
        predictor.predict(multiTimeSeries);
        for(int i = 0; i < multiTimeSeries.getNLen(); i++) {
            assertFalse(Double.isNaN(multiTimeSeries.getValue(multiTimeSeries.getNumAttributes() - 1,
                    i)));
        }
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public static Test suite() {
        return new TestSuite(ExtrapolationPredictorTest.class);
    }
}
