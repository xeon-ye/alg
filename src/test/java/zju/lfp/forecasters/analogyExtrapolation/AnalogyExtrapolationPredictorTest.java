package zju.lfp.forecasters.analogyExtrapolation;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import zju.lfp.utils.MultiTimeSeries;

/**
 * AnalogyExtrapolationPredictor Tester.
 *
 * @author <Authors name>
 * @since <pre>12/29/2007</pre>
 * @version 1.0
 */
public class AnalogyExtrapolationPredictorTest extends TestCase {
    private AnalogyExtrapolationPredictor predictor;
    private MultiTimeSeries multiTimeSeries;
    private int shortPeriod = 10;
    private int longPeriod = shortPeriod * 5;
    private int usefulPointsNum = 3;
    private int lenUnKnown = 10;
    private int numShortPeriod = 6;

    public AnalogyExtrapolationPredictorTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
        predictor = new AnalogyExtrapolationPredictor(longPeriod, shortPeriod, usefulPointsNum);
        multiTimeSeries = MultiTimeSeries.multiDoublePeriodInstance(3, longPeriod, shortPeriod,
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
        return new TestSuite(AnalogyExtrapolationPredictorTest.class);
    }
}
