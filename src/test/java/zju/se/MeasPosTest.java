package zju.se;

import junit.framework.TestCase;
import zju.measure.DefaultMeasParser;
import zju.measure.MeasVector;
import zju.measure.MeasVectorCreator;
import zju.measure.SystemMeasure;

/**
 * Created by arno on 16-7-31.
 */
public class MeasPosTest extends TestCase {


    void testCase4() {
        MeasPosOpt mpo = new MeasPosOpt(SeTest_case4.island);
        int[] candPos = new int[4];
        for(int i = 0; i < 4; i++) {
            candPos[i] = i + 1;
        }
        mpo.setCandPos(candPos);
        //mpo.setMeasTypesPerPos();
        //mpo.setMeasWeight();

        SystemMeasure sm = DefaultMeasParser.parse(this.getClass().getResourceAsStream("/measfiles/case4_meas.txt"));
        int[] existMeasTypes = new int[sm.getMeasureNum()];
        int[] existMeasPos = new int[existMeasTypes.length];
        double[] existMeasWeight = new double[existMeasPos.length];

        MeasVectorCreator mc = new MeasVectorCreator();
        MeasVector v = mc.getMeasureVector(sm);

        //mpo.setExistMeasPos(v.setWeight());
        mpo.setExistMeasTypes(existMeasTypes);
        mpo.setExistMeasWeight(existMeasWeight);
    }
}
