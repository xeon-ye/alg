package zju.se;

import junit.framework.TestCase;
import zju.measure.*;

/**
 * Created by arno on 16-7-31.
 */
public class MeasPosOptTest extends TestCase implements MeasTypeCons {


    public void testCase4() {
        MeasPosOpt mpo = new MeasPosOpt(SeTest_case4.island);
        int[] candPos = new int[0];
        for(int i = 0; i < 0; i++)
            candPos[i] = i + 1;

        int[][] measTypePerPos = new int[candPos.length][];
        double[][] weights = new double[candPos.length][];
        for(int i = 0; i < measTypePerPos.length; i++) {
            measTypePerPos[i] = new int[1];
            measTypePerPos[i][0] = TYPE_BUS_ACTIVE_POWER;
            //measTypePerPos[i][1] = TYPE_BUS_REACTIVE_POWER;
            weights[i] = new double[1];
            weights[i][0] = 1.0;
           // weights[i][1] = 1.0;
        }

        SystemMeasure sm = DefaultMeasParser.parse(this.getClass().getResourceAsStream("/measfiles/case4_meas.txt"));
        MeasVectorCreator mc = new MeasVectorCreator();
        mc.getMeasureVector(sm);

        mpo.setCandPos(candPos);
        mpo.setMeasTypesPerPos(measTypePerPos);
        mpo.setMeasWeight(weights);
        mpo.setExistMeasPos(mc.measPos);
        mpo.setExistMeasTypes(mc.measTypes);
        mpo.setExistMeasWeight(mc.weights);

        mpo.doOpt();
    }
}
