package zju.se;

import junit.framework.TestCase;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsTopoIsland;
import zju.dsmodel.IeeeDsInHand;
import zju.dspf.DsPowerflowTest;
import zju.dspf.DsSimuMeasMaker;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVectorCreator;
import zju.measure.MeasureFileRw;
import zju.measure.SystemMeasure;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by arno on 16-7-31.
 * @author Dong Shufeng
 */
public class MeasPosOptTest extends TestCase implements MeasTypeCons {


    public void testCase4() {
        MeasPosOpt mpo = new MeasPosOpt(SeTest_case4.island);

        SystemMeasure sm = MeasureFileRw.parse(this.getClass().getResourceAsStream("/measfiles/case4_meas.txt"));
        MeasVectorCreator mc = new MeasVectorCreator();
        mc.getMeasureVector(sm);

        int[] candPos = new int[2];
        candPos[0] = 2;
        candPos[1] = 4;
        int[][] measTypePerPos = new int[candPos.length][];
        double[][] weights = new double[candPos.length][];
        for(int i = 0; i < measTypePerPos.length; i++) {
            measTypePerPos[i] = new int[3];
            measTypePerPos[i][0] = TYPE_BUS_ACTIVE_POWER;
            measTypePerPos[i][1] = TYPE_BUS_REACTIVE_POWER;
            measTypePerPos[i][2] = TYPE_BUS_VOLOTAGE;
            weights[i] = new double[measTypePerPos[i].length];
            weights[i][0] = 0.58;
            weights[i][1] = 0.50;
            weights[i][2] = 0.80;
        }

        mpo.setCandPos(candPos);
        mpo.setMeasTypesPerPos(measTypePerPos);
        mpo.setMeasWeight(weights);
        mpo.setExistMeasPos(mc.measPos);
        mpo.setExistMeasTypes(mc.measTypes);
        mpo.setExistMeasWeight(mc.weights);
        mpo.setMaxDevNum(1);

        mpo.doOpt(false);
    }

    public void testDscase4() {
        InputStream ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsfiles/case4-notrans.txt");
        DistriSys sys = IeeeDsInHand.createDs(ieeeFile, "1", 12.47 / IeeeDsInHand.sqrt3);
        //计算潮流
        DsPowerflowTest.testConverged(sys, false);
        DsTopoIsland dsIsland = sys.getActiveIslands()[0];
        //形成量测
        //DsSimuMeasMaker smMaker = new DsSimuMeasMaker();
        //SystemMeasure sm = smMaker.createFullMeasure(dsIsland, 1, 0.02);
        //MeasureFileRw.writeFileFully(sm, "/home/arno/alg/src/test/resources/dsfiles/case4-notrans-measure.txt");
        //读取量测
        SystemMeasure sm = MeasureFileRw.parse(getClass().getResourceAsStream("/dsfiles/case4-notrans-measure.txt"));
        MeasVectorCreator mc = new MeasVectorCreator();
        mc.getMeasureVector(sm, true);

        MeasPosOpt mpo = new MeasPosOpt(dsIsland);

        String[][] candPos = new String[2][];
        candPos[0] = new String[3];
        candPos[0][0] = "1-0";
        candPos[0][1] = "1-1";
        candPos[0][2] = "1-2";
        candPos[1] = new String[3];
        candPos[1][0] = "3-0";
        candPos[1][1] = "3-1";
        candPos[1][2] = "3-2";
        List<int[][]> measTypePerPos = new ArrayList<>(candPos.length);
        double[][] weights = new double[candPos.length][];
        for (int i = 0; i < candPos.length; i++) {
            int[][] ts = new int[candPos[i].length][3];
            weights[i] = new double[candPos[i].length * 3];
            measTypePerPos.add(ts);
            int count = 0;
            for (int j = 0; j < candPos[i].length; j++) {
                ts[j] = new int[3];
                ts[j][0] = TYPE_BUS_ACTIVE_POWER;
                ts[j][1] = TYPE_BUS_REACTIVE_POWER;
                ts[j][2] = TYPE_BUS_VOLOTAGE;
                weights[i][count] = 0.58;
                weights[i][count + 1] = 0.50;
                weights[i][count + 2] = 0.80;
                count++;
            }
        }

        mpo.setDs_candPos(candPos);
        mpo.setDs_measTypesPerPos(measTypePerPos);
        mpo.setDs_measWeight(weights);
        mpo.setDs_existMeasPos(mc.measPosWithPhase);
        mpo.setDs_existMeasTypes(mc.measTypes);
        mpo.setDs_existMeasWeight(mc.weights);
        mpo.setMaxDevNum(1);

        mpo.doOpt(true);
    }

    public void testDscase13() {
        DistriSys sys = IeeeDsInHand.FEEDER13.clone();
        DsPowerflowTest.testConverged(sys, false);
        DsSimuMeasMaker smMaker = new DsSimuMeasMaker();
        DsTopoIsland dsIsland = sys.getActiveIslands()[0];
        SystemMeasure sm = smMaker.createFullMeasure(dsIsland, 1, 0.02);

        MeasVectorCreator mc = new MeasVectorCreator();
        mc.getMeasureVector(sm);

        MeasPosOpt mpo = new MeasPosOpt(dsIsland);
    }
}
