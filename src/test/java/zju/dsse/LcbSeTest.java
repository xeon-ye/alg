package zju.dsse;

import junit.framework.TestCase;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsModelCons;
import zju.dsmodel.IeeeDsInHand;
import zju.dsntp.IpoptLcbSe;
import zju.dspf.DsPowerflowTest;
import zju.dsntp.DsSimuMeasMaker;
import zju.measure.MeasVector;
import zju.measure.MeasVectorCreator;
import zju.measure.SystemMeasure;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2016/7/12
 */
public class LcbSeTest extends TestCase implements DsModelCons {

    public LcbSeTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testCaseTrue() {
        testTrueCase(IeeeDsInHand.FEEDER4_DGrY_B.clone(), 0);
        testTrueCase(IeeeDsInHand.FEEDER4_DGrY_UNB.clone(), 0);
        testTrueCase(IeeeDsInHand.FEEDER4_GrYGrY_B.clone(), 0);
        testTrueCase(IeeeDsInHand.FEEDER4_GrYGrY_UNB.clone(), 0);
        testTrueCase(IeeeDsInHand.FEEDER4_DD_B.clone(),  0);
        testTrueCase(IeeeDsInHand.FEEDER4_DD_UNB.clone(), 0);
    }

    public static void testTrueCase(DistriSys sys, double error) {
        DsPowerflowTest.testConverged(sys, false);
        DsSimuMeasMaker smMaker = new DsSimuMeasMaker();
        SystemMeasure sm = smMaker.createFullMeasure(sys.getActiveIslands()[0], 1, error);

        DistriSys ds = sys.clone();
        ds.buildDynamicTopo();
        ds.createCalDevModel();

        MeasVector meas = new MeasVectorCreator().getMeasureVector(sm, true);

        IpoptLcbSe alg = new IpoptLcbSe();
        alg.setDsIsland(ds.getActiveIslands()[0]);
        alg.setMeas(meas);
        alg.initial();
        //todo: 测试Jacobian和Hessian矩阵的正确性
    }
}
