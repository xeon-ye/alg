package zju.dsse;

import junit.framework.TestCase;
import zju.dsmodel.*;
import zju.dspf.DsPowerflowTest;
import zju.dspf.DsSimuMeasMaker;
import zju.measure.SystemMeasure;
import zju.se.SeObjective;

/**
 * DsStateEstimator Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>11/11/2013</pre>
 */
public class DsStateEstimatorTest extends TestCase implements DsModelCons {
    public DsStateEstimatorTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testCaseTrue() {
        testTrueCase(IeeeDsInHand.FEEDER4_DGrY_B.clone(), 1.0, 0);
        testTrueCase(IeeeDsInHand.FEEDER4_DGrY_UNB.clone(), 1.0, 0);
        testTrueCase(IeeeDsInHand.FEEDER4_GrYGrY_B.clone(), 1.0, 0);
        testTrueCase(IeeeDsInHand.FEEDER4_GrYGrY_UNB.clone(), 1.0, 0);
        testTrueCase(IeeeDsInHand.FEEDER4_DD_B.clone(), 9.0, 0);
        testTrueCase(IeeeDsInHand.FEEDER4_DD_UNB.clone(), 9.0, 0);

        testTrueCase(IeeeDsInHand.FEEDER13.clone(), 1.0, 0);
        testTrueCase(IeeeDsInHand.FEEDER34.clone(), 1.0, 0);
        testTrueCase(IeeeDsInHand.FEEDER37.clone(), 10.0, 0);
        testTrueCase(IeeeDsInHand.FEEDER123.clone(), 1e-10, 0);
    }

    public void testCaseGauss() {
        testTrueCase(IeeeDsInHand.FEEDER4_DGrY_B.clone(), 0, 0.05);
        testTrueCase(IeeeDsInHand.FEEDER4_DGrY_UNB.clone(), 0, 0.05);
        testTrueCase(IeeeDsInHand.FEEDER4_GrYGrY_B.clone(), 0, 0.05);
        testTrueCase(IeeeDsInHand.FEEDER4_GrYGrY_UNB.clone(), 0, 0.05);
        testTrueCase(IeeeDsInHand.FEEDER4_DD_B.clone(), 0, 0.05);
        testTrueCase(IeeeDsInHand.FEEDER4_DD_UNB.clone(), 0, 0.05);

        testTrueCase(IeeeDsInHand.FEEDER13.clone(), 0, 0.05);
        testTrueCase(IeeeDsInHand.FEEDER34.clone(), 0, 0.05);
        testTrueCase(IeeeDsInHand.FEEDER37.clone(), 0, 0.05);
        testTrueCase(IeeeDsInHand.FEEDER123.clone(), 0, 0.05);
    }

    public static void testTrueCase(DistriSys sys, double tol, double error) {
        DsPowerflowTest.testConverged(sys, false);
        DsSimuMeasMaker smMaker = new DsSimuMeasMaker();
        SystemMeasure sm = smMaker.createFullMeasure(sys.getActiveIslands()[0], 1, error);

        if (error < 1e-3) {
            DistriSys ds = sys.clone();
            ds.buildDynamicTopo();
            ds.createCalDevModel();
            doDsSe(sys, tol, ds, sm);
        } else
            doDsSe(null, tol, sys, sm);
    }

    public static void doDsSe(DistriSys dsRef, double tol, DistriSys ds, SystemMeasure sm) {
        long start = System.currentTimeMillis();
        DsStateEstimator estimator = new DsStateEstimator(ds, sm);
        estimator.setAlg(new IpoptDsSe());
        estimator.getAlg().getObjFunc().setObjType(SeObjective.OBJ_TYPE_WLS);
        estimator.doSe();
        assertTrue(estimator.getAlg().isConverged());
        System.out.println("计算WLS状态估计用时：" + (System.currentTimeMillis() - start) + "ms.");
        if (dsRef != null && tol > 0)
            assertTrueSe(dsRef, ds, tol);


        estimator.setAlg(new IpoptLcbSe());
        estimator.getAlg().getObjFunc().setObjType(SeObjective.OBJ_TYPE_WLS);
        estimator.doSe();
        assertTrue(estimator.getAlg().isConverged());
        System.out.println("计算WLS状态估计用时：" + (System.currentTimeMillis() - start) + "ms.");
        if (dsRef != null && tol > 0)
            assertTrueSe(dsRef, ds, tol);

        //start = System.currentTimeMillis();
        //estimator.getAlg().getObjFunc().setObjType(SeObjective.OBJ_TYPE_SIGMOID);
        //estimator.doSe();
        //assertTrue(estimator.getAlg().isConverged());
        //System.out.println("计算MNMR状态估计用时：" + (System.currentTimeMillis() - start) + "ms.");
        //if(dsRef != null && tol > 0)
        //    assertTrueSe(dsRef, ds, tol);
    }

    public static void assertTrueSe(DistriSys ref, DistriSys ds, double tol) {
        DsTopoIsland island = ds.getActiveIslands()[0];
        DsTopoIsland refIsland = ref.getActiveIslands()[0];
        for (DsTopoNode tn : island.getTns()) {
            double[][] a = island.getBusV().get(tn);
            double[][] b = refIsland.getBusV().get(refIsland.getTnNoToTn().get(tn.getTnNo()));
            assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(a, b, tol));
        }
    }
}
