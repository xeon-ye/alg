package zju.se;

import junit.framework.TestCase;
import zju.ieeeformat.DefaultIcfParser;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.measure.MeasTypeCons;
import zju.measure.SystemMeasure;
import zju.pf.PolarPf;
import zju.pf.SimuMeasMaker;

import java.io.InputStream;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/5/6
 */
public class WlsSeTest_IeeeCase extends TestCase implements MeasTypeCons {

    StateEstimator se;
    NewtonWlsSe alg;

    public WlsSeTest_IeeeCase(String name) {
        super(name);
        se = new StateEstimator();
        alg = new NewtonWlsSe();
        se.setAlg(alg);
    }

    private void doPf(IEEEDataIsland island) {
        //先算一遍潮流
        PolarPf pf = new PolarPf();
        pf.setTol_p(1e-5);
        pf.setTol_q(1e-5);
        pf.setTolerance(1e-5);
        pf.setOriIsland(island);
        pf.setDecoupledPqNum(0);
        //计算潮流
        pf.doPf();
        assertTrue(pf.isConverged());
        pf.fillOriIslandPfResult();
    }

    public void setUp() throws Exception {
    }

    public void testOneCase() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/matpower/case3120sp.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");
        SystemMeasure sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doPf(island);
        doSE(island, sm, null);
    }

    public void testCaseTrue() {
        IEEEDataIsland island;
        SystemMeasure sm;
        island = IcfDataUtil.ISLAND_14.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_14);
        island = IcfDataUtil.ISLAND_30.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_30);

        island = IcfDataUtil.ISLAND_39.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_39);

        island = IcfDataUtil.ISLAND_57.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_57);

        island = IcfDataUtil.ISLAND_118.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_118);

        island = IcfDataUtil.ISLAND_300.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_300);
    }

    public void testCaseGauss() {
        IEEEDataIsland island;
        SystemMeasure sm;
        island = IcfDataUtil.ISLAND_14.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, null);
        island = IcfDataUtil.ISLAND_30.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, null);
        island = IcfDataUtil.ISLAND_39.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, null);
        island = IcfDataUtil.ISLAND_57.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, null);
        island = IcfDataUtil.ISLAND_118.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, null);
        island = IcfDataUtil.ISLAND_300.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, null);
    }

    public void doSE(IEEEDataIsland island, SystemMeasure sm, IEEEDataIsland ref) {
        //alg.setPrintPath(false);
        double slackBusAngle = island.getBus(island.getSlackBusNum()).getFinalAngle() * Math.PI / 180.;
        alg.setSlackBusAngle(slackBusAngle);

        se.setOriIsland(island);
        se.setSm(sm);
        se.setFlatStart(false);

        SeResultInfo r;

        //下面这一段测试传统最小二乘的效果
        se.doSe();
        assertTrue(alg.isConverged());
        r = se.createPfResult();
        assertNotNull(r);
        if (ref != null)
            SeTest_IeeeCase.assertTrueSe(ref, r);
        //printS1(ref, r, isTrueValue);
        System.out.println("WLS状态估计迭代次数：" + alg.getIterNum() + "\t用时：" + alg.getTimeUsed() + "\t");
    }

    public NewtonWlsSe getAlg() {
        return alg;
    }
}
