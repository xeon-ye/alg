package zju.se;

import junit.framework.TestCase;
import zju.ieeeformat.DefaultIcfParser;
import zju.ieeeformat.IEEEDataIsland;
import zju.measure.MeasTypeCons;
import zju.measure.SystemMeasure;
import zju.pf.PolarPf;
import zju.pf.SimuMeasMaker;

import java.io.InputStream;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-10-22
 */
public class SeTest_RealCase extends TestCase implements MeasTypeCons {

    public SeTest_RealCase(String name) {
        super(name);
    }

    public void setUp() throws Exception {
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

    public void testOneCase() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/matpower/case3120sp.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");
        SystemMeasure sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doPf(island);
        doTrueSe(island);
    }

    public void testCaseDb_true() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/ieee_orginal.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "GBK");
        //先算一遍潮流
        doTrueSe(island);
    }

    public void testCaseAnhui_true() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/sdxx201307081415.txt");
        //InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/ahxx201312041630.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");
        doTrueSe(island);
    }

    private void doTrueSe(IEEEDataIsland island) {
        //先算一遍潮流
        doPf(island);

        SeTest_IeeeCase seTest = new SeTest_IeeeCase("RealCase");
        seTest.alg.setTol_p(1e-5);
        seTest.alg.setTol_q(1e-5);
        IEEEDataIsland clonedIsland = island.clone();
        SystemMeasure sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        seTest.doSE(clonedIsland, sm, island, false, true);
    }

    public void testCaseAnhui() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/sdxx201307081415.txt");
//        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/ahxx201312041630.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");
        //先算一遍潮流
        doPf(island);
        SeTest_IeeeCase.doSeStudy(island, 0.005, 0.005);
    }

    public void testCaseDb() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/ieee_orginal.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "GBK");
        //先算一遍潮流
        doPf(island);
        SeTest_IeeeCase.doSeStudy(island, 0.005, 0.005);
    }

    public void testCaseShanghai() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/shanghai_ieee.dat");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "GBK");
        //先算一遍潮流
        doPf(island);
        SeTest_IeeeCase.doSeStudy(island, 0.005, 0.005);
    }
}
