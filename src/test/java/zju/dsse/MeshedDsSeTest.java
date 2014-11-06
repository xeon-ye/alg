package zju.dsse;

import junit.framework.TestCase;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsModelCons;
import zju.dsmodel.IeeeDsInHand;
import zju.dspf.MeshedDsPfTest;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/3/25
 */
public class MeshedDsSeTest extends TestCase implements DsModelCons {
    public MeshedDsSeTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testLoopedCase34() {
        DistriSys sys = IeeeDsInHand.FEEDER34.clone();
        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "826", "858");
        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "822", "848");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        int branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        int busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsStateEstimatorTest.testTrueCase(sys, 0, 0.02);
        //DsPowerflowWithDGTest.printBusV(sys.getActiveIslands()[0], false);
    }

    public void testLoopedCase123() {
        DistriSys ds = MeshedDsPfTest.getLoopedCase123();
        DistriSys sys = ds.clone();
        sys.buildDynamicTopo();
        int branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        int busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsStateEstimatorTest.testTrueCase(sys, 0, 0.02);

        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "85", "75");
        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "36", "57");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsStateEstimatorTest.testTrueCase(sys, 0, 0.02);

        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "56", "90");
        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "39", "66");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsStateEstimatorTest.testTrueCase(sys, 0, 0.02);

        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "23", "44");
        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "62", "101");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsStateEstimatorTest.testTrueCase(sys, 0, 0.02);

        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "81", "86");
        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "70", "100");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsStateEstimatorTest.testTrueCase(sys, 0, 0.02);

        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "9", "18");
        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "30", "47");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsStateEstimatorTest.testTrueCase(sys, 0, 0.02);

        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "34", "94");
        MeshedDsPfTest.combineTwoNode(sys.getDevices(), "64", "300");
        sys.buildOrigTopo(sys.getDevices());
        sys.fillCnBaseKv();
        sys.buildDynamicTopo();
        branchNum = sys.getActiveIslands()[0].getGraph().edgeSet().size();
        busNum = sys.getActiveIslands()[0].getGraph().vertexSet().size();
        System.out.println("回路个数：" + (branchNum - busNum + 1));
        DsStateEstimatorTest.testTrueCase(sys, 0, 0.02);
    }
}
