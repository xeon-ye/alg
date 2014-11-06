package zju.pf;

import junit.framework.TestCase;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.util.NumberOptHelper;

/**
 * Created by IntelliJ IDEA.
 *
 * @author zhangxiao
 *         Date: 2010-5-12
 */
public class NumberOptTest extends TestCase {

    public NumberOptTest(String name) {
        super(name);
    }

    @Override
    public void setUp() throws Exception {
    }

    public void testStandardCases() {
        testNumberOpt(IcfDataUtil.ISLAND_14.clone());
        testNumberOpt(IcfDataUtil.ISLAND_30.clone());
        testNumberOpt(IcfDataUtil.ISLAND_39.clone());
        testNumberOpt(IcfDataUtil.ISLAND_57.clone());
        testNumberOpt(IcfDataUtil.ISLAND_118.clone());
        testNumberOpt(IcfDataUtil.ISLAND_300.clone());
    }


    //public void testCaseDB() {
    //    System.out.println("CaseDB");
    //    testNumberOpt(this.getClass().getResource("/db/ieee.txt").getFile());
    //}

    private void testNumberOpt(IEEEDataIsland island) {

        IEEEDataIsland island0 = island.clone();
        IEEEDataIsland island1 = island.clone();
        IEEEDataIsland island2_1 = island.clone();

        NumberOptHelper numOpt0 = new NumberOptHelper();
        System.out.print("SimpleSort : ");
        long start = System.nanoTime();
        numOpt0.simple(island0);
        numOpt0.trans(island0);
        System.out.print((System.nanoTime() - start) / 1000 + "us ");

        //tinney1
        NumberOptHelper numOpt1 = new NumberOptHelper();
        System.out.print("Tinney1 : ");
        start = System.nanoTime();
        numOpt1.tinney1(island1);
        numOpt1.trans(island1);
        System.out.print((System.nanoTime() - start) / 1000 + "us ");

        //tinney2 using connected buses
        NumberOptHelper numOpt2_1 = new NumberOptHelper();
        System.out.print("Tinney2 : ");
        start = System.nanoTime();
        numOpt2_1.tinney2(island2_1);
        numOpt2_1.trans(island2_1);
        System.out.print((System.nanoTime() - start) / 1000 + "us ");
    }
}

