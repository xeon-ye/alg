package zju.se.bdfilter;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.measure.MeasTypeCons;
import zju.measure.MeasureInfo;
import zju.pf.SimuMeasMaker;
import zju.se.SeStatistics;

import java.util.List;
import java.util.Map;

/**
 * BusBasedFilter Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>11/26/2013</pre>
 */
public class BusBasedFilterTest extends TestCase implements MeasTypeCons {
    public BusBasedFilterTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public static Test suite() {
        return new TestSuite(BusBasedFilterTest.class);
    }

    /**
     * 测试数据的是4节点系统中的节点1
     */
    public void testBus1() {
        double baseV = SeStatistics.base_voltage_110;
        double baseMva = 100.;
        double baseZ = baseV * baseV / baseMva;
        BranchData[] branches = new BranchData[2];
        branches[0] = new BranchData();
        branches[0].setBranchR(3.05 / baseZ);
        branches[0].setBranchX(8.37 / baseZ);
        branches[0].setLineB(2 * 0.00056 * baseZ);
        branches[0].setId(1);
        branches[0].setTapBusNumber(1);
        branches[0].setZBusNumber(3);

        branches[1] = new BranchData();
        branches[1].setBranchR(0.52 / baseZ);
        branches[1].setBranchX(2.66 / baseZ);
        branches[1].setLineB(2 * 0.00014 * baseZ);
        branches[1].setId(2);
        branches[1].setTapBusNumber(1);
        branches[1].setZBusNumber(2);

        MeasureInfo[] measures = new MeasureInfo[5];
        //measures[0] = new MeasureInfo("1", TYPE_BUS_ACTIVE_POWER, 20.86 / baseMva);
        //measures[1] = new MeasureInfo("1", TYPE_LINE_FROM_ACTIVE, -19.47 / baseMva);
        //measures[2] = new MeasureInfo("1", TYPE_LINE_TO_ACTIVE, 19.57/ baseMva);
        //measures[3] = new MeasureInfo("2", TYPE_LINE_FROM_ACTIVE, 40.33/ baseMva);
        //measures[4] = null;
        //for (MeasureInfo m : measures) {
        //    if(m == null)
        //        continue;
        //    double trueValue = m.getValue();
        //    m.setValue(trueValue + RandomMaker.randomNorm(0, 0.05 * trueValue));
        //}

        measures[0] = new MeasureInfo("1", TYPE_BUS_ACTIVE_POWER, 18.71 / baseMva);
        measures[1] = new MeasureInfo("1", TYPE_LINE_FROM_ACTIVE, -19.1 / baseMva);
        measures[2] = new MeasureInfo("1", TYPE_LINE_TO_ACTIVE, 17.55 / baseMva);
        measures[3] = new MeasureInfo("2", TYPE_LINE_FROM_ACTIVE, 41.79 / baseMva);
        measures[4] = null;
        BusBasedFilter filter = new BusBasedFilter(branches, measures);

        //measures[0].setValue(10.8 /baseMva);
        //measures[1].setValue(-30.1 / baseMva);
        measures[2].setValue(27.55 / baseMva);
        //measures[3].setValue(55.79 / baseMva);
        filter.doFilter();
    }

    /**
     * 测试数据的是4节点系统中的节点3
     */
    public void testBus3() {
        double baseV = SeStatistics.base_voltage_110;
        double baseMva = 100.;
        double baseZ = baseV * baseV / baseMva;
        BranchData[] branches = new BranchData[3];
        branches[0] = new BranchData();
        branches[0].setBranchR(3.05 / baseZ);
        branches[0].setBranchX(8.37 / baseZ);
        branches[0].setLineB(2 * 0.00056 * baseZ);
        branches[0].setId(1);
        branches[0].setTapBusNumber(1);
        branches[0].setZBusNumber(3);

        branches[1] = new BranchData();
        branches[1].setBranchR(0.41 / baseZ);
        branches[1].setBranchX(2.15 / baseZ);
        branches[1].setLineB(2 * 0.00044 * baseZ);
        branches[1].setId(2);
        branches[1].setTapBusNumber(2);
        branches[1].setZBusNumber(3);

        branches[2] = new BranchData();
        branches[2].setBranchR(0.0 / baseZ);
        branches[2].setBranchX(7.5 / baseZ);
        branches[2].setLineB(0.0);
        branches[2].setId(3);
        branches[2].setTapBusNumber(3);
        branches[2].setZBusNumber(4);

        MeasureInfo[] measures = new MeasureInfo[7];
        //measures[0] = new MeasureInfo("1", TYPE_BUS_ACTIVE_POWER, -50 / baseMva);
        //measures[1] = new MeasureInfo("1", TYPE_LINE_FROM_ACTIVE, 19.57/ baseMva);
        //measures[2] = new MeasureInfo("1", TYPE_LINE_TO_ACTIVE, -19.47 / baseMva);
        //measures[3] = new MeasureInfo("2", TYPE_LINE_FROM_ACTIVE, 130.42/ baseMva);
        //measures[4] = null;
        //measures[5] = new MeasureInfo("3", TYPE_LINE_FROM_ACTIVE, -199.99/ baseMva);
        //measures[6] = null;
        //for (MeasureInfo m : measures) {
        //    if(m == null)
        //        continue;
        //    double trueValue = m.getValue();
        //    m.setValue(trueValue + RandomMaker.randomNorm(0, 0.05 * trueValue));
        //}

        measures[0] = new MeasureInfo("1", TYPE_BUS_ACTIVE_POWER, -49.49 / baseMva);
        measures[1] = new MeasureInfo("1", TYPE_LINE_FROM_ACTIVE, 17.55 / baseMva);
        measures[2] = new MeasureInfo("1", TYPE_LINE_TO_ACTIVE, -19.1 / baseMva);
        measures[3] = new MeasureInfo("2", TYPE_LINE_FROM_ACTIVE, 132.57 / baseMva);
        measures[4] = null;
        measures[5] = new MeasureInfo("3", TYPE_LINE_FROM_ACTIVE, -193.22 / baseMva);
        measures[6] = null;
        BusBasedFilter filter = new BusBasedFilter(branches, measures);

        //measures[0].setValue(10.8 /baseMva);
        measures[1].setValue(27.55 / baseMva);
        //measures[2].setValue(-30.1 / baseMva);
        //measures[3].setValue(55.79 / baseMva);
        //measures[3].setValue(60.33 / baseMva);
        filter.doFilter();

        //for(int i = 0; i < 10; i++)
        //    filter.doFilter();
        //long start = System.currentTimeMillis();
        //for(int i = 0; i < 10; i++)
        //    filter.doFilter();
        //System.out.println("Time used: " + (System.currentTimeMillis() - start) + "ms");
    }

    public void testCase_Ieee() throws Exception {
        //testIeeeIsland(IcfDataUtil.ISLAND_14);
        //testIeeeIsland(IcfDataUtil.ISLAND_30);
        testIeeeIsland(IcfDataUtil.ISLAND_39);
        //testIeeeIsland(IcfDataUtil.ISLAND_57);
        testIeeeIsland(IcfDataUtil.ISLAND_118);
        testIeeeIsland(IcfDataUtil.ISLAND_300);
    }

    public void testIeeeIsland(IEEEDataIsland island) throws Exception {
        BusBasedFilter filter = new BusBasedFilter();

        double[] x = new double[3];
        Map<Integer, BusData> busMap = island.getBusMap();
        Map<Integer, List<BranchData>> busToBranches = IcfDataUtil.getBus2Branches(island);
        int count = 0, total = 0;
        long start = System.currentTimeMillis();
        for (BusData b : island.getBuses()) {
            filter.setBranches(busToBranches.get(b.getBusNumber()).toArray(new BranchData[0]));
            filter.setMeases(getEstValue(b, busToBranches.get(b.getBusNumber())));
            filter.doFilter();
            total++;
        }
        double v = (double) count / (double) total;
        System.out.println("共检测" + total + "条母线,用时" + (System.currentTimeMillis() - start) + "ms,成功率：" + v);
    }

    public MeasureInfo[] getEstValue(BusData bus, List<BranchData> branches) {
        MeasureInfo[] meas = new MeasureInfo[2 * branches.size() + 1];
        double trueValue = 0.0;//todo: wrong
        meas[0] = new MeasureInfo();
        SimuMeasMaker.formMeasure(trueValue, 1, 0.0, meas[0]);
        for (int i = 0; i < branches.size(); i++) {
            meas[2 * i + 1] = new MeasureInfo(String.valueOf(branches.get(i).getId()), TYPE_LINE_FROM_ACTIVE, 0.0);
            SimuMeasMaker.formMeasure(trueValue, 1, 0.0, meas[i + 1]);
            meas[2 * i + 2] =  new MeasureInfo(String.valueOf(branches.get(i).getId()), TYPE_LINE_TO_ACTIVE, 0.0);
            SimuMeasMaker.formMeasure(trueValue, 1, 0.0, meas[i + 1]);
        }
        return meas;
    }
}
