package zju.se.bdfilter;

import junit.framework.TestCase;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.measure.MeasTypeCons;
import zju.measure.MeasureInfo;
import zju.pf.SimuMeasMaker;
import zju.se.SeStatistics;
import zju.util.StateCalByPolar;

import java.util.Map;

/**
 * BranchBasedFilter Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>11/24/2013</pre>
 */
public class BranchBasedFilterTest extends TestCase implements MeasTypeCons {
    public BranchBasedFilterTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    /**
     * 测试数据的是4节点系统中的一条线路1-3
     */
    public void testBranch() throws Exception {
        BranchBasedFilter filter = new BranchBasedFilter();
        double baseV = SeStatistics.base_voltage_110;
        double baseMva = 100.;
        double baseZ = baseV * baseV / baseMva;
        BranchData branch = new BranchData();
        branch.setBranchR(3.05 / baseZ);
        branch.setBranchX(8.37 / baseZ);
        branch.setLineB(2 * 0.00056 * baseZ);
        branch.setId(1);
        branch.setTapBusNumber(1);
        branch.setZBusNumber(3);

        MeasureInfo[] measures = new MeasureInfo[6];
        measures[0] = new MeasureInfo("1", TYPE_LINE_FROM_ACTIVE, -19.47 / baseMva);
        measures[1] = new MeasureInfo("1", TYPE_LINE_FROM_REACTIVE, -2.56 / baseMva);
        measures[2] = new MeasureInfo("1", TYPE_LINE_TO_ACTIVE, 19.57 / baseMva);
        measures[3] = new MeasureInfo("1", TYPE_LINE_TO_REACTIVE, -11.12 / baseMva);
        measures[4] = new MeasureInfo("1", TYPE_BUS_VOLOTAGE, 111.5 / baseV);
        measures[5] = new MeasureInfo("3", TYPE_BUS_VOLOTAGE, 111.71 / baseV);

        filter.setBranch(branch);
        filter.setMeases(measures);

        //case a
        System.out.println("==================== case a) Pij ===============");
        double origV = measures[0].getValue();
        measures[0].setValue(-10.47 / baseMva);
        filter.doFilter();
        assertTrue(filter.isHasBadP());
        measures[0].setValue(origV);
        //case b
        System.out.println("==================== case b) Pji ===============");
        origV = measures[2].getValue();
        measures[2].setValue(10.47 / baseMva);
        filter.doFilter();
        assertTrue(filter.isHasBadP());
        measures[2].setValue(origV);
        //case c
        System.out.println("==================== case c) Qij ===============");
        origV = measures[1].getValue();
        measures[1].setValue(2.56 / baseMva);
        filter.doFilter();
        assertTrue(filter.isHasBadQ());
        measures[1].setValue(origV);
        //case d
        System.out.println("==================== case d) Qji ===============");
        origV = measures[3].getValue();
        measures[3].setValue(-5.12 / baseMva);
        filter.doFilter();
        assertTrue(filter.isHasBadQ());
        measures[3].setValue(origV);
        //case e
        System.out.println("==================== case e) Ui ===============");
        origV = measures[4].getValue();
        measures[4].setValue(101.5 / baseMva);
        filter.doFilter();
        //assertTrue(filter.isViBad());
        measures[4].setValue(origV);
        //case f
        System.out.println("==================== case f) Uj ===============");
        origV = measures[5].getValue();
        measures[5].setValue(120.99 / baseMva);
        filter.doFilter();
        assertTrue(filter.isVjBad());
        measures[5].setValue(origV);
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
        MeasureInfo[] meas = new MeasureInfo[6];
        meas[0] = new MeasureInfo("1", TYPE_LINE_FROM_ACTIVE, 0);
        meas[1] = new MeasureInfo("1", TYPE_LINE_FROM_REACTIVE, 0);
        meas[2] = new MeasureInfo("1", TYPE_LINE_TO_ACTIVE, 0);
        meas[3] = new MeasureInfo("1", TYPE_LINE_TO_REACTIVE, 0);
        meas[4] = new MeasureInfo("1", TYPE_BUS_VOLOTAGE, 0);
        meas[5] = new MeasureInfo("3", TYPE_BUS_VOLOTAGE, 0);
        BranchBasedFilter filter = new BranchBasedFilter();
        filter.setMeases(meas);
        double[] x = new double[3];
        Map<Integer, BusData> busMap = island.getBusMap();
        int count = 0, total = 0;
        long start = System.currentTimeMillis();
        for (BranchData b : island.getBranches()) {
            if (b.getType() == BranchData.BRANCH_TYPE_ACLINE) {
                filter.setBranch(b);
                meas[4].setPositionId(String.valueOf(b.getTapBusNumber()));
                meas[5].setPositionId(String.valueOf(b.getZBusNumber()));
                x[0] = busMap.get(b.getTapBusNumber()).getFinalVoltage();
                x[1] = busMap.get(b.getZBusNumber()).getFinalVoltage();
                x[2] = busMap.get(b.getTapBusNumber()).getFinalAngle() - busMap.get(b.getZBusNumber()).getFinalAngle();
                x[2] = x[2] * Math.PI / 180.;
                getEstValue(b, meas, x);
                double v = Math.random() > 0.5 ? 1 : -1;
                double rand = Math.random();
                if (rand < 0.33) {
                    if (total % 2 == 0)
                        meas[1].setValue(meas[1].getValue_true() * (1.0 + 0.3 * v));
                    else
                        meas[3].setValue(meas[3].getValue_true() * (1.0 + 0.3 * v));
                    filter.doFilter();
                    if (filter.isHasBadQ())
                        count++;
                } else if (rand < 0.66) {
                    if (total % 2 == 0)
                        meas[0].setValue(meas[0].getValue_true() * (1.0 + 0.2 * v));
                    else
                        meas[2].setValue(meas[2].getValue_true() * (1.0 + 0.2 * v));
                    filter.doFilter();
                    if (filter.isHasBadP())
                        count++;
                } else {
                    if (total % 2 == 0) {
                        meas[4].setValue(meas[4].getValue_true() * (1.0 + 0.3 * v));
                        filter.doFilter();
                        if (filter.isViBad())
                            count++;
                    } else {
                        meas[5].setValue(meas[5].getValue_true() * (1.0 + 0.3 * v));
                        filter.doFilter();
                        if (filter.isVjBad())
                            count++;
                    }
                }
                total++;
            }
        }
        double v = (double) count / (double) total;
        System.out.println("共检测" + total + "条支路,用时" + (System.currentTimeMillis() - start) + "ms,成功率：" + v);
    }

    public void getEstValue(BranchData branch, MeasureInfo[] measures, double[] x) {
        for (MeasureInfo info : measures) {
            int type = info.getMeasureType();
            double trueValue = 0.0;
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    if (info.getPositionId().equals(String.valueOf(branch.getTapBusNumber()))) {
                        trueValue = x[0];
                    } else {
                        trueValue = x[1];
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    trueValue = StateCalByPolar.calLinePFrom(branch, x);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    trueValue = StateCalByPolar.calLinePTo(branch, x);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    trueValue = StateCalByPolar.calLineQFrom(branch, x);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    trueValue = StateCalByPolar.calLineQTo(branch, x);
                    break;
                default:
                    break;
            }
            SimuMeasMaker.formMeasure(trueValue, 1, 0.0, info);
        }
    }
}