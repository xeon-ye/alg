package zju.pf;

import junit.framework.TestCase;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.measure.SystemMeasure;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-12
 */
public class MeasureMakerTest extends TestCase {

    protected void setUp() throws Exception {
    }

    public void testCreateFullMeasure(IEEEDataIsland island, String targetFile) {
        //SystemMeasure sm = SimuMeasMaker.createFullMeasure(island, 1);
        SystemMeasure sm = SimuMeasMaker.createFullMeasure_withBadData(island, 1, 0.05, 0.02);
        assertNotNull(sm);
        //DefaultMeasWriter.writeFullyWithTrueValue(sm, targetFile);
    }

    public void testCreateWithBad(IEEEDataIsland island, String targetFile) {
        for (int i = 0; i <= 5; i++) {
            SystemMeasure sm = SimuMeasMaker.createFullMeasure_withBadData(island, 1, 0.018 * i, 0.02);
            assertNotNull(sm);
            //DefaultMeasWriter.writeFullyWithTrueValue(sm, targetFile + "" + i + ".txt");
        }
    }

    public void test14() {
        testCreateFullMeasure(IcfDataUtil.ISLAND_14.clone(), "case14_meas.txt");
        testCreateWithBad(IcfDataUtil.ISLAND_14.clone(), "case14_withBad");
    }

    public void test30() {
        testCreateFullMeasure(IcfDataUtil.ISLAND_30.clone(), "case30_meas.txt");
        testCreateWithBad(IcfDataUtil.ISLAND_30.clone(), "case30_withBad");
    }

    public void test39() {
        testCreateFullMeasure(IcfDataUtil.ISLAND_39.clone(), "case39_meas.txt");
        testCreateWithBad(IcfDataUtil.ISLAND_39.clone(), "case39_withBad");
    }

    public void test57() {
        testCreateFullMeasure(IcfDataUtil.ISLAND_57.clone(), "case57_meas.txt");
        testCreateWithBad(IcfDataUtil.ISLAND_57.clone(), "case57_withBad");
    }

    public void test118() {
        testCreateFullMeasure(IcfDataUtil.ISLAND_118.clone(), "case118_meas.txt");
        testCreateWithBad(IcfDataUtil.ISLAND_118.clone(), "case118_withBad");
    }

    public void test300() {
        testCreateFullMeasure(IcfDataUtil.ISLAND_300.clone(), "case300_meas.txt");
        testCreateWithBad(IcfDataUtil.ISLAND_300.clone(), "case300_withBad");
    }

    //public void testAll() throws IOException {
    //    test14();
    //    test30();
    //    test39();
    //    test57();
    //    test118();
    //    test300();
    //}
}
