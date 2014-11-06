package zju.util;

import junit.framework.TestCase;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.matrix.AVector;

import java.util.List;
import java.util.Map;

/**
 * StateCalByRC Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>07/19/2010</pre>
 */
public class StateCalByRCTest extends TestCase {
    public StateCalByRCTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
    }

    public void testStandardCases() {
        doCal(IcfDataUtil.ISLAND_14.clone(), 1e-5);
        doCal(IcfDataUtil.ISLAND_30.clone(), 1e-5);
        doCal(IcfDataUtil.ISLAND_39.clone(), 1e-5);
        doCal(IcfDataUtil.ISLAND_57.clone(), 1e-5);
        doCal(IcfDataUtil.ISLAND_118.clone(), 1e-5);
        doCal(IcfDataUtil.ISLAND_300.clone(), 1e-5);
    }

    public static AVector getTrueUI(IEEEDataIsland island, YMatrixGetter y) {
        int n = island.getBuses().size();
        AVector vTheta = new AVector(n * 2);
        AVector ui = new AVector(n * 4);
        getTrueUI(island, y, vTheta, ui);
        return ui;
    }

    public static void getTrueUI(IEEEDataIsland island, YMatrixGetter y, AVector vTheta, AVector ui) {
        int n = island.getBuses().size();

        for (BusData bus : island.getBuses()) {
            double v = bus.getFinalVoltage();
            double theta = Math.PI * bus.getFinalAngle() / 180.0;
            vTheta.setValue(bus.getBusNumber() - 1, v);
            vTheta.setValue(bus.getBusNumber() - 1 + n, theta);
            ui.setValue(bus.getBusNumber() - 1, v * Math.cos(theta));
            ui.setValue(bus.getBusNumber() - 1 + n, v * Math.sin(theta));
        }
        Map<Integer, List<BranchData>> bus2Branches = IcfDataUtil.getBus2Branches(island);
        Map<Integer, BusData> busMap = island.getBusMap();
        for (Integer busNum : bus2Branches.keySet()) {
            //ieee 300 buses system has phase shifter, StateCalByPolar.calLineCurrent is not suitable for phase shifter
            //double Ix = 0;
            //double Iy = 0;
            //for (BranchData branch : bus2Branches.get(busNum)) {
            //    int fromOrTo;
            //    if (branch.getTapBusNumber() == busNum)
            //        fromOrTo = YMatrixGetter.LINE_FROM;
            //    else
            //        fromOrTo = YMatrixGetter.LINE_TO;
            //    double[] c = StateCalByPolar.calLineCurrent(branch.getId(), y, vTheta, fromOrTo);
            //    Ix += c[0] * Math.cos(c[1]);
            //    Iy += c[0] * Math.sin(c[1]);
            //}
            //
            //BusData bus = busMap.get(busNum);
            //double[] uiaiujaj = new double[]{vTheta.getValue(busNum - 1), vTheta.getValue(busNum - 1 + n), 0, 0};
            //double[] gbg1b1 = new double[]{bus.getShuntConductance(), bus.getShuntSusceptance(), 0, 0};
            //double[] c = StateCalByPolar.calLineCurrent(gbg1b1, uiaiujaj);
            //ui.setValue(busNum - 1 + 2 * n, Ix + c[0] * Math.cos(c[1]));
            //ui.setValue(busNum - 1 + 3 * n, Iy + c[0] * Math.sin(c[1]));
            //System.out.println(" ============== " + busNum + "\t" + ui.getValue(busNum - 1 + 2 * n) + "\t" + ui.getValue(busNum - 1 + 3 * n));
        }
        double[] u = new double[2 * n];
        System.arraycopy(ui.getValues(), 0, u, 0, u.length);
        double[] c = new double[2 * n];
        StateCalByRC.calI(y, u, c);
        System.arraycopy(c, 0, ui.getValues(), u.length, c.length);
    }

    synchronized public void doCal(IEEEDataIsland island, double tolerance) {
        int n = island.getBuses().size();
        AVector vTheta = new AVector(n * 2);
        AVector ui = new AVector(n * 4);
        NumberOptHelper numOpt = new NumberOptHelper();
        numOpt.simple(island);
        numOpt.trans(island);

        YMatrixGetter y = new YMatrixGetter(island);
        y.formYMatrix();
        getTrueUI(island, y, vTheta, ui);

        System.out.println("Calculation result:\tBus Number\tBus P\tBus Q");
        for (BusData bus : island.getBuses()) {
            double p = StateCalByPolar.calBusP(bus.getBusNumber(), y, vTheta);
            double q = StateCalByPolar.calBusQ(bus.getBusNumber(), y, vTheta);

            double p2 = StateCalByRC.calBusP_UI(bus.getBusNumber(), n, ui);
            double q2 = StateCalByRC.calBusQ_UI(bus.getBusNumber(), n, ui);

            double p3 = StateCalByRC.calBusP_U(bus.getBusNumber(), y, ui);
            double q3 = StateCalByRC.calBusQ_U(bus.getBusNumber(), y, ui);
            Integer busNumber = numOpt.getNew2old().get(bus.getBusNumber());
            System.out.println("cal by v theta:\t" + busNumber + "\t" + p + "\t" + q);
            System.out.println("cal by u i:\t" + busNumber + "\t" + p2 + "\t" + q2);
            System.out.println("cal by u :\t" + busNumber + "\t" + p3 + "\t" + q3);
            System.out.println("--------------------------------------------");
            assertTrue(Math.abs(p - p2) < tolerance);
            assertTrue(Math.abs(q - q2) < tolerance);
            assertTrue(Math.abs(p - p3) < tolerance);
            assertTrue(Math.abs(q - q3) < tolerance);
        }

        for (BranchData branch : island.getBranches()) {
            double p1 = StateCalByPolar.calLinePFrom(branch.getId(), y, vTheta);
            double q1 = StateCalByPolar.calLineQFrom(branch.getId(), y, vTheta);
            double p2 = StateCalByPolar.calLinePTo(branch.getId(), y, vTheta);
            double q2 = StateCalByPolar.calLineQTo(branch.getId(), y, vTheta);

            double p3 = StateCalByRC.calLinePFrom(branch.getId(), y, ui);
            double p5 = StateCalByRC.calLinePFrom2(branch.getId(), y, ui);
            double q3 = StateCalByRC.calLineQFrom(branch.getId(), y, ui);
            double q5 = StateCalByRC.calLineQFrom2(branch.getId(), y, ui);
            double p4 = StateCalByRC.calLinePTo(branch.getId(), y, ui);
            double p6 = StateCalByRC.calLinePTo2(branch.getId(), y, ui);
            double q4 = StateCalByRC.calLineQTo(branch.getId(), y, ui);
            double q6 = StateCalByRC.calLineQTo2(branch.getId(), y, ui);
            assertTrue(Math.abs(p3 - p1) < tolerance);
            assertTrue(Math.abs(p5 - p1) < tolerance);
            assertTrue(Math.abs(q3 - q1) < tolerance);
            assertTrue(Math.abs(q5 - q1) < tolerance);
            assertTrue(Math.abs(p4 - p2) < tolerance);
            assertTrue(Math.abs(p6 - p2) < tolerance);
            assertTrue(Math.abs(q4 - q2) < tolerance);
            assertTrue(Math.abs(q6 - q2) < tolerance);
        }
    }
}
