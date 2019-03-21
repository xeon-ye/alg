package zju.pf;

import junit.framework.TestCase;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.ieeeformat.*;
import zju.matrix.AVector;
import zju.util.NumberOptHelper;
import zju.util.StateCalByPolar;
import zju.util.YMatrixGetter;

import java.io.InputStream;

/**
 * Created by IntelliJ IDEA.
 *
 * @author zhangxiao
 *         Date: 2010-5-26
 */

public class SystemStatTest extends TestCase {

    public SystemStatTest(String name) {
        super(name);
    }


    public void testStandardCases() {
        systemStatistic(IcfDataUtil.ISLAND_14.clone(), true);
        systemStatistic(IcfDataUtil.ISLAND_30.clone(), true);
        systemStatistic(IcfDataUtil.ISLAND_39.clone(), true);
        systemStatistic(IcfDataUtil.ISLAND_57.clone(), true);
        systemStatistic(IcfDataUtil.ISLAND_118.clone(), true);
        systemStatistic(IcfDataUtil.ISLAND_300.clone(), true);
    }

    public void ieeefileFinalVTheta(String ieeeFile) {
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile);
        ieeefileFinalVTheta(island);
    }

    public void ieeefileFinalVTheta(IEEEDataIsland island) {
        NumberOptHelper numOpt = new NumberOptHelper();
        numOpt.simple(island);
        YMatrixGetter Y = new YMatrixGetter(island);
        Y.formYMatrix();

        int N = island.getBuses().size();
        AVector ieeefileFinal = new AVector(2 * N);
        for (BusData bus : island.getBuses()) {
            int busNum = bus.getBusNumber();
            ieeefileFinal.setValue(busNum - 1, bus.getFinalVoltage());
            ieeefileFinal.setValue(busNum - 1 + N, bus.getFinalAngle() * Math.PI / 180);   //ieeefile : degrees
        }

        //all MVar
        double errMvar = 1;
        int crossNum = 0;
        System.out.println("PV Bus Cross Q Record. MVarErrGate = " + errMvar + "MVar");
        System.out.println("busNum, busName  - [ deltaQ, measQ, minQ, maxQ ]");
        for (BusData bus : island.getBuses()) {
            if (bus.getType() == BusData.BUS_TYPE_GEN_PV) {  //PV nodes
                int busNo = bus.getBusNumber();
                double measQ = StateCalByPolar.calBusQ(busNo, Y, ieeefileFinal);
                measQ *= island.getTitle().getMvaBase();
                double maxQ = bus.getMaximum();
                double minQ = bus.getMinimum();
                if (measQ > maxQ + errMvar) {     //at last, measQ near maxQ ,
                    double deltaQ = maxQ - measQ;
                    System.out.println(busNo + ", " + island.getBus(busNo).getName() + " - [" + deltaQ
                            + "," + measQ + ", " + minQ + ", " + maxQ + " ]");
                    crossNum++;
                } else if (measQ < minQ - errMvar) {
                    double deltaQ = minQ - measQ;
                    System.out.println(busNo + ", " + island.getBus(busNo).getName() + " - [" + deltaQ
                            + "," + measQ + ", " + minQ + ", " + maxQ + " ]");
                    crossNum++;
                }
            }
        }
        System.out.println("Total Cross Number = " + crossNum);
    }

    public void testAnhui() {
        String resource = "/ieeefiles/ahxx201309251500.txt";
        InputStream ieeeFile = this.getClass().getResourceAsStream(resource);
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");
        systemStatistic(island, false);

        resource = "/ieeefiles/ahxx201312041630.txt";
        ieeeFile = this.getClass().getResourceAsStream(resource);
        island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");
        systemStatistic(island, false);
    }

    public void systemStatistic(IEEEDataIsland island, boolean isPfResult) {
        System.out.println("Analyze system statistic ");
        System.out.println("island buses number = " + island.getBuses().size());
        System.out.println("island branches number = " + island.getBranches().size());

        double totalGenP = 0;
        double totalLoadP = 0;
        double totalGenQ = 0;
        double totalLoadQ = 0;
        double totalCalP = 0;

        if (isPfResult) {
            NumberOptHelper numOpt = new NumberOptHelper();
            numOpt.simple2(island);
            numOpt.trans(island);
            int busNumber = island.getBuses().size();
            double[] pfSolution = new double[2 * busNumber];
            for (int i = 0; i < busNumber; i++) {
                pfSolution[i] = island.getBus(i + 1).getFinalVoltage();
                pfSolution[i + busNumber] = island.getBus(i + 1).getFinalAngle() * Math.PI / 180;
            }
            YMatrixGetter Y = new YMatrixGetter(island);
            Y.formYMatrix();
            //AVector pfSolutionVector = new AVector(pfSolution);
            for (BusData bus : island.getBuses()) {
                double calP = StateCalByPolar.calBusP(bus.getBusNumber(), Y, pfSolution);
                //double calQ = StateCalByPolar.calBusQ(bus.getBusNumber(), Y, pfSolution);
                calP *= island.getTitle().getMvaBase();
                //calQ *= island.getTitle().getMvaBase();
                //System.out.println(String.format("%4d  %4d  %8.4f   %8.4f   %8.4f   %8.4f ", bus.getBusNumber(), bus.getType(), bus.getGenerationMW(), bus.getLoadMW(), calP, calQ));
                totalCalP += calP;
                totalGenP += bus.getGenerationMW();
                totalLoadP += bus.getLoadMW();
            }
        }
        for (BusData bus : island.getBuses()) {
            totalGenP += bus.getGenerationMW();
            totalLoadP += bus.getLoadMW();
            totalGenQ += bus.getGenerationMVAR();
            totalLoadQ += bus.getLoadMVAR();
            if(bus.getType() == BusData.BUS_TYPE_GEN_PV && bus.getFinalVoltage() < 0.8)
                System.out.println("Too samall voltage:" + bus.getBusNumber() + "\t" + bus.getName() + "\t" + bus.getFinalVoltage());
            if(bus.getType() == BusData.BUS_TYPE_GEN_PV && bus.getFinalVoltage() > 1.2)
                System.out.println("Too big voltage:" + bus.getBusNumber() + "\t" + bus.getName() + "\t" + bus.getFinalVoltage());
            if(bus.getType() == BusData.BUS_TYPE_GEN_PV && bus.getGenerationMW() < 0)
                System.out.println("Negative gen mw output:" + bus.getBusNumber() + "\t" + bus.getName() + "\t" + bus.getGenerationMW());
        }
        for(BranchData b : island.getBranches()) {
            //if(b.getBranchX() < b.getBranchR()) {
            //    System.out.println("Branch X is smaller than branch R.");
            //    System.out.println(b);
            //}
            if(Math.abs(b.getBranchX()) < 1e-4) {
                System.out.println("Too small branch x: " + b.getBranchX());
                System.out.println(b);
            }
        }
        if (isPfResult) {
            System.out.println("total cal P = " + totalCalP + " MW");
        }
        System.out.println("total Gen = " + totalGenP + " MW");
        System.out.println("total Load = " + totalLoadP + " MW");
        System.out.println("power loss = " + (totalGenP - totalLoadP) + " MW");
        System.out.println("total Gen = " + totalGenQ + " Mvar");
        System.out.println("total Load = " + totalLoadQ + " Mvar");
        System.out.println("-------------------------");
        System.out.println("");
    }
}
