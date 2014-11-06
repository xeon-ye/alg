package zju.ieeeformat;

import junit.framework.TestCase;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: Su Xinyi
 * Date: 2007-12-10
 * Time: 13:30:36
 */
public class Make4BusIsland extends TestCase {

    protected void setUp() throws Exception {
    }

    public static IEEEDataIsland testBuildCase4() {

        IEEEDataIsland island4bus = new IEEEDataIsland();
        TitleData title = new TitleData();
        title.setMvaBase(100);
        title.setDate("07/12/10");
        island4bus.setTitle(title);

        List<BusData> buses = new ArrayList<BusData>();
        for (int i = 0; i < 4; i++) {
            BusData bus = new BusData();
            bus.setBusNumber(i + 1);
            bus.setName("Node" + String.valueOf(i + 1));
            bus.setShuntSusceptance(0);
            bus.setShuntConductance(0);
            bus.setBaseVoltage(100);
            bus.setDesiredVolt(1);
            bus.setArea(1);
            bus.setType(1);
            bus.setFinalAngle(0);
            bus.setFinalVoltage(1);
            bus.setGenerationMVAR(0);
            bus.setGenerationMW(0);
            bus.setLoadMVAR(0);
            bus.setLoadMW(0);
            bus.setLossZone(1);
            bus.setRemoteControlBusNumber(0);
            buses.add(bus);
        }
        buses.get(0).setType(3);
        buses.get(0).setGenerationMW(20.86);
        buses.get(0).setGenerationMVAR(35.79);
        buses.get(1).setLoadMW(170);
        buses.get(1).setLoadMVAR(90);
        buses.get(2).setLoadMW(50);
        buses.get(2).setLoadMVAR(40);
        buses.get(3).setType(2);
        buses.get(3).setGenerationMW(200);
        buses.get(3).setGenerationMVAR(100);

        List<BranchData> branches = new ArrayList<BranchData>();
        for (int i = 0; i < 4; i++) {
            BranchData branch = new BranchData();
            branch.setArea(1);
            branch.setCircuit(1);
            branch.setControlBusNumber(0);
            branch.setLossZone(1);
            branches.add(branch);
        }
        branches.get(0).setType(0);
        branches.get(0).setTapBusNumber(1);
        branches.get(0).setZBusNumber(2);
        branches.get(0).setBranchR(0.0052);
        branches.get(0).setBranchX(0.0266);
        branches.get(0).setLineB(0.028);
        branches.get(1).setType(0);
        branches.get(1).setTapBusNumber(1);
        branches.get(1).setZBusNumber(3);
        branches.get(1).setBranchR(0.0305);
        branches.get(1).setBranchX(0.0837);
        branches.get(1).setLineB(0.112);
        branches.get(2).setType(0);
        branches.get(2).setTapBusNumber(2);
        branches.get(2).setZBusNumber(3);
        branches.get(2).setBranchR(0.0041);
        branches.get(2).setBranchX(0.0215);
        branches.get(2).setLineB(0.088);
        branches.get(3).setType(1);
        branches.get(3).setTapBusNumber(3);
        branches.get(3).setZBusNumber(4);
        branches.get(3).setBranchR(0.0000);
        branches.get(3).setBranchX(0.0750);
        branches.get(3).setLineB(0.000);
        branches.get(3).setTransformerRatio(1.05);

        island4bus.setBuses(buses);
        island4bus.setBranches(branches);
        island4bus.setInterchanges(new ArrayList<InterchangeData>());
        island4bus.setLossZones(new ArrayList<LossZoneData>());
        island4bus.setTieLines(new ArrayList<TieLineData>());

        assertNotNull(island4bus);
        return island4bus;
    }
}

