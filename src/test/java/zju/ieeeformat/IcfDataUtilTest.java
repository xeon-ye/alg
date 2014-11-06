package zju.ieeeformat;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;

import java.util.ArrayList;
import java.util.List;

/**
 * IcfDataUtil Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>03/28/2007</pre>
 */
public class IcfDataUtilTest extends TestCase {
    public IcfDataUtilTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public static Test suite() {
        return new TestSuite(IcfDataUtilTest.class);
    }

    public void testSort() {
        IEEEDataIsland island = new DefaultIcfParser().parse(this.getClass().getResource("/ieeefiles/ieeeData.dat").getPath());
        island = IcfDataUtil.sortByBusData(island, new String[]{BusData.VAR_TYPE, BusData.VAR_PGEN, BusData.VAR_PLOAD});
        island = IcfDataUtil.sortByBrandData(island, new String[]{BranchData.VAR_TYPE, BranchData.VAR_R, BranchData.VAR_X});
        System.out.println(new IcfWriter(island).toString());

        //the following lines make sure the sort action is correctly done
        BusData preBus = null;
        for (BusData bus : island.getBuses()) {
            //System.out.println(bus.getType() + "\t" + bus.getGenerationMW() + "\t" + bus.getLoadMW());
            if (preBus != null) {
                assertTrue(bus.getType() >= preBus.getType());
                if (bus.getType() == preBus.getType())
                    assertTrue(bus.getGenerationMW() >= preBus.getGenerationMW());
                if (bus.getType() == preBus.getType() && bus.getGenerationMW() == preBus.getGenerationMW())
                    assertTrue(bus.getLoadMW() >= preBus.getLoadMW());
            }
            preBus = bus;
        }

        BranchData preBranch = null;
        for (BranchData branch : island.getBranches()) {
            //System.out.println(branch.getType() + "\t" + branch.getBranchR() + "\t" + branch.getBranchX());
            if (preBranch != null) {
                assertTrue(branch.getType() >= preBranch.getType());
                if (branch.getType() == preBranch.getType())
                    assertTrue(branch.getBranchR() >= preBranch.getBranchR());
                if (branch.getType() == preBranch.getType() && branch.getBranchR() == preBranch.getBranchR())
                    assertTrue(branch.getBranchX() >= preBranch.getBranchX());
            }
            preBranch = branch;
        }
    }

    public void testSetSubareas() {
        IEEEDataIsland island = IcfDataUtil.ISLAND_14.clone();
        List<BranchData> ties = new ArrayList<BranchData>();
        for (BranchData branch : island.getBranches()) {
            if (branch.getTapBusNumber() == 5 && branch.getZBusNumber() == 6)
                ties.add(branch);
            if (branch.getTapBusNumber() == 4 && branch.getZBusNumber() == 9)
                ties.add(branch);
            if (branch.getTapBusNumber() == 4 && branch.getZBusNumber() == 7)
                ties.add(branch);
        }
        SubareaModel model = IcfDataUtil.buildSubarea(island, ties);
        assertNotNull(model);
        assertEquals(2, model.getIslands().size());
    }

    public void testSplitTwoLevels() {
        IEEEDataIsland island = new DefaultIcfParser().parse(this.getClass().
                getResourceAsStream("/ieeefiles/20101016_0050_island.txt"), "GBK");
        SubareaModel model = IcfDataUtil.splitTowLayers(island,
                KVLevelPicker.KV_500, KVLevelPicker.KV_220);
        assertNotNull(model);
        assertEquals(7, model.getIslands().size());
    }
}
