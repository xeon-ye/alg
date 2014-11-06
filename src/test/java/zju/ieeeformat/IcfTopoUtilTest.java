package zju.ieeeformat;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import org.jgrapht.graph.DefaultEdge;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 * IcfTopoUtil Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>12/02/2013</pre>
 */
public class IcfTopoUtilTest extends TestCase {
    public IcfTopoUtilTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public static Test suite() {
        return new TestSuite(IcfTopoUtilTest.class);
    }

    public void testTopo() {
        //假设island是n-1之后的网络
        IEEEDataIsland island = IcfDataUtil.ISLAND_14.clone();
        IcfTopoUtil icfTopo = new IcfTopoUtil(island);
        icfTopo.createGraph();
        //首先判断是否联通
        assertEquals(true, icfTopo.isIslandConnected());

        //除去2-3, 3-4两条支路
        //如果检查单一设备故障，可以使用方法icfTopo.getSubIslands(BranchData branch)
        List<Set<BusData>> subIslands = icfTopo.getSubIslands(new BranchData[]{
                island.getBranches().get(2),
                island.getBranches().get(5),
        });
        //如果得到两个岛
        assertEquals(2, subIslands.size());

        //得到节点数少的岛
        Set<BusData> smallIsland;
        if (subIslands.get(0).size() < subIslands.get(1).size()) {
            smallIsland = subIslands.get(0);
            assertEquals(1, subIslands.get(0).size());
            assertEquals(13, subIslands.get(1).size());
        } else {
            smallIsland = subIslands.get(1);
            assertEquals(13, subIslands.get(0).size());
            assertEquals(1, subIslands.get(1).size());
        }

        //检查较小的电气岛，并检查故障严重情况
        int loss110BusCount = 0;
        int loss220BusCount = 0;
        int loss500BusCount = 0;
        for (BusData bus : smallIsland) {
            if (bus.getBaseVoltage() < 600 && bus.getBaseVoltage() > 500) {
                loss500BusCount++;
            } else if (bus.getBaseVoltage() < 300 && bus.getBaseVoltage() > 200) {
                loss220BusCount++;
            } else if (bus.getBaseVoltage() < 150 && bus.getBaseVoltage() > 100) {
                loss110BusCount++;
            }
        }
        assertEquals(0, loss220BusCount);
        assertEquals(0, loss500BusCount);
        if (loss220BusCount > 0 && smallIsland.size() > 1) {
            //导致220千伏及以上局部电网解列
        }
        if (loss220BusCount > 3) {
            //导致3个以上220千伏厂站全停
        }
        if (loss500BusCount > 0) {
            //导致500千伏厂站全停
        }
        if (loss110BusCount > 0) {
            //导致110千伏及以下局部电网解列
        }
    }

    public void testCase_cq() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/ieee-cq-2014-03.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");

        IcfTopoUtil icfTopo = new IcfTopoUtil(island);
        icfTopo.createGraph();

        List<Set<BusData>> subgraphes;
        Set<BusData> equivalentLoad = null;
        Set<DefaultEdge> edges;
        BusData b1, b2, bus1, bus2, tmpBus;
        DefaultEdge e;
        List<BranchData> toBreakBranches = new ArrayList<BranchData>();
        for (BranchData b : island.getBranches()) {
            if (b.getType() == BranchData.BRANCH_TYPE_TF_FIXED_TAP) {
                e = icfTopo.getBranchToEdge().get(b);
                b1 = icfTopo.getGraph().getEdgeSource(e);
                b2 = icfTopo.getGraph().getEdgeTarget(e);
                if (b1.getBaseVoltage() < b2.getBaseVoltage()) {
                    tmpBus = b1;
                    b1 = b2;
                    b2 = tmpBus;
                }
                if (b1.getBaseVoltage() < 200 || b1.getBaseVoltage() > 250)
                    continue;
                toBreakBranches.add(b);
                //subgraphes = icfTopo.getSubIslands(b);
                //assertEquals(2, subgraphes.size());


                ////如果是和变压器中心点之间
                //if (b2.getBaseVoltage() < 1e-2) {
                //    edges = icfTopo.getGraph().edgesOf(b2);
                //    boolean isUnder220 = false;
                //    for (DefaultEdge edge : edges) {
                //        if (edge == e)
                //            continue;
                //        bus1 = icfTopo.getGraph().getEdgeSource(edge);
                //        bus2 = icfTopo.getGraph().getEdgeTarget(edge);
                //        if (bus1 == b2 && bus2.getBaseVoltage() < b1.getBaseVoltage()) {
                //            isUnder220 = true;
                //            //subgraphes = icfTopo.getSubIslands(b);
                //            //assertEquals(2, subgraphes.size());
                //        }
                //        if (bus2 == b2 && bus1.getBaseVoltage() < b1.getBaseVoltage())
                //            isUnder220 = true;
                //
                //    }
                //    assertTrue(isUnder220);
                //    continue;
                //}
                //
                //subgraphes = icfTopo.getSubIslands(b);
                ////判断网络是否解裂
                //assertEquals(2, subgraphes.size());
                //
                ////判断子图中哪些是主网，哪些是等效负荷
                //for (BusData bus : subgraphes.get(0)) {
                //    if (bus == b1) {
                //        equivalentLoad = subgraphes.get(1);
                //        break;
                //    } else if (bus == b2) {
                //        equivalentLoad = subgraphes.get(0);
                //        break;
                //    }
                //}
                //for (BusData bus : equivalentLoad)
                //    assertTrue(bus.getBaseVoltage() < b1.getBaseVoltage());
            }
        }
        //subgraphes =  icfTopo.getSubIslands(toBreakBranches.toArray(new BranchData[]{}));
        //assertEquals(2, subgraphes.size());
        //icfTopo.doEquivalentLoad(220);
    }
}
