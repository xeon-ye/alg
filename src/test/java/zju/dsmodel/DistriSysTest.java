package zju.dsmodel;

import junit.framework.TestCase;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.alg.ConnectivityInspector;
import zju.devmodel.MapObject;
import zju.dspf.LcbPfModel;

/**
 * CalModeBridge Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>04/21/2011</pre>
 */
public class DistriSysTest extends TestCase implements DsModelCons {

    public DistriSysTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }


    public void testParse() {
        assertNotNull(IeeeDsInHand.FEEDER4_DGrY_B);
        IeeeDsInHand.FEEDER4_DGrY_B.buildDynamicTopo();
        assertEquals(1, IeeeDsInHand.FEEDER4_DGrY_B.getActiveIslands().length);
        assertTrue(IeeeDsInHand.FEEDER4_DGrY_B.getActiveIslands()[0].isRadical());

        assertNotNull(IeeeDsInHand.FEEDER13);
        IeeeDsInHand.FEEDER13.buildDynamicTopo();
        assertEquals(1, IeeeDsInHand.FEEDER13.getActiveIslands().length);
        assertTrue(IeeeDsInHand.FEEDER13.getActiveIslands()[0].isRadical());

        assertNotNull(IeeeDsInHand.FEEDER34);
        IeeeDsInHand.FEEDER34.buildDynamicTopo();
        assertEquals(1, IeeeDsInHand.FEEDER34.getActiveIslands().length);
        assertTrue(IeeeDsInHand.FEEDER34.getActiveIslands()[0].isRadical());

        assertNotNull(IeeeDsInHand.FEEDER37);
        IeeeDsInHand.FEEDER37.buildDynamicTopo();
        assertEquals(1, IeeeDsInHand.FEEDER37.getActiveIslands().length);
        assertTrue(IeeeDsInHand.FEEDER37.getActiveIslands()[0].isRadical());

        assertNotNull(IeeeDsInHand.FEEDER123);
        IeeeDsInHand.FEEDER123.buildDynamicTopo();
        assertEquals(1, IeeeDsInHand.FEEDER123.getActiveIslands().length);
        assertTrue(IeeeDsInHand.FEEDER123.getActiveIslands()[0].isRadical());
    }

    public void testLoop123() {
        DistriSys ds = IeeeDsInHand.FEEDER123.clone();
        for (MapObject obj : ds.getDevices().getSwitches())
            obj.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
        ds.buildDynamicTopo();
        DsTopoIsland[] activeIslands = ds.getActiveIslands();
        assertEquals(1, activeIslands.length);
        ds.createCalDevModel();

        DsTopoIsland island = activeIslands[0];
        assertFalse(island.isRadical());
    }

    public void testCase4_TDD() {
        DistriSys ds = IeeeDsInHand.FEEDER4_DD_B;
        for (MapObject obj : ds.getDevices().getSwitches())
            obj.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
        ds.buildDynamicTopo();

        DsTopoIsland[] activeIslands = ds.getActiveIslands();
        assertEquals(1, activeIslands.length);
        ds.createCalDevModel();

        DsTopoIsland island = activeIslands[0];

        assertEquals(3, island.getBranches().size());
        assertEquals(1, island.getLoads().size());
        assertEquals(1, ds.getDevices().getTransformers().size());
        GeneralBranch transformerBranch = island.getBranches().get(ds.getDevices().getTransformers().get(0));
        assertNotNull(transformerBranch);
        assertTrue(transformerBranch instanceof Transformer);

        //测试细化后图形的情况
        island.buildDetailedGraph();
        UndirectedGraph<String, DetailedEdge> g = island.getDetailedG();
        assertEquals(13, g.vertexSet().size());
        assertEquals(18, g.edgeSet().size());
        ConnectivityInspector<String, DetailedEdge> inspector = new ConnectivityInspector<String, DetailedEdge>(g);
        assertFalse(inspector.isGraphConnected());
        assertEquals(2, inspector.connectedSets().size());

        LcbPfModel model = new LcbPfModel(island, 50, 1e-4);
        assertEquals(7, model.getLoopSize());
    }

    public void testCase4_TGrYGrY() {
        DistriSys ds = IeeeDsInHand.FEEDER4_GrYGrY_UNB;
        for (MapObject obj : ds.getDevices().getSwitches())
            obj.setProperty(KEY_SWITCH_STATUS, SWITCH_ON);
        ds.buildDynamicTopo();

        DsTopoIsland[] activeIslands = ds.getActiveIslands();
        assertEquals(1, activeIslands.length);
        ds.createCalDevModel();

        DsTopoIsland island = activeIslands[0];

        assertEquals(3, island.getBranches().size());
        assertEquals(1, island.getLoads().size());
        assertEquals(1, ds.getDevices().getTransformers().size());
        GeneralBranch transformerBranch = island.getBranches().get(ds.getDevices().getTransformers().get(0));
        assertNotNull(transformerBranch);
        assertTrue(transformerBranch instanceof Transformer);

        //测试细化后图形的情况
        island.buildDetailedGraph();
        UndirectedGraph<String, DetailedEdge> g = island.getDetailedG();
        assertEquals(13, g.vertexSet().size());
        assertEquals(18, g.edgeSet().size());
        ConnectivityInspector<String, DetailedEdge> inspector = new ConnectivityInspector<String, DetailedEdge>(g);
        assertTrue(inspector.isGraphConnected());

        LcbPfModel model = new LcbPfModel(island, 50, 1e-4);
        assertEquals(6, model.getLoopSize());
        assertEquals(12, model.getDimension());
    }
}
