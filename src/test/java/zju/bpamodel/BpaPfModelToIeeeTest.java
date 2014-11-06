package zju.bpamodel;

import junit.framework.TestCase;
import zju.bpamodel.pf.ElectricIsland;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfTopoUtil;
import zju.pf.PfAlgorithmTest;

/**
 * BpaPfModelToIeee Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>07/25/2013</pre>
 */
public class BpaPfModelToIeeeTest extends TestCase {
    public BpaPfModelToIeeeTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testTransform() {
        testTransform("/bpafiles/anhui/sdxx201307081415.dat");
        testTransform("/bpafiles/anhui/ahxx201309251500.dat");
        IEEEDataIsland island = testTransform("/bpafiles/anhui/ahxx201312041630.dat");

        IcfTopoUtil topo = new IcfTopoUtil(island);
        topo.createGraph();
        assertTrue(topo.isIslandConnected());

        assertEquals(10045, island.getBuses().size());
        assertEquals(535, island.getPvBusSize());
        assertEquals(12368, topo.getGraph().edgeSet().size());
        PfAlgorithmTest.testPf_anhui(island);
    }

    public IEEEDataIsland testTransform(String resource) {
        ElectricIsland island = BpaPfModelParser.parse(this.getClass().getResourceAsStream(resource), "GBK");

        BpaPfModelToIeee trans = new BpaPfModelToIeee();
        IEEEDataIsland ieeeIsland = trans.createIeeeIsland(island);

        assertNotNull(ieeeIsland);
        assertEquals(island.getBuses().size(), ieeeIsland.getBuses().size());
        assertEquals(island.getAclines().size() + island.getTransformers().size(), ieeeIsland.getBranches().size());

        //int index1= resource.lastIndexOf("/");
        //int index2= resource.lastIndexOf(".");
        //String fileName = resource.substring(index1 + 1, index2);
        //IcfWriter ieeeCommonFormatWriter = new IcfWriter(ieeeIsland);
        //ieeeCommonFormatWriter.write(fileName + ".txt", "UTF-8");

        return ieeeIsland;
    }
}
