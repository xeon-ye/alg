package zju.dsntp;

import junit.framework.TestCase;
import zju.dsmodel.DsModelCons;
import zju.dsmodel.IeeeDsInHand;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2016/9/12
 */
public class PathBasedModelTest extends TestCase implements DsModelCons {

    public PathBasedModelTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testDscase13() {
        PathBasedModel model = new PathBasedModel(IeeeDsInHand.FEEDER13);
        model.buildPathes();
        assertEquals(12, model.getPathes().size());
    }

    public void testDscase34() {
        PathBasedModel model = new PathBasedModel(IeeeDsInHand.FEEDER34);
        model.buildPathes();
        assertEquals(33, model.getPathes().size());
    }

    public void testDscase37() {
        PathBasedModel model = new PathBasedModel(IeeeDsInHand.FEEDER37);
        model.buildPathes();
        assertEquals(36, model.getPathes().size());
    }

    public void testDscase4() {
        PathBasedModel model = new PathBasedModel(IeeeDsInHand.FEEDER4_DD_B);
        model.buildPathes();
        assertEquals(3, model.getPathes().size());
        model = new PathBasedModel(IeeeDsInHand.FEEDER4_DD_UNB);
        model.buildPathes();
        assertEquals(3, model.getPathes().size());
        model = new PathBasedModel(IeeeDsInHand.FEEDER4_DGrY_B);
        model.buildPathes();
        assertEquals(3, model.getPathes().size());
        model = new PathBasedModel(IeeeDsInHand.FEEDER4_DGrY_UNB);
        model.buildPathes();
        assertEquals(3, model.getPathes().size());
        model = new PathBasedModel(IeeeDsInHand.FEEDER4_GrYGrY_B);
        model.buildPathes();
        assertEquals(3, model.getPathes().size());
        model = new PathBasedModel(IeeeDsInHand.FEEDER4_GrYGrY_UNB);
        model.buildPathes();
        assertEquals(3, model.getPathes().size());
    }

    public void testDscase123() {
        PathBasedModel model = new PathBasedModel(IeeeDsInHand.FEEDER123);
        model.buildPathes();
        assertEquals(193, model.getPathes().size());
    }
}
