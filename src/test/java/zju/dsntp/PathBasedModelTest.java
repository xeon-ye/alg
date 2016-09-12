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
        assertEquals(10, model.getPathes().size());//
    }
}
