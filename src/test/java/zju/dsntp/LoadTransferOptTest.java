package zju.dsntp;

import junit.framework.TestCase;
import zju.dsmodel.DsModelCons;
import zju.dsmodel.IeeeDsInHand;

/**
 * Created by xcs on 2016/10/5.
 */
public class LoadTransferOptTest extends TestCase implements DsModelCons {

    public LoadTransferOptTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testcase() {
        LoadTransferOpt model = new LoadTransferOpt(IeeeDsInHand.FEEDER13);
        model.doOpt();

    }
}