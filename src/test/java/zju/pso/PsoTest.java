package zju.pso;

import org.junit.Test;
import zju.pso.model.*;

/**
 * @Author: Fang Rui
 * @Date: 2018/6/20
 * @Time: 21:13
 */
public class PsoTest {

    @Test
    public void testSimpleModel() {
        new PsoProcess(new SimpleModel_1(), 50).execute();
        new PsoProcess(new SimpleModel_2(), 50).execute();
    }

    @Test
    public void testDifficultModel() {
        new PsoProcess(new DifficultModel_1(), 1000).execute();
        new PsoProcess(new DifficultModel_2(), 1000).execute();
        new PsoProcess(new DifficultModel_3(), 30).execute();
    }

    @Test
    public void testHybridPso() {
        new HybridPso(new SimpleModel_1(), 50).execute();
        new HybridPso(new SimpleModel_2(), 50).execute();
        new HybridPso(new DifficultModel_1(), 40).execute();
    }

}
