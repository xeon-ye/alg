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
    public void testHybridPso() {
        new HybridPso(new DifficultModel_1(), 17).execute();
    }

    @Test
    public void testParallelPso() {
        new ParallelPso(new DifficultModel_1()).execute();
    }

}
