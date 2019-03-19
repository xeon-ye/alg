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
        new HybridPSO(new DifficultModel_1(), 17).execute();
    }

    @Test
    public void testParallelPso() {
        new HybridPSO(new DifficultModel_1()).execute();
    }

}
