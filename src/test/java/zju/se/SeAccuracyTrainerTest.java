package zju.se;

import org.junit.Before;
import org.junit.Test;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;

import static org.junit.Assert.*;

public class SeAccuracyTrainerTest {

    SeAccuracyTrainer trainer;

    @Before
    public void setUp() throws Exception {
        trainer = new SeAccuracyTrainer(null);
    }

    @Test
    public void trainModel() {
        trainer.trainModel(IcfDataUtil.ISLAND_30, 1000);
    }
}