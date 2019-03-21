package zju.pf;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;

import java.util.List;
import java.util.Random;

/**
 * 使用蒙特卡洛方式模拟多断面
 */
public class MonteCarloCaseBuilder {

    private final static Logger logger = LogManager.getLogger(MonteCarloCaseBuilder.class);

    public static IEEEDataIsland[] simulatePowerFlow(final IEEEDataIsland oriIsland, final int num) {
        Random random = new Random();
        IEEEDataIsland[] islands = new IEEEDataIsland[num];
        for (int i = 0; i < num; ) {
            IEEEDataIsland newIsland = oriIsland.clone();
            List<BusData> busDataList = newIsland.getBuses();
            for (BusData busData : busDataList) {
                double loadMW = busData.getLoadMW();
                double loadMVAR = busData.getLoadMVAR();
                double generationMW = busData.getGenerationMW();
                double generationMVAR = busData.getGenerationMVAR();
                // 生成以该功率为中心，方差为10%的正态分布数据
                if (loadMW != 0)
                    busData.setLoadMW(random.nextGaussian() * loadMW * 0.1 + loadMW);
                if (loadMVAR != 0)
                    busData.setLoadMVAR(random.nextGaussian() * loadMVAR * 0.1 + loadMVAR);
                if (generationMW != 0)
                    busData.setGenerationMW(random.nextGaussian() * generationMW * 0.1 + generationMW);
                if (generationMVAR != 0)
                    busData.setGenerationMVAR(random.nextGaussian() * generationMVAR * 0.1 + generationMVAR);
            }
            PolarPf pf = new PolarPf();
            pf.setTol_p(1e-4);
            pf.setTol_q(1e-4);
            pf.setOriIsland(newIsland);
            pf.setDecoupledPqNum(0);
            //计算潮流
            pf.doPf();
            if (pf.isConverged) {
                islands[i++] = newIsland;
                pf.fillOriIslandPfResult();
            }

        }
        return islands;
    }
}