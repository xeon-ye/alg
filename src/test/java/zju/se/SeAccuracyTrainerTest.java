package zju.se;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.Before;
import org.junit.Test;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.measure.MeasVector;
import zju.measure.SystemMeasure;
import zju.pf.MonteCarloCaseBuilder;
import zju.pf.SimuMeasMaker;

import java.util.List;

public class SeAccuracyTrainerTest {

    private final static Logger logger = LogManager.getLogger(SeAccuracyTrainerTest.class);

    private SeAccuracyTrainer trainer;

    @Before
    public void setUp() throws Exception {
        trainer = new SeAccuracyTrainer(new SeAccuracyTrainMlpModel());
    }

    @Test
    public void testCase30() {
        trainer.trainModel(IcfDataUtil.ISLAND_30, 1000);
        doSeAccuracyPredict(IcfDataUtil.ISLAND_30, 20);
    }

    @Test
    public void testCase30ByFile() {
        trainer.trainModel(this.getClass().getResourceAsStream("/sefiles/train/case30.txt"));
        List<Double> res = trainer.predict(this.getClass().getResourceAsStream("/sefiles/test/case30.txt"));
    }

    private void doSeAccuracyPredict(final IEEEDataIsland oriIsland, int num) {
        IEEEDataIsland[] islands = MonteCarloCaseBuilder.simulatePowerFlow(oriIsland, num);

        StateEstimator se = new StateEstimator();
        IpoptSeAlg alg = new IpoptSeAlg();
        se.setAlg(alg);
        se.setFlatStart(true); // 平启动

        for (int i = 0; i < islands.length; i++) {
            IEEEDataIsland ref = islands[i], island = ref.clone();
            int n = island.getBuses().size();
            double[] trueVar = new double[2 * n];
            for (int j = 0; j < n; j++) {
                BusData bus = island.getBuses().get(j);
                trueVar[bus.getBusNumber() - 1] = bus.getFinalVoltage();
                trueVar[bus.getBusNumber() - 1 + n] = bus.getFinalAngle() * Math.PI / 180.0;
            }

            SystemMeasure sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05); // 加入5%的高斯噪声
            double slackBusAngle = island.getBus(island.getSlackBusNum()).getFinalAngle() * Math.PI / 180.;
            alg.setSlackBusAngle(slackBusAngle);

            se.setOriIsland(island); // 初始化电气岛
            se.setSm(sm);

            alg.getObjFunc().setObjType(SeObjective.OBJ_TYPE_WLS); // 设置目标函数

            //传统最小二乘法
            //SeAccuracyTrainer.dealZeroInjection(se, alg, sm, ref, false);
            alg.setVariable_type(IpoptSeAlg.VARIABLE_VTHETA);
            se.doSe();

            if (alg.isConverged) {
                MeasVector meas = alg.getMeas();
                double[] attribute = meas.z.getValues();
                double[] estVar = alg.getVariableState();
                double label = 0;
                for (int j = 0; j < trueVar.length; j++) {
                    label += Math.pow(estVar[j] - trueVar[j], 2);
                }
                label = Math.sqrt(label);
                double ans = trainer.predict(attribute);
                logger.info("当前预测偏差为：" + Math.abs(ans - label));
            }
        }
    }
}