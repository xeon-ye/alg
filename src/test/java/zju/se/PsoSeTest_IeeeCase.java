package zju.se;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.junit.*;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.measure.MeasTypeCons;
import zju.measure.MeasureInfo;
import zju.measure.SystemMeasure;
import zju.pf.SimuMeasMaker;
import zju.util.StateCalByPolar;

import java.util.Iterator;
import java.util.Map;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertNotNull;
import static junit.framework.TestCase.assertTrue;


/**
 * @Author: Fang Rui
 * @Date: 2018/7/20
 * @Time: 15:03
 */
public class PsoSeTest_IeeeCase implements MeasTypeCons {

    private static Logger log = LogManager.getLogger(PsoSeTest_IeeeCase.class);

    private StateEstimator se;
    private IpoptSeAlg ipoptSeAlg;
    private PsoSeAlg psoSeAlg;

    @Before
    public void setUp() throws Exception {
        se = new StateEstimator();
        ipoptSeAlg = new IpoptSeAlg();
        psoSeAlg = new PsoSeAlg();
    }

    /**
     * 所有量测均为真值
     */
    @Test
    public void testCaseTrue() {
        ipoptSeAlg.setTol_p(1e-5);
        ipoptSeAlg.setTol_q(1e-5);
        psoSeAlg.setTol_p(1e-5);
        psoSeAlg.setTol_q(1e-5);

        IEEEDataIsland island;
        SystemMeasure sm;

        island = IcfDataUtil.ISLAND_14.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_14, true, false);

        island = IcfDataUtil.ISLAND_30.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_30, true, false);

        island = IcfDataUtil.ISLAND_39.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_39, true, false);

        island = IcfDataUtil.ISLAND_57.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_57, true, false);

        island = IcfDataUtil.ISLAND_118.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_118, true, false);

        island = IcfDataUtil.ISLAND_300.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_300, true, false);
    }

    /**
     * 给所有量测加入2%的高斯噪声
     */
    @Test
    public void testCaseGauss() {
        ipoptSeAlg.setTol_p(0.005);
        ipoptSeAlg.setTol_q(0.005);
        psoSeAlg.setTol_p(0.005);
        psoSeAlg.setTol_q(0.005);

        IEEEDataIsland island;
        SystemMeasure sm;

        island = IcfDataUtil.ISLAND_14.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_14, false, false);

        island = IcfDataUtil.ISLAND_30.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_30, false, false);

        island = IcfDataUtil.ISLAND_39.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_39, false, false);

        island = IcfDataUtil.ISLAND_57.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_57, false, false);

        island = IcfDataUtil.ISLAND_118.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_118, false, false);

        island = IcfDataUtil.ISLAND_300.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_300, false, false);
    }


    /**
     * 所有量测均为真值，且加入零功率注入节点等式约束
     */
    @Test
    public void testCaseTrue_zeroInjection() {
        ipoptSeAlg.setTol_p(1e-5);
        ipoptSeAlg.setTol_q(1e-5);
        psoSeAlg.setTol_p(1e-5);
        psoSeAlg.setTol_q(1e-5);

        IEEEDataIsland island;
        SystemMeasure sm;

        island = IcfDataUtil.ISLAND_14.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_14, true, true);

        island = IcfDataUtil.ISLAND_30.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_30, true, true);

        island = IcfDataUtil.ISLAND_39.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_39, true, true);

        island = IcfDataUtil.ISLAND_57.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_57, true, true);

        island = IcfDataUtil.ISLAND_118.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_118, true, true);

        island = IcfDataUtil.ISLAND_300.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0);
        doSE(island, sm, IcfDataUtil.ISLAND_300, true, true);
    }

    /**
     * 给所有量测加入2%的高斯噪声，且加入零功率注入节点等式约束
     */
    @Test
    public void testCaseGauss_zeroInjection() {
        ipoptSeAlg.setTol_p(5e-3);
        ipoptSeAlg.setTol_q(5e-3);
        psoSeAlg.setTol_p(5e-3);
        psoSeAlg.setTol_q(5e-3);

        IEEEDataIsland island;
        SystemMeasure sm;

        island = IcfDataUtil.ISLAND_14.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_14, false, true);

        island = IcfDataUtil.ISLAND_30.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_30, false, true);

        island = IcfDataUtil.ISLAND_39.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_39, false, true);

        island = IcfDataUtil.ISLAND_57.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_57, false, true);

        island = IcfDataUtil.ISLAND_118.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_118, false, true);

        island = IcfDataUtil.ISLAND_300.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.02);
        doSE(island, sm, IcfDataUtil.ISLAND_300, false, true);
    }

    /**
     * 给所有量测加入2%的高斯噪声，且加入零功率注入节点等式约束
     */
    @Test
    public void testCaseGauss_zeroInjection_withBadData() {
        ipoptSeAlg.setTol_p(5e-3);
        ipoptSeAlg.setTol_q(5e-3);
        psoSeAlg.setTol_p(5e-3);
        psoSeAlg.setTol_q(5e-3);

        IEEEDataIsland island;
        SystemMeasure sm;
        SystemMeasure smRef;

        island = IcfDataUtil.ISLAND_14.clone();
        smRef = SimuMeasMaker.createFullMeasure(island, 1, 0);
        sm = SimuMeasMaker.createFullMeasure_withBadData(island, 1, 0.02, 0.06);
        doSE(island, sm, smRef, IcfDataUtil.ISLAND_14, false, true);

        island = IcfDataUtil.ISLAND_30.clone();
        smRef = SimuMeasMaker.createFullMeasure(island, 1, 0);
        sm = SimuMeasMaker.createFullMeasure_withBadData(island, 1, 0.02, 0.06);
        doSE(island, sm, smRef, IcfDataUtil.ISLAND_30, false, true);

        island = IcfDataUtil.ISLAND_39.clone();
        smRef = SimuMeasMaker.createFullMeasure(island, 1, 0);
        sm = SimuMeasMaker.createFullMeasure_withBadData(island, 1, 0.02, 0.06);
        doSE(island, sm, smRef, IcfDataUtil.ISLAND_39, false, true);

        island = IcfDataUtil.ISLAND_57.clone();
        smRef = SimuMeasMaker.createFullMeasure(island, 1, 0);
        sm = SimuMeasMaker.createFullMeasure_withBadData(island, 1, 0.02, 0.06);
        doSE(island, sm, smRef, IcfDataUtil.ISLAND_57, false, true);

        island = IcfDataUtil.ISLAND_118.clone();
        smRef = SimuMeasMaker.createFullMeasure(island, 1, 0);
        sm = SimuMeasMaker.createFullMeasure_withBadData(island, 1, 0.02, 0.06);
        doSE(island, sm, smRef, IcfDataUtil.ISLAND_118, false, true);

        island = IcfDataUtil.ISLAND_300.clone();
        smRef = SimuMeasMaker.createFullMeasure(island, 1, 0);
        sm = SimuMeasMaker.createFullMeasure_withBadData(island, 1, 0.02, 0.06);
        doSE(island, sm, smRef, IcfDataUtil.ISLAND_300, false, true);
    }
    /**
     * 单个算例测试
     */
    @Test
    public void testOneCase_zeroInjection_withBadData() {
        ipoptSeAlg.setTol_p(5e-3);
        ipoptSeAlg.setTol_q(5e-3);
        psoSeAlg.setTol_p(5e-3);
        psoSeAlg.setTol_q(5e-3);

        IEEEDataIsland island;
        SystemMeasure sm;
        SystemMeasure smRef;

        island = IcfDataUtil.ISLAND_118.clone();
        smRef = SimuMeasMaker.createFullMeasure(island, 1, 0);
        sm = SimuMeasMaker.createFullMeasure_withBadData(island, 1, 0.02, 0.06);
        doSE(island, sm, smRef, IcfDataUtil.ISLAND_118, false, true);
    }

    private void doSE(IEEEDataIsland island, SystemMeasure sm,
                      IEEEDataIsland ref, boolean isTrueValue, boolean isZeroInjection) {
        int[] variables_types = {
                IpoptSeAlg.VARIABLE_VTHETA,
        }; // 状态变量类型
        doIpoptSE(island, sm, ref, variables_types, isTrueValue, isZeroInjection, SeObjective.OBJ_TYPE_SIGMOID);
        double[] variableState = ipoptSeAlg.getVariableState();
        psoSeAlg.setInitVariableState(variableState);
        psoSeAlg.setWarmStart(true);
        doPsoSE(island, sm, ref, variables_types, isTrueValue, isZeroInjection);
    }

    private void doSE(IEEEDataIsland island, SystemMeasure sm, SystemMeasure smRef,
                      IEEEDataIsland ref, boolean isTrueValue, boolean isZeroInjection) {
        int[] variables_types = {
                IpoptSeAlg.VARIABLE_VTHETA,
        }; // 状态变量类型
        log.debug("先算一遍MNMR方法：");
        doIpoptSE(island, sm, ref, variables_types, isTrueValue, isZeroInjection, SeObjective.OBJ_TYPE_SIGMOID);
        log.debug("计算真值：");
        doIpoptSE(island, smRef, ref, variables_types, isTrueValue, isZeroInjection, SeObjective.OBJ_TYPE_WLS);
        double[] variableState = ipoptSeAlg.getVariableState();
        psoSeAlg.setInitVariableState(variableState);
        psoSeAlg.setWarmStart(true);
        log.debug("开始粒子群算法：");
        doPsoSE(island, sm, ref, variables_types, isTrueValue, isZeroInjection);
    }

    private void doIpoptSE(IEEEDataIsland island, SystemMeasure sm,
                           IEEEDataIsland ref, int[] variables_types,
                           boolean isTrueValue, boolean isZeroInjection, int objType) {
        //alg.setPrintPath(false);
        se.setAlg(ipoptSeAlg);
        double slackBusAngle = island.getBus(island.getSlackBusNum()).getFinalAngle() * Math.PI / 180.;
        ipoptSeAlg.setSlackBusAngle(slackBusAngle);

        se.setOriIsland(island); // 初始化电气岛
        se.setSm(sm);
        se.setFlatStart(true); // 平启动

        long start = System.currentTimeMillis();
        ipoptSeAlg.getObjFunc().setObjType(objType); // 设置目标函数

        SeResultInfo r;

        //下面这一段测试所有模型的最小二乘效果
        if (ref != null) // ref是潮流计算的结果
            dealZeroInjection(sm, ref, isZeroInjection, ipoptSeAlg); // 将零注入功率节点的节点注入功率量测从sm中剔除
        for (int variable_type : variables_types) {
            ipoptSeAlg.setVariable_type(variable_type);
            se.doSe();
            assertTrue(ipoptSeAlg.isConverged());
            r = se.createPfResult();
            assertNotNull(r);
            printS1AndS2(ref, r, isTrueValue);
            printS1_AndS2_(ref, r, sm, isTrueValue);
            printXi(sm);
            printZeroInjectionDelta(ref, r, ipoptSeAlg);
            System.out.println("WLS状态估计迭代次数：" + ipoptSeAlg.getIterNum() + "\t用时：" + ipoptSeAlg.getTimeUsed() + "\t");
            System.out.println("WLS状态估计用时：" + (System.currentTimeMillis() - start) + "ms");
        }
    }

    private void doPsoSE(IEEEDataIsland island, SystemMeasure sm,
                         IEEEDataIsland ref, int[] variables_types,
                         boolean isTrueValue, boolean isZeroInjection) {
        //alg.setPrintPath(false);
        se.setAlg(psoSeAlg);
        double slackBusAngle = island.getBus(island.getSlackBusNum()).getFinalAngle() * Math.PI / 180.;
        psoSeAlg.setSlackBusAngle(slackBusAngle);

        se.setOriIsland(island); // 初始化电气岛
        se.setSm(sm);
        se.setFlatStart(true); // 平启动

        long start = System.currentTimeMillis();
        psoSeAlg.getObjFunc().setObjType(SeObjective.OBJ_TYPE_MNMR); // 设置目标函数

        SeResultInfo r;

        //下面这一段测试所有模型的最小二乘效果
        if (ref != null) // ref是潮流计算的结果
            dealZeroInjection(sm, ref, isZeroInjection, psoSeAlg); // 将零注入功率节点的节点注入功率量测从sm中剔除
        for (int variable_type : variables_types) {
            psoSeAlg.setVariable_type(variable_type);
            psoSeAlg.setConverged(true);
            se.doSe();
//            delBadDataResult();
            r = se.createPfResult();
            assertNotNull(r);
            printS1AndS2(ref, r, isTrueValue);
            printS1_AndS2_(ref, r, sm, isTrueValue);
            printXi(sm);
            printZeroInjectionDelta(ref, r, psoSeAlg);
            System.out.println("MNMR状态估计用时：" + (System.currentTimeMillis() - start) + "ms");
        }
    }

    private void delBadDataResult() {
        SystemMeasure sm = se.getSm();
        int count = 0;
        log.info("粒子群算法计算后识别的坏数据为：");
        // map里面循环又要删除元素，如果使用foreach会抛出异常，应该使用iterator
        Iterator<Map.Entry<String, MeasureInfo>> iterator = sm.getLine_from_p().entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, MeasureInfo> entry = iterator.next();
            MeasureInfo info = entry.getValue();
            double d = (info.getValue_est() - info.getValue()) / (info.getSigma() * 3); // 测点相对偏移
            if (Math.abs(d) > 1) {
                count++;
                log.info("Branch " + info.getPositionId() + " " + info.getValue_est() + " " + info.getValue());
                iterator.remove();
            }
        }
        iterator = sm.getLine_from_q().entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, MeasureInfo> entry = iterator.next();
            MeasureInfo info = entry.getValue();
            double d = (info.getValue_est() - info.getValue()) / (info.getSigma() * 3); // 测点相对偏移
            if (Math.abs(d) > 1) {
                count++;
                log.info("Branch " + info.getPositionId() + " " + info.getValue_est() + " " + info.getValue());
                iterator.remove();
            }
        }
        iterator = sm.getLine_to_p().entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, MeasureInfo> entry = iterator.next();
            MeasureInfo info = entry.getValue();
            double d = (info.getValue_est() - info.getValue()) / (info.getSigma() * 3); // 测点相对偏移
            if (Math.abs(d) > 1) {
                count++;
                log.info("Branch " + info.getPositionId() + " " + info.getValue_est() + " " + info.getValue());
                iterator.remove();
            }
        }
        iterator = sm.getLine_to_q().entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, MeasureInfo> entry = iterator.next();
            MeasureInfo info = entry.getValue();
            double d = (info.getValue_est() - info.getValue()) / (info.getSigma() * 3); // 测点相对偏移
            if (Math.abs(d) > 1) {
                count++;
                log.info("Branch " + info.getPositionId() + " " + info.getValue_est() + " " + info.getValue());
                iterator.remove();
            }
        }
        iterator = sm.getBus_p().entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, MeasureInfo> entry = iterator.next();
            MeasureInfo info = entry.getValue();
            double d = (info.getValue_est() - info.getValue()) / (info.getSigma() * 3); // 测点相对偏移
            if (Math.abs(d) > 1) {
                count++;
                log.info("Bus " + info.getPositionId() + " " + info.getValue_est() + " " + info.getValue());
                iterator.remove();
            }
        }
        iterator = sm.getBus_q().entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, MeasureInfo> entry = iterator.next();
            MeasureInfo info = entry.getValue();
            double d = (info.getValue_est() - info.getValue()) / (info.getSigma() * 3); // 测点相对偏移
            if (Math.abs(d) > 1) {
                count++;
                log.info("Bus " + info.getPositionId() + " " + info.getValue_est() + " " + info.getValue());
                iterator.remove();
            }
        }
        iterator = sm.getBus_v().entrySet().iterator();
        while (iterator.hasNext()) {
            Map.Entry<String, MeasureInfo> entry = iterator.next();
            MeasureInfo info = entry.getValue();
            double d = (info.getValue_est() - info.getValue()) / (info.getSigma() * 3); // 测点相对偏移
            if (Math.abs(d) > 1) {
                count++;
                log.info("Bus " + info.getPositionId() + " " + info.getValue_est() + " " + info.getValue());
                iterator.remove();
            }
        }
        log.info("不良数据个数：" + count);
    }

    private void dealZeroInjection(SystemMeasure sm, IEEEDataIsland ref, boolean isAddConstraint, AbstractSeAlg alg) {
        int busNumber = se.getOriIsland().getBuses().size();
        double[] vtheta = new double[busNumber * 2];
        for (BusData b : ref.getBuses()) {
            int num = b.getBusNumber();
            int newNum = se.getNumOpt().getOld2new().get(num);
            vtheta[newNum - 1] = b.getFinalVoltage();
            vtheta[newNum - 1 + busNumber] = b.getFinalAngle() * Math.PI / 180.0;
        }
        int count1 = 0, count2 = 0;
        for (BusData b : se.getClonedIsland().getBuses()) {
            int num = b.getBusNumber();
            double p = StateCalByPolar.calBusP(num, se.getY(), vtheta);
            double q = StateCalByPolar.calBusQ(num, se.getY(), vtheta);
            // 将零注入功率节点从注入功率量测中除去
            if (Math.abs(p) < alg.getTol_p()) {//是否是有功零注入节点
                String key = String.valueOf(se.getNumOpt().getNew2old().get(num)); // clonedIsland已经重新编号，这里获取原始编号
                sm.getBus_p().remove(key);
                count1++;
            }
            if (Math.abs(q) < alg.getTol_q()) {//是否是无功零注入节点
                String key = String.valueOf(se.getNumOpt().getNew2old().get(num));
                sm.getBus_q().remove(key);
                count2++;
            }
        }
        System.out.println("共有" + count1 + "零有功注入节点.");
        System.out.println("共有" + count2 + "零无功注入节点.");
        if (isAddConstraint) {
            int[] zeroPInjection = new int[count1];
            int[] zeroQInjection = new int[count2];
            count1 = 0;
            count2 = 0;
            for (BusData b : se.getClonedIsland().getBuses()) {
                int num = b.getBusNumber();
                double p = StateCalByPolar.calBusP(num, se.getY(), vtheta);
                double q = StateCalByPolar.calBusQ(num, se.getY(), vtheta);
                if (Math.abs(p) < alg.getTol_p())
                    zeroPInjection[count1++] = num; // 标记零注入功率节点(重新编号后)
                if (Math.abs(q) < alg.getTol_q())
                    zeroQInjection[count2++] = num;
            }
            alg.setZeroPBuses(zeroPInjection);
            alg.setZeroQBuses(zeroQInjection);
        }
    }

    private static void printS1AndS2(IEEEDataIsland ref, SeResultInfo r, boolean isTrueValue) {
        if (ref != null) {
            if (isTrueValue)
                assertTrueSe(ref, r);
            Map<Integer, Double> busV = r.getPfResult().getBusV();
            Map<Integer, Double> busTheta = r.getPfResult().getBusTheta();
            double s1 = 0.0;
            double s2 = 0.0;
            for (BusData b : ref.getBuses()) {
                double v = busV.get(b.getBusNumber());
                double theta = busTheta.get(b.getBusNumber());
                double trueV = b.getFinalVoltage();
                double trueTheta = b.getFinalAngle() * Math.PI / 180.;
                s1 += Math.abs(v - trueV);
                s1 += Math.abs(theta - trueTheta);
                s2 = Math.max(s2, Math.max(Math.abs(v - trueV), Math.abs(theta - trueTheta)));
            }
            log.info("计算指标S1：" + s1);
            log.info("计算指标S2：" + s2);
        }
    }

    private static void printS1_AndS2_(IEEEDataIsland ref, SeResultInfo r, SystemMeasure sm, boolean isTrueValue) {
        if (ref != null) {
            if (isTrueValue)
                assertTrueSe(ref, r);
            double s1 = 0.0;
            double s2 = 0.0;
            for (MeasureInfo info : sm.getBus_p().values()) {
                s1 += Math.abs(info.getValue_est() - info.getValue_true());
                s2 = Math.max(s2, Math.abs(info.getValue_est() - info.getValue_true()));
            }
            for (MeasureInfo info : sm.getBus_q().values()) {
                s1 += Math.abs(info.getValue_est() - info.getValue_true());
                s2 = Math.max(s2, Math.abs(info.getValue_est() - info.getValue_true()));
            }
            for (MeasureInfo info : sm.getLine_from_p().values()) {
                s1 += Math.abs(info.getValue_est() - info.getValue_true());
                s2 = Math.max(s2, Math.abs(info.getValue_est() - info.getValue_true()));
            }
            for (MeasureInfo info : sm.getLine_from_q().values()) {
                s1 += Math.abs(info.getValue_est() - info.getValue_true());
                s2 = Math.max(s2, Math.abs(info.getValue_est() - info.getValue_true()));
            }
            for (MeasureInfo info : sm.getLine_to_p().values()) {
                s1 += Math.abs(info.getValue_est() - info.getValue_true());
                s2 = Math.max(s2, Math.abs(info.getValue_est() - info.getValue_true()));
            }
            for (MeasureInfo info : sm.getLine_to_q().values()) {
                s1 += Math.abs(info.getValue_est() - info.getValue_true());
                s2 = Math.max(s2, Math.abs(info.getValue_est() - info.getValue_true()));
            }
            log.info("计算指标S1_：" + s1);
            log.info("计算指标S2_：" + s2);
        }
    }

    private void printXi(SystemMeasure sm) {
        int totalNum = sm.getBus_v().size() + sm.getBus_p().size() + sm.getBus_q().size()
                + sm.getLine_from_p().size() + sm.getLine_from_q().size()
                + sm.getLine_to_p().size() + sm.getLine_to_q().size();
        int N3sigma = 0;
        int N2sigma = 0;
        int N1sigma = 0;
        for (MeasureInfo info : sm.getBus_v().values()) {
            double deviation = Math.abs(info.getValue_est() - info.getValue_true());
            if (deviation <= 3 * info.getSigma())
                N3sigma++;
            if (deviation <= 2 * info.getSigma())
                N2sigma++;
            if (deviation <= info.getSigma())
                N1sigma++;
        }
        for (MeasureInfo info : sm.getBus_p().values()) {
            double deviation = Math.abs(info.getValue_est() - info.getValue_true());
            if (deviation <= 3 * info.getSigma())
                N3sigma++;
            if (deviation <= 2 * info.getSigma())
                N2sigma++;
            if (deviation <= info.getSigma())
                N1sigma++;
        }
        for (MeasureInfo info : sm.getBus_q().values()) {
            double deviation = Math.abs(info.getValue_est() - info.getValue_true());
            if (deviation <= 3 * info.getSigma())
                N3sigma++;
            if (deviation <= 2 * info.getSigma())
                N2sigma++;
            if (deviation <= info.getSigma())
                N1sigma++;
        }
        for (MeasureInfo info : sm.getLine_from_p().values()) {
            double deviation = Math.abs(info.getValue_est() - info.getValue_true());
            if (deviation <= 3 * info.getSigma())
                N3sigma++;
            if (deviation <= 2 * info.getSigma())
                N2sigma++;
            if (deviation <= info.getSigma())
                N1sigma++;
        }
        for (MeasureInfo info : sm.getLine_from_q().values()) {
            double deviation = Math.abs(info.getValue_est() - info.getValue_true());
            if (deviation <= 3 * info.getSigma())
                N3sigma++;
            if (deviation <= 2 * info.getSigma())
                N2sigma++;
            if (deviation <= info.getSigma())
                N1sigma++;
        }
        for (MeasureInfo info : sm.getLine_to_p().values()) {
            double deviation = Math.abs(info.getValue_est() - info.getValue_true());
            if (deviation <= 3 * info.getSigma())
                N3sigma++;
            if (deviation <= 2 * info.getSigma())
                N2sigma++;
            if (deviation <= info.getSigma())
                N1sigma++;
        }
        for (MeasureInfo info : sm.getLine_to_q().values()) {
            double deviation = Math.abs(info.getValue_est() - info.getValue_true());
            if (deviation <= 3 * info.getSigma())
                N3sigma++;
            if (deviation <= 2 * info.getSigma())
                N2sigma++;
            if (deviation <= info.getSigma())
                N1sigma++;
        }
        log.info("计算指标Xi3sigma：" + ((double) N3sigma / totalNum));
        log.info("计算指标Xi2sigma：" + ((double) N2sigma / totalNum));
        log.info("计算指标Xi1sigma：" + ((double) N1sigma / totalNum));
    }


    private static void assertTrueSe(IEEEDataIsland ref, SeResultInfo r) {
        assertEquals(1.0, r.getEligibleRate());
        for (BusData bus : ref.getBuses()) {
            double v = r.getPfResult().getBusV().get(bus.getBusNumber());
            double theta = r.getPfResult().getBusTheta().get(bus.getBusNumber());
            double refTheta = bus.getFinalAngle() * Math.PI / 180.;
            assertTrue(Math.abs(v - bus.getFinalVoltage()) < 1e-2);
            if (bus.getType() == BusData.BUS_TYPE_SLACK)
                assertTrue(Math.abs(theta - refTheta) < 1e-4);
            else
                assertTrue(Math.abs(theta - refTheta) < 1e-1);
        }
    }

    private void printZeroInjectionDelta(IEEEDataIsland ref, SeResultInfo r, AbstractSeAlg alg) {
        if (ref == null)
            return;
        int busNumber = se.getClonedIsland().getBuses().size();
        double[] vtheta = new double[busNumber * 2];
        for (BusData b : ref.getBuses()) {
            int num = b.getBusNumber();
            int newNum = se.getNumOpt().getOld2new().get(num);
            vtheta[newNum - 1] = b.getFinalVoltage();
            vtheta[newNum - 1 + busNumber] = b.getFinalAngle() * Math.PI / 180.0;
        }
        double pDelta = 0.0, qDelta = 0.0;
        double baseMva = se.getClonedIsland().getTitle().getMvaBase();
        for (BusData b : se.getClonedIsland().getBuses()) {
            int num = b.getBusNumber();
            int oldNum = se.getNumOpt().getNew2old().get(num);
            double p = StateCalByPolar.calBusP(num, se.getY(), vtheta);
            double q = StateCalByPolar.calBusQ(num, se.getY(), vtheta);
            if (Math.abs(p) < alg.getTol_p()) {//是否是有功零注入节点
                pDelta += Math.abs(r.getPfResult().getBusP().get(oldNum) / baseMva - p);
            }
            if (Math.abs(q) < alg.getTol_q()) {//是否是无功零注入节点
                qDelta += Math.abs(r.getPfResult().getBusQ().get(oldNum) / baseMva - q);
            }
        }
        System.out.println("零有功注入节点有功偏差和：" + pDelta);
        System.out.println("零有功注入节点无功偏差和：" + qDelta);
    }
}
