package zju.se;

import junit.framework.TestCase;
import org.junit.Test;
import zju.ieeeformat.BusData;
import zju.ieeeformat.DefaultIcfParser;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.measure.MeasTypeCons;
import zju.measure.SystemMeasure;
import zju.pf.SimuMeasMaker;
import zju.util.StateCalByPolar;

import java.io.InputStream;
import java.util.Map;

/**
 * @author <Authors name>
 * @version 1.0
 * @since <pre>07/28/2010</pre>
 */
public class SeTest_IeeeCase extends TestCase implements MeasTypeCons {

    StateEstimator se;
    AbstractSeAlg alg;

    public SeTest_IeeeCase(String name) {
        super(name);
        se = new StateEstimator();
        // alg = new IpoptSeAlg();
        alg = new NewtonWlsSe();
        se.setAlg(alg); // se中也有这个变量
    }

    public void setUp() throws Exception {
    }

    public void testNewCase() {
        System.out.println("=========开始计算33节点配电网==========");
        alg.setTol_p(1e-5);
        alg.setTol_q(1e-5);
        InputStream node33File = this.getClass().getResourceAsStream("/ieeefiles/33node-cdf.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(node33File, "GBK");
        SystemMeasure sm;
        IEEEDataIsland islandClone = island.clone();
        sm = SimuMeasMaker.createFullMeasure(islandClone, 1, 0.05);
        doSE(islandClone, sm, island, false, false);
        System.out.println("==========33节点配电网计算结束==========");
        System.out.println("=========开始计算90节点配电网==========");
        alg.setTol_p(1e-5);
        alg.setTol_q(1e-5);
        InputStream node90File = this.getClass().getResourceAsStream("/ieeefiles/90node-cdf.txt");
        IEEEDataIsland island2 = new DefaultIcfParser().parse(node90File, "GBK");
        SystemMeasure sm2;
        IEEEDataIsland islandClone2 = island2.clone();
        sm2 = SimuMeasMaker.createFullMeasure(islandClone2, 1, 0.05);
        doSE(islandClone2, sm2, island2, false, false);
        System.out.println("==========90节点配电网计算结束==========");
    }

    public void testCaseTrue() {
        alg.setTol_p(1e-5);
        alg.setTol_q(1e-5);
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

    public void testCaseGauss() {
        alg.setTol_p(0.005);
        alg.setTol_q(0.005);
        IEEEDataIsland island;
        SystemMeasure sm;
        island = IcfDataUtil.ISLAND_14.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, IcfDataUtil.ISLAND_14, false, true);
        island = IcfDataUtil.ISLAND_30.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, IcfDataUtil.ISLAND_30, false, true);
        island = IcfDataUtil.ISLAND_39.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, IcfDataUtil.ISLAND_39, false, true);
        island = IcfDataUtil.ISLAND_57.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, IcfDataUtil.ISLAND_57, false, true);
        island = IcfDataUtil.ISLAND_118.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, IcfDataUtil.ISLAND_118, false, true);
        island = IcfDataUtil.ISLAND_300.clone();
        sm = SimuMeasMaker.createFullMeasure(island, 1, 0.05);
        doSE(island, sm, IcfDataUtil.ISLAND_300, false, true);
    }

    public void testCaseTrue_zeroInjection() {
        alg.setTol_p(1e-5);
        alg.setTol_q(1e-5);
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

    public void testCaseGauss_zeroInjection() {
        doSeStudy(IcfDataUtil.ISLAND_14.clone(), 0.005, 0.005);
        doSeStudy(IcfDataUtil.ISLAND_30.clone(), 0.005, 0.005);
        doSeStudy(IcfDataUtil.ISLAND_39.clone(), 0.005, 0.005);
        doSeStudy(IcfDataUtil.ISLAND_57.clone(), 0.005, 0.005);
        doSeStudy(IcfDataUtil.ISLAND_118.clone(), 0.005, 0.005);
        doSeStudy(IcfDataUtil.ISLAND_300.clone(), 0.005, 0.005);
    }

    public void doSE(IEEEDataIsland island, SystemMeasure sm,
                     IEEEDataIsland ref, boolean isTrueValue, boolean isZeroInjection) {
        int[] variables_types = {
                IpoptSeAlg.VARIABLE_VTHETA,
                IpoptSeAlg.VARIABLE_VTHETA_PQ,
                IpoptSeAlg.VARIABLE_U,
                IpoptSeAlg.VARIABLE_UI,
        }; // 状态变量类型
        doSE(island, sm, ref, variables_types, isTrueValue, isZeroInjection);
    }

    public void doSE(IEEEDataIsland island, SystemMeasure sm,
                     IEEEDataIsland ref, int[] variables_types,
                     boolean isTrueValue, boolean isZeroInjection) {
        //alg.setPrintPath(false);
        double slackBusAngle = island.getBus(island.getSlackBusNum()).getFinalAngle() * Math.PI / 180.;
        alg.setSlackBusAngle(slackBusAngle);

        se.setOriIsland(island); // 初始化电气岛
        se.setSm(sm);
        se.setFlatStart(true); // 平启动

        long start = System.currentTimeMillis();
        if (alg instanceof IpoptSeAlg)
            ((IpoptSeAlg) alg).getObjFunc().setObjType(SeObjective.OBJ_TYPE_WLS); // 设置目标函数
        else if (alg instanceof PsoSeAlg)
            ((PsoSeAlg) alg).getObjFunc().setObjType(SeObjective.OBJ_TYPE_MNMR); // 设置目标函数

        SeResultInfo r;

        //下面这一段测试传统最小二乘的效果
        //dealZeroInjection(sm, ref, false);
        //alg.setVariable_type(IpoptSeAlg.VARIABLE_VTHETA);
        //se.doSe();
        //assertTrue(alg.isConverged());
        //r = se.createPfResult();
        //printS1(ref, se.createPfResult(), isTrueValue);
        //printZeroInjectionDelta(ref, r);

        //下面这一段测试所有模型的最小二乘效果
        if (ref != null) // ref是潮流计算的结果
            dealZeroInjection(sm, ref, isZeroInjection); // 将零注入功率节点的节点注入功率量测从sm中剔除
        for (int variable_type : variables_types) {
            alg.setVariable_type(variable_type);
            se.doSe();
            assertTrue(alg.isConverged());
            r = se.createPfResult();
            assertNotNull(r);
            //printS1(ref, r, isTrueValue);
            printZeroInjectionDelta(ref, r);
            //System.out.println("WLS状态估计迭代次数：" + alg.getIterNum() + "\t用时：" + alg.getTimeUsed() + "\t");
            //System.out.println("WLS状态估计用时：" + (System.currentTimeMillis() - start) + "ms");
        }

        //下面这一段测试sigmoid函数的效果
//        start = System.currentTimeMillis();
//        alg.getObjFunc().setObjType(SeObjective.OBJ_TYPE_SIGMOID);
//        dealZeroInjection(sm, ref, false);
//        alg.setVariable_type(IpoptSeAlg.VARIABLE_VTHETA);
//        se.doSe();
//        assertTrue(alg.isConverged());
//        r = se.createPfResult();
//        printS1(ref, se.createPfResult(), isTrueValue);
//        printZeroInjectionDelta(ref, r);

        //for (int variable_type : variables_types) {
        //    alg.setVariable_type(variable_type);
        //    r = se.doSe();
        //    assertTrue(r.isConverged());
        //    if (ref != null)
        //        assertTrueSe(ref, r);
        //    System.out.println("Sigmoid状态估计用时：" + (System.currentTimeMillis() - start) + "ms");
        //}
    }

    private void printZeroInjectionDelta(IEEEDataIsland ref, SeResultInfo r) {
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

    public static void printS1(IEEEDataIsland ref, SeResultInfo r, boolean isTrueValue) {
        if (ref != null) {
            if (isTrueValue)
                assertTrueSe(ref, r);
            Map<Integer, Double> busV = r.getPfResult().getBusV();
            Map<Integer, Double> busTheta = r.getPfResult().getBusTheta();
            double s1 = 0.0;
            for (BusData b : ref.getBuses()) {
                double v = busV.get(b.getBusNumber());
                double theta = busTheta.get(b.getBusNumber());
                double trueV = b.getFinalVoltage();
                double trueTheta = b.getFinalAngle() * Math.PI / 180.;
                s1 += Math.abs(v - trueV);
                s1 += Math.abs(theta - trueTheta);
            }
            System.out.println("计算指标S1：" + s1);
        }
    }

    public void dealZeroInjection(SystemMeasure sm, IEEEDataIsland ref, boolean isAddConstraint) {
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

    public static void assertTrueSe(IEEEDataIsland ref, SeResultInfo r) {
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

    public static void doSeStudy(IEEEDataIsland island, double tol_p, double tol_q) {
        SeTest_IeeeCase seTest = new SeTest_IeeeCase("do 20 time study.");
        AbstractSeAlg alg = seTest.alg;
        alg.setTol_p(tol_p);
        alg.setTol_q(tol_q);
        alg.setPrintPath(false);

        long totalTimeUsed = 0;
        int totalIterNum = 0;
        int studyCount = 0;
        for (int i = 0; i <= studyCount; i++) {
            IEEEDataIsland clonedIsland = island.clone();
            SystemMeasure sm = SimuMeasMaker.createFullMeasure(clonedIsland, 1, 0.09);
            seTest.doSE(clonedIsland, sm, island, false, true);
            if (i == 0)
                continue;
            totalTimeUsed += alg.getTimeUsed();
            totalIterNum += alg.getIterNum();
        }
        if (studyCount == 0)
            return;
        System.out.println("Average time used: " + (totalTimeUsed / studyCount) + "ms");
        System.out.println("Average iter number: " + (totalIterNum / studyCount));
    }

    public StateEstimator getSe() {
        return se;
    }

    public AbstractSeAlg getAlg() {
        return alg;
    }
}
