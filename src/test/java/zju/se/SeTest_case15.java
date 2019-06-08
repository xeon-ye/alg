package zju.se;

import com.csvreader.CsvReader;
import org.junit.Assert;
import org.junit.Test;
import zju.ieeeformat.BusData;
import zju.ieeeformat.DefaultIcfParser;
import zju.ieeeformat.IEEEDataIsland;
import zju.measure.*;
import zju.pf.PfResultInfo;
import zju.pf.PolarPf;
import zju.pf.SimuMeasMaker;
import zju.util.*;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.*;

public class SeTest_case15 implements SeConstants, MeasTypeCons {

    private static final IEEEDataIsland island;
    private static YMatrixGetter y;

    StateEstimator se;
    NewtonWlsSe alg;

    public SeTest_case15() {
        se = new StateEstimator();
        alg = new NewtonWlsSe();
        se.setAlg(alg);
    }


    static {
        island = new DefaultIcfParser().parse(SeTest_case15.class.getResourceAsStream("/ieeefiles/nanjing/case15.txt"));
        y = new YMatrixGetter(island);
        y.formYMatrix();
    }

    @Test
    public void test5000Cases() throws IOException {
        InputStream stream = this.getClass().getResourceAsStream("/ieeefiles/nanjing/5000_sample.csv");
        CsvReader reader = new CsvReader(stream, Charset.forName("UTF-8"));
        reader.readHeaders();

        int cnt = 0;
        double rate = 0, mse = 0;
        List<String[]> injections = new ArrayList<>(15);
        while (reader.readRecord()) {
            injections.add(reader.getValues());
            if (injections.size() == 15) {
                SystemMeasure sm = getSm(injections);
                if (sm != null) {
                    Iterator<Map.Entry<String, MeasureInfo>> it = sm.getLine_to_p().entrySet().iterator();
                    while (it.hasNext()) {
                        String branchId = it.next().getKey();
                        if (!branchId.equals("1") && !branchId.equals("7"))
                            it.remove();
                    }
                    it = sm.getLine_to_q().entrySet().iterator();
                    while (it.hasNext()) {
                        String branchId = it.next().getKey();
                        if (!branchId.equals("1") && !branchId.equals("7"))
                            it.remove();
                    }
                    SeResultInfo res = doSE(island.clone(), sm);
                    rate += res.getEligibleRate();
                    for (int i = 0; i < alg.getMeas().getZ_estimate().getN(); i++)
                        mse += Math.pow(alg.getMeas().getZ_estimate().getValue(i) - alg.getMeas().getZ_true().getValue(i), 2);
                    ++cnt;
                }
                injections.clear();
            }
        }
        rate /= cnt;
        mse = Math.sqrt(mse / cnt);
        System.out.println("平均合格率为：" + rate);
        System.out.println("MSE为：" + mse);

    }

    @Test
    public void testMeasPlacement() throws IOException {
        PolarPf pf = new PolarPf();
        pf.setTol_p(1e-4);
        pf.setTol_q(1e-4);
        pf.setOriIsland(island);
        pf.setDecoupledPqNum(0);
        pf.doPf();
        if (pf.isConverged())
            pf.fillOriIslandPfResult();
        SystemMeasure sm = SimuMeasMaker.createMeasureOfTypes(island, new int[]{
                TYPE_BUS_VOLOTAGE,
                TYPE_BUS_ACTIVE_POWER,
                TYPE_BUS_REACTIVE_POWER,
                TYPE_LINE_TO_ACTIVE,
                TYPE_LINE_TO_REACTIVE
        }, 1, 0.02);
        Iterator<Map.Entry<String, MeasureInfo>> it = sm.getLine_to_p().entrySet().iterator();
        while (it.hasNext()) {
            String branchId = it.next().getKey();
            if (!branchId.equals("1") && !branchId.equals("7"))
                it.remove();
        }
        it = sm.getLine_to_q().entrySet().iterator();
        while (it.hasNext()) {
            String branchId = it.next().getKey();
            if (!branchId.equals("1") && !branchId.equals("7"))
                it.remove();
        }
        List<String> ans = new ArrayList<>();
        for (int i = 2; i <= 15; i++) {
            String id = String.valueOf(i);
            MeasureInfo[] infos = deleteOneMeas(id, sm);
            SeResultInfo res = doSE(island.clone(), sm);
            if (res.getEligibleRate() < 0.8 || !alg.isConverged())
                ans.add("测点位置：" + i);
            restoreMeas(id, sm, infos);
        }
        for (int i = 2; i <= 15; i++) {
            String id1 = String.valueOf(i);
            MeasureInfo[] infos1 = deleteOneMeas(id1, sm);
            for (int j = i + 1; j <= 15; j++) {
                String id2 = String.valueOf(j);
                MeasureInfo[] infos2 = deleteOneMeas(id2, sm);
                SeResultInfo res = doSE(island.clone(), sm);
                if (res.getEligibleRate() < 0.8 || !alg.isConverged())
                    ans.add("测点位置：" + i + ";" + j);
                restoreMeas(id2, sm, infos2);
            }
            restoreMeas(id1, sm, infos1);
        }
        for (int i = 2; i <= 15; i++) {
            String id1 = String.valueOf(i);
            MeasureInfo[] infos1 = deleteOneMeas(id1, sm);
            for (int j = i + 1; j <= 15; j++) {
                String id2 = String.valueOf(j);
                MeasureInfo[] infos2 = deleteOneMeas(id2, sm);
                for (int k = j + 1; k <= 15; k++) {
                    String id3 = String.valueOf(k);
                    MeasureInfo[] infos3 = deleteOneMeas(id3, sm);
                    SeResultInfo res = doSE(island.clone(), sm);
                    if (res.getEligibleRate() < 0.8 || !alg.isConverged())
                        ans.add("测点位置：" + i + ";" + j + ";" + k);
                    restoreMeas(id3, sm, infos3);
                }
                restoreMeas(id2, sm, infos2);
            }
            restoreMeas(id1, sm, infos1);
        }
        System.out.println("当以下测点数据不存在时，系统无法进行状态估计：");
        System.out.println("共有情况" + ans.size() + "种");
        ans.forEach(System.out::println);
    }

    private MeasureInfo[] deleteOneMeas(String id, SystemMeasure sm) {
        MeasureInfo[] infos = new MeasureInfo[3];
        MeasureInfo v = sm.getBus_v().get(id);
        MeasureInfo p = sm.getBus_p().get(id);
        MeasureInfo q = sm.getBus_q().get(id);

        sm.getBus_v().remove(id);
        sm.getBus_p().remove(id);
        sm.getBus_q().remove(id);

        infos[0] = v;
        infos[1] = p;
        infos[2] = q;
        return infos;
    }

    private void restoreMeas(String id, SystemMeasure sm, MeasureInfo[] infos) {
        sm.getBus_v().put(id, infos[0]);
        sm.getBus_p().put(id, infos[1]);
        sm.getBus_q().put(id, infos[2]);
    }


    private SeResultInfo doSE(IEEEDataIsland island, SystemMeasure sm) {
        //alg.setPrintPath(false);
        double slackBusAngle = island.getBus(island.getSlackBusNum()).getFinalAngle() * Math.PI / 180.;
        alg.setSlackBusAngle(slackBusAngle);

        se.setOriIsland(island);
        se.setSm(sm);
        se.setFlatStart(true);

        SeResultInfo r;

        //下面这一段测试传统最小二乘的效果
        se.doSe();
        r = se.createPfResult();
        Assert.assertNotNull(r);
        //printS1(ref, r, isTrueValue);
        System.out.println("WLS状态估计迭代次数：" + alg.getIterNum() + "\t用时：" + alg.getTimeUsed() + "\t");
        return r;
    }


    private SystemMeasure getSm(List<String[]> injections) {
        IEEEDataIsland island = SeTest_case15.island.clone();
        for (String[] s : injections) {
            int busNum = Integer.parseInt(s[0]) + 1;
            double p = Double.parseDouble(s[3]) / 1000, q = Double.parseDouble(s[4]) / 1000;

            BusData bus = island.getBus(busNum);
            if (p > 0) {
                bus.setLoadMW(p);
                bus.setLoadMVAR(q);
            } else {
                bus.setGenerationMW(-p);
                bus.setGenerationMVAR(-q);
            }

        }
        PolarPf pf = new PolarPf();
        pf.setTol_p(1e-4);
        pf.setTol_q(1e-4);
        pf.setOriIsland(island);
        pf.setDecoupledPqNum(0);
        pf.doPf();
        if (pf.isConverged()) {
            pf.fillOriIslandPfResult();
        } else
            return null;

        PfResultInfo result = pf.createPfResult();
        Assert.assertNotNull(result);

        System.out.println("==打印结果==");
        System.out.println("==BUS==");
        for (Integer i : result.getBusV().keySet()) {
            System.out.println(i + " " + result.getBusV().get(i) + " " + result.getBusTheta().get(i) + " " + result.getBusPGen().get(i)
                    + " " + result.getBusQGen().get(i) + " " + result.getBusPLoad().get(i) + " " + result.getBusQLoad().get(i) + " " + result.getBusP().get(i)
                    + " " + result.getBusQ().get(i));
        }
        System.out.println("==BRANCH==");
        for (Integer i : result.getBranchPLoss().keySet()) {
            System.out.println(i + " " + result.getBranchPLoss().get(i) + " " + result.getBranchQLoss().get(i));
        }
        System.out.println("\n" + result.getGenPCapacity() + "\n" + result.getGenQCapacity() + "\n" + result.getGenPTotal());

        return SimuMeasMaker.createMeasureOfTypes(island, new int[]{
                TYPE_BUS_VOLOTAGE,
                TYPE_BUS_ACTIVE_POWER,
                TYPE_BUS_REACTIVE_POWER,
                TYPE_LINE_TO_ACTIVE,
                TYPE_LINE_TO_REACTIVE
        }, 1, 0.02);
    }


    public static SystemMeasure setMeasValue(Map<Integer, List<String[]>> measValue) throws IOException {
        SystemMeasure meas = new SystemMeasure();
        for (Map.Entry<Integer, List<String[]>> entry : measValue.entrySet()) {
            setMeasValue(entry.getValue(), meas, entry.getKey());
        }
        return meas;
    }

    private static void setMeasValue(List<String[]> measValue, SystemMeasure meas, int measType) {
        Map<String, MeasureInfo> container = meas.getContainer(measType);
        List<MeasureInfo> list = new ArrayList<>();
        setMeasValue(measValue, list, measType);
        for (MeasureInfo info : list)
            container.put(info.getPositionId(), info);
    }

    private static void setMeasValue(List<String[]> measValue, List<MeasureInfo> container, int measType) {
        for (String[] s : measValue) {
            MeasureInfo measure = new MeasureInfo();
            measure.setPositionId(s[0]);
            measure.setValue(Double.parseDouble(s[1]));
            container.add(measure);
            if (s.length == 3 || s.length > 4)
                measure.setGenMVA(Double.parseDouble(s[s.length - 1]));
            if (s.length > 3) {
                measure.setSigma(Double.parseDouble(s[2]));
                measure.setWeight(Double.parseDouble(s[3]));
                if (s.length > 4)
                    measure.setValue_true(Double.parseDouble(s[4]));
            }
            measure.setMeasureType(measType);
        }
    }
}
