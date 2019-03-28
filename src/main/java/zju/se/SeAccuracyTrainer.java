package zju.se;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.measure.MeasVector;
import zju.measure.SystemMeasure;
import zju.pf.MonteCarloCaseBuilder;
import zju.pf.SimuMeasMaker;
import zju.util.StateCalByPolar;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class SeAccuracyTrainer {

    private final static Logger logger = LogManager.getLogger(SeAccuracyTrainer.class);

    private SeAccuracyTrainModel model;

    public SeAccuracyTrainer() {
    }

    public SeAccuracyTrainer(SeAccuracyTrainModel model) {
        this.model = model;
    }

    /**
     * 产生样本进行模型训练
     *
     * @param oriIsland 初始电气岛
     * @param num       测试样本个数
     */
    public void trainModel(final IEEEDataIsland oriIsland, int num) {
        IEEEDataIsland[] islands = MonteCarloCaseBuilder.simulatePowerFlow(oriIsland, num);
        List<double[]> attributes = new ArrayList<>();
        List<Double> labels = new ArrayList<>();

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
            //dealZeroInjection(se, alg, sm, ref, false);
            alg.setVariable_type(IpoptSeAlg.VARIABLE_VTHETA);
            se.doSe();

            if (alg.isConverged) {
                MeasVector meas = alg.getMeas();
                attributes.add(meas.z.getValues());
                double[] estVar = alg.getVariableState();
                double label = 0;
                for (int j = 0; j < trueVar.length; j++) {
                    label += Math.pow(estVar[j] - trueVar[j], 2);
                }
                label = Math.sqrt(label);
                labels.add(label);
            }
        }
        model.fit(attributes, labels);
    }

    public void trainModel(InputStream stream) {
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        List<double[]> attributes = new ArrayList<>();
        List<Double> labels = new ArrayList<>();

        try {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                String[] ss = line.split(" ");
                double[] attribute = new double[ss.length - 1];
                for (int i = 0; i < attribute.length; i++)
                    attribute[i] = Double.parseDouble(ss[i]);
                attributes.add(attribute);
                labels.add(Double.parseDouble(ss[ss.length - 1]));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        model.fit(attributes, labels);
    }


    /**
     * 预测模型
     *
     * @param attribute 样本特征向量
     * @return 预测值
     */
    public double predict(double[] attribute) {
        return model.predict(attribute);
    }

    public List<Double> predict(List<double[]> attributes) {
        List<Double> list = new ArrayList<>();
        for (double[] attribute : attributes) {
            list.add(model.predict(attribute));
        }
        return list;
    }

    public List<Double> predict(InputStream stream) {
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        List<double[]> attributes = new ArrayList<>();

        try {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                String[] ss = line.split(" ");
                double[] attribute = new double[ss.length - 1];
                for (int i = 0; i < attribute.length; i++)
                    attribute[i] = Double.parseDouble(ss[i]);
                attributes.add(attribute);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return predict(attributes);
    }

    public void setModel(SeAccuracyTrainModel model) {
        this.model = model;
    }

    public static void writeDataSet(final IEEEDataIsland oriIsland, int num, String path, int measNum) {
        IEEEDataIsland[] islands = MonteCarloCaseBuilder.simulatePowerFlow(oriIsland, num);
        List<String> lines = new ArrayList<>();

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
            //dealZeroInjection(se, alg, sm, ref, false);
            alg.setVariable_type(IpoptSeAlg.VARIABLE_VTHETA);
            se.doSe();

            if (alg.isConverged && alg.getMeas().getZ().getN() == measNum) {
                MeasVector meas = alg.getMeas();
                StringBuilder sb = new StringBuilder();
                double[] z = meas.getZ().getValues();
                for (double v : z) {
                    sb.append(v);
                    sb.append(" ");
                }
                double[] estVar = alg.getVariableState();
                double label = 0;
                for (int j = 0; j < trueVar.length; j++) {
                    label += Math.pow(estVar[j] - trueVar[j], 2);
                }
                label = Math.sqrt(label);
                sb.append(label);
                lines.add(sb.toString());
            }
        }

        try {
            Path file = Paths.get(path);
            Files.write(file, lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void dealZeroInjection(StateEstimator se, IpoptSeAlg alg, SystemMeasure sm, IEEEDataIsland ref, boolean isAddConstraint) {
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

    public static void main(String[] args) {
        writeDataSet(IcfDataUtil.ISLAND_30, 10000, "src/test/resources/sefiles/train/case30.txt", 190);
        writeDataSet(IcfDataUtil.ISLAND_30, 100, "src/test/resources/sefiles/test/case30.txt", 190);
    }
}
