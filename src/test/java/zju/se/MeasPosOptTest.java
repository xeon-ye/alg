package zju.se;

import junit.framework.TestCase;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsTopoIsland;
import zju.dsmodel.DsTopoNode;
import zju.dsmodel.IeeeDsInHand;
import zju.dspf.DsPowerflowTest;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.AVector;
import zju.measure.MeasTypeCons;
import zju.measure.MeasVectorCreator;
import zju.measure.MeasureFileRw;
import zju.measure.SystemMeasure;
import zju.planning.MeasPosOpt;
import zju.planning.MeasPosOptByBonmin;
import zju.util.MathUtil;
import zju.util.StateCalByPolar;
import zju.util.YMatrixGetter;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static zju.dsmodel.DsModelCons.sqrt3;

/**
 * Created by arno on 16-7-31.
 * @author Dong Shufeng
 */
public class MeasPosOptTest extends TestCase implements MeasTypeCons {


    public void testCase4() {
        MeasPosOpt mpo = new MeasPosOptByBonmin(SeTest_case4.island);

        SystemMeasure sm = MeasureFileRw.parse(this.getClass().getResourceAsStream("/measfiles/case4_meas.txt"));
        MeasVectorCreator mc = new MeasVectorCreator();
        mc.getMeasureVector(sm);

        int[] candPos = new int[2];
        candPos[0] = 2;
        candPos[1] = 4;
        int[][] measTypePerPos = new int[candPos.length][];
        double[][] weights = new double[candPos.length][];
        for(int i = 0; i < measTypePerPos.length; i++) {
            measTypePerPos[i] = new int[3];
            measTypePerPos[i][0] = TYPE_BUS_ACTIVE_POWER;
            measTypePerPos[i][1] = TYPE_BUS_REACTIVE_POWER;
            measTypePerPos[i][2] = TYPE_BUS_VOLOTAGE;
            weights[i] = new double[measTypePerPos[i].length];
            weights[i][0] = 0.58;
            weights[i][1] = 0.50;
            weights[i][2] = 0.80;
        }

        mpo.setCandPos(candPos);
        mpo.setMeasTypesPerPos(measTypePerPos);
        mpo.setMeasWeight(weights);
        mpo.setExistMeasPos(mc.measPos);
        mpo.setExistMeasTypes(mc.measTypes);
        mpo.setExistMeasWeight(mc.weights);
        mpo.setMaxDevNum(1);

        mpo.doOpt(false);
    }

    public void testDscase4Trans() {
        InputStream ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsfiles/case4-notrans.txt");
        DistriSys sys = IeeeDsInHand.createDs(ieeeFile, "1", 12.47 / sqrt3);
        sys.setPerUnitSys(true); //设置使用标幺值标志
        sys.setBaseKva(1000.); //设置基准功率
        //计算潮流
        DsPowerflowTest.testConverged(sys, false);
        //得到电气岛
        DsTopoIsland dsIsland = sys.getActiveIslands()[0];
        //生成详细拓扑图
        dsIsland.buildDetailedGraph();
        //将配电网转换成等效的IEEE格式
        HashMap<String, BranchData[]> devIdToBranch = new HashMap<>(dsIsland.getBranches().size());
        HashMap<String, BusData> vertexToBus = new HashMap<>(dsIsland.getDetailedG().vertexSet().size());
        IEEEDataIsland island = dsIsland.toIeeeIsland(devIdToBranch, vertexToBus, false);
        //将潮流结果赋值到转换后的模型
        int n = island.getBuses().size();
        AVector state = new AVector(n * 2);
        for(DsTopoNode tn : dsIsland.getBusV().keySet()) {
            double[][] busV = dsIsland.getBusV().get(tn);
            for(int phase : tn.getPhases()) {
                BusData bus = vertexToBus.get(tn.getTnNo() + "-" + phase);
                double[] v = new double[]{busV[phase][0], busV[phase][1]};
                //转换成极坐标
                MathUtil.trans_rect2polar(v);
                //已经是标幺值
                state.setValue(bus.getBusNumber() - 1, v[0]);
                state.setValue(bus.getBusNumber() + n - 1, v[1]);
            }
        }

        //比较潮流结果
        YMatrixGetter Y = new YMatrixGetter(island);
        Y.formYMatrix();
        for(String devId: devIdToBranch.keySet()) {
            MapObject br = dsIsland.getIdToBranch().get(Integer.parseInt(devId));
            DsTopoNode tn = dsIsland.getGraph().getEdgeSource(br);
            double[][] v = dsIsland.getBusV().get(tn);
            for(int phase : tn.getPhases()) {
                BusData bus = vertexToBus.get(tn.getTnNo() + "-" + phase);
                double p = 0, q = 0;
                for (BranchData branch : devIdToBranch.get(devId)) {
                    if (branch.getTapBusNumber() == bus.getBusNumber()) {
                        p += StateCalByPolar.calLinePFrom(branch.getId(), Y, state);
                        q += StateCalByPolar.calLineQFrom(branch.getId(), Y, state);
                    } else if (branch.getZBusNumber() == bus.getBusNumber()) {
                        p += StateCalByPolar.calLinePTo(branch.getId(), Y, state);
                        q += StateCalByPolar.calLineQTo(branch.getId(), Y, state);
                    }
                }
                double[][] c = dsIsland.getBranchHeadI().get(br);
                double p1 = v[phase][0] * c[phase][0] + v[phase][1] * c[phase][1];
                double q1 = v[phase][1] * c[phase][0] - v[phase][0] * c[phase][1];
                assertEquals(p, p1, 1e-6);
                assertEquals(q, q1, 1e-6);
            }
        }
    }

    public void testDscase4() {
        InputStream ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsfiles/case4-notrans.txt");
        DistriSys sys = IeeeDsInHand.createDs(ieeeFile, "1", 12.47 / sqrt3);
        sys.setPerUnitSys(true); //设置使用标幺值标志
        sys.setBaseKva(1000.); //设置基准功率
        //计算潮流
        //DsPowerflowTest.testConverged(sys, false);
        sys.buildDynamicTopo();
        sys.createCalDevModel();
        DsTopoIsland dsIsland = sys.getActiveIslands()[0];
        //形成量测
        //DsSimuMeasMaker smMaker = new DsSimuMeasMaker();
        //SystemMeasure sm = smMaker.createFullMeasure(dsIsland, 1, 0.02);
        //MeasureFileRw.writeFileSimple(sm, "/home/arno/alg/src/test/resources/dsfiles/case4-notrans-measure.txt");
        //读取量测
        SystemMeasure sm = MeasureFileRw.parse(getClass().getResourceAsStream("/dsfiles/case4-notrans-measure.txt"));
        MeasVectorCreator mc = new MeasVectorCreator();
        mc.getMeasureVector(sm, true);

        MeasPosOpt mpo = new MeasPosOpt(dsIsland, false);
        mpo.setDs_existMeasPos(mc.measPosWithPhase);
        mpo.setDs_existMeasTypes(mc.measTypes);
        mpo.setDs_existMeasWeight(mc.weights);

        setIduMeasures(dsIsland, new int[]{1, 3}, new int[]{1, 2}, new double[]{0.58, 0.5, 0.5, 0.58, 0.58, 0.5, 0.3, 0.58,}, mpo);
        //setIduMeasures(dsIsland, new int[]{}, new int[]{}, new double[]{}, mpo);
        mpo.setMaxDevNum(1);
        mpo.doOpt(true);
    }

    public void testDscase13() {
        InputStream ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsfiles/case13-notrans.txt");
        DistriSys sys = IeeeDsInHand.createDs(ieeeFile, "650", 4.16 / sqrt3);
        sys.setPerUnitSys(true); //设置使用标幺值标志
        sys.setBaseKva(1000.); //设置基准功率
        //计算潮流
        //DsPowerflowTest.testConverged(sys, false);
        sys.buildDynamicTopo();
        sys.createCalDevModel();
        DsTopoIsland dsIsland = sys.getActiveIslands()[0];
        //形成量测
        //DsSimuMeasMaker smMaker = new DsSimuMeasMaker();
        //SystemMeasure sm = smMaker.createFullMeasure(dsIsland, 1, 0.02);
        //MeasureFileRw.writeFileSimple(sm, "/home/arno/alg/src/test/resources/dsfiles/case13-notrans-measure.txt");
        //读取量测
        SystemMeasure sm = MeasureFileRw.parse(getClass().getResourceAsStream("/dsfiles/case13-notrans-measure.txt"));
        MeasVectorCreator mc = new MeasVectorCreator();
        mc.getMeasureVector(sm, true);

        MeasPosOpt mpo = new MeasPosOptByBonmin(dsIsland, true);
        //mpo.setVAmplOnly(true);
        mpo.setDs_existMeasPos(mc.measPosWithPhase);
        mpo.setDs_existMeasTypes(mc.measTypes);
        mpo.setDs_existMeasWeight(mc.weights);

        setIduMeasures(dsIsland, new int[]{1,2,3,4}, new int[]{2,3,5,3}, new double[]{0.58, 0.5, 0.5, 0.58, 0.58, 0.5, 0.3, 0.58,}, mpo);
        mpo.setMaxDevNum(2);
        mpo.doOpt(true);
    }

    public void testDscase34() {
        InputStream ieeeFile = IeeeDsInHand.class.getResourceAsStream("/dsfiles/case34-notrans.txt");
        DistriSys sys = IeeeDsInHand.createDs(ieeeFile, "800", 24.9 / sqrt3);
        sys.setPerUnitSys(true); //设置使用标幺值标志
        sys.setBaseKva(1000.); //设置基准功率
        //计算潮流
        //DsPowerflowTest.testConverged(sys, false);
        sys.buildDynamicTopo();
        sys.createCalDevModel();
        DsTopoIsland dsIsland = sys.getActiveIslands()[0];
        //形成量测
        //DsSimuMeasMaker smMaker = new DsSimuMeasMaker();
        //SystemMeasure sm = smMaker.createFullMeasure(dsIsland, 1, 0.02);
        //MeasureFileRw.writeFileSimple(sm, "/home/arno/alg/src/test/resources/dsfiles/case34-notrans-measure.txt");
        //读取量测
        SystemMeasure sm = MeasureFileRw.parse(getClass().getResourceAsStream("/dsfiles/case34-notrans-measure.txt"));
        MeasVectorCreator mc = new MeasVectorCreator();
        mc.getMeasureVector(sm, true);

        MeasPosOpt mpo = new MeasPosOpt(dsIsland, false);
        mpo.setDs_existMeasPos(mc.measPosWithPhase);
        mpo.setDs_existMeasTypes(mc.measTypes);
        mpo.setDs_existMeasWeight(mc.weights);

        setIduMeasures(dsIsland, new int[]{}, new int[]{}, new double[]{}, mpo);
        mpo.setMaxDevNum(1);
        mpo.doOpt(true);
    }

    /**
     * 设置量测位置，以及权重
     * @param island 配电网拓扑分析之后的电气岛
     * @param tnIds 量测所在的节点（节点编号通过广度优先搜索而来，见DsIsland的initialIsland方法）
     * @param branchIds 量测所在的支路（编号方法见上）
     * @param weight 每种量测的权重
     * @param mpo 需要设置的对象
     */
    private void setIduMeasures(DsTopoIsland island, int[] tnIds, int[] branchIds, double[] weight, MeasPosOpt mpo) {

        int busMeasureNum = 4; //某一节点某一相对应的量测数目
        int branchMeasureNum = 4; //某一支路某一相对应的量测数目
        String[][] candPos = new String[tnIds.length][];
        List<int[][]> measTypePerPos = new ArrayList<>(candPos.length);
        double[][] weights = new double[candPos.length][];
        for(int i = 0; i < tnIds.length; i++) {
            int count1 = 0;
            int count2 = 0;
            //首先统计有多少个位置可以装量测
            DsTopoNode tn = island.getTnNoToTn().get(tnIds[i]);
            if(tn != null)
                for(int phase = 0; phase < 3; phase++)
                    if(tn.containsPhase(phase))
                        count1 ++;
            MapObject branchObj = island.getIdToBranch().get(branchIds[i]);
            if(branchObj != null)
                for(int phase = 0; phase < 3; phase++)
                    if(island.getBranches().get(branchObj).containsPhase(phase))
                        count2 ++;
            //开辟内存
            candPos[i] = new String[count1 + count2]; //量测的位置
            weights[i] = new double[count1 * busMeasureNum + count2 * branchMeasureNum];
            int[][] ts = new int[candPos[i].length][];
            measTypePerPos.add(ts);
            int count = 0;
            count1 = 0; count2 = 0;
            //开始设置量测的类型
            if(tn != null) {
                for (int phase = 0; phase < 3; phase++) {
                    if (tn.containsPhase(phase)) {
                        int j = 0;
                        candPos[i][count] = tnIds[i] + "_" + phase;
                        ts[count] = new int[busMeasureNum];
                        ts[count][j++] = TYPE_BUS_ACTIVE_POWER;
                        ts[count][j++] = TYPE_BUS_REACTIVE_POWER;
                        ts[count][j] = TYPE_BUS_VOLOTAGE;
                        ts[count][j] = TYPE_BUS_ANGLE;
                        weights[i][count1 * busMeasureNum] = weight[0];
                        weights[i][count1 * busMeasureNum + 1] = weight[1];
                        weights[i][count1 * busMeasureNum + 2] = weight[2];
                        weights[i][count1 * busMeasureNum + 3] = weight[3];
                        count++;
                        count1++;
                    }
                }
            }
            if(branchObj != null) {
                for (int phase = 0; phase < 3; phase++) {
                    int j = 0;
                    if (island.getBranches().get(branchObj).containsPhase(phase)) {
                        candPos[i][count] = branchIds[i] + "_" + phase;
                        ts[count] = new int[branchMeasureNum];
                        if(island.getGraph().getEdgeSource(branchObj) == tn) {
                            ts[count][j++] = TYPE_LINE_FROM_ACTIVE;
                            ts[count][j++] = TYPE_LINE_FROM_REACTIVE;
                            ts[count][j++] = TYPE_LINE_FROM_CURRENT_REAL;
                            ts[count][j] = TYPE_LINE_FROM_CURRENT_IMAG;
                        } else {
                            ts[count][j++] = TYPE_LINE_TO_ACTIVE;
                            ts[count][j++] = TYPE_LINE_TO_REACTIVE;
                            ts[count][j++] = TYPE_LINE_TO_CURRENT_REAL;
                            ts[count][j] = TYPE_LINE_TO_CURRENT_IMAG;
                        }
                        j = 0;
                        weights[i][count1 * busMeasureNum + count2 * branchMeasureNum + j++] = weight[4];
                        weights[i][count1 * busMeasureNum + count2 * branchMeasureNum + j++] = weight[5];
                        weights[i][count1 * busMeasureNum + count2 * branchMeasureNum + j++] = weight[6];
                        weights[i][count1 * busMeasureNum + count2 * branchMeasureNum + j] = weight[7];
                        count++;
                        count2++;
                    }
                }
            }
        }

        mpo.setDs_candPos(candPos);
        mpo.setDs_measTypesPerPos(measTypePerPos);
        mpo.setDs_measWeight(weights);
    }
}
