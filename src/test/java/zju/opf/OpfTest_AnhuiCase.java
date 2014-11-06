package zju.opf;

import junit.framework.TestCase;
import zju.ieeeformat.BusData;
import zju.ieeeformat.DefaultIcfParser;
import zju.ieeeformat.IEEEDataIsland;
import zju.pf.IpoptPf;
import zju.pf.PfAlgorithmTest;
import zju.pf.PfResultInfo;
import zju.pf.PolarPf;

import java.io.InputStream;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-10-15
 */
public class OpfTest_AnhuiCase extends TestCase {

    public void testAnhuiCase2() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/ahxx201312041630.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");

        //先算一遍潮流
        PolarPf pf = new IpoptPf();
        //设置是否进行PV转PQ的过程
        pf.setOriIsland(island);
        //计算潮流
        pf.doPf();
        assertTrue(pf.isConverged());
        //将潮流结果赋值到原系统中
        pf.fillOriIslandPfResult();

        //开始做最优潮流
        IpoptOpf opf = new IpoptOpf();
        opf.setOriIsland(island.clone());
        //设置不是做潮流
        opf.setPfOnly(false);
             /* 设置OPF收敛精度,实际上是设置了Ipopt中"tol"的数值*/
        opf.setTolerance(1e-4);
        //设置最大迭代次数
        opf.setMaxIter(100);
        //opf.setpLowerLimPer(0.8);
        //opf.setpUpperLimPer(1.2);
        //设置电压上下限
        opf.setvLowerLimPu(0.8);
        opf.setvUpperLimPu(1.2);

        //设置有功、无功、电压可控的母线，这里认为所有PV节点和平衡节点都是可控的
        int size = island.getPvBusSize() + island.getSlackBusSize();
        int[] pControllableBus = new int[size];
        int[] qControllableBus = new int[size];
        int[] vControllableBus = new int[size];
        int i = 0;
        for (BusData busdata : island.getBuses()) {
            int type = busdata.getType();
            if (type == BusData.BUS_TYPE_GEN_PV || type == BusData.BUS_TYPE_SLACK) {
                pControllableBus[i] = busdata.getBusNumber();
                qControllableBus[i] = busdata.getBusNumber();
                vControllableBus[i] = busdata.getBusNumber();
                i++;
            }
        }
        //最优潮流的参数
        OpfPara opfPara = new OpfPara();
        opfPara.setP_ctrl_busno(pControllableBus);
        opfPara.setQ_ctrl_busno(qControllableBus);
        opfPara.setV_ctrl_busno(vControllableBus);
        //opfPara.setObjFunction(OpfPara.OBJ_MIN_SUM_P);
        opfPara.setObjFunction(OpfPara.OBJ_MIN_VOLTAGE_OUTLIMIT);

        //设置参数
        opf.setPara(opfPara);
        long start = System.currentTimeMillis();
        opf.doOpf();
        System.out.println("Total Time used for opf: " + (System.currentTimeMillis() - start) + "ms");
        assertTrue(opf.isConverged());

        //获得优化后的潮流结果，注意该方法会将潮流结果赋值到IeeeIsland中
        PfResultInfo result = opf.createPfResult();
        assertNotNull(result);
    }

    public void testAnhuiCase() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/sdxx201307081415.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");

        //先算一遍潮流
        PolarPf pf = new PolarPf();
        pf.setDecoupledPqNum(0);
        //设置是否进行PV转PQ的过程
        //pf.setHandleQLim(true);
        //pf.setMaxQLimWatchTime(20);

        pf.setOriIsland(island);
        //计算潮流
        pf.doPf();
        assertTrue(pf.isConverged());
        //将潮流结果赋值到原系统中
        pf.fillOriIslandPfResult();

        //开始做最优潮流
        IpoptOpf opf = new IpoptOpf();
        opf.setOriIsland(island.clone());
        //设置不是做潮流
        opf.setPfOnly(false);
       /* 设置OPF收敛精度,实际上是设置了Ipopt中"tol"的数值*/
        opf.setTolerance(1e-4);
        //设置最大迭代次数
        opf.setMaxIter(100);
        //opf.setpLowerLimPer(0.8);
        //opf.setpUpperLimPer(1.2);
        //设置电压上下限
        opf.setvLowerLimPu(0.8);
        opf.setvUpperLimPu(1.2);

        //测试最优潮流退化为潮流的情况
        //这种情况下没有可以调节的资源，计算结果应该和潮流一致
        OpfPara opfPara = new OpfPara();
        opfPara.setP_ctrl_busno(new int[0]);
        opfPara.setQ_ctrl_busno(new int[0]);
        opfPara.setV_ctrl_busno(new int[0]);
        opf.setPara(opfPara);
        opf.setTol_p(0.);
        opf.setTol_q(0.);
        opf.doOpf();
        assertTrue(opf.isConverged());
        opf.fillOriIslandPfResult();
        PfResultInfo pfResult = opf.createPfResult();
        PfAlgorithmTest.assertStandardCasePf(pfResult.getIsland(), island);
        //测试结束，这一段在实际中是不需要的，只是为了测试程序的正确性

        //设置有功、无功、电压可控的母线，这里认为所有PV节点和平衡节点都是可控的
        int size = island.getPvBusSize() + island.getSlackBusSize();
        int[] pControllableBus = new int[size];
        int[] qControllableBus = new int[size];
        int[] vControllableBus = new int[size];
        int i = 0;
        for (BusData busdata : island.getBuses()) {
            int type = busdata.getType();
            if (type == BusData.BUS_TYPE_GEN_PV || type == BusData.BUS_TYPE_SLACK) {
                pControllableBus[i] = busdata.getBusNumber();
                qControllableBus[i] = busdata.getBusNumber();
                vControllableBus[i] = busdata.getBusNumber();
                i++;
            }
        }
        //最优潮流的参数
        opfPara = new OpfPara();
        opfPara.setP_ctrl_busno(pControllableBus);
        opfPara.setQ_ctrl_busno(qControllableBus);
        opfPara.setV_ctrl_busno(vControllableBus);
        opfPara.setObjFunction(OpfPara.OBJ_MIN_SUM_P);
        //opfPara.setObjFunction(OpfPara.OBJ_MIN_VOLTAGE_OUTLIMIT);

        //设置参数
        opf.setPara(opfPara);
        long start = System.currentTimeMillis();
        opf.doOpf();
        System.out.println("Total Time used for opf: " + (System.currentTimeMillis() - start) + "ms");
        assertTrue(opf.isConverged());

        //获得优化后的潮流结果，注意该方法会将潮流结果赋值到IeeeIsland中
        PfResultInfo result = opf.createPfResult();
        assertNotNull(result);

        int qOverLimCount = 0;
        for (BusData bus : result.getIsland().getBuses()) {
            if (bus.getType() == BusData.BUS_TYPE_GEN_PV) {  //PV nodes
                int busNum = bus.getBusNumber();
                double measQ = result.getBusQGen().get(busNum);
                double maxQ = bus.getMaximum();
                double minQ = bus.getMinimum();
                //判断该节点是否正确设置了无功上下限
                if (maxQ - minQ < 1e-1)
                    continue;
                if (measQ - maxQ > 1) {
                    qOverLimCount++;
                } else if (minQ - measQ > 1) {
                    qOverLimCount++;
                }
            }
        }
        System.out.println("共有" + qOverLimCount + "个PV节点无功越限.");
    }
}

