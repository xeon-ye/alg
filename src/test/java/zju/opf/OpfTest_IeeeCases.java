package zju.opf;


import junit.framework.TestCase;
import org.junit.Test;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.pf.PfAlgorithmTest;
import zju.pf.PfResultInfo;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2009-3-24
 */
public class OpfTest_IeeeCases extends TestCase {

    @Test
    public void testStandardCases_pf() {
        IpoptOpf opf = new IpoptOpf();
        opf.setFlatStart(true);
        opf.setTol_p(0.0);
        opf.setTol_q(0.0);
        doPf(IcfDataUtil.ISLAND_14.clone(), opf);
        doPf(IcfDataUtil.ISLAND_39.clone(), opf);
        doPf(IcfDataUtil.ISLAND_57.clone(), opf);
        //下面三个系统都有PV节点Q越限的问题
        doPf(IcfDataUtil.ISLAND_30.clone(), opf);
        //doPf(IcfDataUtil.ISLAND_118.clone(), opf);
        doPf(IcfDataUtil.ISLAND_300.clone(), opf);
    }

    public void testStandardCases() {
        IpoptOpf opf = new IpoptOpf();
        //doOpf(IcfDataUtil.ISLAND_14.clone(), opf);
        //doOpf(IcfDataUtil.ISLAND_30.clone(), opf);
        //doOpf(IcfDataUtil.ISLAND_39.clone(), opf);
        //doOpf(IcfDataUtil.ISLAND_57.clone(), opf);
        //doOpf(IcfDataUtil.ISLAND_118.clone(), opf);
        doOpf(IcfDataUtil.ISLAND_300.clone(), opf);
    }

    public void doPf(IEEEDataIsland island, IpoptOpf opf) {
        IEEEDataIsland refence = island.clone();
        OpfPara opfPara = new OpfPara();
        opfPara.setP_ctrl_busno(new int[0]);
        opfPara.setQ_ctrl_busno(new int[0]);
        opfPara.setV_ctrl_busno(new int[0]);
        opfPara.setObjFunction(OpfPara.OBJ_MIN_SUM_P);

        opf.setOriIsland(island);
        opf.setPara(opfPara);
        //opf.doPf();
        //opf.fillOriIslandPfResult();

        //30,118,300三个系统均有PV节点无功越限的情况，这里放开此约束
        for (BusData bus : opf.getClonedIsland().getBuses()) {
            bus.setMinimum(-99999);
            bus.setMaximum(99999);
        }
        long start = System.currentTimeMillis();
        opf.doOpf();
        System.out.println("Total Time used for opf: " + (System.currentTimeMillis() - start) + "ms");
        assertTrue(opf.isConverged());
        opf.fillOriIslandPfResult();

        PfResultInfo result = opf.createPfResult();
        assertNotNull(result);
        PfAlgorithmTest.assertStandardCasePf(island, refence);
    }

    public void doOpf(IEEEDataIsland island, IpoptOpf opf) {
        opf.setvLowerLimPu(0.9);
        opf.setvUpperLimPu(1.1);
        opf.setOriIsland(island);
        OpfPara opfPara = new OpfPara();

        int size = island.getPvBusSize() + island.getSlackBusSize();
        //int pvBusSize = 0;
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
        opfPara.setP_ctrl_busno(pControllableBus);
        opfPara.setQ_ctrl_busno(qControllableBus);
        opfPara.setV_ctrl_busno(vControllableBus);
        opfPara.setObjFunction(OpfPara.OBJ_MIN_SUM_P);

        opf.setPara(opfPara);
        long start = System.currentTimeMillis();
        opf.doOpf();
        System.out.println("Total Time used for opf: " + (System.currentTimeMillis() - start) + "ms");
        assertTrue(opf.isConverged());

        PfResultInfo result = opf.createPfResult();
        assertNotNull(result);
    }
}

