package zju.pf;

import junit.framework.TestCase;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.alg.ConnectivityInspector;
import org.jgrapht.graph.DefaultEdge;
import zju.ieeeformat.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Powerflow Tester.
 *
 * @author <Dong Shufeng>
 * @version 1.0
 * @since <pre>08/09/2008</pre>
 */
public class PfAlgorithmTest extends TestCase {
    public PfAlgorithmTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
    }

    public void testStandardCases_ipopt() {
        PolarPf pf = new IpoptPf();
        pf.setTol_p(0.0);
        pf.setTol_q(0.0);
        pf.setTol_v(0.0);

        standardCasePf(IcfDataUtil.ISLAND_14.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_30.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_39.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_57.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_118.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_300.clone(), pf);
    }

    public void testStandardCases_OutagePf_ipopt() {
        PolarPf pf = new IpoptPf();
        pf.setPrintPath(false);
        pf.setTol_p(0.0);
        pf.setTol_q(0.0);
        pf.setTol_v(0.0);
        testOutagePf(IcfDataUtil.ISLAND_14.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_30.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_39.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_57.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_118.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_300.clone(), pf);
    }

    public void testStandardCases_newton() {
        PolarPf pf = new PolarPf(PfConstants.ALG_NEWTON);
        pf.setTol_p(0.001);
        pf.setTol_q(0.001);
        pf.setDecoupledPqNum(0);
        standardCasePf(IcfDataUtil.ISLAND_14.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_30.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_39.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_57.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_118.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_300.clone(), pf);
    }

    public void testStandardCases_OutagePf_newton() {
        PolarPf pf = new PolarPf(PfConstants.ALG_NEWTON);
        pf.setDecoupledPqNum(0);
        testOutagePf(IcfDataUtil.ISLAND_14.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_30.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_39.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_57.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_118.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_300.clone(), pf);
    }

    public void testStandardCases_pq() {
        PolarPf pf = new PolarPf(PfConstants.ALG_PQ_DECOUPLED);
        pf.setXB(true);
        standardCasePf(IcfDataUtil.ISLAND_14.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_30.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_39.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_57.clone(), pf);
        //standardCasePf(IcfDataUtil.ISLAND_118.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_300.clone(), pf);

        pf.setXB(false);
        standardCasePf(IcfDataUtil.ISLAND_14.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_30.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_39.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_57.clone(), pf);
        //standardCasePf(IcfDataUtil.ISLAND_118.clone(), pf);
        standardCasePf(IcfDataUtil.ISLAND_300.clone(), pf);
    }

    public void testStandardCases_OutagePf_pq() {
        PolarPf pf = new PolarPf(PfConstants.ALG_PQ_DECOUPLED);

        testOutagePf(IcfDataUtil.ISLAND_14.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_30.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_39.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_57.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_118.clone(), pf);
        testOutagePf(IcfDataUtil.ISLAND_300.clone(), pf);
    }

    public void testCaseDB() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/ieee_orginal.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "GBK");

        PolarPf pf = new PolarPf(PfConstants.ALG_PQ_DECOUPLED);
        pf.setXB(true);
        testPfConvergence(island, pf);
        pf.setXB(false);
        testPfConvergence(island, pf);
        pf.setPfMethod(PfConstants.ALG_NEWTON);
        testPfConvergence(island, pf);
    }

    public void testCaseSJ() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/caseSJ.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "GBK");

        PolarPf pf = new PolarPf(PfConstants.ALG_NEWTON);
        testPfConvergence(island, pf);
    }

    public void testCaseSh() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/shanghai_ieee.dat");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "GBK");

        PolarPf pf = new PolarPf(PfConstants.ALG_PQ_DECOUPLED);
        pf.setXB(true);
        testPfConvergence(island, pf);
        pf.setXB(false);
        testPfConvergence(island, pf);
    }

    public void testCaseCq() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/ieee-cq-2014-03.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile,"UTF-8");
        //for (BusData bus : island.getBuses())
        //    if (bus.getType() == BusData.BUS_TYPE_GEN_PV)
        //        bus.setType(BusData.BUS_TYPE_LOAD_PQ);
        for (BranchData b : island.getBranches()) {
            //if (Math.abs(b.getBranchR()) > 1)
            //    b.setBranchR(0.0001);
            //if (Math.abs(b.getBranchX()) > 1 || b.getBranchX() < 0.0)
            //    b.setBranchX(0.01);
            if (b.getType() == BranchData.BRANCH_TYPE_TF_FIXED_TAP)
                b.setBranchR(0.0001);
            if (b.getType() == BranchData.BRANCH_TYPE_TF_FIXED_TAP && Math.abs(b.getBranchX()) > 0.8)
                b.setBranchX(0.05);
        }

        PolarPf pf = new PolarPf();
        pf.setPfMethod(PfConstants.ALG_NEWTON);
        testPfConvergence(island, pf);
    }
    public void testCaseSZ2015() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/SZ_IEEE.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile,"UTF-8");
        for (BranchData b : island.getBranches()) {
            //if (b.getType() == BranchData.BRANCH_TYPE_TF_FIXED_TAP)
            //    b.setBranchR(0.0001);
            //if (b.getType() == BranchData.BRANCH_TYPE_TF_FIXED_TAP && Math.abs(b.getBranchX()) > 0.8)
            //    b.setBranchX(0.05);
            //b.setBranchR(0.0001);
            //b.setBranchX(0.05);
        }

        PolarPf pf = new PolarPf();
        pf.setPfMethod(PfConstants.ALG_NEWTON);
        testPfConvergence(island, pf);
        assertNotNull(pf.createPfResult());
    }

    //public void testCase2746() {
    //    InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/case2746wop_ieee.txt");
    //    IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "GBK");
    //    testPfConvergence(island);
    //}

    public void testPf_anhui() {
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/sdxx201307081415.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");
        testPf_anhui(island);

        //ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/ahxx201309251500.txt");
        //island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");
        //testPf_anhui(island);

        //ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/ahxx201312041630.txt");
        //island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");
        //testPf_anhui(island);
    }

    public static void testPf_anhui(IEEEDataIsland island) {
        IpoptPf pf = new IpoptPf();

        pf.setPfMethod(PfConstants.ALG_IPOPT);
        pf.setTol_p(0.005); //????????????????????????
        pf.setTol_q(0.001); //????????????????????????
        pf.setTol_v(0.001); //????????????????????????
        pf.setTolerance(0.001);
        pf.setPrintPath(false);
        IEEEDataIsland island1 = island.clone();
        testPfConvergence(island1, pf);
        pf.fillOriIslandPfResult();

        pf.setPfMethod(PfConstants.ALG_NEWTON);
        pf.setDecoupledPqNum(1);
        pf.setXB(false);
        IEEEDataIsland island2 = island.clone();
        testPfConvergence(island2, pf);
        pf.fillOriIslandPfResult();

        ////???????????????????????????????????????????????????
        for (int i = 0; i < island1.getBuses().size(); i++) {
            BusData bus1 = island1.getBuses().get(i);
            BusData bus2 = island2.getBuses().get(i);
            assertEquals(bus1.getBusNumber(), bus2.getBusNumber());
            assertEquals(bus1.getType(), bus2.getType());
            assertEquals(bus1.getMaximum(), bus2.getMaximum());
            assertEquals(bus1.getMinimum(), bus2.getMinimum());
            double v1 = bus1.getFinalVoltage();
            double a1 = bus1.getFinalAngle();
            double genP1 = bus1.getGenerationMW();
            double genQ1 = bus1.getGenerationMVAR();
            double loadP1 = bus1.getLoadMW();
            double loadQ1 = bus1.getLoadMVAR();

            double v2 = bus2.getFinalVoltage();
            double a2 = bus2.getFinalAngle();
            double genP2 = bus2.getGenerationMW();
            double genQ2 = bus2.getGenerationMVAR();
            double loadP2 = bus2.getLoadMW();
            double loadQ2 = bus2.getLoadMVAR();

            //assertTrue(Math.abs(v1 - v2) < 1e-2);
            //assertTrue(Math.abs(a1 - a2) < 1e-2);
            //assertTrue(Math.abs(genP1 - genP2) / island.getTitle().getMvaBase() < 1e-2);
            //assertTrue(Math.abs(genQ1 - genQ2) / island.getTitle().getMvaBase() < 1e-2);
            //assertTrue(Math.abs(loadP1 - loadP2) / island.getTitle().getMvaBase() < 1e-2);
            //assertTrue(Math.abs(loadQ1 - loadQ2) / island.getTitle().getMvaBase() < 1e-2);
        }
    }

    public void testOutagePf_anhui() throws IOException {
        //InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/sdxx201307081415.txt");
        InputStream ieeeFile = this.getClass().getResourceAsStream("/ieeefiles/ahxx201312041630.txt");
        IEEEDataIsland island = new DefaultIcfParser().parse(ieeeFile, "UTF-8");
        InputStream outageBranches = this.getClass().getResourceAsStream("/other/anhuiBranchIds.txt");
        BufferedReader r = new BufferedReader(new InputStreamReader(outageBranches));
        List<Integer> branchIds = new ArrayList<Integer>(2000);
        String str;
        while ((str = r.readLine()) != null)
            branchIds.add(Integer.parseInt(str));
        r.close();

        //????????????????????????PQ???????????????????????????????????????????????????????????????????????????????????????
        PolarPf pf = new IpoptPf();
        pf.setPrintPath(false);
        //pf.setPfMethod(PfConstants.ALG_NEWTON);

        //??????????????????PV???PQ?????????
        //pf.setHandleQLim(true);
        //????????????PV???PQ??????????????????????????????
        //pf.setMaxQLimWatchTime(15);
        //??????????????????(N - 1)
        testOutagePf(island, branchIds, pf);
    }

    public void testOutagePf(IEEEDataIsland island, PolarPf pf) {
        testOutagePf(island, null, pf);
    }

    public void testOutagePf(IEEEDataIsland island, List<Integer> branchIds, PolarPf pf) {
        //?????????????????????????????????????????????
        pf.setOriIsland(island);
        pf.doPf();
        assertTrue(pf.isConverged());
        //????????????????????????PQ????????????????????????????????????????????????????????????
        //?????????????????????PQ?????????????????????
        pf.setDecoupledPqNum(0);
        //???????????????????????????????????????????????????
        pf.setPfMethod(PfConstants.ALG_NEWTON);

        //???????????????????????????????????????---???
        UndirectedGraph<BusData, DefaultEdge> g = IcfTopoUtil.createGraph(island);
        ConnectivityInspector<BusData, DefaultEdge> inspector = new ConnectivityInspector<BusData, DefaultEdge>(g);
        //????????????????????????
        assertEquals(true, inspector.isGraphConnected());

        Map<Integer, BusData> busMap = island.getBusMap();
        long start = System.currentTimeMillis();
        pf.setOutagePf(true);
        pf.setHandleQLim(false);
        //pf.setMaxIter(10);
        int count = 0;
        int converged = 0;
        if (branchIds == null) {
            branchIds = new ArrayList<Integer>(island.getBranches().size());
            for (BranchData b : island.getBranches())
                branchIds.add(b.getId());
        }
        for (int branchId : branchIds) {
            BranchData branch = island.getBranches().get(branchId - 1);

            BusData bus1 = busMap.get(branch.getTapBusNumber());
            BusData bus2 = busMap.get(branch.getZBusNumber());
            DefaultEdge edge = g.getEdge(bus1, bus2);

            //??????????????? ?????????/???????????????????????????????????????????????????????????????
            //????????????????????????????????????????????????????????????????????????????????????
            if (g.getEdgeWeight(edge) < 1.5) {
                assertEquals(1.0, g.getEdgeWeight(edge));
                //?????????????????????
                g.removeEdge(edge);
                //????????????????????????????????????????????????????????????????????????
                inspector = new ConnectivityInspector<BusData, DefaultEdge>(g);
                if (!inspector.isGraphConnected()) {
                    //?????????????????????????????????????????????????????????????????????????????????????????????????????????
                    //????????????????????????????????????????????????????????????????????????????????????????????????
                    //List<Set<BusData>> islands = inspector.connectedSets();
                    //for(Set<BusData> aIsland : islands) {
                    //    IEEEDataIsland aIeeeIsland = new IEEEDataIsland();
                    //    aIeeeIsland.setBuses(new ArrayList<BusData>(aIsland));
                    //    ArrayList<BranchData> branches = new ArrayList<BranchData>();
                    //    aIeeeIsland.setBranches(branches);
                    //    System.out.println(aIsland.size());
                    //}
                    g.addEdge(bus1, bus2, edge);
                    //inspector = new ConnectivityInspector<BusData, DefaultEdge>(g);
                    //assertTrue(inspector.isGraphConnected());
                    continue;
                }
            }
            //pf.setOutageBranches(new int[0]);
            //?????????????????????????????????????????????????????????????????????N - 2????????????????????????N - 1
            pf.setOutageBranches(new int[]{branch.getId()});
            pf.doPf();
            count++;
            //?????????????????????????????????????????????
            g.addEdge(bus1, bus2, edge);
            if (pf.isConverged())
                converged++;
            System.out.println("???" + count + "???????????????.");
            //if (count > 1000)
            //    break;
        }
        System.out.println("????????????" + count + "(" + island.getBranches().size() + "?????????)?????????,??????" + converged + "?????????.");
        System.out.println("??????N-1????????????: " + (System.currentTimeMillis() - start) + " ms");
    }

    public static void testPfConvergence(IEEEDataIsland island, PolarPf pf) {
        pf.setOriIsland(island);
        long start = System.currentTimeMillis();
        System.out.println("?????????: " + island.getBuses().size() + ", ????????????...");
        pf.doPf();
        System.out.println("??????????????????: " + (System.currentTimeMillis() - start) + " ms");
        assertTrue(pf.isConverged());
        pf.fillOriIslandPfResult();
        //assertNotNull(pf.createPfResult());
    }

    public void standardCasePf(IEEEDataIsland island, PolarPf pf) {
        IEEEDataIsland refence = island.clone();

        pf.setOriIsland(island);
        pf.setHandleQLim(false);
        long start = System.currentTimeMillis();
        pf.doPf();
        System.out.println("??????????????????: " + (System.currentTimeMillis() - start) + " ms");
        assertTrue(pf.isConverged());

        PfResultInfo result = pf.createPfResult();
        assertNotNull(result);
        assertStandardCasePf(island, refence);

        System.out.println("==????????????==");
        System.out.println("==BUS==");
        for (Integer i : result.getBusV().keySet()){
            System.out.println(i + " "+ result.getBusV().get(i) + " "+result.getBusTheta().get(i)+" "+result.getBusPGen().get(i)
            +" "+result.getBusQGen().get(i)+" " +result.getBusPLoad().get(i)+" "+result.getBusQLoad().get(i)+" "+result.getBusP().get(i)
            +" "+result.getBusQ().get(i));
        }
        System.out.println("==BRANCH==");
        for (Integer i : result.getBranchPLoss().keySet()){
            System.out.println(i+" "+result.getBranchPLoss().get(i)+" "+result.getBranchQLoss().get(i));
        }

        System.out.println("\n"+result.getGenPCapacity()+"\n"+result.getGenQCapacity()+"\n"+result.getGenPTotal());

    }

    public static void assertStandardCasePf(IEEEDataIsland island, IEEEDataIsland refence) {
        //testResultGetter(result);
        //new IcfWriter(pf.getResult().getIsland()).writeFile("result.txt");
        for (int i = 0; i < island.getBuses().size(); i++) {
            BusData referBus = refence.getBuses().get(i);
            BusData bus = island.getBuses().get(i);
            double deltaV = Math.abs(bus.getFinalVoltage() - referBus.getFinalVoltage());
            double deltaTheta = Math.abs(bus.getFinalAngle() - referBus.getFinalAngle()) * Math.PI / 180.0;
            if (deltaV > 1e-2 || deltaTheta > 1e-1) {
                System.out.println("refrence :" + referBus.getFinalVoltage() + "\t" + referBus.getFinalAngle());
                System.out.println("pf result:" + bus.getFinalVoltage() + "\t" + bus.getFinalAngle());
            }
            if (referBus.getType() == BusData.BUS_TYPE_SLACK) {
                assertTrue(deltaTheta < 1e-6);
                assertTrue(deltaV < 1e-5);
            } else
                assertTrue(deltaTheta < 1e-1);
            assertTrue(deltaV < 0.02);
        }
    }

    public static void testResultGetter(PfResultInfo result) {
        double baseMVA = result.getIsland().getTitle().getMvaBase();
        double pGround = 0;
        double qGround = 0;
        for (BusData bus : result.getIsland().getBuses()) {
            Double v = result.getBusV().get(bus.getBusNumber());
            qGround += -v * v * bus.getShuntSusceptance() * baseMVA;
            pGround += v * v * bus.getShuntConductance() * baseMVA;
        }
        double v = Math.abs(result.getGenPTotal() - result.getLoadPTotal() - result.getPLossTotal() - pGround);
        assertTrue(v < 0.01);
        double v1 = Math.abs(result.getGenQTotal() - result.getLoadQTotal() - result.getQLossTotal() - qGround);
        assertTrue(v1 < 0.01);
    }
}
