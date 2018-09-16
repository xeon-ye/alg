package zju.dsntp;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.common.NewtonSolver;
import zju.devmodel.MapObject;
import zju.dsmodel.*;
import zju.util.DoubleMatrixToolkit;

import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-10-31
 */
public class DsPowerflow implements DsModelCons {
    private static Logger log = LogManager.getLogger(DsPowerflow.class);

    private int maxIter = 50;
    private double tolerance = 1e-5;
    private boolean isConverged = false;

    private DistriSys distriSys;
    private Map<String, double[][]> rootBusV;
    private double[][] tmpLoadI = new double[3][2];

    public DsPowerflow() {
    }

    public DsPowerflow(DistriSys distriSys) {
        this.distriSys = distriSys;
    }

    public void doPf() {
        for (DsTopoIsland island : distriSys.getActiveIslands()) {
            isConverged = false;
            if (island.isRadical()) {
                //初始化, 开辟内存
                island.initialVariables();
                doRadicalPf(island);
            } else
                doLcbPf(island);
        }
    }

    public void doLcbPf() {
        for (DsTopoIsland island : distriSys.getActiveIslands()) {
            isConverged = false;
            doLcbPf(island);
        }
    }

    public void doOutagePf() {

    }

    /**
     * 使用Loop current based算法的潮流
     * @param island
     */
    public void doLcbPf(DsTopoIsland island) {
        island.initialBusV();
        initialRootV(island);
        island.buildDetailedGraph();
        LcbPfModel newtonModel = new LcbPfModel(island, maxIter, tolerance);
        NewtonSolver solver = new NewtonSolver(newtonModel);
        //solver.setLinearSolver(NewtonSolver.LINEAR_SOLVER_COLT);
        long start = System.nanoTime();
        isConverged = solver.solve();
        log.debug("Time used for 潮流计算 : " + (System.nanoTime() - start) / 1000 + "us");
        if (isConverged) {
            //计算节点电压
            newtonModel.fillStateInIsland();
            island.setVCartesian(true);
            island.setICartesian(true);
            //log.info("潮流计算收敛, 迭代次数: " + solver.getIterNum());
        }
    }

    public void doRadicalPf(DsTopoIsland island) {
        initialRootV(island);

        //计算支路电流
        calBranchI(island);
        int iter = 0;
        while (!isConverged) {
            iter++;
            double maxDelta = Double.MIN_VALUE;

            //计算节点电压
            int size = island.isBalanced() ? 1 : 3;
            double[][] v0 = new double[size][2];
            for (int bus = 1; bus < island.getTns().size(); bus++) {
                DsTopoNode tn = island.getTnNoToTn().get(bus);
                for (int son : tn.getConnectedBusNo()) {
                    if (son < bus)
                        continue;
                    double[][] sonV = island.getBusV().get(island.getTnNoToTn().get(son));
                    for (int i = 0; i < sonV.length; i++) {
                        v0[i][0] = sonV[i][0];
                        v0[i][1] = sonV[i][1];
                    }

                    double[][] fatherV = island.getBusV().get(tn);
                    MapObject edge = island.getGraph().getEdge(tn, island.getTnNoToTn().get(son));
                    GeneralBranch branch = island.getBranches().get(edge);
                    branch.calTailV(fatherV, island.getBranchTailI().get(edge), sonV);

                    for (int i = 0; i < sonV.length; i++) {
                        v0[i][0] -= sonV[i][0];
                        v0[i][1] -= sonV[i][1];
                        double delta = Math.sqrt(v0[i][0] * v0[i][0] + v0[i][1] * v0[i][1]);
                        log.debug("Now voltage delta" + " " + i + " " + delta);
                        if (delta > maxDelta)
                            maxDelta = delta;
                    }
                }
            }
            //判断是否收敛
            if (maxDelta <= tolerance) {
                isConverged = true;
                break;
            }

            if (iter >= maxIter && !isConverged) {
                log.info("达到最大迭代次数仍未收敛.");
                return;
            }
            calBranchI(island);
        }
        island.setVCartesian(true);
        island.setICartesian(true);
        log.info("潮流计算收敛, 迭代次数: " + iter);
    }

    private void initialRootV(DsTopoIsland island) {
        //给跟节点赋电压初值
        if (rootBusV != null) {
            for (DsTopoNode tn : island.getSupplyTns()) {
                for (DsConnectNode cn : tn.getConnectivityNodes()) {
                    if (!rootBusV.containsKey(cn.getId()))
                        continue;
                    double[][] v = rootBusV.get(cn.getId());
                    double baseKv = 1.0;
                    if (distriSys.isPerUnitSys())
                        baseKv = tn.getBaseKv();
                    double vReal = v[0][0] * Math.cos(v[0][1] * Math.PI / 180.0) / baseKv;
                    double vImag = v[0][0] * Math.sin(v[0][1] * Math.PI / 180.0) / baseKv;
                    if (distriSys.isBalanced()) {
                        island.getBusV().get(tn)[0][0] = vReal;
                        island.getBusV().get(tn)[0][1] = vImag;
                        island.getBusV().put(tn, new double[][]{{vReal, vImag}});
                    } else {
                        double vReal2 = v[1][0] * Math.cos(v[1][1] * Math.PI / 180.0) / baseKv;
                        double vImag2 = v[1][0] * Math.sin(v[1][1] * Math.PI / 180.0) / baseKv;
                        double vReal3 = v[2][0] * Math.cos(v[2][1] * Math.PI / 180.0) / baseKv;
                        double vImag3 = v[2][0] * Math.sin(v[2][1] * Math.PI / 180.0) / baseKv;
                        island.getBusV().get(tn)[0][0] = vReal;
                        island.getBusV().get(tn)[0][1] = vImag;
                        island.getBusV().get(tn)[1][0] = vReal2;
                        island.getBusV().get(tn)[1][1] = vImag2;
                        island.getBusV().get(tn)[2][0] = vReal3;
                        island.getBusV().get(tn)[2][1] = vImag3;
                    }
                    break;
                }
            }
        }
    }

    public void calBranchI(DsTopoIsland island) {
        MapObject curEdge = null;
        double[][] tailI;
        double[][] v;
        GeneralBranch branch;
        for (int bus = island.getTns().size(); bus > 1; bus--) {
            DsTopoNode tn = island.getTnNoToTn().get(bus);
            for (int anotherBus : tn.getConnectedBusNo()) {
                if (tn.getTnNo() > anotherBus) {
                    curEdge = island.getGraph().getEdge(tn, island.getTnNoToTn().get(anotherBus));
                    break;
                }
            }
            tailI = island.getBranchTailI().get(curEdge);
            v = island.getBusV().get(tn);
            DoubleMatrixToolkit.makeZero(tailI);
            //计算流入负荷的电流
            for (MapObject obj : tn.getConnectedDev()) {
                if (island.getLoads().containsKey(obj)) {
                    island.getLoads().get(obj).calI(v, tmpLoadI);
                    DoubleMatrixToolkit.selfAdd(tailI, tmpLoadI);
                } else if(island.getDispersedGens().containsKey(obj)) {
                    island.getDispersedGens().get(obj).calI(v, tmpLoadI);
                    DoubleMatrixToolkit.selfAdd(tailI, tmpLoadI);
                }
            }
            //计算流入子节点的电流
            for (int anotherBus : tn.getConnectedBusNo()) {
                if (tn.getTnNo() < anotherBus) {
                    MapObject subEdge = island.getGraph().getEdge(tn, island.getTnNoToTn().get(anotherBus));
                    DoubleMatrixToolkit.selfAdd(tailI, island.getBranchHeadI().get(subEdge));
                }
            }
            //计算首端的电流
            branch = island.getBranches().get(curEdge);
            branch.calHeadI(v, tailI, island.getBranchHeadI().get(curEdge));
        }
    }

    public int getMaxIter() {
        return maxIter;
    }

    public void setMaxIter(int maxIter) {
        this.maxIter = maxIter;
    }

    public double getTolerance() {
        return tolerance;
    }

    public void setTolerance(double tolerance) {
        this.tolerance = tolerance;
    }

    public boolean isConverged() {
        return isConverged;
    }

    public Map<String, double[][]> getRootBusV() {
        return rootBusV;
    }

    public void setRootBusV(Map<String, double[][]> rootBusV) {
        this.rootBusV = rootBusV;
    }

    public DistriSys getDistriSys() {
        return distriSys;
    }

    public void setDistriSys(DistriSys distriSys) {
        this.distriSys = distriSys;
    }
}
