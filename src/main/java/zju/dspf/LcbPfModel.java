package zju.dspf;

import cern.colt.matrix.DoubleMatrix2D;
import org.apache.log4j.Logger;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.event.EdgeTraversalEvent;
import org.jgrapht.event.TraversalListenerAdapter;
import org.jgrapht.event.VertexTraversalEvent;
import org.jgrapht.traverse.BreadthFirstIterator;
import zju.common.NewtonModel;
import zju.devmodel.MapObject;
import zju.dsmodel.*;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;

import java.util.*;

/**
 * Loop-Current-Based power flow model
 *
 * @author : Dong Shufeng
 *         Date: 14-1-16
 */
public class LcbPfModel implements NewtonModel, DsModelCons {
    private static Logger log = Logger.getLogger(LcbPfModel.class);

    //最大迭代次数，复变量个数
    private int maxIter, dimension;
    //收敛精度
    private double tol;
    //电气岛
    private DsTopoIsland island;
    //变压器原边所在支路
    private List<DetailedEdge> windingEdges;
    //非恒阻抗支路
    private List<DetailedEdge> nonZLoadEdges;
    //分布式电源支路
    private List<DetailedEdge> dgEdges;
    //存储计算值,环路电压实部，环路电压虚部，变压器电流方程
    private AVector z_est;
    //存储状态变量
    private AVector state;
    //存储偏差
    public double[] delta;
    //编号和id之间的关联
    private Map<Integer, DetailedEdge> noToEdge;
    //id和编号之间的关联
    private Map<DetailedEdge, Integer> edgeToNo;
    //回路矩阵
    private ASparseMatrixLink2D B;
    //Jacobian矩阵的结构
    private ASparseMatrixLink2D jacStruc;
    //Jacobian矩阵的数值
    private MySparseDoubleMatrix2D jacobian;
    //支路id与其电压在状态变量中的位置，针对变压器支路和非恒阻抗负荷
    private Map<DetailedEdge, Integer> edgeIndex;
    //节点id与其编号的关联关系，这个编号不是DsTopoNode
    private Map<String, Integer> vertexIdToNo;
    private Map<Integer, String> vertexNoToId;
    //树节点与其父节点之间的关联关系
    private Map<Integer, Integer> sonToFather;
    //回路个数
    private int loopSize;
    public int varSize;

    //private int[] motorVarNum;

    public LcbPfModel(DsTopoIsland island) {
        this.island = island;
        initialB();

        dimension = B.getM();
        dimension += windingEdges != null ? windingEdges.size() : 0;
        //非恒阻抗支路的电压作为状态变量
        dimension += nonZLoadEdges != null ? nonZLoadEdges.size() : 0;
        varSize = dimension * 2;
        for (DispersedGen dg : island.getDispersedGens().values()) {
            dg.setStateIndex(dimension);
            dimension += dg.getStateSize() / 2;
            varSize += dg.getStateSize();
        }
        int slipCount = 2 * dimension;
        for (DispersedGen dg : island.getDispersedGens().values())
            if (dg.getMode() == DispersedGen.MODE_IM)
                dg.setSlipIndex(slipCount++);
        delta = new double[varSize];
        z_est = new AVector(varSize);
        state = new AVector(varSize);
    }

    public LcbPfModel(DsTopoIsland island, int maxIter, double tol) {
        this(island);
        this.maxIter = maxIter;
        this.tol = tol;
    }

    private void initialB() {
        UndirectedGraph<String, DetailedEdge> g = island.getDetailedG();
        edgeToNo = new HashMap<DetailedEdge, Integer>(g.edgeSet().size());
        noToEdge = new HashMap<Integer, DetailedEdge>(g.edgeSet().size());

        //广度优先便利之后，edgeIdToNo中会填好树枝的数据
        Set<String> vertexSet = g.vertexSet();
        if (vertexSet.size() > 0) {
            BreadthFirstIterator<String, DetailedEdge> iter =
                    new BreadthFirstIterator<String, DetailedEdge>(g, null);
            iter.addTraversalListener(new MyTraversalListener<String, DetailedEdge>(g));
            while (iter.hasNext())
                iter.next();
        }

        loopSize = g.edgeSet().size() - edgeToNo.size();

        int tfWindingCount = 0, loadCount = 0, dgCount = 0;
        //统计变压器支路和非恒阻抗负荷支路的个数
        for (DetailedEdge edge : g.edgeSet()) {
            switch (edge.getEdgeType()) {
                case DetailedEdge.EDGE_TYPE_TF_WINDING:
                case DetailedEdge.EDGE_TYPE_LOAD_TF_MIX:
                    if (edge.isSource())
                        tfWindingCount++;
                    break;
                case DetailedEdge.EDGE_TYPE_LOAD:
                    if (Math.abs(edge.getS_real()) > ZERO_LIMIT
                            || Math.abs(edge.getS_image()) > ZERO_LIMIT
                            || Math.abs(edge.getI_ampl()) > ZERO_LIMIT)
                        loadCount++;
                    break;
                case DetailedEdge.EDGE_TYPE_DG:
                    dgCount++;
                default:
                    break;
            }
        }
        //将变压器支路和非恒阻抗支路电压在状态变量中的位置进行编号
        windingEdges = new ArrayList<DetailedEdge>(tfWindingCount);
        nonZLoadEdges = new ArrayList<DetailedEdge>(loadCount);
        dgEdges = new ArrayList<DetailedEdge>(dgCount);
        edgeIndex = new HashMap<DetailedEdge, Integer>(tfWindingCount + loadCount);
        int tfIndex = 0, loadIndex = 0;
        //状态变量顺序为：回路电流，变压器绕组电压，负荷支路电压，异步电机序电压
        for (DetailedEdge edge : g.edgeSet()) {
            switch (edge.getEdgeType()) {
                case DetailedEdge.EDGE_TYPE_TF_WINDING:
                case DetailedEdge.EDGE_TYPE_LOAD_TF_MIX:
                    if (edge.isSource()) {
                        windingEdges.add(edge);
                        edgeIndex.put(edge, loopSize + tfIndex);
                        tfIndex++;
                    }
                    break;
                case DetailedEdge.EDGE_TYPE_LOAD:
                    if (Math.abs(edge.getS_real()) > ZERO_LIMIT
                            || Math.abs(edge.getS_image()) > ZERO_LIMIT
                            || Math.abs(edge.getI_ampl()) > ZERO_LIMIT) {
                        nonZLoadEdges.add(edge);
                        edgeIndex.put(edge, loopSize + tfWindingCount + loadIndex);
                        loadIndex++;
                    }
                    break;
                case DetailedEdge.EDGE_TYPE_DG:
                    dgEdges.add(edge);
                    break;
                default:
                    break;
            }
        }
        int count = 0, vertexNo1, vertexNo2, tmpNo;
        DetailedEdge tmpEdge;
        B = new ASparseMatrixLink2D(loopSize, g.edgeSet().size());
        //更新回路矩阵
        for (DetailedEdge edge : g.edgeSet()) {
            //该支路是树枝
            if (edgeToNo.containsKey(edge))
                continue;
            vertexNo1 = vertexIdToNo.get(g.getEdgeSource(edge));
            vertexNo2 = vertexIdToNo.get(g.getEdgeTarget(edge));
            if (vertexNo1 < vertexNo2) {
                B.setValue(count, count, getDirection(edge, vertexNo1, vertexNo2, 1.0));
                while (vertexNo1 != vertexNo2) {
                    if (vertexNo1 > vertexNo2) {
                        tmpNo = sonToFather.get(vertexNo1);
                        tmpEdge = g.getEdge(vertexNoToId.get(tmpNo), vertexNoToId.get(vertexNo1));
                        B.setValue(count, edgeToNo.get(tmpEdge) + loopSize, getDirection(tmpEdge, tmpNo, vertexNo1, 1.0));
                        vertexNo1 = sonToFather.get(vertexNo1);
                    } else {
                        tmpNo = sonToFather.get(vertexNo2);
                        tmpEdge = g.getEdge(vertexNoToId.get(tmpNo), vertexNoToId.get(vertexNo2));
                        B.setValue(count, edgeToNo.get(tmpEdge) + loopSize, getDirection(tmpEdge, tmpNo, vertexNo2, -1.0));
                        vertexNo2 = sonToFather.get(vertexNo2);
                    }
                }
            } else {
                B.setValue(count, count, getDirection(edge, vertexNo2, vertexNo1, 1.0));
                while (vertexNo1 != vertexNo2) {
                    if (vertexNo1 > vertexNo2) {
                        tmpNo = sonToFather.get(vertexNo1);
                        tmpEdge = g.getEdge(vertexNoToId.get(tmpNo), vertexNoToId.get(vertexNo1));
                        B.setValue(count, edgeToNo.get(tmpEdge) + loopSize, getDirection(tmpEdge, tmpNo, vertexNo1, -1.0));
                        vertexNo1 = sonToFather.get(vertexNo1);
                    } else {
                        tmpNo = sonToFather.get(vertexNo2);
                        tmpEdge = g.getEdge(vertexNoToId.get(tmpNo), vertexNoToId.get(vertexNo2));
                        B.setValue(count, edgeToNo.get(tmpEdge) + loopSize, getDirection(tmpEdge, tmpNo, vertexNo2, 1.0));
                        vertexNo2 = sonToFather.get(vertexNo2);
                    }
                }
            }
            noToEdge.put(count, edge);
            count++;
        }
        //更新存储支路编号的HashMap
        for (DetailedEdge edge : edgeToNo.keySet())
            noToEdge.put(loopSize + edgeToNo.get(edge), edge);
        for (int edgeNo : noToEdge.keySet())
            edgeToNo.put(noToEdge.get(edgeNo), edgeNo);
    }

    //loopDirect,如果回路方向是节点号小->大，则为1，否则为-1
    private double getDirection(DetailedEdge edge, int vertexNo1, int vertexNo2, double loopDirect) {
        switch (edge.getEdgeType()) {
            case DetailedEdge.EDGE_TYPE_SUPPLIER:
                if (DsTopoIsland.EARTH_NODE_ID.equals(vertexNoToId.get(vertexNo1))) {
                    return loopDirect;
                } else
                    return -loopDirect;
                //变压器电流正方向和书《Distribution System Modeling and Analysis》第二版相同
                //D-GrY接法的电压正方向和书中相反，其余接法均与书中一致
            case DetailedEdge.EDGE_TYPE_TF_WINDING:
            case DetailedEdge.EDGE_TYPE_LOAD_TF_MIX:
                String id1 = vertexNoToId.get(vertexNo1);
                String id2 = vertexNoToId.get(vertexNo2);
                int phase1 = Integer.parseInt(id1.substring(id1.length() - 1));
                int phase2 = Integer.parseInt(id2.substring(id2.length() - 1));
                int mode = ((Transformer) island.getBranches().get(island.getDevices().get(edge.getDevId()))).getConnType();
                switch (mode) {
                    case Transformer.CONN_TYPE_D_GrY:
                        if (phase1 == 2 && phase2 == 0)
                            return -loopDirect;
                        else if (phase2 == 2 && phase1 == 0) {
                            return loopDirect;
                        } else if (phase1 < phase2) {
                            return -loopDirect;
                        } else
                            return loopDirect;
                    case Transformer.CONN_TYPE_Y_D:
                    case Transformer.CONN_TYPE_GrY_GrY:
                    case Transformer.CONN_TYPE_D_D:
                        if (edge.isSource()) {
                            if (phase1 == 2 && phase2 == 0)
                                return loopDirect;
                            else if (phase2 == 2 && phase1 == 0) {
                                return -loopDirect;
                            } else if (phase1 < phase2) {
                                return loopDirect;
                            } else
                                return -loopDirect;
                        } else {
                            if (phase1 == 2 && phase2 == 0)
                                return -loopDirect;
                            else if (phase2 == 2 && phase1 == 0) {
                                return loopDirect;
                            } else if (phase1 < phase2) {
                                return -loopDirect;
                            } else
                                return loopDirect;
                        }
                    default:
                        return loopDirect;
                }
            case DetailedEdge.EDGE_TYPE_FEEDER:
            case DetailedEdge.EDGE_TYPE_LOAD:
            case DetailedEdge.EDGE_TYPE_DG:
                id1 = vertexNoToId.get(vertexNo1);
                id2 = vertexNoToId.get(vertexNo2);
                phase1 = Integer.parseInt(id1.substring(id1.length() - 1));
                phase2 = Integer.parseInt(id2.substring(id2.length() - 1));
                if (phase1 == 2 && phase2 == 0)
                    return loopDirect;
                else if (phase2 == 2 && phase1 == 0) {
                    return -loopDirect;
                } else if (phase1 <= phase2) {//负荷电流电压正方向和书中一致
                    return loopDirect;
                } else
                    return -loopDirect;
            default:
                break;
        }
        return loopDirect;
    }

    @Override
    public int getMaxIter() {
        return maxIter;
    }

    @Override
    public double getTolerance() {
        return tol;
    }

    @Override
    public boolean isTolSatisfied(double[] delta) {
        return false;
    }

    @Override
    public AVector getInitial() {
        fillVInState(windingEdges);
        fillVInState(nonZLoadEdges);
        for (DispersedGen dg : island.getDispersedGens().values())
            dg.initialState(this);
        for (DispersedGen dg : island.getDispersedGens().values())
            dg.updateOuterState(this);
        return state;
    }

    private void fillVInState(List<DetailedEdge> edges) {
        int pos, phase, phase1, phase2, tmpPhase, tnNo1, tnNo2;
        String id1, id2;
        DsTopoNode tn1, tn2, tmpTn;
        for (DetailedEdge edge : edges) {
            id1 = island.getDetailedG().getEdgeSource(edge);
            id2 = island.getDetailedG().getEdgeTarget(edge);
            phase1 = Integer.parseInt(id1.substring(id1.length() - 1));
            phase2 = Integer.parseInt(id2.substring(id2.length() - 1));
            phase = edge.getPhase();
            pos = edgeIndex.get(edge);
            if (phase1 == 3) {//大地节点或是不接地Y接法中心点
                tnNo2 = Integer.parseInt(id2.substring(0, id2.lastIndexOf("-")));
                tn2 = island.getBusNoToTn().get(tnNo2);
                state.setValue(pos, island.getBusV().get(tn2)[phase2][0]);
                state.setValue(pos + dimension, island.getBusV().get(tn2)[phase2][1]);
            } else if (phase2 == 3) {
                tnNo1 = Integer.parseInt(id1.substring(0, id1.lastIndexOf("-")));
                tn1 = island.getBusNoToTn().get(tnNo1);
                state.setValue(pos, island.getBusV().get(tn1)[phase1][0]);
                state.setValue(pos + dimension, island.getBusV().get(tn1)[phase1][1]);
            } else {
                tnNo1 = Integer.parseInt(id1.substring(0, id1.lastIndexOf("-")));
                tnNo2 = Integer.parseInt(id2.substring(0, id2.lastIndexOf("-")));
                tn1 = island.getBusNoToTn().get(tnNo1);
                tn2 = island.getBusNoToTn().get(tnNo2);
                if (phase1 > phase2) {
                    tmpTn = tn1;
                    tn1 = tn2;
                    tn2 = tmpTn;
                    tmpPhase = phase1;
                    phase1 = phase2;
                    phase2 = tmpPhase;
                }
                if (phase != 2) {
                    state.setValue(pos, island.getBusV().get(tn1)[phase1][0] - island.getBusV().get(tn2)[phase2][0]);
                    state.setValue(pos + dimension, island.getBusV().get(tn1)[phase1][1] - island.getBusV().get(tn2)[phase2][1]);
                } else {
                    state.setValue(pos, island.getBusV().get(tn2)[phase2][0] - island.getBusV().get(tn1)[phase1][0]);
                    state.setValue(pos + dimension, island.getBusV().get(tn2)[phase2][1] - island.getBusV().get(tn1)[phase1][1]);
                }
            }
        }
    }

    @Override
    public DoubleMatrix2D getJocobian(AVector state) {
        formJacStruc();
        int index = 2 * getLoopSize(), pos, branchNo;
        double[] tempI = new double[2];
        double vx, vy, r, x, tmp;
        double dvx1, dvy1, dvx2, dvy2;
        //变压器原副边电流方程
        fillTfCurrentJac(jacobian, index);
        index += 2 * windingEdges.size();
        //负荷功率平衡方程
        for (DetailedEdge edge : nonZLoadEdges) {
            pos = edgeIndex.get(edge);
            vx = state.getValue(pos);
            vy = state.getValue(pos + dimension);
            branchNo = edgeToNo.get(edge);
            calCurrent(branchNo, state, tempI);

            dvx1 = tempI[0];
            dvy1 = tempI[1];
            dvx2 = -tempI[1];
            dvy2 = tempI[0];
            if (Math.abs(edge.getZ_real()) > ZERO_LIMIT
                    || Math.abs(edge.getZ_image()) > ZERO_LIMIT) {
                r = edge.getZ_real();
                x = edge.getZ_image();
                tmp = r * r + x * x;
                dvx1 -= 2. * vx * r / tmp;
                dvy1 -= 2. * vy * r / tmp;
                dvx2 -= 2. * vx * x / tmp;
                dvy2 -= 2. * vy * x / tmp;
            }
            if (Math.abs(edge.getI_ampl()) > ZERO_LIMIT) {
                double v_ampl = Math.sqrt(vx * vx + vy * vy);
                dvx1 -= vx * edge.getI_ampl() * Math.cos(edge.getI_angle()) / v_ampl;
                dvy1 -= vy * edge.getI_ampl() * Math.cos(edge.getI_angle()) / v_ampl;
                dvx2 -= vx * edge.getI_ampl() * Math.sin(edge.getI_angle()) / v_ampl;
                dvy2 -= vy * edge.getI_ampl() * Math.sin(edge.getI_angle()) / v_ampl;
            }
            fillJac(branchNo, index, vx, 0);
            fillJac(branchNo, index, vy, dimension);
            jacobian.setQuick(index, pos, dvx1);
            jacobian.setQuick(index, pos + dimension, dvy1);
            index++;
            fillJac(branchNo, index, vy, 0);
            fillJac(branchNo, index, -vx, dimension);
            jacobian.setQuick(index, pos, dvx2);
            jacobian.setQuick(index, pos + dimension, dvy2);
            index++;
        }
        //分布式电源的方程
        for (DispersedGen dg : island.getDispersedGens().values()) {
            dg.updateState(this);
            index += dg.fillJac(this, index);
        }
        return jacobian;
    }

    public void fillTfCurrentJac(DoubleMatrix2D jac, int index) {
        Transformer tf;
        double[] tempI = new double[2];
        double[] tempI2 = new double[2];
        int pos, branchNo, phase;
        double vx, vy, r, x, rt, xt, tmp;
        double dvx1, dvy1, dvx2, dvy2;
        //变压器支路电流方程
        for (DetailedEdge e : windingEdges) {
            DetailedEdge otherE = e.getOtherEdgeOfTf();
            if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX) {
                tf = (Transformer) island.getBranches().get(island.getDevices().get(e.getDevId()));
                pos = edgeIndex.get(e);
                dvx1 = 0.;
                dvy1 = 0.;
                dvx2 = 0.;
                dvy2 = 0.;
                if (Math.abs(e.getS_real()) > ZERO_LIMIT
                        || Math.abs(e.getS_image()) > ZERO_LIMIT) {
                    vx = state.getValue(edgeIndex.get(e));
                    vy = state.getValue(edgeIndex.get(e) + dimension);
                    double t = vx * vx + vy * vy;
                    dvx1 -= tf.getNt() * (-2.0 * vx * (e.getS_real() * vx + e.getS_image() * vy) / (t * t)
                            + e.getS_real() / t);
                    dvy1 -= tf.getNt() * (-2.0 * vy * (e.getS_real() * vx + e.getS_image() * vy) / (t * t)
                            + e.getS_image() / t);
                    dvx2 -= tf.getNt() * (-2.0 * vx * (e.getS_real() * vy - e.getS_image() * vx) / (t * t)
                            - e.getS_image() / t);
                    dvy2 -= tf.getNt() * (-2.0 * vy * (e.getS_real() * vy - e.getS_image() * vx) / (t * t)
                            + e.getS_real() / t);
                }
                if (Math.abs(e.getZ_real()) > ZERO_LIMIT
                        || Math.abs(e.getZ_image()) > ZERO_LIMIT) {
                    double t = e.getZ_real() * e.getZ_real() + e.getZ_image() * e.getZ_image();
                    dvx1 -= tf.getNt() * e.getZ_real() / t;
                    dvy1 -= tf.getNt() * e.getZ_image() / t;
                    dvx2 -= -tf.getNt() * e.getZ_image() / t;
                    dvy2 -= tf.getNt() * e.getZ_real() / t;
                }
                if (Math.abs(e.getI_ampl()) > ZERO_LIMIT) {
                    log.warn("Not support, Not support, Not support!!!");
                    //todo:
                }
                jac.setQuick(index, pos, dvx1);
                jac.setQuick(index, pos + dimension, dvy1);
                index++;
                jac.setQuick(index, pos, dvx2);
                jac.setQuick(index, pos + dimension, dvy2);
                index++;
            } else if (otherE.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX &&
                    (Math.abs(otherE.getS_real()) > ZERO_LIMIT || Math.abs(otherE.getS_image()) > ZERO_LIMIT)) {
                tf = (Transformer) island.getBranches().get(island.getDevices().get(e.getDevId()));
                int j1 = edgeToNo.get(e);
                calCurrent(j1, state, tempI);
                pos = edgeIndex.get(e);
                branchNo = edgeToNo.get(e);
                phase = otherE.getPhase();
                tempI2[0] = tempI[0] * tf.getNt();
                tempI2[1] = tempI[1] * tf.getNt();
                //目前认为出现这种情况的变压器只有D-GrY或GrY-GrY两种接法会有这种情况
                vx = state.getValue(pos) / tf.getNt();
                vy = state.getValue(pos + dimension) / tf.getNt();

                rt = tf.getR()[phase];
                xt = tf.getX()[phase];
                if (Math.abs(e.getZ_real()) > ZERO_LIMIT
                        || Math.abs(e.getZ_image()) > ZERO_LIMIT) {
                    //todo: not tested
                    r = e.getZ_real();
                    x = e.getZ_image();
                    tmp = r * r + x * x;

                    fillJac(branchNo, index, tf.getNt() * (vx - 2. * rt * tempI2[0] +
                            2. * ((rt * rt + xt * xt) * tempI2[0] - rt * vx - xt * vy) * r / tmp), 0);
                    fillJac(branchNo, index, tf.getNt() * (vy - 2. * rt * tempI2[1] +
                            2. * ((rt * rt + xt * xt) * tempI2[1] - xt * vx - rt * vy) * r / tmp), dimension);
                    jac.setQuick(index, pos, tempI2[0] - 2. * (vx + xt * tempI2[1] - rt * tempI2[0]) * r / tmp / tf.getNt());
                    jac.setQuick(index, pos + dimension, tempI2[1] - 2. * (vy - rt * tempI2[1] - xt * tempI2[0]) * r / tmp / tf.getNt());
                    index++;
                    fillJac(branchNo, index, tf.getNt() * (vy - 2. * rt * tempI2[0] - xt * tempI2[1] + rt * tempI2[1] +
                            2. * ((rt * rt + xt * xt) * tempI2[0] - rt * vx - xt * vy) * x / tmp), 0);
                    fillJac(branchNo, index, tf.getNt() * (-vx - 2. * xt * tempI2[1] - xt * tempI2[0] + rt * tempI2[0] +
                            2. * ((rt * rt + xt * xt) * tempI2[1] - xt * vx - rt * vy) * x / tmp), dimension);
                    jac.setQuick(index, pos, -tempI2[1] - 2. * (vx + xt * tempI2[1] - rt * tempI2[0]) * x / tmp / tf.getNt());
                    jac.setQuick(index, pos + dimension, tempI2[0] - 2. * (vy - rt * tempI2[1] - xt * tempI2[0]) * x / tmp / tf.getNt());
                    index++;
                } else {
                    fillJac(branchNo, index, tf.getNt() * (vx - 2. * rt * tempI2[0]), 0);
                    fillJac(branchNo, index, tf.getNt() * (vy - 2. * rt * tempI2[1]), dimension);
                    jac.setQuick(index, pos, tempI[0]);
                    jac.setQuick(index, pos + dimension, tempI[1]);
                    index++;
                    fillJac(branchNo, index, tf.getNt() * (vy - 2. * rt * tempI2[0] - xt * tempI2[1] + rt * tempI2[1]), 0);
                    fillJac(branchNo, index, tf.getNt() * (-vx - 2. * xt * tempI2[1] - xt * tempI2[0] + rt * tempI2[0]), dimension);
                    jac.setQuick(index, pos, -tempI[1]);
                    jac.setQuick(index, pos + dimension, tempI[0]);
                    index++;
                }
            } else
                index += 2;
        }
    }

    @Override
    public ASparseMatrixLink2D getJacobianStruc() {
        formJacStruc();
        return jacStruc;
    }

    @Override
    public AVector getZ() {
        return null;
    }

    @Override
    public double[] getDeltaArray() {
        return delta;
    }

    @Override
    public AVector calZ(AVector state) {
        return calZ(state, true);
    }

    /**
     * 计算回路KVL的偏差
     *
     * @param result     存储结果的数组
     * @param index      存储的起始位置
     * @param isUpdateDG 是否更新DG的状态
     */
    public void calLoopKVL(double[] result, int index, boolean isUpdateDG) {
        int k, j;
        double rValue, iValue, value;
        double[] tempI = new double[2];
        double[] vDrop = new double[2];
        DetailedEdge edge;
        if (isUpdateDG)
            for (DispersedGen dg : island.getDispersedGens().values())
                dg.updateState(this);
        for (int i = 0; i < B.getM(); i++) {
            k = B.getIA()[i];
            rValue = 0.0;
            iValue = 0.0;
            while (k != -1) {
                j = B.getJA().get(k);
                value = B.getVA().get(k);
                k = B.getLINK().get(k);

                edge = noToEdge.get(j);
                calVoltDrop(edge, vDrop, tempI);
                if (value < 0) {
                    rValue += -vDrop[0];
                    iValue += -vDrop[1];
                } else {
                    rValue += vDrop[0];
                    iValue += vDrop[1];
                }
            }
            result[i + index] = rValue;
            result[i + index + B.getM()] = iValue;
        }
    }

    public AVector calZ(AVector state, boolean isUpdateDG) {
        calLoopKVL(z_est.getValues(), 0, isUpdateDG);
        calTfCurrent(z_est.getValues(), 2 * getLoopSize());
        int index = 2 * (getLoopSize() + windingEdges.size()), pos;
        double[] tempI = new double[2];
        double vx, vy, r, x, tmp, p, q;
        for (DetailedEdge e : nonZLoadEdges) {
            pos = edgeIndex.get(e);
            vx = state.getValue(pos);
            vy = state.getValue(pos + dimension);
            calCurrent(edgeToNo.get(e), state, tempI);
            p = vx * tempI[0] + vy * tempI[1];
            q = vy * tempI[0] - vx * tempI[1];
            if (Math.abs(e.getZ_real()) > ZERO_LIMIT
                    || Math.abs(e.getZ_image()) > ZERO_LIMIT) {
                r = e.getZ_real();
                x = e.getZ_image();
                tmp = r * r + x * x;
                p -= (vx * vx + vy * vy) * r / tmp;
                q -= (vx * vx + vy * vy) * x / tmp;
            }
            if (Math.abs(e.getI_ampl()) > ZERO_LIMIT) {
                double v_ampl = Math.sqrt(vx * vx + vy * vy);
                p -= v_ampl * e.getI_ampl() * Math.cos(e.getI_angle());
                q -= v_ampl * e.getI_ampl() * Math.sin(e.getI_angle());
            }
            z_est.setValue(index++, p - e.getS_real());
            z_est.setValue(index++, q - e.getS_image());
        }
        //分布式电源对应的方程
        for (DispersedGen dg : island.getDispersedGens().values())
            index += dg.calZ(this, index);
        return z_est;
    }

    public void fillJac(int branchNo, int row, double v, int offset) {
        int k = B.getJA2()[branchNo];
        while (k != -1) {
            int i = B.getIA2().get(k);
            jacobian.setQuick(row, i + offset, B.getVA().get(k) * v);
            k = B.getLINK2().get(k);
        }
    }

    /**
     * 置零
     *
     * @param branchNo 支路编号
     * @param row      行
     * @param offset   偏移量
     */
    public void cleanJac(int branchNo, int row, int offset) {
        int k = B.getJA2()[branchNo];
        while (k != -1) {
            int i = B.getIA2().get(k);
            jacobian.setQuick(row, i + offset, 0.0);
            k = B.getLINK2().get(k);
        }
    }

    /**
     * 将值追加到Jacobian中
     *
     * @param branchNo 支路编号
     * @param row      行
     * @param v        值
     * @param offset   偏移量
     */
    public void fillJac2(int branchNo, int row, double v, int offset) {
        int k = B.getJA2()[branchNo];
        while (k != -1) {
            int i = B.getIA2().get(k);
            jacobian.addQuick(row, i + offset, B.getVA().get(k) * v);
            k = B.getLINK2().get(k);
        }
    }

    public void fillJacStruc(DoubleMatrix2D jacStruc, int branchNo, int row, double v, int offset) {
        int k = B.getJA2()[branchNo], i;
        while (k != -1) {
            i = B.getIA2().get(k);
            jacStruc.setQuick(row, i + offset, B.getVA().get(k) * v); //todo:
            k = B.getLINK2().get(k);
        }
    }

    public void fillJacStruc(ASparseMatrixLink2D jacStruc, int branchNo, int row, double v, int offset) {
        int k = B.getJA2()[branchNo], i;
        while (k != -1) {
            i = B.getIA2().get(k);
            jacStruc.setValue(row, i + offset, B.getVA().get(k) * v, true);
            k = B.getLINK2().get(k);
        }
    }

    public void fillJacStruc(int branchNo, int row, double v, int offset) {
        int k = B.getJA2()[branchNo], i;
        while (k != -1) {
            i = B.getIA2().get(k);
            jacStruc.setValue(row, i + offset, B.getVA().get(k) * v, true);
            k = B.getLINK2().get(k);
        }
    }

    public void calCurrent(int branchNo, double[] c) {
        calCurrent(branchNo, state, c);
    }

    public void calCurrent(int branchNo, AVector state, double[] c) {
        int k = B.getJA2()[branchNo], i;
        c[0] = 0;
        c[1] = 0;
        while (k != -1) {
            i = B.getIA2().get(k);
            c[0] += B.getVA().get(k) * state.getValue(i);
            c[1] += B.getVA().get(k) * state.getValue(i + dimension);
            k = B.getLINK2().get(k);
        }
    }

    @Override
    public boolean isJacStrucChange() {
        return false;
    }

    public void setDimension(int dimension) {
        this.dimension = dimension;
    }

    public int getVarSize() {
        return varSize;
    }

    /**
     * 计算变压器原副边电流方程
     *
     * @param g     用于存储结算结果的数组
     * @param index g中存储时的起始位置
     */
    public void calTfCurrent(double[] g, int index) {
        double[] tempI = new double[2];
        double[] vDrop = new double[2];
        int j1, j2;
        double[] tempI2 = new double[2];
        //变压器电流的关系
        Transformer tf;
        for (DetailedEdge e : windingEdges) {
            tf = (Transformer) island.getBranches().get(island.getDevices().get(e.getDevId()));
            j1 = edgeToNo.get(e);
            calCurrent(j1, state, tempI);
            DetailedEdge otherE = e.getOtherEdgeOfTf();
            if (e.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX) {
                vDrop[0] = state.getValue(edgeIndex.get(e));
                vDrop[1] = state.getValue(edgeIndex.get(e) + dimension);
                if (Math.abs(e.getS_real()) > ZERO_LIMIT
                        || Math.abs(e.getS_image()) > ZERO_LIMIT) {
                    double t = vDrop[0] * vDrop[0] + vDrop[1] * vDrop[1];
                    tempI[0] -= (e.getS_real() * vDrop[0] + e.getS_image() * vDrop[1]) / t;
                    tempI[1] -= (e.getS_real() * vDrop[1] - e.getS_image() * vDrop[0]) / t;
                }
                if (Math.abs(e.getZ_real()) > ZERO_LIMIT
                        || Math.abs(e.getZ_image()) > ZERO_LIMIT) {
                    double t = e.getZ_real() * e.getZ_real() + e.getZ_image() * e.getZ_image();
                    tempI[0] -= (e.getZ_real() * vDrop[0] + e.getZ_image() * vDrop[1]) / t;
                    tempI[1] -= (e.getZ_real() * vDrop[1] - e.getZ_image() * vDrop[0]) / t;
                }
                if (Math.abs(e.getI_ampl()) > ZERO_LIMIT) {
                    log.warn("Not support, Not support, Not support!!!");
                    //todo:
                }
                j2 = edgeToNo.get(otherE);
                calCurrent(j2, state, tempI2);
                g[index++] = tempI[0] * tf.getNt() - tempI2[0];
                g[index++] = tempI[1] * tf.getNt() - tempI2[1];
            } else if (otherE.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX) {
                tempI2[0] = tempI[0] * tf.getNt();
                tempI2[1] = tempI[1] * tf.getNt();
                double vx = state.getValue(edgeIndex.get(e)) / tf.getNt();
                double vy = state.getValue(edgeIndex.get(e) + dimension) / tf.getNt();

                if (Math.abs(otherE.getS_real()) > ZERO_LIMIT
                        || Math.abs(otherE.getS_image()) > ZERO_LIMIT) {
                    vx -= tempI2[0] * tf.getR()[otherE.getPhase()];
                    vx += tempI2[1] * tf.getX()[otherE.getPhase()];
                    vy -= tempI2[0] * tf.getX()[otherE.getPhase()];
                    vy -= tempI2[1] * tf.getR()[otherE.getPhase()];
                    if (Math.abs(e.getZ_real()) > ZERO_LIMIT
                            || Math.abs(e.getZ_image()) > ZERO_LIMIT) {
                        double r = e.getZ_real();
                        double x = e.getZ_image();
                        double tmp = r * r + x * x;
                        g[index++] = vx * tempI2[0] + vy * tempI2[1] - (vx * vx + vy * vy) * r / tmp - otherE.getS_real();
                        g[index++] = vy * tempI2[0] - vx * tempI2[1] - (vx * vx + vy * vy) * x / tmp - otherE.getS_image();
                    } else {
                        g[index++] = vx * tempI2[0] + vy * tempI2[1] - otherE.getS_real();
                        g[index++] = vy * tempI2[0] - vx * tempI2[1] - otherE.getS_image();
                    }
                } else {
                    double r = e.getZ_real() + tf.getR()[otherE.getPhase()];
                    double x = e.getZ_image() + tf.getX()[otherE.getPhase()];
                    double tmp = r * r + x * x;
                    g[index++] = tempI2[0] - (vx * r + vy * x) / tmp;
                    g[index++] = tempI2[1] - (vy * r - vx * x) / tmp;
                }
            } else {
                j2 = edgeToNo.get(otherE);
                calCurrent(j2, state, tempI2);
                g[index++] = tempI[0] * tf.getNt() - tempI2[0];
                g[index++] = tempI[1] * tf.getNt() - tempI2[1];
            }
        }
    }

    private class MyTraversalListener<V, E> extends TraversalListenerAdapter<V, E> {
        private MyTraversalListener(UndirectedGraph<V, E> g) {
            this.g = g;
            vertexIdToNo = new HashMap<String, Integer>(g.vertexSet().size());
            vertexNoToId = new HashMap<Integer, String>(g.vertexSet().size());
            sonToFather = new HashMap<Integer, Integer>(g.vertexSet().size());
        }

        private UndirectedGraph<V, E> g;
        private int treeEdgeCount = 0;
        private int verteNoCount = 0;

        @Override
        public void edgeTraversed(EdgeTraversalEvent<V, E> e) {
            V v1 = g.getEdgeSource(e.getEdge());
            V v2 = g.getEdgeTarget(e.getEdge());
            if (!vertexIdToNo.containsKey(v1.toString())) {
                V tmpV = v2;
                v2 = v1;
                v1 = tmpV;
            }
            if (!vertexIdToNo.containsKey(v2.toString())) {
                edgeToNo.put((DetailedEdge) e.getEdge(), treeEdgeCount);
                treeEdgeCount++;

                vertexIdToNo.put((String) v2, verteNoCount);
                vertexNoToId.put(verteNoCount, (String) v2);
                sonToFather.put(verteNoCount, vertexIdToNo.get(v1.toString()));
                verteNoCount++;
            }
        }

        /**
         * @see TraversalListenerAdapter#vertexTraversed(VertexTraversalEvent)
         */
        public void vertexTraversed(VertexTraversalEvent<V> e) {
            if (!vertexIdToNo.containsKey(e.getVertex().toString())) {
                vertexIdToNo.put((String) e.getVertex(), verteNoCount);
                vertexNoToId.put(verteNoCount, (String) e.getVertex());
                sonToFather.put(verteNoCount, -1);
                verteNoCount++;
            }
        }
    }

    /**
     * 该类在潮流结算结束后给节点电压赋值
     *
     * @param <V>
     * @param <E>
     */
    private class MyTraversalListener2<V, E> extends TraversalListenerAdapter<V, E> {
        String v1, v2, tmpV;
        DetailedEdge edge;
        int phase1, phase2, busNo;
        double[] tempI = new double[2];
        double[] vDrop = new double[2];
        DsTopoNode tn;
        Map<String, double[][]> vertexIdToV;

        private MyTraversalListener2() {
            vertexIdToV = new HashMap<String, double[][]>(island.getDetailedG().vertexSet().size());
            vertexIdToV.put(DsTopoIsland.EARTH_NODE_ID, new double[][]{{0., 0.}});
        }

        @Override
        public void edgeTraversed(EdgeTraversalEvent<V, E> e) {
            edge = (DetailedEdge) e.getEdge();
            v1 = island.getDetailedG().getEdgeSource(edge);
            v2 = island.getDetailedG().getEdgeTarget(edge);
            if (vertexIdToV.containsKey(v1) && vertexIdToV.containsKey(v2))
                return;
            //保证v1是已知电压的顶点，v2是待求的顶点
            if (vertexIdToV.containsKey(v2)) {
                tmpV = v1;
                v1 = v2;
                v2 = tmpV;
            }
            switch (edge.getEdgeType()) {
                case DetailedEdge.EDGE_TYPE_SUPPLIER:
                    tn = island.getBusNoToTn().get(edge.getTnNo1());
                    vertexIdToV.put(v2, island.getBusV().get(tn));
                    return;
            }
            phase1 = Integer.parseInt(v1.substring(v1.length() - 1));
            phase2 = Integer.parseInt(v2.substring(v2.length() - 1));
            if (phase2 > 2)
                return;
            busNo = Integer.parseInt(v2.substring(0, v2.length() - 2));
            tn = island.getBusNoToTn().get(busNo);
            if (!island.getBusV().containsKey(tn))
                island.getBusV().put(tn, new double[3][2]);
            vertexIdToV.put(v2, island.getBusV().get(tn));
            calVoltDrop(edge, v1, v2, vDrop, tempI);
            vertexIdToV.get(v2)[phase2][0] = vertexIdToV.get(v1)[phase1 % 3][0] - vDrop[0];
            vertexIdToV.get(v2)[phase2][1] = vertexIdToV.get(v1)[phase1 % 3][1] - vDrop[1];
        }
    }

    //方法求边上的电压降落，电压降方向为v1->v2
    public void calVoltDrop(DetailedEdge edge, String v1, String v2, double[] vDrop, double[] tempI) {
        calVoltDrop(edge, vDrop, tempI);
        int vertexNo1 = vertexIdToNo.get(v1);
        int vertexNo2 = vertexIdToNo.get(v2);
        if (vertexNo1 > vertexNo2) {
            if (getDirection(edge, vertexNo2, vertexNo1, -1.) < 0) {
                vDrop[0] = -vDrop[0];
                vDrop[1] = -vDrop[1];
            }
        } else if (getDirection(edge, vertexNo1, vertexNo2, 1.) < 0) {
            vDrop[0] = -vDrop[0];
            vDrop[1] = -vDrop[1];
        }
    }

    /**
     * 方法求边上的电压降落，电压降正方向和该边电流正方向相同
     *
     * @param edge  待求电压降落的边
     * @param v     存放电压降落的变量
     * @param tempI 存放电流中间计算量的变量
     */
    public void calVoltDrop(DetailedEdge edge, double[] v, double[] tempI) {
        v[0] = 0;
        v[1] = 0;
        switch (edge.getEdgeType()) {
            case DetailedEdge.EDGE_TYPE_SUPPLIER:
                DsTopoNode tn = island.getBusNoToTn().get(edge.getTnNo1());
                v[0] -= island.getBusV().get(tn)[edge.getPhase()][0];
                v[1] -= island.getBusV().get(tn)[edge.getPhase()][1];
                break;
            case DetailedEdge.EDGE_TYPE_FEEDER:
                String v1 = island.getDetailedG().getEdgeSource(edge);
                String v2 = island.getDetailedG().getEdgeTarget(edge);
                int vertexNo1 = vertexIdToNo.get(v1);
                int vertexNo2 = vertexIdToNo.get(v2);
                String tmpV;
                Feeder feeder = (Feeder) island.getBranches().get(island.getDevices().get(edge.getDevId()));
                if (vertexNo1 < vertexNo2) {
                    tmpV = v1;
                } else
                    tmpV = v2;
                calCurrent(edgeToNo.get(edge), state, tempI);
                v[0] += tempI[0] * feeder.getZ_real()[edge.getPhase()][edge.getPhase()]
                        - tempI[1] * feeder.getZ_imag()[edge.getPhase()][edge.getPhase()];
                v[1] += tempI[0] * feeder.getZ_imag()[edge.getPhase()][edge.getPhase()]
                        + tempI[1] * feeder.getZ_real()[edge.getPhase()][edge.getPhase()];
                //当前馈线电流方向是从tmpV流出的
                tmpV = tmpV.substring(0, tmpV.length() - 1);

                for (DetailedEdge otherE : edge.getOtherEdgesOfSameFeeder()) {
                    calCurrent(edgeToNo.get(otherE), state, tempI);
                    vertexNo1 = vertexIdToNo.get(island.getDetailedG().getEdgeSource(otherE));
                    vertexNo2 = vertexIdToNo.get(island.getDetailedG().getEdgeTarget(otherE));
                    if (vertexNo1 < vertexNo2) {
                        //如果该相馈线电流方向不是从tmpV流出,与当前phase1相正方向不同
                        if (!vertexNoToId.get(vertexNo1).startsWith(tmpV)) {
                            v[0] -= tempI[0] * feeder.getZ_real()[edge.getPhase()][otherE.getPhase()]
                                    - tempI[1] * feeder.getZ_imag()[edge.getPhase()][otherE.getPhase()];
                            v[1] -= tempI[0] * feeder.getZ_imag()[edge.getPhase()][otherE.getPhase()]
                                    + tempI[1] * feeder.getZ_real()[edge.getPhase()][otherE.getPhase()];
                            continue;
                        }
                    } else {
                        //如果该p相馈线电流方向不是从tmpV流出
                        if (!vertexNoToId.get(vertexNo2).startsWith(tmpV)) {
                            v[0] -= tempI[0] * feeder.getZ_real()[edge.getPhase()][otherE.getPhase()]
                                    - tempI[1] * feeder.getZ_imag()[edge.getPhase()][otherE.getPhase()];
                            v[1] -= tempI[0] * feeder.getZ_imag()[edge.getPhase()][otherE.getPhase()]
                                    + tempI[1] * feeder.getZ_real()[edge.getPhase()][otherE.getPhase()];
                            continue;
                        }
                    }
                    v[0] += tempI[0] * feeder.getZ_real()[edge.getPhase()][otherE.getPhase()]
                            - tempI[1] * feeder.getZ_imag()[edge.getPhase()][otherE.getPhase()];
                    v[1] += tempI[0] * feeder.getZ_imag()[edge.getPhase()][otherE.getPhase()]
                            + tempI[1] * feeder.getZ_real()[edge.getPhase()][otherE.getPhase()];
                }
                break;
            case DetailedEdge.EDGE_TYPE_TF_WINDING:
                Transformer tf = (Transformer) island.getBranches().get(island.getDevices().get(edge.getDevId()));
                if (edge.isSource()) {
                    int pos = edgeIndex.get(edge);
                    v[0] += state.getValue(pos);
                    v[1] += state.getValue(pos + dimension);
                } else {
                    int pos = edgeIndex.get(edge.getOtherEdgeOfTf());
                    calCurrent(edgeToNo.get(edge), state, tempI);
                    v[0] += (-state.getValue(pos) / tf.getNt()
                            + tempI[0] * tf.getR()[edge.getPhase()] - tempI[1] * tf.getX()[edge.getPhase()]);
                    v[1] += (-state.getValue(pos + dimension) / tf.getNt()
                            + tempI[0] * tf.getX()[edge.getPhase()] + tempI[1] * tf.getR()[edge.getPhase()]);
                }
                break;
            case DetailedEdge.EDGE_TYPE_LOAD:
                if (Math.abs(edge.getS_real()) > ZERO_LIMIT
                        || Math.abs(edge.getS_image()) > ZERO_LIMIT
                        || Math.abs(edge.getI_ampl()) > ZERO_LIMIT) {
                    int pos = edgeIndex.get(edge);
                    v[0] += state.getValue(pos);
                    v[1] += state.getValue(pos + dimension);
                } else {
                    int pos = edgeToNo.get(edge);
                    calCurrent(pos, state, tempI);
                    v[0] += (tempI[0] * edge.getZ_real() - tempI[1] * edge.getZ_image());
                    v[1] += (tempI[0] * edge.getZ_image() + tempI[1] * edge.getZ_real());
                }
                break;
            case DetailedEdge.EDGE_TYPE_LOAD_TF_MIX:
                tf = (Transformer) island.getBranches().get(island.getDevices().get(edge.getDevId()));
                if (edge.isSource()) {
                    int pos = edgeIndex.get(edge);
                    v[0] += state.getValue(pos);
                    v[1] += state.getValue(pos + dimension);
                    break;
                }
                int pos = edgeIndex.get(edge.getOtherEdgeOfTf());
                calCurrent(edgeToNo.get(edge.getOtherEdgeOfTf()), state, tempI);
                tempI[0] *= tf.getNt();
                tempI[1] *= tf.getNt();
                v[0] += (-state.getValue(pos) / tf.getNt()
                        + tempI[0] * tf.getR()[edge.getPhase()] - tempI[1] * tf.getX()[edge.getPhase()]);
                v[1] += (-state.getValue(pos + dimension) / tf.getNt()
                        + tempI[0] * tf.getX()[edge.getPhase()] + tempI[1] * tf.getR()[edge.getPhase()]);
                break;
            case DetailedEdge.EDGE_TYPE_DG:
                island.getDispersedGens().get(island.getDevices().get(edge.getDgId())).calVDrop(this, edge, v);
                break;
            default:
                break;
        }
    }

    public void formJacStrucOfKVL(ASparseMatrixLink2D m, int index) {
        int k, row, j;
        DetailedEdge e;
        for (int i = 0; i < B.getM(); i++) {
            row = i + index;
            k = B.getIA()[i];

            while (k != -1) {
                j = B.getJA().get(k);
                e = noToEdge.get(j);
                fillVDropJac(m, e, row, B.getVA().get(k), 1.0, 1.0);
                k = B.getLINK().get(k);
            }
        }
        DispersedGen dg;
        for (DetailedEdge edge : dgEdges) {
            dg = island.getDispersedGens().get(island.getDevices().get(edge.getDgId()));
            dg.fillJacStrucOfVDrop(m, this, edge, index);
        }
    }

    /**
     *
     * @param m 存储Jacobian矩阵
     * @param e 支路
     * @param row 当前的Jacobian矩阵中的行
     * @param value 回路矩阵中该支路所对应的值
     */
    public void fillVDropJac(ASparseMatrixLink2D m, DetailedEdge e, int row, double value, double a, double b) {
        switch (e.getEdgeType()) {
            case DetailedEdge.EDGE_TYPE_FEEDER:
                Feeder feeder = (Feeder) island.getBranches().get(island.getDevices().get(e.getDevId()));
                int v1 = vertexIdToNo.get(island.getDetailedG().getEdgeSource(e));
                int v2 = vertexIdToNo.get(island.getDetailedG().getEdgeTarget(e));
                String tnId;
                if (v1 < v2) {
                    tnId = vertexNoToId.get(v1);
                } else
                    tnId = vertexNoToId.get(v2);
                tnId = tnId.substring(0, tnId.length() - 1);

                int pos = edgeToNo.get(e);
                fillJacStruc(m, pos, row, a * value * feeder.getZ_real()[e.getPhase()][e.getPhase()], 0);
                fillJacStruc(m, pos, row, - a * value * feeder.getZ_imag()[e.getPhase()][e.getPhase()], dimension);
                fillJacStruc(m, pos, row + B.getM(), b * value * feeder.getZ_imag()[e.getPhase()][e.getPhase()], 0);
                fillJacStruc(m, pos, row + B.getM(), b * value * feeder.getZ_real()[e.getPhase()][e.getPhase()], dimension);
                for (DetailedEdge otherE : e.getOtherEdgesOfSameFeeder()) {
                    pos = edgeToNo.get(otherE);
                    v1 = vertexIdToNo.get(island.getDetailedG().getEdgeSource(otherE));
                    v2 = vertexIdToNo.get(island.getDetailedG().getEdgeTarget(otherE));
                    if (v1 < v2) {
                        if (!vertexNoToId.get(v1).startsWith(tnId)) {
                            fillJacStruc(m, pos, row, -a * value * feeder.getZ_real()[e.getPhase()][otherE.getPhase()], 0);
                            fillJacStruc(m, pos, row, a * value * feeder.getZ_imag()[e.getPhase()][otherE.getPhase()], dimension);
                            fillJacStruc(m, pos, row + B.getM(), -b * value * feeder.getZ_imag()[e.getPhase()][otherE.getPhase()], 0);
                            fillJacStruc(m, pos, row + B.getM(), -b * value * feeder.getZ_real()[e.getPhase()][otherE.getPhase()], dimension);
                            continue;
                        }
                    } else {
                        if (!vertexNoToId.get(v2).startsWith(tnId)) {
                            fillJacStruc(m, pos, row, -a * value * feeder.getZ_real()[e.getPhase()][otherE.getPhase()], 0);
                            fillJacStruc(m, pos, row, a * value * feeder.getZ_imag()[e.getPhase()][otherE.getPhase()], dimension);
                            fillJacStruc(m, pos, row + B.getM(), -b * value * feeder.getZ_imag()[e.getPhase()][otherE.getPhase()], 0);
                            fillJacStruc(m, pos, row + B.getM(), -b * value * feeder.getZ_real()[e.getPhase()][otherE.getPhase()], dimension);
                            continue;
                        }
                    }
                    fillJacStruc(m, pos, row, a * value * feeder.getZ_real()[e.getPhase()][otherE.getPhase()], 0);
                    fillJacStruc(m, pos, row, -a * value * feeder.getZ_imag()[e.getPhase()][otherE.getPhase()], dimension);
                    fillJacStruc(m, pos, row + B.getM(), b * value * feeder.getZ_imag()[e.getPhase()][otherE.getPhase()], 0);
                    fillJacStruc(m, pos, row + B.getM(), b * value * feeder.getZ_real()[e.getPhase()][otherE.getPhase()], dimension);
                }
                break;
            case DetailedEdge.EDGE_TYPE_LOAD_TF_MIX:
                if (e.isSource()) {
                    pos = edgeIndex.get(e);
                    m.setValue(row, pos, a * value);
                    m.setValue(row + B.getM(), pos + dimension, b * value);
                } else {
                    log.warn("Not supported, Not supported, Not supported!!!");
                }
                break;
            case DetailedEdge.EDGE_TYPE_TF_WINDING:
                Transformer tf = (Transformer) island.getBranches().get(island.getDevices().get(e.getDevId()));
                if (e.isSource()) {
                    pos = edgeIndex.get(e);
                    m.setValue(row, pos, a * value);
                    m.setValue(row + B.getM(), pos + dimension, b * value);
                } else {
                    pos = edgeIndex.get(e.getOtherEdgeOfTf());
                    m.setValue(row, pos, -a * value / tf.getNt());
                    m.setValue(row + B.getM(), pos + dimension, -b * value / tf.getNt());
                    pos = edgeToNo.get(e);
                    fillJacStruc(m, pos, row, a * value * tf.getR()[e.getPhase()], 0);
                    fillJacStruc(m, pos, row, -a * value * tf.getX()[e.getPhase()], dimension);
                    fillJacStruc(m, pos, row + B.getM(), b * value * tf.getX()[e.getPhase()], 0);
                    fillJacStruc(m, pos, row + B.getM(), b * value * tf.getR()[e.getPhase()], dimension);
                }
                break;
            case DetailedEdge.EDGE_TYPE_LOAD:
                if (Math.abs(e.getS_real()) > ZERO_LIMIT
                        || Math.abs(e.getS_image()) > ZERO_LIMIT
                        || Math.abs(e.getI_ampl()) > ZERO_LIMIT) {
                    pos = edgeIndex.get(e);
                    m.setValue(row, pos, a * value);
                    m.setValue(row + B.getM(), pos + dimension, b * value);
                } else {
                    pos = edgeToNo.get(e);
                    fillJacStruc(m, pos, row, a * value * e.getZ_real(), 0);
                    fillJacStruc(m, pos, row, -a * value * e.getZ_image(), dimension);
                    fillJacStruc(m, pos, row + B.getM(), b * value * e.getZ_image(), 0);
                    fillJacStruc(m, pos, row + B.getM(), b * value * e.getZ_real(), dimension);
                }
                break;
            default:
                break;
        }
    }

    public void formJacStruc() {
        if (jacStruc != null)
            return;
        jacStruc = new ASparseMatrixLink2D(varSize, varSize);
        //回路KVL方程的Jacobian
        formJacStrucOfKVL(jacStruc, 0);

        Transformer tf;
        int index = 2 * B.getM(), j1, j2, pos;
        double r, x;
        //变压器电流的关系
        for (DetailedEdge edge : windingEdges) {
            tf = (Transformer) island.getBranches().get(island.getDevices().get(edge.getDevId()));
            DetailedEdge otherE = edge.getOtherEdgeOfTf();
            j1 = edgeToNo.get(edge);
            if (edge.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX) {
                pos = edgeIndex.get(edge);
                j2 = edgeToNo.get(otherE);
                fillJacStruc(j1, index, tf.getNt(), 0);
                fillJacStruc(j2, index, -1.0, 0);
                jacStruc.setValue(index, pos, 1.0);
                jacStruc.setValue(index, pos + dimension, 1.0);
                index++;
                fillJacStruc(j1, index, tf.getNt(), dimension);
                fillJacStruc(j2, index, -1.0, dimension);
                jacStruc.setValue(index, pos, 1.0);
                jacStruc.setValue(index, pos + dimension, 1.0);
                index++;
            } else if (otherE.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX) {
                pos = edgeIndex.get(edge);
                if (Math.abs(otherE.getS_real()) > ZERO_LIMIT
                        || Math.abs(otherE.getS_image()) > ZERO_LIMIT) {
                    fillJacStruc(j1, index, 1.0, 0);
                    fillJacStruc(j1, index, 1.0, dimension);
                    jacStruc.setValue(index, pos, 1.0);
                    jacStruc.setValue(index, pos + dimension, 1.0);
                    index++;
                    fillJacStruc(j1, index, 1.0, 0);
                    fillJacStruc(j1, index, 1.0, dimension);
                    jacStruc.setValue(index, pos, 1.0);
                    jacStruc.setValue(index, pos + dimension, 1.0);
                    index++;
                } else {
                    r = edge.getZ_real() + tf.getR()[otherE.getPhase()];
                    x = edge.getZ_image() + tf.getX()[otherE.getPhase()];
                    double tmp = r * r + x * x;
                    fillJacStruc(j1, index, tf.getNt(), 0);
                    jacStruc.setValue(index, pos, -r / tmp / tf.getNt());
                    jacStruc.setValue(index, pos + dimension, -x / tmp / tf.getNt());
                    index++;
                    fillJacStruc(j1, index, tf.getNt(), dimension);
                    jacStruc.setValue(index, pos, x / tmp / tf.getNt());
                    jacStruc.setValue(index, pos + dimension, -r / tmp / tf.getNt());
                    index++;
                }
            } else {
                j2 = edgeToNo.get(otherE);
                fillJacStruc(j1, index, tf.getNt(), 0);
                fillJacStruc(j2, index, -1.0, 0);
                index++;
                fillJacStruc(j1, index, tf.getNt(), dimension);
                fillJacStruc(j2, index, -1.0, dimension);
                index++;
            }
        }
        //负荷功率平衡方程
        int branchNo;
        for (DetailedEdge edge : nonZLoadEdges) {
            pos = edgeIndex.get(edge);
            branchNo = edgeToNo.get(edge);
            fillJacStruc(branchNo, index, 1.0, 0);
            fillJacStruc(branchNo, index, 1.0, dimension);
            jacStruc.setValue(index, pos, 1.0);
            jacStruc.setValue(index, pos + dimension, 1.0);
            index++;
            fillJacStruc(branchNo, index, 1.0, 0);
            fillJacStruc(branchNo, index, 1.0, dimension);
            jacStruc.setValue(index, pos, 1.0);
            jacStruc.setValue(index, pos + dimension, 1.0);
            index++;
        }
        //分布式电源
        for (DispersedGen dg : island.getDispersedGens().values())
            index += dg.fillJacStruc(this, index);
        jacobian = new MySparseDoubleMatrix2D(jacStruc.getM(), jacStruc.getN(), jacStruc.getVA().size(), 0.2, 0.9);
        jacStruc.toColteMatrix(jacobian);
    }

    public int getLoopSize() {
        return loopSize;
    }

    public AVector getState() {
        return state;
    }

    public int getDimension() {
        return dimension;
    }

    /**
     * 将计算结果存储在island的电压、电流的Map中
     */
    public void fillStateInIsland() {
        if (!vertexIdToNo.containsKey(DsTopoIsland.EARTH_NODE_ID))
            return;
        UndirectedGraph<String, DetailedEdge> g = island.getDetailedG();
        BreadthFirstIterator<String, DetailedEdge> iter =
                new BreadthFirstIterator<String, DetailedEdge>(g, DsTopoIsland.EARTH_NODE_ID);
        MyTraversalListener2<String, DetailedEdge> busVFiller = new MyTraversalListener2<String, DetailedEdge>();
        iter.addTraversalListener(busVFiller);
        while (iter.hasNext())
            iter.next();
        List<DsTopoNode> unFillTns = new ArrayList<DsTopoNode>();
        boolean isFilled;
        for (DsTopoNode tn : island.getBusV().keySet()) {
            isFilled = false;
            for (int i = 0; i < 3; i++) {
                if (busVFiller.vertexIdToV.containsKey(tn.getBusNo() + "-" + i)) {
                    isFilled = true;
                    break;
                }
            }
            if (!isFilled)
                unFillTns.add(tn);
        }
        for (DsTopoNode tn : unFillTns)
            island.getBusV().remove(tn);

        island.setBranchHeadI(new HashMap<MapObject, double[][]>(island.getBranches().size()));
        island.setBranchTailI(new HashMap<MapObject, double[][]>(island.getBranches().size()));
        for (MapObject obj : island.getBranches().keySet()) {
            island.getBranchHeadI().put(obj, new double[3][2]);
            if (island.getBranches().get(obj) instanceof Transformer) {
                island.getBranchTailI().put(obj, new double[3][2]);
            } else
                island.getBranchTailI().put(obj, island.getBranchHeadI().get(obj));
        }

        MapObject obj;
        GeneralBranch gb;
        String v1, v2;
        int tnNo1, tnNo2;
        double[] c = new double[2];
        for (DetailedEdge edge : island.getDetailedG().edgeSet()) {
            obj = island.getDevices().get(edge.getDevId());
            if (obj == null)
                continue;
            gb = island.getBranches().get(obj);
            if (gb == null)
                continue;
            if (edge.getEdgeType() == DetailedEdge.EDGE_TYPE_LOAD_TF_MIX) {
                if (edge.isSource()) {
                    calCurrent(edgeToNo.get(edge), state, c);
                    double vx = state.getValue(edgeIndex.get(edge));
                    double vy = state.getValue(edgeIndex.get(edge) + dimension);
                    if (Math.abs(edge.getS_real()) > ZERO_LIMIT
                            || Math.abs(edge.getS_image()) > ZERO_LIMIT) {
                        double t = vx * vx + vy * vy;
                        c[0] -= (edge.getS_real() * vx + edge.getS_image() * vy) / t;
                        c[1] -= (edge.getS_real() * vy - edge.getS_image() * vx) / t;
                    }
                    if (Math.abs(edge.getZ_real()) > ZERO_LIMIT
                            || Math.abs(edge.getZ_image()) > ZERO_LIMIT) {
                        double t = edge.getZ_real() * edge.getZ_real() + edge.getZ_image() * edge.getZ_image();
                        c[0] -= (edge.getZ_real() * vx + edge.getZ_image() * vy) / t;
                        c[1] -= (edge.getZ_real() * vy - edge.getZ_image() * vx) / t;
                    }
                    if (Math.abs(edge.getI_ampl()) > ZERO_LIMIT) {
                        log.warn("Not support, Not support, Not support!!!");
                        //todo:
                    }
                } else {
                    calCurrent(edgeToNo.get(edge.getOtherEdgeOfTf()), state, c);
                    c[0] *= ((Transformer) gb).getNt();
                    c[1] *= ((Transformer) gb).getNt();
                }
            } else
                calCurrent(edgeToNo.get(edge), state, c);
            if (gb instanceof Feeder) {
                v1 = island.getDetailedG().getEdgeSource(edge);
                v2 = island.getDetailedG().getEdgeTarget(edge);
                tnNo1 = Integer.parseInt(v1.substring(0, v1.indexOf("-")));
                tnNo2 = Integer.parseInt(v2.substring(0, v2.indexOf("-")));
                if ((vertexIdToNo.get(v1) - vertexIdToNo.get(v2)) * ((tnNo1 - tnNo2)) > 0) {
                    island.getBranchHeadI().get(obj)[edge.getPhase()][0] = c[0];
                    island.getBranchHeadI().get(obj)[edge.getPhase()][1] = c[1];
                } else {
                    island.getBranchHeadI().get(obj)[edge.getPhase()][0] = -c[0];
                    island.getBranchHeadI().get(obj)[edge.getPhase()][1] = -c[1];
                }
            } else if (gb instanceof Transformer) {
                Transformer tf = (Transformer) gb;
                if (edge.isSource()) {
                    switch (tf.getConnType()) {
                        case Transformer.CONN_TYPE_Y_D:
                        case Transformer.CONN_TYPE_GrY_GrY:
                            island.getBranchHeadI().get(obj)[edge.getPhase()][0] = c[0];
                            island.getBranchHeadI().get(obj)[edge.getPhase()][1] = c[1];
                            break;
                        case Transformer.CONN_TYPE_D_GrY:
                            island.getBranchHeadI().get(obj)[edge.getPhase()][0] -= c[0];
                            island.getBranchHeadI().get(obj)[edge.getPhase()][1] -= c[1];
                            island.getBranchHeadI().get(obj)[(edge.getPhase() + 1) % 3][0] += c[0];
                            island.getBranchHeadI().get(obj)[(edge.getPhase() + 1) % 3][1] += c[1];
                            break;
                        case Transformer.CONN_TYPE_D_D:
                            island.getBranchHeadI().get(obj)[edge.getPhase()][0] += c[0];
                            island.getBranchHeadI().get(obj)[edge.getPhase()][1] += c[1];
                            island.getBranchHeadI().get(obj)[(edge.getPhase() + 1) % 3][0] -= c[0];
                            island.getBranchHeadI().get(obj)[(edge.getPhase() + 1) % 3][1] -= c[1];
                            break;
                        default:
                            break;
                    }
                } else {
                    switch (tf.getConnType()) {
                        case Transformer.CONN_TYPE_D_GrY:
                        case Transformer.CONN_TYPE_GrY_GrY:
                            island.getBranchTailI().get(obj)[edge.getPhase()][0] = c[0];
                            island.getBranchTailI().get(obj)[edge.getPhase()][1] = c[1];
                            break;
                        case Transformer.CONN_TYPE_Y_D:
                        case Transformer.CONN_TYPE_D_D:
                            island.getBranchTailI().get(obj)[edge.getPhase()][0] += c[0];
                            island.getBranchTailI().get(obj)[edge.getPhase()][1] += c[1];
                            island.getBranchTailI().get(obj)[(edge.getPhase() + 1) % 3][0] -= c[0];
                            island.getBranchTailI().get(obj)[(edge.getPhase() + 1) % 3][1] -= c[1];
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }

    //该方法将Island中潮流结果中获得回路电流等状态量的值
    //仅仅用在测试中
    public void fillState() {
        int pos, phase, busNo1, busNo2;
        double value, r, x, tmp;
        double[][] tnV, branchI;
        String v1, v2, tmpId;
        DetailedEdge e;
        DsTopoNode tn;
        Transformer tf;
        for (int i = 0; i < B.getM(); i++) {
            e = noToEdge.get(i);
            v1 = island.getDetailedG().getEdgeSource(e);
            v2 = island.getDetailedG().getEdgeTarget(e);
            if (vertexIdToNo.get(v2) < vertexIdToNo.get(v1)) {
                tmpId = v1;
                v1 = v2;
                v2 = tmpId;
            }
            value = 1.0;
            phase = e.getPhase();
            MapObject obj = island.getDevices().get(e.getDevId());
            switch (e.getEdgeType()) {
                case DetailedEdge.EDGE_TYPE_TF_WINDING:
                    tf = (Transformer) island.getBranches().get(obj);
                    if (e.isSource()) {
                        //回路的方向和变压器绕组方向不一致,绕组方向是固定的AN,BN,CN, AB,BC,CA
                        if (Integer.parseInt(v1.substring(v1.length() - 1)) != e.getPhase())
                            value = -1.0;
                        switch (tf.getConnType()) {
                            case Transformer.CONN_TYPE_D_GrY:
                                branchI = island.getBranchTailI().get(obj);
                                state.setValue(i, -value * branchI[(e.getPhase() + 1) % 3][0] / tf.getNt());
                                state.setValue(i + dimension, -value * branchI[(e.getPhase() + 1) % 3][1] / tf.getNt());
                                break;
                            case Transformer.CONN_TYPE_D_D:
                                DsTopoNode tn1 = island.getBusNoToTn().get(e.getTnNo1());
                                DsTopoNode tn2 = island.getBusNoToTn().get(e.getOtherEdgeOfTf().getTnNo1());
                                double vx1 = island.getBusV().get(tn1)[phase][0] - island.getBusV().get(tn1)[(phase + 1) % 3][0];
                                double vy1 = island.getBusV().get(tn1)[phase][1] - island.getBusV().get(tn1)[(phase + 1) % 3][1];
                                double vx2 = island.getBusV().get(tn2)[phase][0] - island.getBusV().get(tn2)[(phase + 1) % 3][0];
                                double vy2 = island.getBusV().get(tn2)[phase][1] - island.getBusV().get(tn2)[(phase + 1) % 3][1];
                                r = tf.getR()[phase];
                                x = tf.getX()[phase];
                                double vx = vx1 / tf.getNt() - vx2;
                                double vy = vy1 / tf.getNt() - vy2;
                                double ix1 = (vx * r + vy * x) / (r * r + x * x) / tf.getNt();
                                double iy1 = (vy * r - vx * x) / (r * r + x * x) / tf.getNt();
                                state.setValue(i, value * ix1);
                                state.setValue(i + dimension, value * iy1);
                                break;
                            case Transformer.CONN_TYPE_Y_D:
                            case Transformer.CONN_TYPE_GrY_GrY:
                                branchI = island.getBranchHeadI().get(obj);
                                state.setValue(i, value * branchI[phase][0]);
                                state.setValue(i + dimension, value * branchI[phase][1]);
                                break;
                            default:
                                break;
                        }
                    } else {
                        if (Integer.parseInt(v1.substring(v1.length() - 1)) == phase)
                            value = -1.0;
                        switch (tf.getConnType()) {
                            case Transformer.CONN_TYPE_D_GrY:
                            case Transformer.CONN_TYPE_GrY_GrY:
                                branchI = island.getBranchTailI().get(obj);
                                state.setValue(i, value * branchI[phase][0]);
                                state.setValue(i + dimension, value * branchI[phase][1]);
                                break;
                            case Transformer.CONN_TYPE_D_D:
                                DsTopoNode tn1 = island.getBusNoToTn().get(e.getOtherEdgeOfTf().getTnNo1());
                                DsTopoNode tn2 = island.getBusNoToTn().get(e.getTnNo1());
                                double vx1 = island.getBusV().get(tn1)[phase][0] - island.getBusV().get(tn1)[(phase + 1) % 3][0];
                                double vy1 = island.getBusV().get(tn1)[phase][1] - island.getBusV().get(tn1)[(phase + 1) % 3][1];
                                double vx2 = island.getBusV().get(tn2)[phase][0] - island.getBusV().get(tn2)[(phase + 1) % 3][0];
                                double vy2 = island.getBusV().get(tn2)[phase][1] - island.getBusV().get(tn2)[(phase + 1) % 3][1];
                                r = tf.getR()[phase];
                                x = tf.getX()[phase];
                                double vx = vx1 / tf.getNt() - vx2;
                                double vy = vy1 / tf.getNt() - vy2;
                                state.setValue(i, value * (vx * r + vy * x) / (r * r + x * x));
                                state.setValue(i + dimension, value * (vy * r - vx * x) / (r * r + x * x));
                                break;
                            case Transformer.CONN_TYPE_Y_D:
                                branchI = island.getBranchHeadI().get(obj);
                                state.setValue(i, value * (branchI[phase][0] / tf.getNt()));
                                state.setValue(i + dimension, value * (branchI[phase][1] / tf.getNt()));
                                break;
                            default:
                                break;
                        }
                    }
                    break;
                case DetailedEdge.EDGE_TYPE_FEEDER:
                    busNo1 = Integer.parseInt(v1.substring(0, v1.length() - 2));
                    busNo2 = Integer.parseInt(v2.substring(0, v2.length() - 2));
                    if (busNo1 > busNo2)
                        value = -1.0;
                    state.setValue(i, value * island.getBranchHeadI().get(obj)[phase][0]);
                    state.setValue(i + dimension, value * island.getBranchHeadI().get(obj)[phase][1]);
                    break;
                case DetailedEdge.EDGE_TYPE_LOAD:
                    busNo1 = e.getTnNo1();
                    tn = island.getBusNoToTn().get(busNo1);
                    tnV = island.getBusV().get(tn);
                    if (Integer.parseInt(v1.substring(v1.length() - 1)) != phase)
                        value = -1.0;
                    double vx, vy, sx, sy;
                    if (e.isLoadD()) {
                        vx = tnV[phase][0] - tnV[(phase + 1) % 3][0];
                        vy = tnV[phase][1] - tnV[(phase + 1) % 3][1];
                    } else {
                        vx = tnV[phase][0];
                        vy = tnV[phase][1];
                    }
                    sx = e.getS_real();
                    sy = e.getS_image();
                    double ix = (sx * vx + sy * vy) / (vx * vx + vy * vy);
                    double iy = (sx * vy - sy * vx) / (vx * vx + vy * vy);
                    if (Math.abs(e.getZ_real()) > ZERO_LIMIT
                            || Math.abs(e.getZ_image()) > ZERO_LIMIT) {
                        r = e.getZ_real();
                        x = e.getZ_image();
                        tmp = r * r + x * x;
                        ix += (vx * r + vy * x) / tmp;
                        iy += (vy * r - vx * x) / tmp;
                    }
                    if (Math.abs(e.getI_ampl()) > ZERO_LIMIT) {
                        ix += e.getI_ampl() * Math.cos(Math.atan2(vy, vx) - e.getI_angle());
                        iy += e.getI_ampl() * Math.sin(Math.atan2(vy, vx) - e.getI_angle());
                    }
                    state.setValue(i, value * ix);
                    state.setValue(i + dimension, value * iy);
                    break;
                case DetailedEdge.EDGE_TYPE_DG:
                    obj = island.getDevices().get(e.getDgId());
                    double[] c = island.getDispersedGens().get(obj).getMotor().calI(e.getPhase());
                    state.setValue(i, value * c[0]);
                    state.setValue(i + dimension, value * c[1]);
                    break;
                case DetailedEdge.EDGE_TYPE_LOAD_TF_MIX:
                    System.out.println("Not support, Not support, Not support!!!");
                    break;
                default:
                    break;
            }
        }
        for (DetailedEdge edge : windingEdges) {
            tf = (Transformer) island.getBranches().get(island.getDevices().get(edge.getDevId()));
            pos = edgeIndex.get(edge);
            tn = island.getBusNoToTn().get(edge.getTnNo1());
            tnV = island.getBusV().get(tn);
            switch (tf.getConnType()) {
                case Transformer.CONN_TYPE_Y_D:
                    //todo:
                    System.out.println("!!! Connection type Y_D Not supported!");
                    break;
                case Transformer.CONN_TYPE_D_GrY:
                    state.setValue(pos, -tnV[edge.getPhase()][0] + tnV[(edge.getPhase() + 1) % 3][0]);
                    state.setValue(pos + dimension, -tnV[edge.getPhase()][1] + tnV[(edge.getPhase() + 1) % 3][1]);
                    break;
                case Transformer.CONN_TYPE_D_D:
                    state.setValue(pos, tnV[edge.getPhase()][0] - tnV[(edge.getPhase() + 1) % 3][0]);
                    state.setValue(pos + dimension, tnV[edge.getPhase()][1] - tnV[(edge.getPhase() + 1) % 3][1]);
                    break;
                case Transformer.CONN_TYPE_GrY_GrY:
                    state.setValue(pos, tnV[edge.getPhase()][0]);
                    state.setValue(pos + dimension, tnV[edge.getPhase()][1]);
                    break;
                default:
                    break;
            }
        }
        for (DetailedEdge edge : nonZLoadEdges) {
            tn = island.getBusNoToTn().get(edge.getTnNo1());
            pos = edgeIndex.get(edge);
            tnV = island.getBusV().get(tn);
            if (edge.isLoadD()) {
                state.setValue(pos, tnV[edge.getPhase()][0] - tnV[(edge.getPhase() + 1) % 3][0]);
                state.setValue(pos + dimension, tnV[edge.getPhase()][1] - tnV[(edge.getPhase() + 1) % 3][1]);
            } else {
                state.setValue(pos, tnV[edge.getPhase()][0]);
                state.setValue(pos + dimension, tnV[edge.getPhase()][1]);
            }
        }
    }

    public Map<DetailedEdge, Integer> getEdgeToNo() {
        return edgeToNo;
    }

    public Map<Integer, DetailedEdge> getNoToEdge() {
        return noToEdge;
    }

    public ASparseMatrixLink2D getB() {
        return B;
    }

    public Map<String, Integer> getVertexIdToNo() {
        return vertexIdToNo;
    }

    public Map<Integer, String> getVertexNoToId() {
        return vertexNoToId;
    }

    public Map<Integer, Integer> getSonToFather() {
        return sonToFather;
    }

    public List<DetailedEdge> getWindingEdges() {
        return windingEdges;
    }

    public List<DetailedEdge> getNonZLoadEdges() {
        return nonZLoadEdges;
    }

    public Map<DetailedEdge, Integer> getEdgeIndex() {
        return edgeIndex;
    }

    public MySparseDoubleMatrix2D getJacobian() {
        return jacobian;
    }

    public AVector getZ_est() {
        return z_est;
    }
}
