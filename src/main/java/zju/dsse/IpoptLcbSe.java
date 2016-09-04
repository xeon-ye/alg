package zju.dsse;

import cern.colt.function.IntIntDoubleFunction;
import zju.dsmodel.DetailedEdge;
import zju.dsmodel.DsTopoIsland;
import zju.dsmodel.DsTopoNode;
import zju.dsmodel.Transformer;
import zju.dspf.LcbPfModel;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.measure.MeasVector;
import zju.se.IpoptSeAlg;

import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 14-2-24
 */
public class IpoptLcbSe extends IpoptSeAlg {
    //电气岛
    protected DsTopoIsland dsIsland;
    //基于环路电流的潮流模型
    private LcbPfModel pfModel;
    //
    private Map<String, DetailedEdge> edgeMap;

    double[] tempI = new double[2];
    double[] tempV = new double[2];

    @Override
    public void initial() {
        dsIsland.setVCartesian(true);
        dsIsland.setICartesian(true);
        busNumber = dsIsland.getTns().size();
        if (!dsIsland.isPerUnitSys())
            setTolerance(1e-1);//todo:
        dsIsland.buildDetailedGraph();
        pfModel = new LcbPfModel(dsIsland);
        dimension = pfModel.getVarSize();

        n = pfModel.getVarSize() + getMeas().getZ().getN();
        m = 2 * pfModel.getLoopSize();              //回路KVL方程
        m += 2 * pfModel.getWindingEdges().size();  //变压器原副边电流方程
        m += getMeas().getZ().getN();            //量测方程约束
        nele_jac = 0;
        if (getSlackBusNum() > 0) {//todo:目前电源支路的电压还没办法处理
            //slackBusCol = getSlackBusNum() - 1;
            //m += 3;
            //nele_jac += 5;
        }
        //初始化潮流约束的Jacobian部分
        ASparseMatrixLink2D kvlJacStruc = new ASparseMatrixLink2D(pfModel.getLoopSize(), pfModel.getVarSize());
        pfModel.formJacStrucOfKVL(kvlJacStruc, 0);

        nele_jac += kvlJacStruc.getVA().size();
        //获得量测相关的Jacobian部分
        nele_jac += getNonZeroNum(meas);
        jacobian = new MySparseDoubleMatrix2D(m, dimension, nele_jac, 0.2, 0.9);
        nele_jac += meas.getZ().getN();

        //先计算jacobian矩阵中不变的部分
        kvlJacStruc.toColteMatrix(jacobian);
        if (getSlackBusNum() > 0) {
            //int row = pfModel.getLoopSize() + meas.getZ().getN();
            //jacobian.setQuick(row, 3 * slackBusCol + vImageOffset, 1.0);
            //jacobian.setQuick(row + 1, 3 * slackBusCol + vImageOffset + 1, 1.0);
            //jacobian.setQuick(row + 1, 3 * slackBusCol + vRealOffset + 1, -DsModelCons.tanB);
            //jacobian.setQuick(row + 2, 3 * slackBusCol + vImageOffset + 2, 1.0);
            //jacobian.setQuick(row + 2, 3 * slackBusCol + vRealOffset + 2, -DsModelCons.tanC);
        }

        initialMeasInObjFunc();
        hessian = new MySparseDoubleMatrix2D(pfModel.getVarSize(), pfModel.getVarSize());
        objFunc.getHessStruc(hessian);
        nele_hess = hessian.cardinality() + measInObjFunc.length;
        //objFunc.setShortenRate(1e-6);
    }

    protected void updateJacobian(double[] x, boolean isNewX) {
        if (isNewX)
            fillState(x);
        pfModel.fillTfCurrentJac(jacobian, 2 * pfModel.getLoopSize());
        fillJacobian(meas, 2 * (pfModel.getLoopSize() + pfModel.getWindingEdges().size()));
    }

    @Override
    protected void updateHessian(double[] x, double obj_factor, double[] lambda) {
        hessian.forEachNonZero(new IntIntDoubleFunction() {
            public double apply(int i, int i2, double v) {
                return 0.0;
            }
        });
        fillHessian(meas, hessian, lambda, 2 * (pfModel.getLoopSize() + pfModel.getWindingEdges().size()));
    }


    @Override
    public AVector getFinalVTheta() {
        return null; //todo:
    }

    public void setStartingPoint(double[] x) {
        System.arraycopy(pfModel.getInitial().getValues(), 0, x, 0, dimension);
        calEstimatedZ();
        for (int i = 0; i < getMeas().getZ().getN(); i++)
            x[i + dimension] = meas.getZ().getValue(i) - meas.getZ_estimate().getValue(i);
    }

    public boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g) {
        //计算回路KVL方程
        pfModel.calLoopKVL(g, 0, true);
        //计算变压器原副边电流方程
        pfModel.calTfCurrent(g, pfModel.getLoopSize() * 2);
        int i = pfModel.getLoopSize();
        //计算量测方程约束
        calEstimatedZ();
        for (int j = 0; j < meas.getZ_estimate().getN(); j++, i++)
            g[i] = meas.getZ_estimate().getValue(j) + x[j + dimension] - meas.getZ().getValue(j);
        return true;
    }

    private void calEstimatedZ() {
        int index = 0;
        int num, phase, branchNo;
        pfModel.fillStateInIsland();
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        num = meas.getBus_v_pos()[i];
                        phase = meas.getBus_v_phase()[i];
                        calV(num + "-" + phase, DsTopoIsland.EARTH_NODE_ID, tempV);
                        meas.getZ_estimate().setValue(index, tempV[0] * tempV[0] + tempV[1] * tempV[1]);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        num = meas.getBus_p_pos()[i];//num starts from 1
                        phase = meas.getBus_p_phase()[i];
                        meas.getZ_estimate().setValue(index, calLoadOfVertex(num + "-" + phase, false));
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        num = meas.getBus_q_pos()[i];
                        phase = meas.getBus_q_phase()[i];
                        meas.getZ_estimate().setValue(index, calLoadOfVertex(num + "-" + phase, true));
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    DsTopoNode tn1, tn2;
                    String key;
                    DetailedEdge e;
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        num = meas.getLine_from_p_pos()[k];
                        phase = meas.getLine_from_p_phase()[k];
                        key = String.valueOf(num);
                        tn1 = dsIsland.getGraph().getEdgeSource(dsIsland.getDevices().get(key));
                        tn2 = dsIsland.getGraph().getEdgeTarget(dsIsland.getDevices().get(key));
                        e = dsIsland.getDetailedG().getEdge(tn1.getTnNo() + "-" + phase, tn2.getTnNo() + "-" + phase);
                        meas.getZ_estimate().setValue(index, calPowerOfBranch(e, false, false));
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        num = meas.getLine_from_q_pos()[k];
                        phase = meas.getLine_from_q_phase()[k];
                        key = String.valueOf(num);
                        tn1 = dsIsland.getGraph().getEdgeSource(dsIsland.getDevices().get(key));
                        tn2 = dsIsland.getGraph().getEdgeTarget(dsIsland.getDevices().get(key));
                        e = dsIsland.getDetailedG().getEdge(tn1.getTnNo() + "-" + phase, tn2.getTnNo() + "-" + phase);
                        meas.getZ_estimate().setValue(index, calPowerOfBranch(e, false, true));
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        num = meas.getLine_to_p_pos()[k];
                        phase = meas.getLine_to_p_phase()[k];
                        key = String.valueOf(num);
                        tn1 = dsIsland.getGraph().getEdgeSource(dsIsland.getDevices().get(key));
                        tn2 = dsIsland.getGraph().getEdgeTarget(dsIsland.getDevices().get(key));
                        e = dsIsland.getDetailedG().getEdge(tn1.getTnNo() + "-" + phase, tn2.getTnNo() + "-" + phase);
                        meas.getZ_estimate().setValue(index, calPowerOfBranch(e, true, false));
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        num = meas.getLine_to_q_pos()[k];
                        phase = meas.getLine_to_q_phase()[k];
                        key = String.valueOf(num);
                        tn1 = dsIsland.getGraph().getEdgeSource(dsIsland.getDevices().get(key));
                        tn2 = dsIsland.getGraph().getEdgeTarget(dsIsland.getDevices().get(key));
                        e = dsIsland.getDetailedG().getEdge(tn1.getTnNo() + "-" + phase, tn2.getTnNo() + "-" + phase);
                        meas.getZ_estimate().setValue(index, calPowerOfBranch(e, true, true));
                    }
                    break;
                case TYPE_LINE_FROM_CURRENT:
                    for (int k = 0; k < meas.getLine_from_i_amp_pos().length; k++, index++) {
                        num = meas.getLine_from_i_amp_pos()[k];//num starts from 1
                        phase = meas.getLine_from_i_amp_phase()[k];//num starts from 1
                        key = String.valueOf(num);
                        tn1 = dsIsland.getGraph().getEdgeSource(dsIsland.getDevices().get(key));
                        tn2 = dsIsland.getGraph().getEdgeTarget(dsIsland.getDevices().get(key));
                        e = dsIsland.getDetailedG().getEdge(tn1.getTnNo() + "-" + phase, tn2.getTnNo() + "-" + phase);
                        branchNo = pfModel.getEdgeToNo().get(e);
                        pfModel.calCurrent(branchNo, tempI);
                        meas.getZ_estimate().setValue(index, tempI[0] * tempI[0] + tempI[1] * tempI[1]);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    /**
     * 用于计算流出节点到负荷的功率
     * @param vertexId 顶点ID
     * @param isQ 是否计算无功
     * @return 计算结果
     */
    private double calLoadOfVertex(String vertexId, boolean isQ) {
        double vx = 0.0, vy = 0.0, r = 0.0;
        int index;
        double[] tempI2 = new double[2];
        for (DetailedEdge e : dsIsland.getDetailedG().edgesOf(vertexId)) {
            if (pfModel.getEdgeIndex().containsKey(e)) {
                index = pfModel.getEdgeIndex().get(e);
                vx = pfModel.getState().getValue(index);
                vy = pfModel.getState().getValue(index + pfModel.getDimension());
            }
            switch (e.getEdgeType()) {
                case DetailedEdge.EDGE_TYPE_LOAD:
                    pfModel.calCurrent(pfModel.getEdgeToNo().get(e), pfModel.getState(), tempI);
                    if (!pfModel.getEdgeIndex().containsKey(e)) {
                        if (isQ)
                            r += (tempI[0] * tempI[0] + tempI[1] * tempI[1]) * e.getZ_image();
                        else
                            r += (tempI[0] * tempI[0] + tempI[1] * tempI[1]) * e.getZ_real();
                        continue;
                    }
                    break;
                case DetailedEdge.EDGE_TYPE_LOAD_TF_MIX:
                    Transformer tf = (Transformer) dsIsland.getBranches().get(dsIsland.getDevices().get(e.getDevId()));
                    if (e.isSource()) {
                        pfModel.calCurrent(pfModel.getEdgeToNo().get(e), pfModel.getState(), tempI);
                        pfModel.calCurrent(pfModel.getEdgeToNo().get(e.getOtherEdgeOfTf()), pfModel.getState(), tempI2);
                        double nt = tf.getNt();
                        tempI[0] -= tempI2[0] / nt;
                        tempI[1] -= tempI2[1] / nt;
                    } else {
                        pfModel.calCurrent(pfModel.getEdgeToNo().get(e.getOtherEdgeOfTf()), pfModel.getState(), tempI2);
                        double nt = tf.getNt();
                        tempI[0] = tempI2[0] * nt;
                        tempI[1] = tempI2[1] * nt;
                        index = pfModel.getEdgeIndex().get(e.getOtherEdgeOfTf());
                        vx = pfModel.getState().getValue(index) / nt;
                        vy = pfModel.getState().getValue(index + pfModel.getDimension()) / nt;
                        vx -= (tempI[0] * tf.getR()[e.getPhase()] - tempI[1] * tf.getX()[e.getPhase()]);
                        vx -= (tempI[0] * tf.getX()[e.getPhase()] + tempI[1] * tf.getR()[e.getPhase()]);
                    }
                    break;
                case DetailedEdge.EDGE_TYPE_DG:
                    break;
                default:
                    break;
            }
            if (isQ)
                r += (vx * tempI[0] + vy * tempI[0]);
            else
                r += (vy * tempI[0] - vx * tempI[1]);
        }
        return r;
    }

    /**
     * @param e 支路
     * @param isTo 是否是节点编号大的->小的
     * @param isQ 是否是计算无功
     * @return 计算结果
     */
    private double calPowerOfBranch(DetailedEdge e, boolean isTo, boolean isQ) {
        int branchNo = pfModel.getEdgeToNo().get(e);
        pfModel.calCurrent(branchNo, pfModel.getState(), tempI);
        String v1 = dsIsland.getDetailedG().getEdgeSource(e);
        String v2 = dsIsland.getDetailedG().getEdgeTarget(e);
        int busNo1 = Integer.parseInt(v2.substring(0, v1.length() - 2));
        int busNo2 = Integer.parseInt(v2.substring(0, v2.length() - 2));
        if (busNo1 < busNo2) {
            if (isTo)
                calV(v2, DsTopoIsland.EARTH_NODE_ID, tempV);
            else
                calV(v1, DsTopoIsland.EARTH_NODE_ID, tempV);
        } else {
            if (isTo)
                calV(v1, DsTopoIsland.EARTH_NODE_ID, tempV);
            else
                calV(v2, DsTopoIsland.EARTH_NODE_ID, tempV);
        }
        if (isQ)
            return tempV[0] * tempI[0] + tempV[1] * tempI[0];
        else
            return tempV[1] * tempI[0] - tempV[0] * tempI[1];
    }

    /**
     * 该方法用于计算两个节点之间的电压差，方法是不断通过树枝回溯，累加电压降
     *
     * @param v1 节点id
     * @param v2 节点id
     * @param v  存储计算结果的数组
     */
    public void calV(String v1, String v2, double[] v) {
        int vertexNo1 = pfModel.getVertexIdToNo().get(v1);
        int vertexNo2 = pfModel.getVertexIdToNo().get(v2);
        int tmpNo;
        String id1, id2;
        DetailedEdge tmpEdge;
        v[0] = 0.0;
        v[1] = 0.0;
        while (vertexNo1 != vertexNo2) {
            if (vertexNo1 > vertexNo2) {
                tmpNo = pfModel.getSonToFather().get(vertexNo1);
                id1 = pfModel.getVertexNoToId().get(tmpNo);
                id2 = pfModel.getVertexNoToId().get(vertexNo1);
                tmpEdge = dsIsland.getDetailedG().getEdge(id1, id2);
                pfModel.calVoltDrop(tmpEdge, id2, id1, tempV, tempI);
                vertexNo1 = pfModel.getSonToFather().get(vertexNo1);
            } else {
                tmpNo = pfModel.getSonToFather().get(vertexNo2);
                id1 = pfModel.getVertexNoToId().get(tmpNo);
                id2 = pfModel.getVertexNoToId().get(vertexNo2);
                tmpEdge = dsIsland.getDetailedG().getEdge(id1, id2);
                pfModel.calVoltDrop(tmpEdge, id1, id2, tempV, tempI);
                vertexNo2 = pfModel.getSonToFather().get(vertexNo2);
            }
            v[0] += tempV[0];
            v[1] += tempV[1];
        }
    }

    /**
     * 该方法用于计算两个节点之间的电压差所对应的Jacobian元素
     *
     * @param v1 节点id
     * @param v2 节点id
     */
    private void fillJacOfV(String v1, String v2, int index, double a, double b) {
        int vertexNo1 = pfModel.getVertexIdToNo().get(v1);
        int vertexNo2 = pfModel.getVertexIdToNo().get(v2);
        int tmpNo;
        String id1, id2;
        DetailedEdge tmpEdge;
        while (vertexNo1 != vertexNo2) {
            if (vertexNo1 > vertexNo2) {
                tmpNo = pfModel.getSonToFather().get(vertexNo1);
                id1 = pfModel.getVertexNoToId().get(tmpNo);
                id2 = pfModel.getVertexNoToId().get(vertexNo1);
                tmpEdge = dsIsland.getDetailedG().getEdge(id1, id2);
                //todo:
                pfModel.fillVDropJac(null, tmpEdge, index, 1.0, a, b);
                vertexNo1 = pfModel.getSonToFather().get(vertexNo1);
            } else {
                tmpNo = pfModel.getSonToFather().get(vertexNo2);
                id1 = pfModel.getVertexNoToId().get(tmpNo);
                id2 = pfModel.getVertexNoToId().get(vertexNo2);
                tmpEdge = dsIsland.getDetailedG().getEdge(id1, id2);
                //todo:
                pfModel.fillVDropJac(null, tmpEdge, index, 1.0, a, b);
                vertexNo2 = pfModel.getSonToFather().get(vertexNo2);
            }
        }
    }


    public void setXLimit(double[] x_L, double[] x_U) {
        int i = 0;
        for (; i < pfModel.getLoopSize(); i++) {
            x_L[i] = -2e15;//todo: this is not perfect
            x_U[i] = 2e15;
            x_L[i + dimension] = -2e15;
            x_U[i + dimension] = 2e15;
        }
        for (int j = 0; j < pfModel.getWindingEdges().size(); i++, j++) {
            x_L[i] = -2.0;
            x_U[i] = 2.;
            x_L[i + dimension] = -2.;
            x_U[i + dimension] = 2.0;
        }
        initialMeasLimit(dimension, x_L, x_U);
    }

    public void fillState(double[] x) {
        System.arraycopy(x, 0, pfModel.getState().getValues(), 0, pfModel.getVarSize());
    }

    private int getNonZeroNum(MeasVector meas) {
        int p_from_size = meas.getLine_from_p_pos().length;
        int q_from_size = meas.getLine_from_q_pos().length;
        int p_to_size = meas.getLine_to_p_pos().length;
        int q_to_size = meas.getLine_to_q_pos().length;
        int i_to_size = meas.getLine_to_i_amp_pos().length;
        int i_from_size = meas.getLine_from_i_amp_pos().length;
        int v_size = meas.getBus_v_pos().length;

        int capacity = 2 * (v_size + i_from_size + i_to_size) + 4 * (p_from_size + q_from_size + p_to_size + q_to_size);
        for (int i = 0; i < meas.getBus_p_pos().length; i++) {
            int num = meas.getBus_p_pos()[i];//num starts from 1
            //DsTopoNode tn = island.getTnNoToTn().get(num);
            //capacity += (2 * tn.getConnectedBusNo().length + 2);
        }
        for (int i = 0; i < meas.getBus_q_pos().length; i++) {
            int num = meas.getBus_q_pos()[i];//num starts from 1
            //DsTopoNode tn = island.getTnNoToTn().get(num);
            //capacity += (2 * tn.getConnectedBusNo().length + 2);
        }
        return capacity;
    }


    private void fillJacobian(MeasVector meas, int index) {
        //todo:
        int num, phase, branchNo;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        num = meas.getBus_v_pos()[i];
                        phase = meas.getBus_v_phase()[i];
                        calV(num + "-" + phase, DsTopoIsland.EARTH_NODE_ID, tempV);
                        fillJacOfV(num + "-" + phase, DsTopoIsland.EARTH_NODE_ID, index, 2.0 * tempV[0],  2.0 * tempV[1]);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        num = meas.getBus_p_pos()[i];//num starts from 1
                        phase = meas.getBus_p_phase()[i];
                        //fillJacobian_bus_p(num, phase, result, index);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        num = meas.getBus_q_pos()[i];
                        phase = meas.getBus_q_phase()[i];
                        //fillJacobian_bus_q(num, phase, result, index);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    DsTopoNode tn1, tn2;
                    String key;
                    DetailedEdge e;
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        num = meas.getLine_from_p_pos()[k];
                        phase = meas.getLine_from_p_phase()[k];
                        key = String.valueOf(num);
                        tn1 = dsIsland.getGraph().getEdgeSource(dsIsland.getDevices().get(key));
                        tn2 = dsIsland.getGraph().getEdgeTarget(dsIsland.getDevices().get(key));
                        e = dsIsland.getDetailedG().getEdge(tn1.getTnNo() + "-" + phase, tn2.getTnNo() + "-" + phase);
                        branchNo = pfModel.getEdgeToNo().get(e);
                        pfModel.calCurrent(branchNo, pfModel.getState(), tempI);
                        calV(num + "-" + phase, DsTopoIsland.EARTH_NODE_ID, tempV);
                        fillJacOfV(num + "-" + phase, DsTopoIsland.EARTH_NODE_ID, index, -tempI[0],  tempI[1]);
                        //pfModel.fillJacStruc(result, branchNo, index, tempV[1], 0);
                        //pfModel.fillJacStruc(result, branchNo, index, -tempV[0], pfModel.getDimension());
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        num = meas.getLine_from_q_pos()[k];
                        phase = meas.getLine_from_q_phase()[k];
                        //fillJacobian_line_from_q(num, phase, result, index);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        num = meas.getLine_to_p_pos()[k];
                        phase = meas.getLine_to_p_phase()[k];
                        //fillJacobian_line_to_p(num, phase, result, index);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        num = meas.getLine_to_q_pos()[k];
                        phase = meas.getLine_to_q_phase()[k];
                        //fillJacobian_line_to_q(num, phase, result, index);
                    }
                    break;
                case TYPE_LINE_FROM_CURRENT:
                    for (int k = 0; k < meas.getLine_from_i_amp_pos().length; k++, index++) {
                        num = meas.getLine_from_i_amp_pos()[k];//num starts from 1
                        phase = meas.getLine_from_i_amp_phase()[k];//num starts from 1
                        //fillJacobian_line_from_i_ampl(num, phase, result, index);
                        key = String.valueOf(num);
                        tn1 = dsIsland.getGraph().getEdgeSource(dsIsland.getDevices().get(key));
                        tn2 = dsIsland.getGraph().getEdgeTarget(dsIsland.getDevices().get(key));
                        e = dsIsland.getDetailedG().getEdge(tn1.getTnNo() + "-" + phase, tn2.getTnNo() + "-" + phase);
                        branchNo = pfModel.getEdgeToNo().get(e);
                        int tmpK = pfModel.getB().getJA2()[branchNo], i;
                        pfModel.calCurrent(branchNo, tempI);
                        while (tmpK != -1) {
                            i = pfModel.getB().getIA2().get(tmpK);
                            tmpK = pfModel.getB().getLINK2().get(tmpK);
                            jacobian.setQuick(index, i, 2 * pfModel.getB().getVA().get(tmpK) * tempI[0]);
                            jacobian.setQuick(index, i + pfModel.getDimension(), 2 * pfModel.getB().getVA().get(tmpK) * tempI[1]);
                        }
                    }
                    break;
                default:
                    break;
            }
        }
    }

    private void fillHessian(MeasVector meas, MySparseDoubleMatrix2D hessian, double[] lambda, int index) {
        //todo:
    }

    public DsTopoIsland getDsIsland() {
        return dsIsland;
    }

    public void setDsIsland(DsTopoIsland dsIsland) {
        this.dsIsland = dsIsland;
    }
}
