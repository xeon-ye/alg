package zju.dsse;

import cern.colt.function.IntIntDoubleFunction;
import zju.devmodel.MapObject;
import zju.dsmodel.*;
import zju.dspf.DsStateCal;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.measure.MeasVector;
import zju.se.IpoptSeAlg;

/**
 * Distribution SE using Cartesian coordinate system<br>
 * using branch current and bus voltage as state variables
 *
 * @author Dong Shufeng
 *         Date: 2009-9-28
 */
public class IpoptDsSe extends IpoptSeAlg {
    //用于计算量测方程
    private DsStateCal stateCal;
    //电气岛
    protected DsTopoIsland dsIsland;

    protected int[] nodeOffset;

    protected int[] branchOffset;

    public double[][] vTemp;

    public double[][] iTemp;

    private int[] tmpPos = new int[6];

    public IpoptDsSe() {
        stateCal = new DsStateCal();
        vTemp = new double[3][2];
        iTemp = new double[3][2];
    }

    public IpoptDsSe(DsTopoIsland dsIsland) {
        setDsIsland(dsIsland);
    }

    @Override
    public void initial() {
        dsIsland.setVCartesian(true);
        dsIsland.setICartesian(true);
        busNumber = dsIsland.getTns().size();
        if (!dsIsland.isPerUnitSys())
            setTolerance(1e-1);//todo:

        nodeOffset = new int[dsIsland.getTns().size()];
        dimension = 0;
        for (int busNo = 1; busNo <= busNumber; busNo++) {
            nodeOffset[busNo - 1] = dimension;
            //电压实部和虚部作为状态变量
            dimension += 2 * dsIsland.getBusNoToTn().get(busNo).getPhases().length;
        }
        int branchNum = dsIsland.getIdToBranch().size();
        MapObject obj;
        GeneralBranch gb;
        branchOffset = new int[branchNum];
        int pfConstraintsNum = 0;
        for (int branchNo = 1; branchNo <= branchNum; branchNo++) {
            obj = dsIsland.getIdToBranch().get(branchNo);
            gb = dsIsland.getBranches().get(obj);
            branchOffset[branchNo - 1] = dimension;
            if (gb instanceof Feeder) {
                //电流实部和虚部作为状态量
                dimension += 2 * ((Feeder) gb).getPhases().length;
                //电压方程 VLNABC = a * VLNabc + b * Iabc的个数
                pfConstraintsNum += 2 * ((Feeder) gb).getPhases().length;
            } else {
                //变压器支路首末端电流都要作为变量
                dimension += 12;
                //电压方程+电流方程 IABC = c * VLNabc + d * Iabc的个数
                pfConstraintsNum += 12;
            }
        }
        //联络节点电流约束方程个数
        for (int busNo : zeroPBuses)
            pfConstraintsNum += 2 * dsIsland.getBusNoToTn().get(busNo).getPhases().length;
        m = getMeas().getZ().getN() + pfConstraintsNum; //量测方程约束个数

        nele_jac = 0;
        n = dimension + getMeas().getZ().getN();

        if (getSlackBusNum() > 0) {
            slackBusCol = getSlackBusNum() - 1;
            m += 3;
            nele_jac += 5;
        }
        /* Number of nonzeros of jacobian matrix the the constraints */
        nele_jac += getNonZeroNum(meas);
        nele_jac += getNonZeroNumOfPfConstraints();
        nele_jac += meas.getZ().getN();

        jacobian = new MySparseDoubleMatrix2D(m, dimension, nele_jac, 0.8, 0.9);
        //先计算jacobian矩阵中不变的部分
        int row = meas.getZ().getN();
        fillPfJacobian(row);
        row += pfConstraintsNum;

        if (getSlackBusNum() > 0) {//假设参考节点没有缺相
            jacobian.setQuick(row, nodeOffset[slackBusCol] + 3, 1.0);
            jacobian.setQuick(row + 1, nodeOffset[slackBusCol] + 4, 1.0);
            jacobian.setQuick(row + 1, nodeOffset[slackBusCol] + 1, -DsModelCons.tanB);
            jacobian.setQuick(row + 2, nodeOffset[slackBusCol] + 5, 1.0);
            jacobian.setQuick(row + 2, nodeOffset[slackBusCol] + 2, -DsModelCons.tanC);
        }

        hessian = new MySparseDoubleMatrix2D(dimension, dimension);
        initialHessian(meas);

        initialMeasInObjFunc();
        objFunc.getHessStruc(hessian);
        nele_hess = hessian.cardinality() + measInObjFunc.length;
        //objFunc.setShortenRate(1e-6);
    }

    protected void updateJacobian(double[] x, boolean isNewX) {
        if (isNewX)
            fillState(x);
        fillJacobian(meas, 0);
    }

    @Override
    protected void updateHessian(double[] x, double obj_factor, double[] lambda) {
        hessian.forEachNonZero(new IntIntDoubleFunction() {
            public double apply(int i, int i2, double v) {
                return 0.0;
            }
        });
        fillHessian(meas, lambda, 0);
    }

    @Override
    public AVector getFinalVTheta() {
        return null; //todo:
    }

    public void setStartingPoint(double[] x) {
        MapObject obj;
        GeneralBranch gb;
        double[][] c, c2;
        for (int branchNo = 1; branchNo <= dsIsland.getIdToBranch().size(); branchNo++) {
            obj = dsIsland.getIdToBranch().get(branchNo);
            gb = dsIsland.getBranches().get(obj);
            c = dsIsland.getBranchHeadI().get(obj);
            if (gb instanceof Feeder) {
                for (int i : ((Feeder) gb).getPhases()) {
                    x[branchOffset[branchNo - 1] + ((Feeder) gb).getPhaseIndex(i)] = c[i][0];
                    x[branchOffset[branchNo - 1] + ((Feeder) gb).getPhaseIndex(i) + ((Feeder) gb).getPhases().length] = c[i][1];
                }
            } else {
                c2 = dsIsland.getBranchTailI().get(obj);
                for (int i = 0; i < c.length; i++) {
                    x[branchOffset[branchNo - 1] + i] = c[i][0];
                    x[branchOffset[branchNo - 1] + i + 6] = c[i][1];
                    x[branchOffset[branchNo - 1] + i + 3] = c2[i][0];
                    x[branchOffset[branchNo - 1] + i + 9] = c2[i][1];
                }
            }
        }
        for (int busNo = 1; busNo <= busNumber; busNo++) {
            double[][] v = dsIsland.getBusV().get(dsIsland.getBusNoToTn().get(busNo));
            DsTopoNode tn = dsIsland.getBusNoToTn().get(busNo);
            for (int i : tn.getPhases()) {
                x[nodeOffset[busNo - 1] + tn.getPhaseIndex(i)] = v[i][0];
                x[nodeOffset[busNo - 1] + tn.getPhaseIndex(i) + tn.getPhases().length] = v[i][1];
            }
        }
        stateCal.getEstimatedZ(meas);
        for (int i = 0; i < getMeas().getZ().getN(); i++)
            x[i + dimension] = meas.getZ().getValue(i) - meas.getZ_estimate().getValue(i);
    }

    public boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g) {
        if (new_x)
            fillState(x);
        int i = 0;
        //计算量测方程约束
        stateCal.getEstimatedZ(meas);
        for (int j = 0; j < meas.getZ_estimate().getN(); j++, i++)
            g[i] = meas.getZ_estimate().getValue(j) + x[j + dimension] - meas.getZ().getValue(j);
        i += fillPfConstrains(g, i);
        //计算角度参考节点的约束
        if (getSlackBusNum() > 0) {
            g[i] = x[nodeOffset[slackBusCol] + 3];
            g[i + 1] = x[nodeOffset[slackBusCol] + 4] - x[nodeOffset[slackBusCol] + 1] * DsModelCons.tanB;
            g[i + 2] = x[nodeOffset[slackBusCol] + 5] - x[nodeOffset[slackBusCol] + 2] * DsModelCons.tanC;
        }
        return true;
    }

    protected int fillPfConstrains(double[] g, int row) {
        int oldRow = row;
        GeneralBranch gb;
        DsTopoNode tn, father, tmpTn;
        MapObject obj;

        for (int branchNo = 1; branchNo <= dsIsland.getIdToBranch().size(); branchNo++) {
            obj = dsIsland.getIdToBranch().get(branchNo);
            tn = dsIsland.getGraph().getEdgeTarget(obj);
            father = dsIsland.getGraph().getEdgeSource(obj);
            //注意电流正方向是节点编号小的流向大的
            //todo: 对于环状网络的变压器支路可能是错误的
            if (father.getBusNo() > tn.getBusNo()) {
                tmpTn = father;
                father = tn;
                tn = tmpTn;
            }
            obj = dsIsland.getGraph().getEdge(father, tn);
            gb = dsIsland.getBranches().get(obj);
            //计算电压方程约束 VLNABC = a * VLNabc + b * Iabc;
            gb.calHeadV(dsIsland.getBusV().get(tn), dsIsland.getBranchTailI().get(obj), vTemp);
            for (int j = 0; j < 3; j++) {
                if (!gb.containsPhase(j))
                    continue;
                g[row++] = vTemp[j][0] - dsIsland.getBusV().get(father)[j][0];
                g[row++] = vTemp[j][1] - dsIsland.getBusV().get(father)[j][1];
            }

            if (dsIsland.getBranchHeadI().get(obj) == dsIsland.getBranchTailI().get(obj))
                continue;
            //注意电流正方向是节点编号小的流向大的
            //todo: 对于环状网络的变压器支路可能是错误的
            //计算电流方程 IABC = c * VLNabc + d * Iabc;
            gb.calHeadI(dsIsland.getBusV().get(tn), dsIsland.getBranchTailI().get(obj), iTemp);
            for (int j = 0; j < 3; j++) {
                if (!gb.containsPhase(j))
                    continue;
                g[row++] = iTemp[j][0] - dsIsland.getBranchHeadI().get(obj)[j][0];
                g[row++] = iTemp[j][1] - dsIsland.getBranchHeadI().get(obj)[j][1];
            }
        }

        //计算联络节点电流约束
        for (int busNo : zeroPBuses) {
            tn = dsIsland.getBusNoToTn().get(busNo);
            for (int anotherBus : tn.getConnectedBusNo()) {
                obj = dsIsland.getGraph().getEdge(tn, dsIsland.getBusNoToTn().get(anotherBus));
                for (int p : tn.getPhases()) {
                    if (anotherBus < busNo) {
                        if (!dsIsland.getBranches().get(obj).containsPhase(p))
                            continue;
                        g[row + 2 * tn.getPhaseIndex(p)] += dsIsland.getBranchTailI().get(obj)[p][0];
                        g[row + 2 * tn.getPhaseIndex(p) + 1] += dsIsland.getBranchTailI().get(obj)[p][1];
                    } else {
                        if (!dsIsland.getBranches().get(obj).containsPhase(p))
                            continue;
                        g[row + 2 * tn.getPhaseIndex(p)] -= dsIsland.getBranchHeadI().get(obj)[p][0];
                        g[row + 2 * tn.getPhaseIndex(p) + 1] -= dsIsland.getBranchHeadI().get(obj)[p][1];
                    }
                }
            }
            row += 2 * tn.getPhases().length;
        }
        return row - oldRow;
    }

    public void setXLimit(double[] x_L, double[] x_U) {
        int size = branchOffset[0];//节点电压的上下限
        int i = 0;
        for (; i < size; i++) {
            x_L[i] = -2e15;
            x_U[i] = 2e15;
        }
        size = dimension;
        for (; i < size; i++) {
            x_L[i] = -2e15;//todo: this is not perfect
            x_U[i] = 2e15;
        }
        initialMeasLimit(dimension, x_L, x_U);
    }

    public void fillState(double[] x) {
        MapObject obj;
        GeneralBranch gb;
        Feeder feeder;
        int index;
        for (int i = 0; i < dsIsland.getBranches().size(); i++) {
            obj = dsIsland.getIdToBranch().get(i + 1);
            gb = dsIsland.getBranches().get(obj);
            if (gb instanceof Feeder) {
                feeder = (Feeder) gb;
                for (int p : feeder.getPhases()) {
                    index = branchOffset[i] + feeder.getPhaseIndex(p);
                    dsIsland.getBranchHeadI().get(obj)[p][0] = x[index];
                    dsIsland.getBranchHeadI().get(obj)[p][1] = x[index + feeder.getPhases().length];
                }
            } else {
                for (int p = 0; p < 3; p++) {
                    index = branchOffset[i] + p;
                    dsIsland.getBranchHeadI().get(obj)[p][0] = x[index];
                    dsIsland.getBranchHeadI().get(obj)[p][1] = x[index + 6];
                    dsIsland.getBranchTailI().get(obj)[p][0] = x[index + 3];
                    dsIsland.getBranchTailI().get(obj)[p][1] = x[index + 9];
                }
            }
        }
        for (int i = 0; i < busNumber; i++) {
            DsTopoNode tn = dsIsland.getBusNoToTn().get(i + 1);
            for (int p : tn.getPhases()) {
                index = nodeOffset[i] + tn.getPhaseIndex(p);
                dsIsland.getBusV().get(tn)[p][0] = x[index];
                dsIsland.getBusV().get(tn)[p][1] = x[index + tn.getPhases().length];

            }
        }
    }

    public DsTopoIsland getDsIsland() {
        return dsIsland;
    }

    public void setDsIsland(DsTopoIsland dsIsland) {
        this.dsIsland = dsIsland;
        stateCal.setIsland(dsIsland);
    }

    //Jacobian计算的方法
    public void fillPfJacobian(int row) {
        GeneralBranch gb;
        DsTopoNode tn, father, tmpTn;
        MapObject obj;
        Feeder feeder;
        Transformer transformer;

        for (int branchId = 1; branchId <= dsIsland.getIdToBranch().size(); branchId++) {
            obj = dsIsland.getIdToBranch().get(branchId);
            tn = dsIsland.getGraph().getEdgeTarget(obj);
            father = dsIsland.getGraph().getEdgeSource(obj);
            //注意电流正方向是节点编号小的流向大的
            //todo: 对于环状网络的变压器支路可能是错误的
            if (father.getBusNo() > tn.getBusNo()) {
                tmpTn = father;
                father = tn;
                tn = tmpTn;
            }
            obj = dsIsland.getGraph().getEdge(father, tn);
            gb = dsIsland.getBranches().get(obj);

            if (gb instanceof Feeder) {
                feeder = (Feeder) gb;
                //计算电压方程约束 VLNABC = a * VLNabc + b * Iabc的Jacobian;
                for (int i : feeder.getPhases()) {
                    this.jacobian.setQuick(row, nodeOffset[tn.getBusNo() - 1] + tn.getPhaseIndex(i), 1.0);
                    this.jacobian.setQuick(row + 1, nodeOffset[tn.getBusNo() - 1] + tn.getPhases().length + tn.getPhaseIndex(i), 1.0);
                    this.jacobian.setQuick(row, nodeOffset[father.getBusNo() - 1] + father.getPhaseIndex(i), -1.0);
                    this.jacobian.setQuick(row + 1, nodeOffset[father.getBusNo() - 1] + father.getPhaseIndex(i) + father.getPhases().length, -1.0);
                    for (int j : feeder.getPhases()) {
                        if (j != i && Math.abs(feeder.getZ_real()[i][j]) < DsModelCons.ZERO_LIMIT
                                && Math.abs(feeder.getZ_imag()[i][j]) < DsModelCons.ZERO_LIMIT)
                            continue;
                        this.jacobian.setQuick(row, branchOffset[branchId - 1] + feeder.getPhaseIndex(j), feeder.getZ_real()[i][j]);
                        this.jacobian.setQuick(row, branchOffset[branchId - 1] + feeder.getPhaseIndex(j) + feeder.getPhases().length, -feeder.getZ_imag()[i][j]);
                        this.jacobian.setQuick(row + 1, branchOffset[branchId - 1] + feeder.getPhaseIndex(j), feeder.getZ_imag()[i][j]);
                        this.jacobian.setQuick(row + 1, branchOffset[branchId - 1] + feeder.getPhaseIndex(j) + feeder.getPhases().length, feeder.getZ_real()[i][j]);
                    }
                    row += 2;
                }
            } else if (gb instanceof Transformer) {
                transformer = (Transformer) gb;
                tmpPos[0] = nodeOffset[tn.getBusNo() - 1];
                tmpPos[1] = nodeOffset[tn.getBusNo() - 1] + 3;
                tmpPos[2] = branchOffset[branchId - 1] + 3;
                tmpPos[3] = branchOffset[branchId - 1] + 9;
                tmpPos[4] = nodeOffset[father.getBusNo() - 1];
                tmpPos[5] = nodeOffset[father.getBusNo() - 1] + 3;
                //计算电压方程约束 VLNABC = a * VLNabc + b * Iabc的Jacobian;
                transformer.fillJacOfHeadV(jacobian, tmpPos, row);
                row += 6;

                if (dsIsland.getBranchHeadI().get(obj) == dsIsland.getBranchTailI().get(obj))
                    continue;
                //计算电流方程 IABC = c * VLNabc + d * Iabc的Jacobian
                tmpPos[4] = branchOffset[branchId - 1];
                tmpPos[5] = branchOffset[branchId - 1] + 6;
                transformer.fillJacOfHeadI(jacobian, tmpPos, row);
                row += 6;
            }
        }
        //计算联络节点电流约束的Jacobian
        double v;
        for (int busNo : zeroPBuses) {
            tn = dsIsland.getBusNoToTn().get(busNo);
            for (int anotherBus : tn.getConnectedBusNo()) {
                obj = dsIsland.getGraph().getEdge(tn, dsIsland.getBusNoToTn().get(anotherBus));
                int branchNo = Integer.parseInt(obj.getId());

                gb = dsIsland.getBranches().get(obj);
                if (gb instanceof Feeder) {
                    if (anotherBus < busNo) {
                        v = 1.0;
                    } else
                        v = -1.0;
                    feeder = (Feeder) gb;
                    for (int p : feeder.getPhases()) {
                        jacobian.setQuick(row + 2 * tn.getPhaseIndex(p),
                                branchOffset[branchNo - 1] + feeder.getPhaseIndex(p), v);
                        jacobian.setQuick(row + 2 * tn.getPhaseIndex(p) + 1,
                                branchOffset[branchNo - 1] + feeder.getPhaseIndex(p) + feeder.getPhases().length, v);
                    }
                } else {//该支路是变压器，目前只能处理三相的变压器
                    if (anotherBus < busNo) {
                        for (int p = 0; p < 3; p++) {
                            jacobian.setQuick(row + 2 * p, branchOffset[branchNo - 1] + 3 + p, 1.0);
                            jacobian.setQuick(row + 2 * p + 1, branchOffset[branchNo - 1] + 9 + p, 1.0);
                        }
                    } else {
                        for (int p = 0; p < 3; p++) {
                            jacobian.setQuick(row + 2 * p, branchOffset[branchNo - 1] + p, -1.0);
                            jacobian.setQuick(row + 2 * p + 1, branchOffset[branchNo - 1] + 6 + p, -1.0);
                        }
                    }
                }
            }
            row += tn.getPhases().length * 2;
        }
    }

    public int getNonZeroNumOfPfConstraints() {
        int capacity = 0;
        for (MapObject obj : dsIsland.getIdToBranch().values())
            capacity += dsIsland.getBranches().get(obj).getNonZeroNumOfJac();
        DsTopoNode tn;
        MapObject obj;
        GeneralBranch gb;
        for (int busNum : zeroPBuses) {
            tn = dsIsland.getBusNoToTn().get(busNum);
            for (int anotherBus : tn.getConnectedBusNo()) {
                obj = dsIsland.getGraph().getEdge(tn, dsIsland.getBusNoToTn().get(anotherBus));
                gb = dsIsland.getBranches().get(obj);
                if (gb instanceof Feeder) {
                    capacity += 2 * ((Feeder) gb).getPhases().length;
                } else
                    capacity += 6;
            }
        }
        return capacity;
    }

    public int getNonZeroNum(MeasVector meas) {
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
            DsTopoNode tn = dsIsland.getBusNoToTn().get(num);
            capacity += (2 * tn.getConnectedBusNo().length + 2);
        }
        for (int i = 0; i < meas.getBus_q_pos().length; i++) {
            int num = meas.getBus_q_pos()[i];//num starts from 1
            DsTopoNode tn = dsIsland.getBusNoToTn().get(num);
            capacity += (2 * tn.getConnectedBusNo().length + 2);
        }
        return capacity;
    }

    private void fillJacobian(MeasVector meas, int index) {
        DsTopoNode tn;
        double[] v, c;
        double jacOfVReal, jacOfVImage;
        int[] iPos = new int[2];
        MapObject obj;
        int branchNo, busNo, phase, vRealPos, vImagPos;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        busNo = meas.getBus_v_pos()[i];
                        phase = meas.getBus_v_phase()[i];
                        tn = dsIsland.getBusNoToTn().get(busNo);
                        v = dsIsland.getBusV().get(tn)[phase];
                        vRealPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase);
                        vImagPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase) + tn.getPhases().length;
                        jacobian.setQuick(index, vRealPos, 2.0 * v[0]);
                        jacobian.setQuick(index, vImagPos, 2.0 * v[1]);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        busNo = meas.getBus_p_pos()[i];//num starts from 1
                        phase = meas.getBus_p_phase()[i];
                        tn = dsIsland.getBusNoToTn().get(busNo);
                        v = dsIsland.getBusV().get(tn)[phase];
                        vRealPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase);
                        vImagPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase) + tn.getPhases().length;
                        jacOfVReal = 0.0;
                        jacOfVImage = 0.0;
                        for (int b : tn.getConnectedBusNo()) {
                            obj = dsIsland.getGraph().getEdge(tn, dsIsland.getBusNoToTn().get(b));
                            branchNo = Integer.parseInt(obj.getId());
                            if(!dsIsland.getBranches().get(obj).containsPhase(phase))
                                continue;
                            getIPos(branchNo, phase, busNo < b, iPos);
                            if (busNo < b) {
                                jacOfVReal += dsIsland.getBranchHeadI().get(obj)[phase][0];
                                jacOfVImage += dsIsland.getBranchHeadI().get(obj)[phase][1];
                                jacobian.setQuick(index, iPos[0], v[0]);
                                jacobian.setQuick(index, iPos[1], v[1]);
                            } else {
                                jacOfVReal -= dsIsland.getBranchTailI().get(obj)[phase][0];
                                jacOfVImage -= dsIsland.getBranchTailI().get(obj)[phase][1];
                                jacobian.setQuick(index, iPos[0], -v[0]);
                                jacobian.setQuick(index, iPos[1], -v[1]);
                            }
                        }
                        jacobian.setQuick(index, vRealPos, jacOfVReal);
                        jacobian.setQuick(index, vImagPos, jacOfVImage);
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        busNo = meas.getBus_q_pos()[i];
                        phase = meas.getBus_q_phase()[i];
                        tn = dsIsland.getBusNoToTn().get(busNo);
                        v = dsIsland.getBusV().get(tn)[phase];
                        vRealPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase);
                        vImagPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase) + tn.getPhases().length;
                        jacOfVReal = 0.0;
                        jacOfVImage = 0.0;
                        for (int b : tn.getConnectedBusNo()) {
                            obj = dsIsland.getGraph().getEdge(tn, dsIsland.getBusNoToTn().get(b));
                            branchNo = Integer.parseInt(obj.getId());
                            if(!dsIsland.getBranches().get(obj).containsPhase(phase))
                                continue;
                            getIPos(branchNo, phase, busNo < b, iPos);
                            if (busNo < b) {
                                jacOfVImage += dsIsland.getBranchHeadI().get(obj)[phase][0];
                                jacOfVReal -= dsIsland.getBranchHeadI().get(obj)[phase][1];
                                jacobian.setQuick(index, iPos[0], v[1]);
                                jacobian.setQuick(index, iPos[1], -v[0]);
                            } else {
                                jacOfVImage -= dsIsland.getBranchTailI().get(obj)[phase][0];
                                jacOfVReal += dsIsland.getBranchTailI().get(obj)[phase][1];
                                jacobian.setQuick(index, iPos[0], -v[1]);
                                jacobian.setQuick(index, iPos[1], v[0]);
                            }
                        }
                        jacobian.setQuick(index, vRealPos, jacOfVReal);
                        jacobian.setQuick(index, vImagPos, jacOfVImage);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        branchNo = meas.getLine_from_p_pos()[k];
                        phase = meas.getLine_from_p_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode src = dsIsland.getGraph().getEdgeSource(obj);
                        if (src.getBusNo() > dsIsland.getGraph().getEdgeTarget(obj).getBusNo())
                            src = dsIsland.getGraph().getEdgeTarget(obj);
                        vRealPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase);
                        vImagPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase) + src.getPhases().length;
                        v = dsIsland.getBusV().get(src)[phase];
                        c = dsIsland.getBranchHeadI().get(obj)[phase];
                        getIPos(branchNo, phase, true, iPos);
                        jacobian.setQuick(index, iPos[0], v[0]);
                        jacobian.setQuick(index, iPos[1], v[1]);
                        jacobian.setQuick(index, vRealPos, c[0]); //todo:
                        jacobian.setQuick(index, vImagPos, c[1]);
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        branchNo = meas.getLine_from_q_pos()[k];
                        phase = meas.getLine_from_q_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode src = dsIsland.getGraph().getEdgeSource(obj);
                        if (src.getBusNo() > dsIsland.getGraph().getEdgeTarget(obj).getBusNo())
                            src = dsIsland.getGraph().getEdgeTarget(obj);
                        vRealPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase);
                        vImagPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase) + src.getPhases().length;
                        v = dsIsland.getBusV().get(src)[phase];
                        c = dsIsland.getBranchHeadI().get(obj)[phase];
                        getIPos(branchNo, phase, true, iPos);
                        jacobian.setQuick(index, iPos[0], v[1]);
                        jacobian.setQuick(index, iPos[1], -v[0]);
                        jacobian.setQuick(index, vRealPos, -c[1]);
                        jacobian.setQuick(index, vImagPos, c[0]);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        branchNo = meas.getLine_to_p_pos()[k];
                        phase = meas.getLine_to_p_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode tar = dsIsland.getGraph().getEdgeTarget(obj);
                        if (dsIsland.getGraph().getEdgeSource(obj).getBusNo() > tar.getBusNo())
                            tar = dsIsland.getGraph().getEdgeSource(obj);
                        vRealPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase);
                        vImagPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase) + tar.getPhases().length;
                        v = dsIsland.getBusV().get(tar)[phase];
                        c = dsIsland.getBranchTailI().get(obj)[phase];
                        getIPos(branchNo, phase, false, iPos);
                        jacobian.setQuick(index, iPos[0], -v[0]);
                        jacobian.setQuick(index, iPos[1], -v[1]);
                        jacobian.setQuick(index, vRealPos, -c[0]);
                        jacobian.setQuick(index, vImagPos, -c[1]);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        branchNo = meas.getLine_to_q_pos()[k];
                        phase = meas.getLine_to_q_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode tar = dsIsland.getGraph().getEdgeTarget(obj);
                        if (dsIsland.getGraph().getEdgeSource(obj).getBusNo() > tar.getBusNo())
                            tar = dsIsland.getGraph().getEdgeSource(obj);
                        vRealPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase);
                        vImagPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase) + tar.getPhases().length;
                        v = dsIsland.getBusV().get(tar)[phase];
                        c = dsIsland.getBranchTailI().get(obj)[phase];
                        getIPos(branchNo, phase, false, iPos);
                        jacobian.setQuick(index, iPos[0], -v[1]);
                        jacobian.setQuick(index, iPos[1], v[0]);
                        jacobian.setQuick(index, vRealPos, c[1]);
                        jacobian.setQuick(index, vImagPos, -c[0]);
                    }
                    break;
                case TYPE_LINE_FROM_CURRENT:
                    for (int k = 0; k < meas.getLine_from_i_amp_pos().length; k++, index++) {
                        branchNo = meas.getLine_from_i_amp_pos()[k];//num starts from 1
                        phase = meas.getLine_from_i_amp_phase()[k];//num starts from 1
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        c = dsIsland.getBranchHeadI().get(obj)[phase];
                        getIPos(branchNo, phase, true, iPos);
                        jacobian.setQuick(index, iPos[0], 2 * c[0]);
                        jacobian.setQuick(index, iPos[1], 2 * c[1]);
                    }
                    break;
                case TYPE_LINE_TO_CURRENT:
                    for (int k = 0; k < meas.getLine_to_i_amp_pos().length; k++, index++) {
                        branchNo = meas.getLine_to_i_amp_pos()[k];//num starts from 1
                        phase = meas.getLine_to_i_amp_phase()[k];//num starts from 1
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        c = dsIsland.getBranchTailI().get(obj)[phase];
                        getIPos(branchNo, phase, false, iPos);
                        jacobian.setQuick(index, iPos[0], 2 * c[0]);
                        jacobian.setQuick(index, iPos[1], 2 * c[1]);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    private void initialHessian(MeasVector meas) {
        int[] iPos = new int[2];
        MapObject obj;
        DsTopoNode tn;
        int branchNo, busNo, phase, vRealPos, vImagPos;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++) {
                        busNo = meas.getBus_v_pos()[i];
                        phase = meas.getBus_v_phase()[i];
                        tn = dsIsland.getBusNoToTn().get(busNo);
                        vRealPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase);
                        vImagPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase) + tn.getPhases().length;
                        hessian.addQuick(vRealPos, vRealPos, 2.0);
                        hessian.addQuick(vImagPos, vImagPos, 2.0);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++) {
                        busNo = meas.getBus_p_pos()[i];//num starts from 1
                        phase = meas.getBus_p_phase()[i];
                        tn = dsIsland.getBusNoToTn().get(busNo);
                        vRealPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase);
                        vImagPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase) + tn.getPhases().length;
                        for (int b : tn.getConnectedBusNo()) {
                            obj = dsIsland.getGraph().getEdge(tn, dsIsland.getBusNoToTn().get(b));
                            branchNo = Integer.parseInt(obj.getId());
                            if(!dsIsland.getBranches().get(obj).containsPhase(phase))
                                continue;
                            getIPos(branchNo, phase, busNo < b, iPos);
                            if (busNo < b) {
                                hessian.addQuick(iPos[0], vRealPos, 1.0);
                                hessian.addQuick(iPos[1], vImagPos, 1.0);
                                hessian.addQuick(vRealPos, iPos[0], 1.0);
                                hessian.addQuick(vImagPos, iPos[1], 1.0);
                            } else {
                                hessian.addQuick(iPos[0], vRealPos, -1.0);
                                hessian.addQuick(iPos[1], vImagPos, -1.0);
                                hessian.addQuick(vRealPos, iPos[0], -1.0);
                                hessian.addQuick(vImagPos, iPos[1], -1.0);
                            }
                        }
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++) {
                        busNo = meas.getBus_q_pos()[i];
                        phase = meas.getBus_q_phase()[i];
                        tn = dsIsland.getBusNoToTn().get(busNo);
                        vRealPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase);
                        vImagPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase) + tn.getPhases().length;
                        for (int b : tn.getConnectedBusNo()) {
                            obj = dsIsland.getGraph().getEdge(tn, dsIsland.getBusNoToTn().get(b));
                            branchNo = Integer.parseInt(obj.getId());
                            if(!dsIsland.getBranches().get(obj).containsPhase(phase))
                                continue;
                            getIPos(branchNo, phase, busNo < b, iPos);
                            if (busNo < b) {
                                hessian.addQuick(iPos[0], vImagPos, 1.0);
                                hessian.addQuick(iPos[1], vRealPos, -1.0);
                                hessian.addQuick(vRealPos, iPos[1], -1.0);
                                hessian.addQuick(vImagPos, iPos[0], 1.0);
                            } else {
                                hessian.addQuick(iPos[0], vImagPos, -1.0);
                                hessian.addQuick(iPos[1], vRealPos, 1.0);
                                hessian.addQuick(vRealPos, iPos[1], 1.0);
                                hessian.addQuick(vImagPos, iPos[0], -1.0);
                            }
                        }
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++) {
                        branchNo = meas.getLine_from_p_pos()[k];
                        phase = meas.getLine_from_p_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode src = dsIsland.getGraph().getEdgeSource(obj);
                        DsTopoNode tar = dsIsland.getGraph().getEdgeTarget(obj);
                        if (src.getBusNo() > tar.getBusNo())
                            src = tar;
                        getIPos(branchNo, phase, true, iPos);
                        vRealPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase);
                        vImagPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase) + src.getPhases().length;
                        hessian.addQuick(iPos[0], vRealPos, 1.0);//(p/(ir,ur)
                        hessian.addQuick(iPos[1], vImagPos, 1.0);//(p/(iim.uim)
                        hessian.addQuick(vRealPos, iPos[0], 1.0); //(p/(ur,ir)
                        hessian.addQuick(vImagPos, iPos[1], 1.0);//(p/(uim,iim)
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++) {
                        branchNo = meas.getLine_from_q_pos()[k];
                        phase = meas.getLine_from_q_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode src = dsIsland.getGraph().getEdgeSource(obj);
                        DsTopoNode tar = dsIsland.getGraph().getEdgeTarget(obj);
                        if (src.getBusNo() > tar.getBusNo())
                            src = tar;
                        getIPos(branchNo, phase, true, iPos);
                        vRealPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase);
                        vImagPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase) + src.getPhases().length;
                        hessian.addQuick(iPos[0], vImagPos, 1.0);
                        hessian.addQuick(iPos[1], vRealPos, -1.0);
                        hessian.addQuick(vRealPos, iPos[1], -1.0);
                        hessian.addQuick(vImagPos, iPos[0], 1.0);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++) {
                        branchNo = meas.getLine_to_p_pos()[k];
                        phase = meas.getLine_to_p_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode src = dsIsland.getGraph().getEdgeSource(obj);
                        DsTopoNode tar = dsIsland.getGraph().getEdgeTarget(obj);
                        if (src.getBusNo() > tar.getBusNo())
                            tar = src;
                        getIPos(branchNo, phase, false, iPos);
                        vRealPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase);
                        vImagPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase) + tar.getPhases().length;
                        hessian.addQuick(iPos[0], vRealPos, -1.0);
                        hessian.addQuick(iPos[1], vImagPos, -1.0);
                        hessian.addQuick(vRealPos, iPos[0], -1.0);
                        hessian.addQuick(vImagPos, iPos[1], -1.0);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++) {
                        branchNo = meas.getLine_to_q_pos()[k];
                        phase = meas.getLine_to_q_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode src = dsIsland.getGraph().getEdgeSource(obj);
                        DsTopoNode tar = dsIsland.getGraph().getEdgeTarget(obj);
                        if (src.getBusNo() > tar.getBusNo())
                            tar = src;
                        getIPos(branchNo, phase, false, iPos);
                        vRealPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase);
                        vImagPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase) + tar.getPhases().length;
                        hessian.addQuick(iPos[0], vImagPos, -1.0);
                        hessian.addQuick(iPos[1], vRealPos, 1.0);
                        hessian.addQuick(vRealPos, iPos[1], 1.0);
                        hessian.addQuick(vImagPos, iPos[0], -1.0);
                    }
                    break;
                case TYPE_LINE_FROM_CURRENT:
                    for (int k = 0; k < meas.getLine_from_i_amp_pos().length; k++) {
                        branchNo = meas.getLine_from_i_amp_pos()[k];
                        phase = meas.getLine_from_i_amp_phase()[k];
                        getIPos(branchNo, phase, true, iPos);
                        hessian.addQuick(iPos[0], iPos[0], 2.0);
                        hessian.addQuick(iPos[1], iPos[1], 2.0);
                    }
                    break;
                case TYPE_LINE_TO_CURRENT:
                    for (int k = 0; k < meas.getLine_to_i_amp_pos().length; k++) {
                        branchNo = meas.getLine_to_i_amp_pos()[k];
                        phase = meas.getLine_to_i_amp_phase()[k];
                        getIPos(branchNo, phase, false, iPos);
                        hessian.addQuick(iPos[0], iPos[0], 2.0);
                        hessian.addQuick(iPos[1], iPos[1], 2.0);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    private void fillHessian(MeasVector meas, double[] lambda, int index) {
        int[] iPos = new int[2];
        MapObject obj;
        DsTopoNode tn;
        int branchNo, busNo, phase, vRealPos, vImagPos;
        for (int type : meas.getMeasureOrder()) {
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    for (int i = 0; i < meas.getBus_v_pos().length; i++, index++) {
                        busNo = meas.getBus_v_pos()[i];
                        phase = meas.getBus_v_phase()[i];
                        tn = dsIsland.getBusNoToTn().get(busNo);
                        vRealPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase);
                        vImagPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase) + tn.getPhases().length;
                        hessian.addQuick(vRealPos, vRealPos, 2.0 * lambda[index]);
                        hessian.addQuick(vImagPos, vImagPos, 2.0 * lambda[index]);
                    }
                    break;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_p_pos().length; i++, index++) {
                        busNo = meas.getBus_p_pos()[i];//num starts from 1
                        phase = meas.getBus_p_phase()[i];
                        tn = dsIsland.getBusNoToTn().get(busNo);
                        vRealPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase);
                        vImagPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase) + tn.getPhases().length;
                        for (int b : tn.getConnectedBusNo()) {
                            obj = dsIsland.getGraph().getEdge(tn, dsIsland.getBusNoToTn().get(b));
                            branchNo = Integer.parseInt(obj.getId());
                            if(!dsIsland.getBranches().get(obj).containsPhase(phase))
                                continue;
                            getIPos(branchNo, phase, busNo < b, iPos);
                            if (busNo < b) {
                                hessian.addQuick(iPos[0], vRealPos, lambda[index]);
                                hessian.addQuick(iPos[1], vImagPos, lambda[index]);
                                hessian.addQuick(vRealPos, iPos[0], lambda[index]);
                                hessian.addQuick(vImagPos, iPos[1], lambda[index]);
                            } else {
                                hessian.addQuick(iPos[0], vRealPos, -lambda[index]);
                                hessian.addQuick(iPos[1], vImagPos, -lambda[index]);
                                hessian.addQuick(vRealPos, iPos[0], -lambda[index]);
                                hessian.addQuick(vImagPos, iPos[1], -lambda[index]);
                            }
                        }
                    }
                    break;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i = 0; i < meas.getBus_q_pos().length; i++, index++) {
                        busNo = meas.getBus_q_pos()[i];
                        phase = meas.getBus_q_phase()[i];
                        tn = dsIsland.getBusNoToTn().get(busNo);
                        vRealPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase);
                        vImagPos = nodeOffset[busNo - 1] + tn.getPhaseIndex(phase) + tn.getPhases().length;
                        for (int b : tn.getConnectedBusNo()) {
                            obj = dsIsland.getGraph().getEdge(tn, dsIsland.getBusNoToTn().get(b));
                            branchNo = Integer.parseInt(obj.getId());
                            if(!dsIsland.getBranches().get(obj).containsPhase(phase))
                                continue;
                            getIPos(branchNo, phase, busNo < b, iPos);
                            if (busNo < b) {
                                hessian.addQuick(iPos[0], vImagPos, lambda[index]);
                                hessian.addQuick(iPos[1], vRealPos, -lambda[index]);
                                hessian.addQuick(vRealPos, iPos[1], -lambda[index]);
                                hessian.addQuick(vImagPos, iPos[0], lambda[index]);
                            } else {
                                hessian.addQuick(iPos[0], vImagPos, -lambda[index]);
                                hessian.addQuick(iPos[1], vRealPos, lambda[index]);
                                hessian.addQuick(vRealPos, iPos[1], lambda[index]);
                                hessian.addQuick(vImagPos, iPos[0], -lambda[index]);
                            }
                        }
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    for (int k = 0; k < meas.getLine_from_p_pos().length; k++, index++) {
                        branchNo = meas.getLine_from_p_pos()[k];
                        phase = meas.getLine_from_p_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode src = dsIsland.getGraph().getEdgeSource(obj);
                        if (src.getBusNo() > dsIsland.getGraph().getEdgeTarget(obj).getBusNo())
                            src = dsIsland.getGraph().getEdgeTarget(obj);
                        getIPos(branchNo, phase, true, iPos);
                        vRealPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase);
                        vImagPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase) + src.getPhases().length;
                        hessian.addQuick(iPos[0], vRealPos, lambda[index]);//(p/(ir,ur)
                        hessian.addQuick(iPos[1], vImagPos, lambda[index]);//(p/(iim.uim)
                        hessian.addQuick(vRealPos, iPos[0], lambda[index]); //(p/(ur,ir)
                        hessian.addQuick(vImagPos, iPos[1], lambda[index]);//(p/(uim,iim)
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    for (int k = 0; k < meas.getLine_from_q_pos().length; k++, index++) {
                        branchNo = meas.getLine_from_q_pos()[k];
                        phase = meas.getLine_from_q_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode src = dsIsland.getGraph().getEdgeSource(obj);
                        if (src.getBusNo() > dsIsland.getGraph().getEdgeTarget(obj).getBusNo())
                            src = dsIsland.getGraph().getEdgeTarget(obj);
                        getIPos(branchNo, phase, true, iPos);
                        vRealPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase);
                        vImagPos = nodeOffset[src.getBusNo() - 1] + src.getPhaseIndex(phase) + src.getPhases().length;
                        hessian.addQuick(iPos[0], vImagPos, lambda[index]);
                        hessian.addQuick(iPos[1], vRealPos, -lambda[index]);
                        hessian.addQuick(vRealPos, iPos[1], -lambda[index]);
                        hessian.addQuick(vImagPos, iPos[0], lambda[index]);
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    for (int k = 0; k < meas.getLine_to_p_pos().length; k++, index++) {
                        branchNo = meas.getLine_to_p_pos()[k];
                        phase = meas.getLine_to_p_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode tar = dsIsland.getGraph().getEdgeTarget(obj);
                        if (dsIsland.getGraph().getEdgeSource(obj).getBusNo() > tar.getBusNo())
                            tar = dsIsland.getGraph().getEdgeSource(obj);
                        getIPos(branchNo, phase, false, iPos);
                        vRealPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase);
                        vImagPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase) + tar.getPhases().length;
                        hessian.addQuick(iPos[0], vRealPos, -lambda[index]);
                        hessian.addQuick(iPos[1], vImagPos, -lambda[index]);
                        hessian.addQuick(vRealPos, iPos[0], -lambda[index]);
                        hessian.addQuick(vImagPos, iPos[1], -lambda[index]);
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    for (int k = 0; k < meas.getLine_to_q_pos().length; k++, index++) {
                        branchNo = meas.getLine_to_q_pos()[k];
                        phase = meas.getLine_to_q_phase()[k];
                        obj = dsIsland.getIdToBranch().get(branchNo);
                        DsTopoNode tar = dsIsland.getGraph().getEdgeTarget(obj);
                        if (dsIsland.getGraph().getEdgeSource(obj).getBusNo() > tar.getBusNo())
                            tar = dsIsland.getGraph().getEdgeSource(obj);
                        getIPos(branchNo, phase, false, iPos);
                        vRealPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase);
                        vImagPos = nodeOffset[tar.getBusNo() - 1] + tar.getPhaseIndex(phase) + tar.getPhases().length;
                        hessian.addQuick(iPos[0], vImagPos, -lambda[index]);
                        hessian.addQuick(iPos[1], vRealPos, lambda[index]);
                        hessian.addQuick(vRealPos, iPos[1], lambda[index]);
                        hessian.addQuick(vImagPos, iPos[0], -lambda[index]);
                    }
                    break;
                case TYPE_LINE_FROM_CURRENT:
                    for (int k = 0; k < meas.getLine_from_i_amp_pos().length; k++, index++) {
                        branchNo = meas.getLine_from_i_amp_pos()[k];
                        phase = meas.getLine_from_i_amp_phase()[k];
                        getIPos(branchNo, phase, true, iPos);
                        hessian.addQuick(iPos[0], iPos[0], 2.0 * lambda[index]);
                        hessian.addQuick(iPos[1], iPos[1], 2.0 * lambda[index]);
                    }
                    break;
                case TYPE_LINE_TO_CURRENT:
                    for (int k = 0; k < meas.getLine_to_i_amp_pos().length; k++, index++) {
                        branchNo = meas.getLine_to_i_amp_pos()[k];
                        phase = meas.getLine_to_i_amp_phase()[k];
                        getIPos(branchNo, phase, false, iPos);
                        hessian.addQuick(iPos[0], iPos[0], 2.0 * lambda[index]);
                        hessian.addQuick(iPos[1], iPos[1], 2.0 * lambda[index]);
                    }
                    break;
                default:
                    break;
            }
        }
    }

    private void getIPos(int branchNo, int phase, boolean isHeadI, int[] iPos) {
        MapObject obj = dsIsland.getIdToBranch().get(branchNo);
        GeneralBranch gb = dsIsland.getBranches().get(obj);
        if (gb instanceof Feeder) {
            Feeder feeder = (Feeder) gb;
            iPos[0] = branchOffset[branchNo - 1] + feeder.getPhaseIndex(phase);
            iPos[1] = branchOffset[branchNo - 1] + feeder.getPhaseIndex(phase) + feeder.getPhases().length;
        } else {
            if (isHeadI) {
                iPos[0] = branchOffset[branchNo - 1] + phase;
                iPos[1] = branchOffset[branchNo - 1] + phase + 6;
            } else {
                iPos[0] = branchOffset[branchNo - 1] + phase + 3;
                iPos[1] = branchOffset[branchNo - 1] + phase + 9;
            }
        }
    }
}
