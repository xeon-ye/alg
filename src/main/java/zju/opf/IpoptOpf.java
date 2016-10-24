package zju.opf;

import cern.colt.function.IntIntDoubleFunction;
import org.apache.log4j.Logger;
import zju.common.IpoptSolver;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.pf.IpoptPf;
import zju.pf.PfResultInfo;
import zju.pf.PfResultMaker;
import zju.util.HessianMakerPC;
import zju.util.JacobianMakerPC;
import zju.util.StateCalByPolar;

import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * using bus's p,q,v,theta as variables
 *
 * @author Dong Shufeng
 *         Date: 2008-11-23
 */
public class IpoptOpf extends IpoptPf {
    private static Logger log = Logger.getLogger(IpoptOpf.class);

    //OPF计算参数，参数中的母线号都是节点编号前的
    protected OpfPara para;
    //母线号和母线之间的对应关系
    protected Map<Integer, BusData> busMap;
    //是否只做潮流
    private boolean isPfOnly = false;
    //是否平启动
    private boolean isFlatStart = false;
    //电压上限
    protected double vUpperLimPu = 2e15;
    //电压下限
    protected double vLowerLimPu = -2e15;

    /**
     * 计算OPF
     */
    public void doOpf() {
        iterNum = 0;
        if (clonedIsland == null) {
            log.warn("电气岛为NULL, 最优潮流计算中止.");
            return;
        }
        //与节点有关的控制信息都应该换成编号后节点号
        for (int i = 0; i < para.getP_ctrl_busno().length; i++)
            para.getP_ctrl_busno()[i] = numberOpt.getOld2new().get(para.getP_ctrl_busno()[i]);
        for (int i = 0; i < para.getV_ctrl_busno().length; i++)
            para.getV_ctrl_busno()[i] = numberOpt.getOld2new().get(para.getV_ctrl_busno()[i]);
        for (int i = 0; i < para.getQ_ctrl_busno().length; i++)
            para.getQ_ctrl_busno()[i] = numberOpt.getOld2new().get(para.getQ_ctrl_busno()[i]);

        initial();
        IpoptSolver solver = new IpoptSolver(this);
        solver.solve(getMaxIter(), tolerance, true);
        setConverged(solver.isConverged());
    }

    public void setOriIsland(IEEEDataIsland oriIsland) {
        super.setOriIsland(oriIsland);
        variableState = new double[4 * busNumber];
    }

    /**
     * 返回优化后的潮流结果
     */
    public PfResultInfo createPfResult() {
        if (isPfOnly())
            return super.createPfResult();
        if (isConverged()) {
            return PfResultMaker.getResult(oriIsland, variableState, Y, numberOpt.getOld2new());
        } else
            return null;
    }

    /**
     * 初始化函数，计算Jacobian和hession中的非零元个数，
     * 设置上下限
     */
    public void initial() {
        if (isPfOnly()) {
            super.initial();
            return;
        }

        busMap = clonedIsland.getBusMap();
        n = 4 * busNumber;
        //计算nele_jac的非零元个数
        nele_jac = JacobianMakerPC.getNonZeroNumOfFullState(Y);
        //约束的个数
        m = 2 * busNumber;//
        //为Jacobian开辟内存
        jacobian = new MySparseDoubleMatrix2D(m, n, nele_jac, 0.2, 0.9);
        //对Jacobian中不变的部分进行赋值
        for (int i = 0; i < busNumber; i++) {
            jacobian.setQuick(i, i + 2 * busNumber, -1.0);
            jacobian.setQuick(i + busNumber, i + 3 * busNumber, -1.0);
        }

        //为Hessian开辟内存
        hessian = new MySparseDoubleMatrix2D(4 * busNumber, 4 * busNumber);
        eval_hessian_struc_obj();
        for (int i = 0; i < busNumber; i++)
            HessianMakerPC.getStrucBusPQ(i + 1, Y, hessian);
        //计算Hessian中非零元个数
        nele_hess = hessian.cardinality();

        //开始对x的初值
        double mvaBase = clonedIsland.getTitle().getMvaBase();
        for (BusData bus : clonedIsland.getBuses()) {
            int i = bus.getBusNumber() - 1;
            if (isFlatStart) {
                variableState[i] = 1.0;
                variableState[i + busNumber] = .0;
            } else {
                variableState[i] = bus.getFinalVoltage();
                variableState[i + busNumber] = bus.getFinalAngle() * Math.PI / 180.0;
            }
            if (variableState[i] > vUpperLimPu || variableState[i] < vLowerLimPu)
                log.info("Over limit voltage, bus number " + (i + 1) + " value: " + variableState[i]);
        }
        double busP, busQ;
        for (BusData bus : clonedIsland.getBuses()) {
            int i = bus.getBusNumber() - 1;
            if (isFlatStart) {
                busP = StateCalByPolar.calBusP(i + 1, Y, variableState);
                busQ = StateCalByPolar.calBusQ(i + 1, Y, variableState);
            } else {
                busP = (bus.getGenerationMW() - bus.getLoadMW()) / mvaBase;
                busQ = (bus.getGenerationMVAR() - bus.getLoadMVAR()) / mvaBase;
            }
            variableState[i + 2 * busNumber] = busP;
            variableState[i + 3 * busNumber] = busQ;
        }
    }

    public boolean eval_f(int n, double[] x, boolean new_x, double[] obj_value) {
        if (isPfOnly()) {
            return super.eval_f(n, x, new_x, obj_value);
        }
        obj_value[0] = 0;
        switch (para.getObjFunction()) {
            case OpfPara.OBJ_MIN_SUM_P:
                for (int i = 0; i < clonedIsland.getPvBusSize() + clonedIsland.getSlackBusSize(); i++)
                    obj_value[0] += x[i + clonedIsland.getPqBusSize() + 2 * busNumber];
                break;
            case OpfPara.OBJ_MIN_VOLTAGE_OUTLIMIT:
                for (int i = 0; i < busNumber; i++)
                    if (x[i] < vLowerLimPu)
                        obj_value[0] += (vLowerLimPu - x[i]) * (vLowerLimPu - x[i]);
                    else if (x[i] > vUpperLimPu) {
                        obj_value[0] += (x[i] - vUpperLimPu) * (x[i] - vUpperLimPu);
                    }
                break;
            default:
                return true;
        }
        return true;
    }

    public boolean eval_grad_f(int n, double[] x, boolean new_x, double[] grad_f) {
        if (isPfOnly()) {
            return super.eval_grad_f(n, x, new_x, grad_f);
        }
        for (int i = 0; i < grad_f.length; i++)
            grad_f[i] = 0;
        switch (para.getObjFunction()) {
            case OpfPara.OBJ_MIN_SUM_P:
                for (int i = 0; i < clonedIsland.getPvBusSize() + clonedIsland.getSlackBusSize(); i++)
                    grad_f[i + clonedIsland.getPqBusSize() + 2 * busNumber] = 1;
                break;
            case OpfPara.OBJ_MIN_VOLTAGE_OUTLIMIT:
                for (int i = 0; i < busNumber; i++) {
                    BusData bus = busMap.get(i + 1);
                    if (bus.getType() == BusData.BUS_TYPE_LOAD_PQ || bus.getType() == BusData.BUS_TYPE_GEN_PQ) {
                        if (bus.getMinimum() > vLowerLimPu && x[i] < bus.getMinimum()) {
                            grad_f[i] = 2.0 * (x[i] - bus.getMinimum());
                            continue;
                        }
                        if (bus.getMaximum() > vLowerLimPu && bus.getMaximum() < vUpperLimPu && x[i] > bus.getMaximum()) {
                            grad_f[i] = 2.0 * (x[i] - bus.getMaximum());
                            continue;
                        }
                    }
                    if (x[i] < vLowerLimPu)
                        grad_f[i] = 2.0 * (x[i] - vLowerLimPu);
                    else if (x[i] > vUpperLimPu) {
                        grad_f[i] = 2.0 * (x[i] - vUpperLimPu);
                    }
                }
                break;
            case OpfPara.OBJ_MIN_ViolationAndAdjustment:

            default:
                return true;
        }
        return true;
    }

    protected void eval_hessian_struc_obj() {
        switch (para.getObjFunction()) {
            case OpfPara.OBJ_MIN_VOLTAGE_OUTLIMIT:
                for (int i = 0; i < busNumber; i++)
                    hessian.setQuick(i, i, 1.0);
                break;
            default:
                break;
        }
    }

    protected void fill_hessian_obj(double[] x, double obj_factor) {
        switch (para.getObjFunction()) {
            case OpfPara.OBJ_MIN_VOLTAGE_OUTLIMIT:
                for (int i = 0; i < busNumber; i++) {
                    BusData bus = busMap.get(i + 1);
                    if (bus.getType() == BusData.BUS_TYPE_LOAD_PQ || bus.getType() == BusData.BUS_TYPE_GEN_PQ) {
                        if (bus.getMinimum() > vLowerLimPu && x[i] < bus.getMinimum()) {
                            hessian.addQuick(i, i, 2.0 * obj_factor);
                            continue;
                        }
                        if (bus.getMaximum() > vLowerLimPu && bus.getMaximum() < vUpperLimPu && x[i] > bus.getMaximum()) {
                            hessian.addQuick(i, i, 2.0 * obj_factor);
                            continue;
                        }
                    }
                    if (x[i] < vLowerLimPu || x[i] > vUpperLimPu)
                        hessian.addQuick(i, i, 2.0 * obj_factor);
                }
                break;
            default:
                break;
        }
    }

    public boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g) {
        if (isPfOnly())
            return super.eval_g(n, x, new_x, m, g);
        for (int i = 0; i < busNumber; i++) {
            g[i] = StateCalByPolar.calBusP(i + 1, Y, x) - x[i + 2 * busNumber];
            g[i + busNumber] = StateCalByPolar.calBusQ(i + 1, Y, x) - x[i + 3 * busNumber];
        }
        return true;
    }

    protected void updateJacobian(double[] x, boolean isNewX) {
        if (isPfOnly()) {
            super.updateJacobian(x, isNewX);
            return;
        }
        for (int i = 0; i < busNumber; i++) {
            JacobianMakerPC.fillJacobian_bus_p(i + 1, Y, x, jacobian, i);
            JacobianMakerPC.fillJacobian_bus_q(i + 1, Y, x, jacobian, i + busNumber);
        }
    }

    protected void updateHessian(double[] x, boolean new_x, double obj_factor, double[] lambda) {
        if (isPfOnly()) {
            super.updateHessian(x, new_x, obj_factor, lambda);
            return;
        }
        //先置零
        hessian.forEachNonZero((i, j, v) -> 0.0);
        fill_hessian_obj(x, obj_factor);
        for (int i = 0; i < busNumber; i++) {
            //if (Math.abs(lambda[i]) > 1e-10)
            HessianMakerPC.getHessianBusP(i + 1, Y, x, hessian, lambda[i]);
            //if (Math.abs(lambda[i + busNumber]) > 1e-10)
            HessianMakerPC.getHessianBusQ(i + 1, Y, x, hessian, lambda[i + busNumber]);
        }
    }

    @Override
    public void setStartingPoint(double[] x) {
        System.arraycopy(variableState, 0, x, 0, variableState.length);
    }

    @Override
    public void fillState(double[] x) {
        System.arraycopy(x, 0, variableState, 0, variableState.length);
    }

    @Override
    public void setXLimit(double[] x_L, double[] x_U) {
        double mvaBase = clonedIsland.getTitle().getMvaBase();
        double busP, busQ;
        for (BusData bus : clonedIsland.getBuses()) {
            int i = bus.getBusNumber() - 1;
            switch (bus.getType()) {
                case BusData.BUS_TYPE_LOAD_PQ:
                case BusData.BUS_TYPE_GEN_PQ:
                    //有功无功是定值
                    busP = (bus.getGenerationMW() - bus.getLoadMW()) / mvaBase;
                    busQ = (bus.getGenerationMVAR() - bus.getLoadMVAR()) / mvaBase;
                    x_L[i + 2 * busNumber] = busP;
                    x_U[i + 2 * busNumber] = busP;
                    x_L[i + 3 * busNumber] = busQ;
                    x_U[i + 3 * busNumber] = busQ;
                    break;
                case BusData.BUS_TYPE_SLACK:
                    x_L[i] = bus.getFinalVoltage(); //v
                    x_U[i] = bus.getFinalVoltage();
                    x_L[i + busNumber] = bus.getFinalAngle() * Math.PI / 180.;
                    x_U[i + busNumber] = x_L[i + busNumber];
                    break;
                case BusData.BUS_TYPE_GEN_PV:
                    x_L[i] = bus.getFinalVoltage(); //v
                    x_U[i] = bus.getFinalVoltage();
                    busP = (bus.getGenerationMW() - bus.getLoadMW()) / mvaBase;
                    x_L[i + 2 * busNumber] = busP;
                    x_U[i + 2 * busNumber] = busP;
                    break;
                default:
                    break;
            }
        }
        for (BusData bus : clonedIsland.getBuses()) {
            int i = bus.getBusNumber() - 1;
            if (bus.getType() == BusData.BUS_TYPE_SLACK || bus.getType() == BusData.BUS_TYPE_GEN_PV) {
                //设置节点有功无功的上下限
                busQ = (bus.getGenerationMVAR() - bus.getLoadMVAR()) / mvaBase;
                x_L[i + 3 * busNumber] = -2e15;
                x_U[i + 3 * busNumber] = 2e15;
                if (bus.getMaximum() - bus.getMinimum() > 1) {
                    x_L[i + 3 * busNumber] = (bus.getMinimum() - bus.getLoadMVAR()) / mvaBase;
                    x_U[i + 3 * busNumber] = (bus.getMaximum() - bus.getLoadMVAR()) / mvaBase;
                    if (x_L[i + 3 * busNumber] - busQ > 0.0) {
                        x_L[i + 3 * busNumber] = busQ;
                        //x_L[i + 3 * busNumber] = -2e15;
                    } else if (busQ - x_U[i + 3 * busNumber] > 0.0) {
                        x_U[i + 3 * busNumber] = busQ;
                        //x_U[i + 3 * busNumber] = 2e15;
                    }
                }
            }
            if (bus.getType() == BusData.BUS_TYPE_LOAD_PQ ||
                    bus.getType() == BusData.BUS_TYPE_GEN_PQ
                    || bus.getType() == BusData.BUS_TYPE_GEN_PV) {
                //设置节点电压相角的上下限
                x_L[i + busNumber] = -Math.PI; //theta
                x_U[i + busNumber] = Math.PI;
            }

            switch (bus.getType()) {
                case BusData.BUS_TYPE_LOAD_PQ:
                case BusData.BUS_TYPE_GEN_PQ:
                    switch (para.getObjFunction()) {
                        case OpfPara.OBJ_MIN_VOLTAGE_OUTLIMIT:
                            x_L[i] = 0; //v
                            x_U[i] = 2.0;
                            break;
                        default:
                            x_L[i] = vLowerLimPu; //v
                            x_U[i] = vUpperLimPu;
                            //load node
                            if (bus.getType() == BusData.BUS_TYPE_LOAD_PQ || bus.getType() == BusData.BUS_TYPE_GEN_PQ) {
                                if (bus.getMinimum() > vLowerLimPu)
                                    x_L[i] = bus.getMinimum();
                                if (bus.getMaximum() > vLowerLimPu && bus.getMaximum() < vUpperLimPu)
                                    x_U[i] = bus.getMaximum();
                            }
                            break;
                    }
                    break;
                case BusData.BUS_TYPE_SLACK:
                    x_L[i + 2 * busNumber] = -2e15;
                    x_U[i + 2 * busNumber] = 2e15;
                    break;
                default:
                    break;
            }
        }

        if (para.getP_ctrl_L() != null && para.getP_ctrl_L().length > 0) {
            for (int i = 0; i < para.getP_ctrl_busno().length; i++) {
                int busNo = para.getP_ctrl_busno()[i];
                x_L[busNo - 1 + 2 * busNumber] = para.getP_ctrl_L()[i];
            }
        } else {
            for (int busNo : para.getP_ctrl_busno())
                x_L[busNo - 1 + 2 * busNumber] = -2e15;
        }
        if (para.getP_ctrl_U() != null && para.getP_ctrl_U().length > 0) {
            for (int i = 0; i < para.getP_ctrl_busno().length; i++) {
                int busNo = para.getP_ctrl_busno()[i];
                x_U[busNo - 1 + 2 * busNumber] = para.getP_ctrl_U()[i];
            }
        } else {
            for (int busNo : para.getP_ctrl_busno())
                x_U[busNo - 1 + 2 * busNumber] = 2e15;
        }

        if (para.getQ_ctrl_L() != null && para.getQ_ctrl_L().length > 0) {
            for (int i = 0; i < para.getQ_ctrl_busno().length; i++) {
                int busNo = para.getQ_ctrl_busno()[i];
                x_L[busNo - 1 + 3 * busNumber] = para.getQ_ctrl_L()[i];
            }
        } else {
            for (int busNo : para.getQ_ctrl_busno())
                x_L[busNo - 1 + 3 * busNumber] = -2e15;
        }
        if (para.getQ_ctrl_U() != null && para.getQ_ctrl_U().length > 0) {
            for (int i = 0; i < para.getQ_ctrl_busno().length; i++) {
                int busNo = para.getQ_ctrl_busno()[i];
                x_U[busNo - 1 + 3 * busNumber] = para.getQ_ctrl_U()[i];
            }
        } else {
            for (int busNo : para.getQ_ctrl_busno())
                x_U[busNo - 1 + 3 * busNumber] = 2e15;
        }

        if (para.getV_ctrl_L() != null && para.getV_ctrl_L().length > 0) {
            for (int i = 0; i < para.getV_ctrl_busno().length; i++) {
                int busNo = para.getV_ctrl_busno()[i];
                x_L[busNo - 1] = para.getV_ctrl_L()[i];
            }
        } else {
            for (int busNo : para.getV_ctrl_busno())
                x_L[busNo - 1] = vLowerLimPu;
        }
        if (para.getV_ctrl_U() != null && para.getV_ctrl_U().length > 0) {
            for (int i = 0; i < para.getV_ctrl_busno().length; i++) {
                int busNo = para.getV_ctrl_busno()[i];
                x_U[busNo - 1] = para.getV_ctrl_U()[i];
            }
        } else {
            for (int busNo : para.getV_ctrl_busno())
                x_U[busNo - 1] = vUpperLimPu;
        }
    }

    public void setGLimit(double[] g_L, double[] g_U) {
        for (int i = 0; i < busNumber; i++) {
            g_L[i] = -tol_p;
            g_U[i] = tol_p;
            g_L[i + busNumber] = -tol_q;
            g_U[i + busNumber] = tol_q;
        }
    }

    public OpfPara getPara() {
        return para;
    }

    public void setPara(OpfPara para) {
        this.para = para;
    }

    public boolean isPfOnly() {
        return isPfOnly;
    }

    public void setPfOnly(boolean pfOnly) {
        isPfOnly = pfOnly;
    }

    public boolean isFlatStart() {
        return isFlatStart;
    }

    public void setFlatStart(boolean flatStart) {
        isFlatStart = flatStart;
    }

    public double getvUpperLimPu() {
        return vUpperLimPu;
    }

    public void setvUpperLimPu(double vUpperLimPu) {
        this.vUpperLimPu = vUpperLimPu;
    }

    public double getvLowerLimPu() {
        return vLowerLimPu;
    }

    public void setvLowerLimPu(double vLowerLimPu) {
        this.vLowerLimPu = vLowerLimPu;
    }
}
