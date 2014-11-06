package zju.opf;

import java.io.Serializable;
import java.util.Date;
import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-1-13
 */
public class OpfPara implements Serializable {
    public final static int OBJ_MIN_P_LOSS = 1;
    public final static int OBJ_MIN_Q_LOSS = 2;
    public final static int OBJ_MIN_TOTAL_LOSS = 3;
    public final static int OBJ_MIN_PNET_LOSS = 4;
    public final static int OBJ_MIN_QNET_LOSS = 5;
    public final static int OBJ_MIN_SUM_P = 6;
    public final static int OBJ_MIN_SUM_Q = 7;
    public final static int OBJ_MIN_ViolationAndAdjustment = 8;
    public final static int OBJ_MAX_RESERVE_REACTIVEPOWER = 9;
    public final static int OBJ_MIN_VOLTAGE_OUTLIMIT = 10;

    public final static String EQUCONSTRAINT = "PowerFlowConstraints";

    public final static String INECONSTRAINT_LINEPOWER = "LinePowerBoundConstraints";
    public final static String INECONSTRAINT_VOLTAGE = "VoltageBoundConstraints";
    public final static String INECONSTRAINT_PGEN = "GeneratorPBoundConstraints";
    public final static String INECONSTRAINT_QGEN = "GeneratorQBoundConstraints";
    public final static String INECONSTRAINT_COMPENSATOR = "CompensatorBoundConstraints";
    public final static String INECONSTRAINT_TAPLIMIT = "TapChangerBoundConstraints";

    public int objFunction;
    /*
      * 本次计算的等式约束，等式约束可是 <1>、PowerFlowConstraints——潮流约束
      */
    public String equalityConstraint;
    /*
      * 本次计算的不等式约束，不等式约束可是
      * <1>LinePowerBoundConstraints——线路功率约束。
      * <2>VoltageBoundConstraints——电压约束。
      * <3>GeneratorPBoundConstraints——发电机有功约束。
      * <4>GeneratorQBoundConstraints——发电机无功约束。
      * <5>CompensatorBoundConstraints——补偿器组数与档位约束。
      * <6>TapChangerBoundConstraints——变压器档位约束。 每次计算可包含多个不等式约束
      */
    public List<String> inequalityConstraints;
    /*
      * 本次OPF计算系统名称
      *
      */
    public String name;
    /*
      * 本次OPF接口数据作者
      */
    public String vender;
    /*
      * OPF版本号
      */
    public String version;
    /*
      * 本次OPF接口数据生成时间
      */
    public Date genTime;
    /*
      * 本次OPF计算的收敛精度
      */
    public double tol;
    /*
      * 本次OPF计算的最大迭代次数
      */
    public int maxIteNum;
    /*
      * 本次OPF数据交换中，IEEE部分数据采用交换方式
      *
      * DataExchangeType内容为File与Data。DataExchangeType为File，<IEEEData>节点内容为数据文件名，数据文件格式为*.dat，
      * 文件内容为IEEE格式的待计算系统信息。当DataExchangeType为Data，<IEEEData>节点内容为IEEE格式的待计算系统
      * 信息。
      */
    public String dataExchangeType;

    private List<String> nodeIds;

    public List<String> pControlGenerator;

    public List<String> vControlGenerator;

    public List<String> vControlBuses;

    public List<String> qControlBuses;

    public Map<String, double[]> agcCapacity;

    public Map<String, double[]> genMwLimit;

    public Map<String, Double> adjustRatioes;

    public int[] p_ctrl_busno = new int[0];

    public int[] q_ctrl_busno = new int[0];

    public int[] v_ctrl_busno = new int[0];

    public double[] v_ctrl_step = new double[0];

    public double[] v_ctrl_L = new double[0];

    public double[] v_ctrl_U = new double[0];

    public double[] q_ctrl_step = new double[0];

    public double[] q_ctrl_L = new double[0];

    public double[] q_ctrl_U = new double[0];

    public double[] p_ctrl_step = new double[0];

    public double[] p_ctrl_L = new double[0];

    public double[] p_ctrl_U = new double[0];


    public List<String> getNodeIds() {
        return nodeIds;
    }

    public void setNodeIds(List<String> nodeIds) {
        this.nodeIds = nodeIds;
    }

    public Map<String, Double> getAdjustRatioes() {
        return adjustRatioes;
    }

    public void setAdjustRatioes(Map<String, Double> adjustRatioes) {
        this.adjustRatioes = adjustRatioes;
    }

    public List<String> getPControllableGenerator() {
        return pControlGenerator;
    }

    public void setPControllableGenerator(List<String> pControllableGenerator) {
        this.pControlGenerator = pControllableGenerator;
    }

    public List<String> getVControllableGenerator() {
        return vControlGenerator;
    }

    public void setVControllableGenerator(List<String> vControllableGenerator) {
        this.vControlGenerator = vControllableGenerator;
    }

    public List<String> getVControllableBuses() {
        return vControlBuses;
    }

    public void setVControllableBuses(List<String> vControllableBuses) {
        this.vControlBuses = vControllableBuses;
    }

    public List<String> getQControllableBuses() {
        return qControlBuses;
    }

    public void setQControllableBuses(List<String> qControllableBuses) {
        this.qControlBuses = qControllableBuses;
    }

    public Map<String, double[]> getAgcCapacity() {
        return agcCapacity;
    }

    public void setAgcCapacity(Map<String, double[]> agcCapacity) {
        this.agcCapacity = agcCapacity;
    }

    public Map<String, double[]> getGenMwLimit() {
        return genMwLimit;
    }

    public void setGenMwLimit(Map<String, double[]> genMwLimit) {
        this.genMwLimit = genMwLimit;
    }

    public String getEqualityConstraint() {
        return equalityConstraint;
    }

    public void setEqualityConstraint(String equalityConstraint) {
        this.equalityConstraint = equalityConstraint;
    }

    public List<String> getInequalityConstraints() {
        return inequalityConstraints;
    }

    public void setInequalityConstraints(List<String> inequalityConstraints) {
        this.inequalityConstraints = inequalityConstraints;
    }

    public int getObjFunction() {
        return objFunction;
    }

    public void setObjFunction(int objFunction) {
        this.objFunction = objFunction;
    }

    public Date getGenTime() {
        return genTime;
    }

    public void setGenTime(Date genTime) {
        this.genTime = genTime;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getVender() {
        return vender;
    }

    public void setVender(String vender) {
        this.vender = vender;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public double getTol() {
        return tol;
    }

    public void setTol(double d) {
        tol = d;
    }

    public int getMaxIteNum() {
        return maxIteNum;
    }

    public void setMaxIteNum(int maxIteNum) {
        this.maxIteNum = maxIteNum;
    }

    public String getDataExchangeType() {
        return dataExchangeType;
    }

    public void setDataExchangeType(String dataExchangeType) {
        this.dataExchangeType = dataExchangeType;
    }

    public List<String> getpControlGenerator() {
        return pControlGenerator;
    }

    public void setpControlGenerator(List<String> pControlGenerator) {
        this.pControlGenerator = pControlGenerator;
    }

    public List<String> getvControlGenerator() {
        return vControlGenerator;
    }

    public void setvControlGenerator(List<String> vControlGenerator) {
        this.vControlGenerator = vControlGenerator;
    }

    public List<String> getvControlBuses() {
        return vControlBuses;
    }

    public void setvControlBuses(List<String> vControlBuses) {
        this.vControlBuses = vControlBuses;
    }

    public List<String> getqControlBuses() {
        return qControlBuses;
    }

    public void setqControlBuses(List<String> qControlBuses) {
        this.qControlBuses = qControlBuses;
    }

    public int[] getP_ctrl_busno() {
        return p_ctrl_busno;
    }

    public void setP_ctrl_busno(int[] p_ctrl_busno) {
        this.p_ctrl_busno = p_ctrl_busno;
    }

    public int[] getQ_ctrl_busno() {
        return q_ctrl_busno;
    }

    public void setQ_ctrl_busno(int[] q_ctrl_busno) {
        this.q_ctrl_busno = q_ctrl_busno;
    }

    public int[] getV_ctrl_busno() {
        return v_ctrl_busno;
    }

    public void setV_ctrl_busno(int[] v_ctrl_busno) {
        this.v_ctrl_busno = v_ctrl_busno;
    }

    public double[] getV_ctrl_step() {
        return v_ctrl_step;
    }

    public void setV_ctrl_step(double[] v_ctrl_step) {
        this.v_ctrl_step = v_ctrl_step;
    }

    public double[] getV_ctrl_L() {
        return v_ctrl_L;
    }

    public void setV_ctrl_L(double[] v_ctrl_L) {
        this.v_ctrl_L = v_ctrl_L;
    }

    public double[] getV_ctrl_U() {
        return v_ctrl_U;
    }

    public void setV_ctrl_U(double[] v_ctrl_U) {
        this.v_ctrl_U = v_ctrl_U;
    }

    public double[] getQ_ctrl_step() {
        return q_ctrl_step;
    }

    public void setQ_ctrl_step(double[] q_ctrl_step) {
        this.q_ctrl_step = q_ctrl_step;
    }

    public double[] getQ_ctrl_L() {
        return q_ctrl_L;
    }

    public void setQ_ctrl_L(double[] q_ctrl_L) {
        this.q_ctrl_L = q_ctrl_L;
    }

    public double[] getQ_ctrl_U() {
        return q_ctrl_U;
    }

    public void setQ_ctrl_U(double[] q_ctrl_U) {
        this.q_ctrl_U = q_ctrl_U;
    }

    public double[] getP_ctrl_step() {
        return p_ctrl_step;
    }

    public void setP_ctrl_step(double[] p_ctrl_step) {
        this.p_ctrl_step = p_ctrl_step;
    }

    public double[] getP_ctrl_L() {
        return p_ctrl_L;
    }

    public void setP_ctrl_L(double[] p_ctrl_L) {
        this.p_ctrl_L = p_ctrl_L;
    }

    public double[] getP_ctrl_U() {
        return p_ctrl_U;
    }

    public void setP_ctrl_U(double[] p_ctrl_U) {
        this.p_ctrl_U = p_ctrl_U;
    }

    public static OpfPara getDefaultParameter() {
        //todo 默认参数的修改
        OpfPara para = new OpfPara();
        para.setDataExchangeType("Data");
        para.setName("default");
        para.setVender("anonymous");
        para.setGenTime(new Date(System.currentTimeMillis()));
        para.setVersion("version 1.0");
        para.setMaxIteNum(50);
        para.setTol(0.0001);
        para.setEqualityConstraint(EQUCONSTRAINT);
        return para;
    }
}
