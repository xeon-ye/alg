package zju.dsmodel;

import java.io.Serializable;

/**
 * 该类用于在表述三相配电网详细拓扑结构中的边
 *
 * @author Dong Shufeng
 * Date: 14-2-18
 */
public class DetailedEdge implements Serializable {
    public static final int EDGE_TYPE_SUPPLIER = 1;
    public static final int EDGE_TYPE_FEEDER = 2;
    public static final int EDGE_TYPE_TF_WINDING = 3;
    public static final int EDGE_TYPE_LOAD = 4;
    public static final int EDGE_TYPE_LOAD_TF_MIX = 5;
    public static final int EDGE_TYPE_DG = 6;
    public static final int EDGE_TYPE_DG_TF_MIX = 7;

    //该标志位用于确定变压器支路是否为源端
    private boolean isSource = false;
    //支路类型
    private int edgeType = -1;
    //相位0,1,2分别代表A,B,C或者AB,BC,CA
    private int phase = -1;
    //相关的拓扑节点编号1
    private int tnNo1 = -1;
    //相关的拓扑节点编号2
    private int tnNo2 = -1;
    //对于非负荷型支路，只对于一个设备，该变量为该设备的id
    private String devId;
    //可能会出现一条支路对应多个设备的情况，其中之一就是DG
    private String dgId;
    //对于馈线型支路，该变量为同一馈线的其他支路
    private DetailedEdge[] otherEdgesOfSameFeeder;
    //对于变压器绕组型支路，该变量是对应的绕组
    private DetailedEdge otherEdgeOfTf;
    //恒阻抗负荷部分
    private double z_real = 0, z_image = 0;
    //恒功率负荷部分
    private double s_real = 0, s_image = 0;
    //恒电流负荷部分
    private double i_ampl = 0, i_angle = 0;
    //该标志位用于确定负荷连接是D还是Y
    private boolean isLoadD = false;

    public DetailedEdge() {
    }

    //可用于电源型支路的生成
    public DetailedEdge(int edgeType, int tnNo1, int phase) {
        this.edgeType = edgeType;
        this.tnNo1 = tnNo1;
        this.phase = phase;
    }

    //可用于变压器绕组型支路的生成
    public DetailedEdge(int edgeType, int tnNo1, int phase, boolean isSource, String devId) {
        this.edgeType = edgeType;
        this.tnNo1 = tnNo1;
        this.phase = phase;
        this.isSource = isSource;
        this.devId = devId;
    }

    //可用于馈线型支路的生成
    public DetailedEdge(int edgeType, int tnNo1, int tnNo2, int phase, String devId) {
        this.edgeType = edgeType;
        this.tnNo1 = tnNo1;
        this.tnNo2 = tnNo2;
        this.phase = phase;
        this.devId = devId;
    }

    public boolean isSource() {
        return isSource;
    }

    public void setSource(boolean isSupplier) {
        this.isSource = isSupplier;
    }

    public int getPhase() {
        return phase;
    }

    public void setPhase(int phase) {
        this.phase = phase;
    }

    public int getTnNo1() {
        return tnNo1;
    }

    public void setTnNo1(int tnNo1) {
        this.tnNo1 = tnNo1;
    }

    public int getTnNo2() {
        return tnNo2;
    }

    public void setTnNo2(int tnNo2) {
        this.tnNo2 = tnNo2;
    }

    public int getEdgeType() {
        return edgeType;
    }

    public void setEdgeType(int edgeType) {
        this.edgeType = edgeType;
    }

    public String getDevId() {
        return devId;
    }

    public void setDevId(String devId) {
        this.devId = devId;
    }

    public DetailedEdge[] getOtherEdgesOfSameFeeder() {
        return otherEdgesOfSameFeeder;
    }

    public void setOtherEdgesOfSameFeeder(DetailedEdge[] otherEdgesOfSameFeeder) {
        this.otherEdgesOfSameFeeder = otherEdgesOfSameFeeder;
    }

    public DetailedEdge getOtherEdgeOfTf() {
        return otherEdgeOfTf;
    }

    public void setOtherEdgeOfTf(DetailedEdge otherEdgeOfTf) {
        this.otherEdgeOfTf = otherEdgeOfTf;
    }

    public double getZ_real() {
        return z_real;
    }

    public void setZ_real(double z_real) {
        this.z_real = z_real;
    }

    public double getZ_image() {
        return z_image;
    }

    public void setZ_image(double z_image) {
        this.z_image = z_image;
    }

    public double getS_real() {
        return s_real;
    }

    public void setS_real(double s_real) {
        this.s_real = s_real;
    }

    public double getS_image() {
        return s_image;
    }

    public void setS_image(double s_image) {
        this.s_image = s_image;
    }

    public boolean isLoadD() {
        return isLoadD;
    }

    public void setLoadD(boolean isLoadD) {
        this.isLoadD = isLoadD;
    }

    public double getI_ampl() {
        return i_ampl;
    }

    public void setI_ampl(double i_ampl) {
        this.i_ampl = i_ampl;
    }

    public double getI_angle() {
        return i_angle;
    }

    public void setI_angle(double i_angle) {
        this.i_angle = i_angle;
    }

    public String getDgId() {
        return dgId;
    }

    public void setDgId(String dgId) {
        this.dgId = dgId;
    }
}
