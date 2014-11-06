package zju.dsmodel;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-6
 */
public interface DsModelCons {
    static final double sqrt3 = Math.sqrt(3.0);
    final static double tanB = Math.tan(Math.PI * -120.0 / 180.0);
    final static double tanC = Math.tan(Math.PI * 120.0 / 180.0);
    final static double cosB = Math.cos(Math.PI * -120.0 / 180.0);
    final static double cosC = Math.cos(Math.PI * 120.0 / 180.0);
    final static double sinB = Math.sin(Math.PI * -120.0 / 180.0);
    final static double sinC = Math.sin(Math.PI * 120.0 / 180.0);

    final static double ZERO_LIMIT = 1e-6;

    String KEY_RESOURCE_TYPE = "ResourceType";

    //key for load
    String KEY_LOAD_TYPE = "LoadType";//spot or distributed
    String KEY_LOAD_MODEL = "LoadModel";//PQ, I or Z
    String KEY_CONN_TYPE = "ConnectType";//Y or D
    String KEY_KW_PH1 = "LoadKW1";
    String KEY_KW_PH2 = "LoadKW2";
    String KEY_KW_PH3 = "LoadKW3";
    String KEY_KVAR_PH1 = "LoadKVar1";
    String KEY_KVAR_PH2 = "LoadKVar2";
    String KEY_KVAR_PH3 = "LoadKVar3";

    String KEY_CONNECTED_NODE = "ConnectedNode";//Node
    //key for DG
    String IM_CONN_Y = "Y";
    String IM_CONN_D = "D";

    String KEY_DG_MODEL = "DGModel";// IM or Load
    String KEY_IM_BASE_KVA = "BaseKva";
    String KEY_IM_BASE_KV = "BaseKv";
    String KEY_IM_P_OUT = "IMPout";
    String KEY_IM_PU_RS = "Rs";
    String KEY_IM_PU_XS = "Xs";
    String KEY_IM_PU_RR = "Rr";
    String KEY_IM_PU_XR = "Xr";
    String KEY_IM_PU_XM = "Xm";
    String KEY_PV_P_PH1 = "DgKw1";
    String KEY_PV_P_PH2 = "DgKw2";
    String KEY_PV_P_PH3 = "DgKw3";
    String KEY_PV_V_PH1 = "DgKv1";
    String KEY_PV_V_PH2 = "DgKv2";
    String KEY_PV_V_PH3 = "DgKv3";

    //key for feeder
    String KEY_LINE_LENGTH = "LineLength";
    String KEY_LINE_CONF = "LineConfigure";
    String KEY_LENGTH_UNIT = "LengthUnit";

    //key for switch
    String KEY_SWITCH_STATUS = "SwitchStatus";

    //key for transformer
    String KEY_KVA_S_TF = "KVA-S-TF";
    String KEY_PU_R_TF = "PU-R-TF";
    String KEY_PU_X_TF = "PU-X-TF";
    String KEY_KV_HIGH = "KV-High"; //USED IN TF
    String KEY_KV_LOW = "KV-Low";   //USED IN TF
    String KEY_KV_MIDDLE = "KV-Middle";
    String KEY_KV_BASE = "KV-Base";

    //key for regulator
    String KEY_LOCATION = "Location";
    String KEY_PHASES = "Phases";
    String KEY_MONITORING_PHASES = "Monitoring_phases";
    String KEY_BANDWIDTH = "Bandwidth";
    String KEY_RG_PT_NUM = "Npt";
    String KEY_RG_CT_NUM = "CTp";
    String KEY_RA_RG = "Ra";
    String KEY_RB_RG = "Rb";
    String KEY_RC_RG = "Rc";
    String KEY_XA_RG = "Xa";
    String KEY_XB_RG = "Xb";
    String KEY_XC_RG = "Xc";
    String KEY_VLA_RG = "VLA";
    String KEY_VLB_RG = "VLB";
    String KEY_VLC_RG = "VLC";

    //constants for transformer
    String TF_CONN_GrY_GrY = "Gr.Y-Gr.Y";
    String TF_CONN_D_D = "D-D";
    String TF_CONN_D_GrY = "D-Gr.Y";
    String TF_CONN_Y_D = "Y-D";
    //String TF_Wye = "3-Ph,LG";
    //String TF_OpenDelta = "AB-CB";

    //contant for load
    String LOAD_Y_PQ = "Y-PQ";
    String LOAD_Y_I = "Y-I";
    String LOAD_Y_Z = "Y-Z";
    String LOAD_D_PQ = "D-PQ";
    String LOAD_D_I = "D-I";
    String LOAD_D_Z = "D-Z";

    //contant for switch
    String SWITCH_ON = "on";
    String SWITCH_OFF = "off";

    //contants for length unit
    String LEN_UNIT_FEET = "feet";
    String LEN_UNIT_MILE = "mile";
    String LEN_UNIT_METER = "meter";
    String LEN_UNIT_KILOMETER = "kilometer";

    //contants for dg
    String DG_MODEL_IM = "IM";
    String DG_MODEL_PV = "PV";

    //resource constants
    String RESOURCE_SWITCH = "Switch";
    String RESOURCE_FEEDER = "Feeder";
    String RESOURCE_SHUNT_CAPACITORS = "ShuntCapacitors";
    String RESOURCE_SPOT_LOAD = "SpotLoad";
    String RESOURCE_DIS_LOAD = "DistributedLoad";
    String RESOURCE_TRANSFORMER = "Transformer";
    String RESOURCE_REGULATOR = "Regulator";
    String RESOURCE_DG = "DistributedGeneration";
}
