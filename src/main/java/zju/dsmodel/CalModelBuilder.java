package zju.dsmodel;

import org.apache.log4j.Logger;
import zju.devmodel.MapObject;
import zju.util.DoubleMatrixToolkit;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 14-1-15
 */
public class CalModelBuilder implements DsModelCons, Serializable {
    private static Logger log = Logger.getLogger(CalModelBuilder.class);

    private FeederConfMgr feederConf;

    boolean isPerUnitSys;

    private Double baseKva = 1000.;

    public void createCalDevModel(DsTopoIsland island) {

        int branchNum = island.getIdToBranch().size();

        int loadNum = island.getSpotLoadNum();
        loadNum += island.getDistriLoadNum();
        loadNum += island.getShuntCapacitorNum();

        HashMap<MapObject, GeneralBranch> branches = new HashMap<>(branchNum);
        HashMap<MapObject, ThreePhaseLoad> loads = new HashMap<>(loadNum);
        HashMap<MapObject, DispersedGen> dgs = new HashMap<>(island.getDistriGenNum());
        for (MapObject obj : island.getIdToBranch().values()) {
            String devType = obj.getProperty(KEY_RESOURCE_TYPE);
            if (RESOURCE_FEEDER.equals(devType)) {
                GeneralBranch feeder = dealFeeder(obj);
                branches.put(obj, feeder);
            } else if (RESOURCE_TRANSFORMER.equals(devType)) {
                GeneralBranch tf = dealTransformer(obj);
                branches.put(obj, tf);
            } else if (RESOURCE_REGULATOR.equals(devType)) {
                GeneralBranch rg = dealRegulator(obj);
                branches.put(obj, rg);
            }
        }
        for (DsTopoNode tn : island.getTns()) {
            for (MapObject obj : tn.getConnectedDev()) {
                String devType = obj.getProperty(KEY_RESOURCE_TYPE);
                if (RESOURCE_SPOT_LOAD.equals(devType)
                        || RESOURCE_SHUNT_CAPACITORS.equals(devType)) {
                    BasicLoad load = dealLoad(obj);
                    loads.put(obj, load);
                } else if (RESOURCE_DIS_LOAD.equals(devType)) {
                    if (loads.containsKey(obj))
                        continue;
                    BasicLoad load = dealLoad(obj);
                    loads.put(obj, load);
                } else if (RESOURCE_DG.equals(devType)) {
                    DispersedGen dg = dealDisGeneration(obj);
                    dg.setTn(tn);
                    dgs.put(obj, dg);
                }
            }
        }
        island.setBranches(branches);
        island.setLoads(loads);
        island.setDispersedGens(dgs);
        for (DsTopoNode tn : island.getTns())
            tn.initialPhases();
    }

    public GeneralBranch dealFeeder(MapObject obj) {
        Feeder feeder = new Feeder();
        String configName = obj.getProperty(KEY_LINE_CONF);
        String length = obj.getProperty(KEY_LINE_LENGTH);
        String unit = obj.getProperty(KEY_LENGTH_UNIT);
        double baseKv = Double.parseDouble(obj.getProperty(KEY_KV_BASE));

        feeder.setZ_real(new double[3][3]);
        feeder.setZ_imag(new double[3][3]);
        feeder.setY_imag(new double[3][3]);

        if (isPerUnitSys)
            feederConf.calPara(configName, length, unit, baseKva, baseKv, feeder);
        else
            feederConf.calPara(configName, length, unit, feeder);
        feeder.initialPhases();
        return feeder;
    }

    public GeneralBranch dealRegulator(MapObject obj) {//todo: not finished
        double[] voltageLevel = new double[3];
        double[] R = new double[3];
        double[] X = new double[3];
        double Bandwidth = Double.parseDouble(obj.getProperties().get(KEY_BANDWIDTH));
        voltageLevel[0] = Double.parseDouble(obj.getProperties().get(KEY_VLA_RG));
        voltageLevel[1] = Double.parseDouble(obj.getProperties().get(KEY_VLB_RG));
        voltageLevel[2] = Double.parseDouble(obj.getProperties().get(KEY_VLC_RG));
        R[0] = Double.parseDouble(obj.getProperties().get(KEY_RA_RG));
        R[1] = Double.parseDouble(obj.getProperties().get(KEY_RB_RG));
        R[2] = Double.parseDouble(obj.getProperties().get(KEY_RC_RG));
        X[0] = Double.parseDouble(obj.getProperties().get(KEY_XA_RG));
        X[1] = Double.parseDouble(obj.getProperties().get(KEY_XB_RG));
        X[2] = Double.parseDouble(obj.getProperties().get(KEY_XC_RG));
        double Npt = Double.parseDouble(obj.getProperties().get(KEY_RG_PT_NUM));
        double CTp = Double.parseDouble(obj.getProperties().get(KEY_RG_CT_NUM));

        if (obj.getProperties().get(KEY_CONN_TYPE).equals("3-Ph,LG")) {
            RG_Wye wye = new RG_Wye();
            wye.initial(Bandwidth, voltageLevel, R, X, Npt, CTp);
            wye.formPara();
            //return wye;
        } else if (obj.getProperties().get(KEY_CONN_TYPE).equals("AB-CB")) {
            RG_OpenDelta od = new RG_OpenDelta();
            od.initial(Bandwidth, voltageLevel, R, X, Npt, CTp);
            od.formPara();
            //return od;
        }
        return null;
    }

    public BasicLoad dealLoad(MapObject obj) {
        double sAr = 1000. * Double.parseDouble(obj.getProperties().get(KEY_KW_PH1));
        double sAi = 1000. * Double.parseDouble(obj.getProperties().get(KEY_KVAR_PH1));
        double sBr = 1000. * Double.parseDouble(obj.getProperties().get(KEY_KW_PH2));
        double sBi = 1000. * Double.parseDouble(obj.getProperties().get(KEY_KVAR_PH2));
        double sCr = 1000. * Double.parseDouble(obj.getProperties().get(KEY_KW_PH3));
        double sCi = 1000. * Double.parseDouble(obj.getProperties().get(KEY_KVAR_PH3));
        double baseV = 1000. * Double.parseDouble(obj.getProperty(KEY_KV_BASE));

        double[][] s = new double[][]{{sAr, sAi}, {sBr, sBi}, {sCr, sCi}};
        String loadModel = obj.getProperties().get(KEY_LOAD_MODEL);
        String resourceType = obj.getProperties().get(KEY_RESOURCE_TYPE);
        if (RESOURCE_SHUNT_CAPACITORS.equals(resourceType)) {
            if (loadModel.equals(LOAD_Y_Z)) {
                DoubleMatrixToolkit.selfMul(s, -1.0);
                BasicLoad load = new BasicLoad(LOAD_Y_Z);
                load.formPara(s, baseV);
                return load;
            } else {
                log.warn("UNKNOWN SHUNT CAPACITORS");
                return null;
            }
        } else {
            if (resourceType.equals(RESOURCE_DIS_LOAD))
                DoubleMatrixToolkit.selfMul(s, 0.5);
            BasicLoad load = new BasicLoad(loadModel);
            if (isPerUnitSys) {
                DoubleMatrixToolkit.selfMul(s, 1.0 / (1000. * baseKva));
                load.formPara(s, 1.0);
            } else {
                load.formPara(s, baseV);
            }
            return load;
        }
    }

    public GeneralBranch dealTransformer(MapObject obj) {
        //turn pu value to real value <R&X>
        //注意文件中读取出来的是百分数
        double r_pu = Double.parseDouble(obj.getProperties().get(KEY_PU_R_TF)) / 100.;
        double x_pu = Double.parseDouble(obj.getProperties().get(KEY_PU_X_TF)) / 100;
        double sn = Double.parseDouble(obj.getProperties().get(KEY_KVA_S_TF));
        double v_high_tf = Double.parseDouble(obj.getProperties().get(KEY_KV_HIGH));
        double v_low_tf = Double.parseDouble(obj.getProperties().get(KEY_KV_LOW));
        Transformer tf = new Transformer(v_high_tf, v_low_tf, sn, r_pu, x_pu);
        if (obj.getProperty(KEY_CONN_TYPE).equals(TF_CONN_D_GrY)) {
            tf.setConnType(Transformer.CONN_TYPE_D_GrY);
            tf.formPara();
            return tf;
        } else if (obj.getProperty(KEY_CONN_TYPE).equals(TF_CONN_GrY_GrY)) {
            tf.setConnType(Transformer.CONN_TYPE_GrY_GrY);
            tf.formPara();
            return tf;
        } else if (obj.getProperty(KEY_CONN_TYPE).equals(TF_CONN_D_D)) {
            tf.setConnType(Transformer.CONN_TYPE_D_D);
            tf.formPara();
            return tf;
        } else if (obj.getProperty(KEY_CONN_TYPE).equals(TF_CONN_Y_D)) {
            tf.setConnType(Transformer.CONN_TYPE_Y_D);
            tf.formPara();
            return tf;
        } else
            log.warn("Not supported connection type: " + obj.getProperty(KEY_CONN_TYPE));
        return null;
    }

    public DispersedGen dealDisGeneration(MapObject obj) {
        if (obj.getProperty(KEY_DG_MODEL).equals(DG_MODEL_PV)) {
            double p1 = Double.parseDouble(obj.getProperties().get(KEY_PV_P_PH1));//Line-to-line voltage
            double p2 = Double.parseDouble(obj.getProperties().get(KEY_PV_P_PH2));
            double p3 = Double.parseDouble(obj.getProperties().get(KEY_PV_P_PH3));
            double v1 = Double.parseDouble(obj.getProperties().get(KEY_PV_V_PH1));
            double v2 = Double.parseDouble(obj.getProperties().get(KEY_PV_V_PH2));
            double v3 = Double.parseDouble(obj.getProperties().get(KEY_PV_V_PH3));
            int count = 0;
            if (p1 > ZERO_LIMIT)
                count++;
            if (p2 > ZERO_LIMIT)
                count++;
            if (p3 > ZERO_LIMIT)
                count++;
            DispersedGen dg = new DispersedGen(DispersedGen.MODE_PV);
            dg.setPhases(new int[count]);
            dg.setpOutput(new double[count]);
            dg.setvAmpl(new double[count]);
            count = 0;
            if (p1 > ZERO_LIMIT) {
                dg.getpOutput()[count] = p1 * 1000;
                dg.getvAmpl()[count] = v1 * 1000;
                dg.getPhases()[count] = 0;
                count++;
            }
            if (p2 > ZERO_LIMIT) {
                dg.getpOutput()[count] = p2 * 1000;
                dg.getvAmpl()[count] = v2 * 1000;
                dg.getPhases()[count] = 1;
                count++;
            }
            if (p3 > ZERO_LIMIT) {
                dg.getpOutput()[count] = p3 * 1000;
                dg.getvAmpl()[count] = v3 * 1000;
                dg.getPhases()[count] = 2;
            }
            return dg;
        } else if (obj.getProperty(KEY_DG_MODEL).equals(DG_MODEL_IM)) {
            double baseKva = Double.parseDouble(obj.getProperties().get(KEY_IM_BASE_KVA));
            double pOut = Double.parseDouble(obj.getProperties().get(KEY_IM_P_OUT));
            double baseKv = Double.parseDouble(obj.getProperties().get(KEY_IM_BASE_KV));//Line-to-line voltage
            double rs_pu = Double.parseDouble(obj.getProperties().get(KEY_IM_PU_RS));
            double xs_pu = Double.parseDouble(obj.getProperties().get(KEY_IM_PU_XS));
            double rr_pu = Double.parseDouble(obj.getProperties().get(KEY_IM_PU_RR));
            double xr_pu = Double.parseDouble(obj.getProperties().get(KEY_IM_PU_XR));
            double xm_pu = Double.parseDouble(obj.getProperties().get(KEY_IM_PU_XM));
            double zbase = baseKv * baseKv / baseKva * 1000.;

            InductionMachine motor = new InductionMachine();
            motor.setUp(rs_pu * zbase, xs_pu * zbase, rr_pu * zbase, xr_pu * zbase, xm_pu * zbase);
            motor.setP(-pOut * 1000.);
            if (obj.getProperty(KEY_CONN_TYPE).equals(IM_CONN_Y)) {
                motor.setConnType(InductionMachine.CONN_TYPE_Y);
            } else if (obj.getProperty(KEY_CONN_TYPE).equals(IM_CONN_D)) {
                motor.setConnType(InductionMachine.CONN_TYPE_D);
            }
            DispersedGen dg = new DispersedGen(DispersedGen.MODE_IM);
            dg.setMotor(motor);
            return dg;
        } else
            return null;//todo
    }

    public FeederConfMgr getFeederConf() {
        return feederConf;
    }

    public void setFeederConf(FeederConfMgr feederConf) {
        this.feederConf = feederConf;
    }

    public boolean isPerUnitSys() {
        return isPerUnitSys;
    }

    public void setPerUnitSys(boolean perUnitSys) {
        isPerUnitSys = perUnitSys;
    }

    public Double getBaseKva() {
        return baseKva;
    }

    public void setBaseKva(double baseKva) {
        this.baseKva = baseKva;
    }
}
