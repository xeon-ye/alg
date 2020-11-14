package zju.hems;

import ilog.concert.IloException;
import ilog.concert.IloNumExpr;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DemandRespModel extends SelfOptModel {
    Map<String, UserResult> selfOptResult;  // 自趋优结果
    Map<String, UserResult> demandRespResult;   // 100%需求响应结果
    int[] peakShaveTime;   // 削峰时段
    double clearingPrice;   // 出清价格
    Map<String, double[]> peakShavePowers;    // 应削峰量
    double totalPeakShaveCap;   // 总应削峰量
    Map<String, Double> peakShaveCapRatios;   // 用户应削峰容量占总削峰容量的比例
    List<Map<String, Double>> timeShaveCapRatios;   // 每个时刻用户应削峰容量占总削峰容量的比例
    Map<String, Double> peakShaveRatios; // 削峰比例
    List<Offer> offers; // 报价
    Map<String, double[]> shaveGatePowers = new HashMap<>();
    Map<String, double[]> mcs;
    Map<String, double[]> peakShaveCaps = new HashMap<>();

    double[] gatePowerSum;  // 总关口功率

    public DemandRespModel(Microgrid microgrid, int periodNum, double t, double[] elecPrices, double[] gasPrices, double[] steamPrices) {
        super(microgrid, periodNum, t, elecPrices, gasPrices, steamPrices);
    }

    public DemandRespModel(Microgrid microgrid, int periodNum, double t, double[] elecPrices, double[] gasPrices,
                           double[] steamPrices, double[] chargePrices) {
        super(microgrid, periodNum, t, elecPrices, gasPrices, steamPrices, chargePrices);
    }

    public void mgOrigDemandResp() {
        microgridResult = new HashMap<>(microgrid.getUsers().size());
        for (User user : microgrid.getUsers().values()) {
//            origDemandResp(user);
            gzTestDR(user);
//            chargingPileDR(user);
        }
    }

    /**
     * 削峰功率在目标函数中
     */
    public void gzTestDR(User user) {
        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
        List<AirCon> airCons = user.getAirCons();
        List<Converter> converters = user.getConverters();
        List<GasBoiler> gasBoilers = user.getGasBoilers();
        List<GasTurbine> gasTurbines = user.getGasTurbines();
        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
        List<Storage> storages = user.getStorages();
        SteamLoad steamLoad = user.getSteamLoad();
        Photovoltaic photovoltaic = user.getPhotovoltaic();
        double[] acLoad = user.acLoad;
        double[] dcLoad = user.dcLoad;
        double[] heatLoad = user.heatLoad;
        double[] coolingLoad1 = user.coolingLoad1;
        double[] coolingLoad2 = user.coolingLoad2;
        double[] gatePowers = user.gatePowers;
        try {
            int peakShaveTimeNum = 0;
            for (int i = 0; i < peakShaveTime.length; i++) {
                peakShaveTimeNum += peakShaveTime[i];
            }
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，|P-Pref|，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率，储能功率变化变量
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 + storages.size();
            int varNum = periodNum * periodVarNum;
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[varNum];
            // 状态变量上限
            double columnUpper[] = new double[varNum];
            // 指明变量类型
            IloNumVarType[] xt = new IloNumVarType[varNum];
            // 约束方程系数
            double[][] coeff = new double[(1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    periodNum +
                    2 * peakShaveTimeNum +
                    3 * periodNum +
                    periodNum + 2 * (periodNum - 1) * storages.size()][varNum];

            // 设置变量上下限和类型
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[j * periodVarNum + i] = 0;
                    columnUpper[j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + converters.size() + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn() + storages.get(i).getMaxPOut();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            double coef1 = 1e6;
            double coef2 = 1e-3;
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    objValue[j * periodVarNum + i] = coef2 * iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    objValue[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = coef2 * gasTurbines.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = coef2 * gasTurbines.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += coef2 * gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = coef2 * storages.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + i] += coef2 * storages.get(i).getCbw() * t;    // 折旧成本
                    objValue[j * periodVarNum + handledVarNum + storages.size() + i] = coef2 * storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                }

                handledVarNum += 2 * storages.size();
                handledVarNum += 2 * converters.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = coef2 * elecPrices[j] * t; // 购电成本

                handledVarNum += 1;
                if (peakShaveTime[j] == 1) {
                    objValue[j * periodVarNum + handledVarNum] = coef1; // 削峰目标
                }

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = coef2 * airCons.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = coef2 * gasBoilers.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = coef2 * gasBoilers.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += coef2 * gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = coef2 * absorptionChillers.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += absorptionChillers.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = coef2 * steamPrices[j] * t; // 购热成本

                handledVarNum += 1;

                if (j > 0) {
                    for (int i = 0; i < storages.size(); i++) {
                        objValue[j * periodVarNum + handledVarNum + i] = 0.01;   // 储能功率变化成本
                    }
                }
//                for (int i = 0; i < storages.size(); i++) {
//                    objValue[j * periodVarNum + handledVarNum + i] = coef2;   // 储能功率变化成本
//                }
            }
            cplex.addMinimize(cplex.scalProd(x, objValue));

            //记录数组中存储元素的个数
            int coeffNum = 0;
            for (int j = 0; j < periodNum; j++) {
                // 交流母线电功率平衡约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                }
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = - 1; // 变流器AC-DC交流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                }
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = - iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = - iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), acLoad[j]);
                coeffNum += 1;

                // 直流母线电功率平衡约束
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = - 1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                }
                for (int i = 0; i < storages.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                }
                double periodDcLoad = dcLoad[j];
                periodDcLoad -= photovoltaic.getPower()[j];
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), periodDcLoad);
                coeffNum += 1;

                // 向电网购电功率与电网输入电功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1; // 电网输入电功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 向园区购热功率与园区输入热功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = - 1; // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 燃气轮机启停功率约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMinP();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气轮机爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMaxRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMinRampRate());
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉启停功率约束
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + i] = - gasBoilers.get(i).getMaxH();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + i] = - gasBoilers.get(i).getMinH();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = - 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                    }
                }

                // 电池储能爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                    }
                }
            }

            // 电池储能容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < storages.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * storages.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * storages.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 电池储能日电量累积约束
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * storages.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 冰蓄冷耗电功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = 1; // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxP());
                    coeffNum += 1;
                }
            }

            // 冰蓄冷容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * iceStorageAcs.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * iceStorageAcs.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 冰蓄冷日累积冰量约束
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * iceStorageAcs.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 关口功率约束
            for (int j = 0; j < periodNum; j++) {
                if (peakShaveTime[j] == 0) {
                    // 非削峰时段关口功率约束
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                } else {
                    // 削峰功率上限约束
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                }

                // |P-Pref|约束
                if (peakShaveTime[j] == 1) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2] = 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1;
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), - gatePowers[j]);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2] = 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                }
            }

            // 热功率约束（不考虑空间热负荷）
            for (int j = 0; j < periodNum; j++) {
                // 总热量约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                            gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + i] = - 1;   // 吸收式制冷机耗热功率
                }
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                coeffNum += 1;

                // 中品味热约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j]);
                coeffNum += 1;
            }

            // 冷功率（左）约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                    coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3] = airCons.get(0).getEffac() * airCons.get(0).getEERc();   // 空调1制冷功率
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), coolingLoad1[j]);
                coeffNum += 1;
            }

            // 冷功率（右）约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                    coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + 1] = airCons.get(1).getEffac() * airCons.get(1).getEERc();   // 空调2制冷功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + 2] = airCons.get(2).getEffac() * airCons.get(2).getEERc();   // 空调3制冷功率
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), coolingLoad2[j]);
                coeffNum += 1;
            }

            // 电池功率变化绝对值
            for (int j = 1; j < periodNum; j++) {
                for (int i = 0; i < storages.size(); i++) {
                    coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 上一时刻储能充电功率
                    coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -1; // 上一时刻储能放电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = -1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 + i] = 1;
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = -1; // 上一时刻储能充电功率
                    coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 上一时刻储能放电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -1; // 储能放电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 + i] = 1;
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double[] result = cplex.getValues(x);
                    double minCost = 0;
                    for (int j = 0; j < periodNum; j++) {
                        int handledVarNum = 0;
                        for (int i = 0; i < iceStorageAcs.size(); i++) {
                            minCost += result[j * periodVarNum + i] * iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                        }

                        handledVarNum += 3 * iceStorageAcs.size();
                        for (int i = 0; i < gasTurbines.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + gasTurbines.size() + i] * gasTurbines.get(i).getCss();  // 启停成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] * gasTurbines.get(i).getCoper() * t;   // 运维成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] * gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                        }

                        handledVarNum += 3 * gasTurbines.size();
                        for (int i = 0; i < storages.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + i] * storages.get(i).getCoper() * t;   // 运维成本
                            minCost += result[j * periodVarNum + handledVarNum + i] * storages.get(i).getCbw() * t;    // 折旧成本
                            minCost += result[j * periodVarNum + handledVarNum + storages.size() + i] * storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                        }

                        handledVarNum += 2 * storages.size();
                        handledVarNum += 2 * converters.size();
                        handledVarNum += 1;
                        minCost += result[j * periodVarNum + handledVarNum] * elecPrices[j] * t; // 购电成本

                        handledVarNum += 1;
                        handledVarNum += 1;
                        for (int i = 0; i < airCons.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + i] * airCons.get(i).getCoper() * t;   // 运维成本
                        }

                        handledVarNum += airCons.size();
                        for (int i = 0; i < gasBoilers.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + gasBoilers.size() + i] * gasBoilers.get(i).getCss();  // 启停成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] * gasBoilers.get(i).getCoper() * t;   // 运维成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] * gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                        }

                        handledVarNum += 3 * gasBoilers.size();
                        for (int i = 0; i < absorptionChillers.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + i] * absorptionChillers.get(i).getCoper() * t;   // 运维成本
                        }

                        handledVarNum += absorptionChillers.size();
                        handledVarNum += 1;
                        minCost += result[j * periodVarNum + handledVarNum] * steamPrices[j] * t; // 购热成本

                        minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];    // 光伏运行成本
                    }

                    createUserResult(user.getUserId(), cplex.getStatus().toString(), minCost, cplex.getValues(x),
                            periodVarNum, iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers,
                            absorptionChillers);
                } else {
                    UserResult userResult = new UserResult(user.getUserId(), cplex.getStatus().toString());
                    microgridResult.put(user.getUserId(), userResult);
                }
            }

            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

//    public void createUserResult(String userId, String status, double minCost, double[] result, int periodVarNum,
//                                 List<IceStorageAc> iceStorageAcs, List<GasTurbine> gasTurbines, List<Storage> storages,
//                                 List<Converter> converters, List<AirCon> airCons, List<GasBoiler> gasBoilers,
//                                 List<AbsorptionChiller> absorptionChillers) {
//        List<double[]> frigesP = new ArrayList<>(iceStorageAcs.size());
//        List<double[]> iceTanksP = new ArrayList<>(iceStorageAcs.size());
//        List<double[]> iceTanksQ = new ArrayList<>(iceStorageAcs.size());
//        for (int i = 0; i < iceStorageAcs.size(); i++) {
//            frigesP.add(new double[periodNum]);
//            iceTanksP.add(new double[periodNum]);
//            iceTanksQ.add(new double[periodNum]);
//        }
//        List<double[]> gasTurbinesState = new ArrayList<>(gasTurbines.size());
//        List<double[]> gasTurbinesP = new ArrayList<>(gasTurbines.size());
//        for (int i = 0; i < gasTurbines.size(); i++) {
//            gasTurbinesState.add(new double[periodNum]);
//            gasTurbinesP.add(new double[periodNum]);
//        }
//        List<double[]> storagesP = new ArrayList<>(storages.size());
//        for (int i = 0; i < storages.size(); i++) {
//            storagesP.add(new double[periodNum]);
//        }
//        List<double[]> convertersP = new ArrayList<>(converters.size());
//        for (int i = 0; i < converters.size(); i++) {
//            convertersP.add(new double[periodNum]);
//        }
//        double[] Pin = new double[periodNum];
//        double[] purP = new double[periodNum];
//        List<double[]> airConsP = new ArrayList<>(airCons.size());
//        for (int i = 0; i < airCons.size(); i++) {
//            airConsP.add(new double[periodNum]);
//        }
//        List<double[]> gasBoilersState = new ArrayList<>(gasBoilers.size());
//        List<double[]> gasBoilersH = new ArrayList<>(gasBoilers.size());
//        for (int i = 0; i < gasBoilers.size(); i++) {
//            gasBoilersState.add(new double[periodNum]);
//            gasBoilersH.add(new double[periodNum]);
//        }
//        List<double[]> absorptionChillersH = new ArrayList<>(absorptionChillers.size());
//        for (int i = 0; i < absorptionChillers.size(); i++) {
//            absorptionChillersH.add(new double[periodNum]);
//        }
//        double[] Hin = new double[periodNum];
//        double[] purH = new double[periodNum];
//        for (int j = 0; j < periodNum; j++) {
//            for (int i = 0; i < iceStorageAcs.size(); i++) {
//                frigesP.get(i)[j] = result[j * periodVarNum + i];
//                iceTanksP.get(i)[j] = result[j * periodVarNum + iceStorageAcs.size() + i];
//                iceTanksQ.get(i)[j] = result[j * periodVarNum + 2 * iceStorageAcs.size() + i];
//            }
//            for (int i = 0; i < gasTurbines.size(); i++) {
//                gasTurbinesState.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + i];
//                gasTurbinesP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i];
//            }
//            for (int i = 0; i < storages.size(); i++) {
//                storagesP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] -
//                        result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i];
//            }
//            for (int i = 0; i < converters.size(); i++) {
//                convertersP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] -
//                        result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i];
//            }
//            Pin[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()];
//            purP[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1];
//            for (int i = 0; i < airCons.size(); i++) {
//                airConsP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + i];
//            }
//            for (int i = 0; i < gasBoilers.size(); i++) {
//                gasBoilersState.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i];
//                gasBoilersH.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
//                        2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i];
//            }
//            for (int i = 0; i < absorptionChillers.size(); i++) {
//                absorptionChillersH.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
//                        2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + i];
//            }
//            Hin[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
//                    2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()];
//            purH[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
//                    2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1];
//        }
//        UserResult userResult = new UserResult(userId, status, minCost,
//                frigesP, iceTanksP, iceTanksQ, gasTurbinesState, gasTurbinesP, storagesP, convertersP, Pin,
//                purP, airConsP, gasBoilersState, gasBoilersH, absorptionChillersH, Hin, purH);
//        microgridResult.put(userId, userResult);
//    }

    /**
     * 削峰功率在目标函数中
     */
    public void origDemandResp(User user) {
        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
        List<AirCon> airCons = user.getAirCons();
        List<Converter> converters = user.getConverters();
        List<GasBoiler> gasBoilers = user.getGasBoilers();
        List<GasTurbine> gasTurbines = user.getGasTurbines();
        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
        List<Storage> storages = user.getStorages();
        SteamLoad steamLoad = user.getSteamLoad();
        Photovoltaic photovoltaic = user.getPhotovoltaic();
        double[] acLoad = user.acLoad;
        double[] dcLoad = user.dcLoad;
        double[] heatLoad = user.heatLoad;
        double[] coolingLoad = user.coolingLoad;
        double[] gatePowers = user.gatePowers;
        try {
            int peakShaveTimeNum = 0;
            for (int i = 0; i < peakShaveTime.length; i++) {
                peakShaveTimeNum += peakShaveTime[i];
            }
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，|P-Pref|，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2;
            int varNum = periodNum * periodVarNum;
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[varNum];
            // 状态变量上限
            double columnUpper[] = new double[varNum];
            // 指明变量类型
            IloNumVarType[] xt = new IloNumVarType[varNum];
            // 约束方程系数
            double[][] coeff = new double[(1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    periodNum +
                    2 * peakShaveTimeNum +
                    3 * periodNum][varNum];

            // 设置变量上下限和类型
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[j * periodVarNum + i] = 0;
                    columnUpper[j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + converters.size() + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            double coef1 = 1e6;
            double coef2 = 1e-3;
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    objValue[j * periodVarNum + i] = coef2 * iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    objValue[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = coef2 * gasTurbines.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = coef2 * gasTurbines.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += coef2 * gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = coef2 * storages.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + i] += coef2 * storages.get(i).getCbw() * t;    // 折旧成本
                    objValue[j * periodVarNum + handledVarNum + storages.size() + i] = coef2 * storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                }

                handledVarNum += 2 * storages.size();
                handledVarNum += 2 * converters.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = coef2 * elecPrices[j] * t; // 购电成本

                handledVarNum += 1;
                if (peakShaveTime[j] == 1) {
                    objValue[j * periodVarNum + handledVarNum] = coef1; // 削峰目标
                }

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = coef2 * airCons.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = coef2 * gasBoilers.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = coef2 * gasBoilers.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += coef2 * gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = coef2 * absorptionChillers.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += absorptionChillers.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = coef2 * steamPrices[j] * t; // 购热成本
            }
            cplex.addMinimize(cplex.scalProd(x, objValue));

            //记录数组中存储元素的个数
            int coeffNum = 0;
            for (int j = 0; j < periodNum; j++) {
                // 交流母线电功率平衡约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                }
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = - 1; // 变流器AC-DC交流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                }
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = - iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = - iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), acLoad[j]);
                coeffNum += 1;

                // 直流母线电功率平衡约束
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = - 1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                }
                for (int i = 0; i < storages.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                }
                double periodDcLoad = dcLoad[j];
                periodDcLoad -= photovoltaic.getPower()[j];
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), periodDcLoad);
                coeffNum += 1;

                // 向电网购电功率与电网输入电功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1; // 电网输入电功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 向园区购热功率与园区输入热功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = - 1; // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 燃气轮机启停功率约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMinP();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气轮机爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMaxRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMinRampRate());
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉启停功率约束
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + i] = - gasBoilers.get(i).getMaxH();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + i] = - gasBoilers.get(i).getMinH();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = - 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                    }
                }

                // 电池储能爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                    }
                }
            }

            // 电池储能容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < storages.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * storages.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * storages.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 电池储能日电量累积约束
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * storages.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 冰蓄冷耗电功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = 1; // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxP());
                    coeffNum += 1;
                }
            }

            // 冰蓄冷容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * iceStorageAcs.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * iceStorageAcs.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 冰蓄冷日累积冰量约束
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * iceStorageAcs.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 关口功率约束
            for (int j = 0; j < periodNum; j++) {
                if (peakShaveTime[j] == 0) {
                    // 非削峰时段关口功率约束
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                } else {
                    // 削峰功率上限约束
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                }

                // |P-Pref|约束
                if (peakShaveTime[j] == 1) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2] = 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1;
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), - gatePowers[j]);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2] = 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                }
            }

            // 热功率约束（不考虑空间热负荷）
            for (int j = 0; j < periodNum; j++) {
                // 总热量约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                            gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + i] = - 1;   // 吸收式制冷机耗热功率
                }
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                coeffNum += 1;

                // 中品味热约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j]);
                coeffNum += 1;
            }

            // 冷功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                    coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + i] = airCons.get(i).getEffac() * airCons.get(i).getEERc();   // 空调制冷功率
                }
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), coolingLoad[j]);
                coeffNum += 1;
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double[] result = cplex.getValues(x);
                    double minCost = 0;
                    for (int j = 0; j < periodNum; j++) {
                        int handledVarNum = 0;
                        for (int i = 0; i < iceStorageAcs.size(); i++) {
                            minCost += result[j * periodVarNum + i] * iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                        }

                        handledVarNum += 3 * iceStorageAcs.size();
                        for (int i = 0; i < gasTurbines.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + gasTurbines.size() + i] * gasTurbines.get(i).getCss();  // 启停成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] * gasTurbines.get(i).getCoper() * t;   // 运维成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] * gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                        }

                        handledVarNum += 3 * gasTurbines.size();
                        for (int i = 0; i < storages.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + i] * storages.get(i).getCoper() * t;   // 运维成本
                            minCost += result[j * periodVarNum + handledVarNum + i] * storages.get(i).getCbw() * t;    // 折旧成本
                            minCost += result[j * periodVarNum + handledVarNum + storages.size() + i] * storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                        }

                        handledVarNum += 2 * storages.size();
                        handledVarNum += 2 * converters.size();
                        handledVarNum += 1;
                        minCost += result[j * periodVarNum + handledVarNum] * elecPrices[j] * t; // 购电成本

                        handledVarNum += 1;
                        handledVarNum += 1;
                        for (int i = 0; i < airCons.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + i] * airCons.get(i).getCoper() * t;   // 运维成本
                        }

                        handledVarNum += airCons.size();
                        for (int i = 0; i < gasBoilers.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + gasBoilers.size() + i] * gasBoilers.get(i).getCss();  // 启停成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] * gasBoilers.get(i).getCoper() * t;   // 运维成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] * gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                        }

                        handledVarNum += 3 * gasBoilers.size();
                        for (int i = 0; i < absorptionChillers.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + i] * absorptionChillers.get(i).getCoper() * t;   // 运维成本
                        }

                        handledVarNum += absorptionChillers.size();
                        handledVarNum += 1;
                        minCost += result[j * periodVarNum + handledVarNum] * steamPrices[j] * t; // 购热成本

                        minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];    // 光伏运行成本
                    }

                    createUserResult(user.getUserId(), cplex.getStatus().toString(), minCost, cplex.getValues(x),
                            periodVarNum, iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers,
                            absorptionChillers);
                } else {
                    UserResult userResult = new UserResult(user.getUserId(), cplex.getStatus().toString());
                    microgridResult.put(user.getUserId(), userResult);
                }
            }

            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    /**
     * 充电桩需求响应，削峰功率在目标函数中
     */
    public void chargingPileDR(User user) {
        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
        List<AirCon> airCons = user.getAirCons();
        List<Converter> converters = user.getConverters();
        List<GasBoiler> gasBoilers = user.getGasBoilers();
        List<GasTurbine> gasTurbines = user.getGasTurbines();
        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
        List<Storage> storages = user.getStorages();
        SteamLoad steamLoad = user.getSteamLoad();
        Photovoltaic photovoltaic = user.getPhotovoltaic();
        double[] acLoad = user.acLoad;
        double[] dcLoad = user.dcLoad;
        double[] heatLoad = user.heatLoad;
        double[] coolingLoad = user.coolingLoad;
        double[] gatePowers = user.gatePowers;
        List<ChargingPile> chargingPiles = user.getChargingPiles();
        try {
            int peakShaveTimeNum = 0;
            for (int i = 0; i < peakShaveTime.length; i++) {
                peakShaveTimeNum += peakShaveTime[i];
            }
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，|P-Pref|，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率，充电桩启停状态，充电桩功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                    2 * chargingPiles.size();
            int varNum = periodNum * periodVarNum;
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[varNum];
            // 状态变量上限
            double columnUpper[] = new double[varNum];
            // 指明变量类型
            IloNumVarType[] xt = new IloNumVarType[varNum];
            // 约束方程系数
            double[][] coeff = new double[(1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    periodNum +
                    2 * peakShaveTimeNum +
                    3 * periodNum +
                    2 * chargingPiles.size() * periodNum + chargingPiles.size()][varNum];

            // 设置变量上下限和类型
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[j * periodVarNum + i] = 0;
                    columnUpper[j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + converters.size() + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < chargingPiles.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + chargingPiles.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + chargingPiles.size() + i] = chargingPiles.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + chargingPiles.size() + i] = IloNumVarType.Float;
                }
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            double coef1 = 1e6;
            double coef2 = 1e-3;
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    objValue[j * periodVarNum + i] = coef2 * iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    objValue[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = coef2 * gasTurbines.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = coef2 * gasTurbines.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += coef2 * gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = coef2 * storages.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + i] += coef2 * storages.get(i).getCbw() * t;    // 折旧成本
                    objValue[j * periodVarNum + handledVarNum + storages.size() + i] = coef2 * storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                }

                handledVarNum += 2 * storages.size();
                handledVarNum += 2 * converters.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = coef2 * elecPrices[j] * t; // 购电成本

                handledVarNum += 1;
                if (peakShaveTime[j] == 1) {
                    objValue[j * periodVarNum + handledVarNum] = coef1; // 削峰目标
                }

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = coef2 * airCons.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = coef2 * gasBoilers.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = coef2 * gasBoilers.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += coef2 * gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = coef2 * absorptionChillers.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += absorptionChillers.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = coef2 * steamPrices[j] * t; // 购热成本

                handledVarNum += 1;
                handledVarNum += chargingPiles.size();
                for (int i = 0; i < chargingPiles.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = - coef2 * chargePrices[j] * t;   // 充电桩收益
                }
            }
            cplex.addMinimize(cplex.scalProd(x, objValue));

            //记录数组中存储元素的个数
            int coeffNum = 0;
            for (int j = 0; j < periodNum; j++) {
                // 交流母线电功率平衡约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                }
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = - 1; // 变流器AC-DC交流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                }
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = - iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = - iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                }
                for (int i = 0; i < chargingPiles.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                            chargingPiles.size() + i] = -1;   // 充电桩耗电功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), acLoad[j]);
                coeffNum += 1;

                // 直流母线电功率平衡约束
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = - 1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                }
                for (int i = 0; i < storages.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                }
                double periodDcLoad = dcLoad[j];
                periodDcLoad -= photovoltaic.getPower()[j];
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), periodDcLoad);
                coeffNum += 1;

                // 向电网购电功率与电网输入电功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1; // 电网输入电功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 向园区购热功率与园区输入热功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = - 1; // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 燃气轮机启停功率约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMinP();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气轮机爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMaxRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMinRampRate());
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉启停功率约束
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + i] = - gasBoilers.get(i).getMaxH();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + i] = - gasBoilers.get(i).getMinH();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i] = - 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                    }
                }

                // 电池储能爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                    }
                }
            }

            // 电池储能容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < storages.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * storages.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * storages.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 电池储能日电量累积约束
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * storages.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 冰蓄冷耗电功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = 1; // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxP());
                    coeffNum += 1;
                }
            }

            // 冰蓄冷容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * iceStorageAcs.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * iceStorageAcs.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 冰蓄冷日累积冰量约束
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * iceStorageAcs.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 关口功率约束
            for (int j = 0; j < periodNum; j++) {
                if (peakShaveTime[j] == 0) {
                    // 非削峰时段关口功率约束
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                } else {
                    // 削峰功率上限约束
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                }

                // |P-Pref|约束
                if (peakShaveTime[j] == 1) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2] = 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1;
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), - gatePowers[j]);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2] = 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                }
            }

            // 热功率约束（不考虑空间热负荷）
            for (int j = 0; j < periodNum; j++) {
                // 总热量约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                            gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + i] = - 1;   // 吸收式制冷机耗热功率
                }
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                coeffNum += 1;

                // 中品味热约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j]);
                coeffNum += 1;
            }

            // 冷功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                    coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + i] = airCons.get(i).getEffac() * airCons.get(i).getEERc();   // 空调制冷功率
                }
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), coolingLoad[j]);
                coeffNum += 1;
            }

            // 充电桩启停功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < chargingPiles.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                            chargingPiles.size() + i] = 1; // 充电桩充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() +
                            absorptionChillers.size() + 2 + i] = - chargingPiles.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                            chargingPiles.size() + i] = 1; // 充电桩充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() +
                            absorptionChillers.size() + 2 + i] = - chargingPiles.get(i).getMinP();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }
            }

            // 充电桩充电模式约束
            for (int i = 0; i < chargingPiles.size(); i++) {
                ChargingPile chargingPile = chargingPiles.get(i);
                if (chargingPile.getMode() == 1) {
                    // 对于处于自动充满模式的充电桩，充电量不超过使电动汽车充满的电量
                    for (int j = 0; j < periodNum; j++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                                chargingPiles.size() + i] = t; // 充电桩充电量
                    }
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), chargingPile.getSm());
                } else if (chargingPile.getMode() == 2) {
                    // 对于处于按时间充电模式的充电桩，在设定的时间过后，充电桩停止充电
                    for (int j = 0; j < periodNum; j++) {
                        if (j >= chargingPile.getTe()) {
                            coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                                    chargingPiles.size() + i] = 1; // 充电桩充电功率
                        }
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), 0);
                    }
                } else if (chargingPile.getMode() == 3) {
                    // 对于处于按金额充电模式的充电桩，充电桩消费金额不超过设定金额
                    for (int j = 0; j < periodNum; j++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                                chargingPiles.size() + i] = chargePrices[j] * t; // 充电费用
                    }
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), chargingPile.getM());
                } else if (chargingPile.getMode() == 4) {
                    // 对于处于按电量充电模式的充电桩，充电量不超过设定的电量
                    for (int j = 0; j < periodNum; j++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                                chargingPiles.size() + i] = t; // 充电桩充电量
                    }
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), chargingPile.getS());
                }
                coeffNum += 1;
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double[] result = cplex.getValues(x);
                    double minCost = 0;
                    for (int j = 0; j < periodNum; j++) {
                        int handledVarNum = 0;
                        for (int i = 0; i < iceStorageAcs.size(); i++) {
                            minCost += result[j * periodVarNum + i] * iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                        }

                        handledVarNum += 3 * iceStorageAcs.size();
                        for (int i = 0; i < gasTurbines.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + gasTurbines.size() + i] * gasTurbines.get(i).getCss();  // 启停成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] * gasTurbines.get(i).getCoper() * t;   // 运维成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] * gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                        }

                        handledVarNum += 3 * gasTurbines.size();
                        for (int i = 0; i < storages.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + i] * storages.get(i).getCoper() * t;   // 运维成本
                            minCost += result[j * periodVarNum + handledVarNum + i] * storages.get(i).getCbw() * t;    // 折旧成本
                            minCost += result[j * periodVarNum + handledVarNum + storages.size() + i] * storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                        }

                        handledVarNum += 2 * storages.size();
                        handledVarNum += 2 * converters.size();
                        handledVarNum += 1;
                        minCost += result[j * periodVarNum + handledVarNum] * elecPrices[j] * t; // 购电成本

                        handledVarNum += 1;
                        handledVarNum += 1;
                        for (int i = 0; i < airCons.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + i] * airCons.get(i).getCoper() * t;   // 运维成本
                        }

                        handledVarNum += airCons.size();
                        for (int i = 0; i < gasBoilers.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + gasBoilers.size() + i] * gasBoilers.get(i).getCss();  // 启停成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] * gasBoilers.get(i).getCoper() * t;   // 运维成本
                            minCost += result[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] * gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                        }

                        handledVarNum += 3 * gasBoilers.size();
                        for (int i = 0; i < absorptionChillers.size(); i++) {
                            minCost += result[j * periodVarNum + handledVarNum + i] * absorptionChillers.get(i).getCoper() * t;   // 运维成本
                        }

                        handledVarNum += absorptionChillers.size();
                        handledVarNum += 1;
                        minCost += result[j * periodVarNum + handledVarNum] * steamPrices[j] * t; // 购热成本

                        handledVarNum += 1;
                        handledVarNum += chargingPiles.size();
                        for (int i = 0; i < chargingPiles.size(); i++) {
                            minCost -= result[j * periodVarNum + handledVarNum + i] * chargePrices[j] * t;   // 充电桩收益
                        }

                        minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];    // 光伏运行成本
                    }

                    createUserResult(user.getUserId(), cplex.getStatus().toString(), minCost, cplex.getValues(x),
                            periodVarNum, iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers,
                            absorptionChillers, chargingPiles);
                } else {
                    UserResult userResult = new UserResult(user.getUserId(), cplex.getStatus().toString());
                    microgridResult.put(user.getUserId(), userResult);
                }
            }

            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    public void createUserResult(String userId, String status, double minCost, double[] result, int periodVarNum,
                                 List<IceStorageAc> iceStorageAcs, List<GasTurbine> gasTurbines, List<Storage> storages,
                                 List<Converter> converters, List<AirCon> airCons, List<GasBoiler> gasBoilers,
                                 List<AbsorptionChiller> absorptionChillers, List<ChargingPile> chargingPiles) {
        List<double[]> frigesP = new ArrayList<>(iceStorageAcs.size());
        List<double[]> iceTanksP = new ArrayList<>(iceStorageAcs.size());
        List<double[]> iceTanksQ = new ArrayList<>(iceStorageAcs.size());
        for (int i = 0; i < iceStorageAcs.size(); i++) {
            frigesP.add(new double[periodNum]);
            iceTanksP.add(new double[periodNum]);
            iceTanksQ.add(new double[periodNum]);
        }
        List<double[]> gasTurbinesState = new ArrayList<>(gasTurbines.size());
        List<double[]> gasTurbinesP = new ArrayList<>(gasTurbines.size());
        for (int i = 0; i < gasTurbines.size(); i++) {
            gasTurbinesState.add(new double[periodNum]);
            gasTurbinesP.add(new double[periodNum]);
        }
        List<double[]> storagesP = new ArrayList<>(storages.size());
        for (int i = 0; i < storages.size(); i++) {
            storagesP.add(new double[periodNum]);
        }
        List<double[]> convertersP = new ArrayList<>(converters.size());
        for (int i = 0; i < converters.size(); i++) {
            convertersP.add(new double[periodNum]);
        }
        double[] Pin = new double[periodNum];
        double[] purP = new double[periodNum];
        List<double[]> airConsP = new ArrayList<>(airCons.size());
        for (int i = 0; i < airCons.size(); i++) {
            airConsP.add(new double[periodNum]);
        }
        List<double[]> gasBoilersState = new ArrayList<>(gasBoilers.size());
        List<double[]> gasBoilersH = new ArrayList<>(gasBoilers.size());
        for (int i = 0; i < gasBoilers.size(); i++) {
            gasBoilersState.add(new double[periodNum]);
            gasBoilersH.add(new double[periodNum]);
        }
        List<double[]> absorptionChillersH = new ArrayList<>(absorptionChillers.size());
        for (int i = 0; i < absorptionChillers.size(); i++) {
            absorptionChillersH.add(new double[periodNum]);
        }
        double[] Hin = new double[periodNum];
        double[] purH = new double[periodNum];
        List<double[]> chargingPilesState = new ArrayList<>(gasBoilers.size());
        List<double[]> chargingPilesP = new ArrayList<>(gasBoilers.size());
        for (int i = 0; i < chargingPiles.size(); i++) {
            chargingPilesState.add(new double[periodNum]);
            chargingPilesP.add(new double[periodNum]);
        }
        for (int j = 0; j < periodNum; j++) {
            for (int i = 0; i < iceStorageAcs.size(); i++) {
                frigesP.get(i)[j] = result[j * periodVarNum + i];
                iceTanksP.get(i)[j] = result[j * periodVarNum + iceStorageAcs.size() + i];
                iceTanksQ.get(i)[j] = result[j * periodVarNum + 2 * iceStorageAcs.size() + i];
            }
            for (int i = 0; i < gasTurbines.size(); i++) {
                gasTurbinesState.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + i];
                gasTurbinesP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i];
            }
            for (int i = 0; i < storages.size(); i++) {
                storagesP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] -
                        result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i];
            }
            for (int i = 0; i < converters.size(); i++) {
                convertersP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] -
                        result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i];
            }
            Pin[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()];
            purP[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1];
            for (int i = 0; i < airCons.size(); i++) {
                airConsP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + i];
            }
            for (int i = 0; i < gasBoilers.size(); i++) {
                gasBoilersState.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + i];
                gasBoilersH.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                        2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 2 * gasBoilers.size() + i];
            }
            for (int i = 0; i < absorptionChillers.size(); i++) {
                absorptionChillersH.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                        2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + i];
            }
            Hin[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()];
            purH[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1];
            for (int i = 0; i < chargingPiles.size(); i++) {
                chargingPilesState.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                        2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() +
                        absorptionChillers.size() + 2 + i];
                chargingPilesP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                        2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size() +
                        absorptionChillers.size() + 2 + chargingPiles.size() + i];
            }
        }
        UserResult userResult = new UserResult(userId, status, minCost,
                frigesP, iceTanksP, iceTanksQ, gasTurbinesState, gasTurbinesP, storagesP, convertersP, Pin, purP,
                airConsP, gasBoilersState, gasBoilersH, absorptionChillersH, Hin, purH, chargingPilesState, chargingPilesP);
        microgridResult.put(userId, userResult);
    }

    public void mgDemandResp() {
        microgridResult = new HashMap<>(microgrid.getUsers().size());
        for (User user : microgrid.getUsers().values()) {
//            demandResp(user);
            demandRespIL(user);
        }
    }

    /**
     * 削峰功率为等式约束
     */
    public void demandResp(User user) {
        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
        List<AirCon> airCons = user.getAirCons();
        List<Converter> converters = user.getConverters();
        List<GasBoiler> gasBoilers = user.getGasBoilers();
        List<GasTurbine> gasTurbines = user.getGasTurbines();
        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
        List<Storage> storages = user.getStorages();
        SteamLoad steamLoad = user.getSteamLoad();
        Photovoltaic photovoltaic = user.getPhotovoltaic();
        double[] acLoad = user.acLoad;
        double[] dcLoad = user.dcLoad;
        double[] heatLoad = user.heatLoad;
        double[] coolingLoad = user.coolingLoad;
        double[] gatePowers = user.gatePowers;
        try {
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2;
            int varNum = periodNum * periodVarNum;
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[varNum];
            // 状态变量上限
            double columnUpper[] = new double[varNum];
            // 指明变量类型
            IloNumVarType[] xt = new IloNumVarType[varNum];
            // 约束方程系数
            double[][] coeff = new double[(1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    4 * periodNum][varNum];

            // 设置变量上下限
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[j * periodVarNum + i] = 0;
                    columnUpper[j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + converters.size() + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    objValue[j * periodVarNum + i] = iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    objValue[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = gasTurbines.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = storages.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + i] += storages.get(i).getCbw() * t;    // 折旧成本
                    objValue[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                }

                handledVarNum += 2 * storages.size();
                handledVarNum += 2 * converters.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = elecPrices[j] * t; // 购电成本

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = airCons.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = gasBoilers.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += absorptionChillers.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = steamPrices[j] * t; // 购热成本
            }
            cplex.addMinimize(cplex.scalProd(x, objValue));

            //记录数组中存储元素的个数
            int coeffNum = 0;
            for (int j = 0; j < periodNum; j++) {
                // 交流母线电功率平衡约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                }
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = - 1; // 变流器AC-DC交流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                }
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = - iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = - iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), acLoad[j]);
                coeffNum += 1;

                // 直流母线电功率平衡约束
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = - 1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                }
                for (int i = 0; i < storages.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                }
                double periodDcLoad = dcLoad[j];
                periodDcLoad -= photovoltaic.getPower()[j];
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), periodDcLoad);
                coeffNum += 1;

                // 向电网购电功率与电网输入电功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1; // 电网输入电功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 向园区购热功率与园区输入热功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = - 1; // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 燃气轮机启停功率约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMinP();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气轮机爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMaxRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMinRampRate());
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉启停功率约束
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMaxH();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMinH();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                    }
                }

                // 电池储能爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                    }
                }
            }

            // 电池储能容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < storages.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * storages.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * storages.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 电池储能日电量累积约束
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * storages.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 冰蓄冷耗电功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = 1; // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxP());
                    coeffNum += 1;
                }
            }

            // 冰蓄冷容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * iceStorageAcs.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * iceStorageAcs.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 冰蓄冷日累积冰量约束
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * iceStorageAcs.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 关口功率约束
            for (int j = 0; j < periodNum; j++) {
                // 非削峰时段关口功率约束
                if (peakShaveTime[j] == 0) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                } else {
                    // 削峰时段功率约束
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                }
            }

            // 热功率约束（不考虑空间热负荷）
            for (int j = 0; j < periodNum; j++) {
                // 总热量约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                            gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = - 1;   // 吸收式制冷机耗热功率
                }
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                coeffNum += 1;

                // 中品味热约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j]);
                coeffNum += 1;
            }

            // 冷功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                    coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + i] = airCons.get(i).getEffac() * airCons.get(i).getEERc();   // 空调制冷功率
                }
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), coolingLoad[j]);
                coeffNum += 1;
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double minCost = cplex.getObjValue();
                    // 加上光伏运行成本
                    for (int j = 0; j < periodNum; j++) {
                        minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];
                    }
                    createUserResult(user.getUserId(), cplex.getStatus().toString(), minCost, cplex.getValues(x),
                            periodVarNum, iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers,
                            absorptionChillers);
                } else {
                    UserResult userResult = new UserResult(user.getUserId(), cplex.getStatus().toString());
                    userResult.setMinCost(Double.MAX_VALUE);
                    microgridResult.put(user.getUserId(), userResult);
                }
            } else {
                UserResult userResult = new UserResult(user.getUserId(), cplex.getStatus().toString());
                userResult.setMinCost(Double.MAX_VALUE);
                microgridResult.put(user.getUserId(), userResult);
            }

            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    /**
     * 削峰功率为等式约束，带可中断负荷
     */
    public void demandRespIL(User user) {
        int peakShaveTimeNum = 0;
        for (int i = 0; i < periodNum; i++) {
            if (peakShaveTime[i] == 1) {
                peakShaveTimeNum++;
            }
        }
        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
        List<AirCon> airCons = user.getAirCons();
        List<Converter> converters = user.getConverters();
        List<GasBoiler> gasBoilers = user.getGasBoilers();
        List<GasTurbine> gasTurbines = user.getGasTurbines();
        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
        List<Storage> storages = user.getStorages();
        SteamLoad steamLoad = user.getSteamLoad();
        Photovoltaic photovoltaic = user.getPhotovoltaic();
        double[] acLoad = user.acLoad;
        double[] dcLoad = user.dcLoad;
        double[] heatLoad = user.heatLoad;
        double[] coolingLoad = user.coolingLoad;
        double[] gatePowers = user.gatePowers;
        InterruptibleLoad interruptibleLoad = user.getInterruptibleLoad();
        try {
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2;
            // 增加中断负荷量
            int varNum = periodNum * periodVarNum + peakShaveTimeNum;
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[varNum];
            // 状态变量上限
            double columnUpper[] = new double[varNum];
            // 指明变量类型
            IloNumVarType[] xt = new IloNumVarType[varNum];
            // 约束方程系数
            double[][] coeff = new double[(1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    4 * periodNum][varNum];

            // 设置变量上下限
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[j * periodVarNum + i] = 0;
                    columnUpper[j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + converters.size() + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;
            }
            for (int j = 0; j < peakShaveTimeNum; j++) {
                columnLower[periodNum * periodVarNum + j] = 0;
                columnUpper[periodNum * periodVarNum + j] = interruptibleLoad.maxP;
                xt[periodNum * periodVarNum + j] = IloNumVarType.Float;
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    objValue[j * periodVarNum + i] = iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    objValue[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = gasTurbines.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = storages.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + i] += storages.get(i).getCbw() * t;    // 折旧成本
                    objValue[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                }

                handledVarNum += 2 * storages.size();
                handledVarNum += 2 * converters.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = elecPrices[j] * t; // 购电成本

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = airCons.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = gasBoilers.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += absorptionChillers.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = steamPrices[j] * t; // 购热成本
            }
            IloNumExpr obj = cplex.numExpr();
            for (int j = 0; j < peakShaveTimeNum; j++) {
                objValue[periodNum * periodVarNum + j] = interruptibleLoad.getB() * t; // 中断负荷成本
                IloNumExpr p = cplex.prod(x[periodNum * periodVarNum + j], x[periodNum * periodVarNum + j]);
                obj = cplex.sum(obj, cplex.prod(interruptibleLoad.getA() * t * t, p));
            }
            cplex.addMinimize(cplex.sum(obj, cplex.scalProd(x, objValue)));

            //记录数组中存储元素的个数
            int coeffNum = 0;
            for (int j = 0; j < periodNum; j++) {
                // 交流母线电功率平衡约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                }
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = - 1; // 变流器AC-DC交流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                }
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = - iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = - iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                }
                if (peakShaveTime[j] == 1) {
                    coeff[coeffNum][periodNum * periodVarNum + j - 13] = 1;   // 中断负荷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), acLoad[j]);
                coeffNum += 1;

                // 直流母线电功率平衡约束
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = - 1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                }
                for (int i = 0; i < storages.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                }
                double periodDcLoad = dcLoad[j];
                periodDcLoad -= photovoltaic.getPower()[j];
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), periodDcLoad);
                coeffNum += 1;

                // 向电网购电功率与电网输入电功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1; // 电网输入电功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 向园区购热功率与园区输入热功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = - 1; // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 燃气轮机启停功率约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMinP();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气轮机爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMaxRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMinRampRate());
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉启停功率约束
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMaxH();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMinH();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                    }
                }

                // 电池储能爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                    }
                }
            }

            // 电池储能容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < storages.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * storages.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * storages.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 电池储能日电量累积约束
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * storages.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 冰蓄冷耗电功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = 1; // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxP());
                    coeffNum += 1;
                }
            }

            // 冰蓄冷容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * iceStorageAcs.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * iceStorageAcs.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 冰蓄冷日累积冰量约束
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * iceStorageAcs.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 关口功率约束
            for (int j = 0; j < periodNum; j++) {
                // 非削峰时段关口功率约束
                if (peakShaveTime[j] == 0) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                } else {
                    // 削峰时段功率约束
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                }
            }

            // 热功率约束（不考虑空间热负荷）
            for (int j = 0; j < periodNum; j++) {
                // 总热量约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                            gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = - 1;   // 吸收式制冷机耗热功率
                }
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                coeffNum += 1;

                // 中品味热约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j]);
                coeffNum += 1;
            }

            // 冷功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                    coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + i] = airCons.get(i).getEffac() * airCons.get(i).getEERc();   // 空调制冷功率
                }
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), coolingLoad[j]);
                coeffNum += 1;
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double minCost = cplex.getObjValue();
                    // 加上光伏运行成本
                    for (int j = 0; j < periodNum; j++) {
                        minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];
                    }
                    createUserResult(user.getUserId(), cplex.getStatus().toString(), minCost, cplex.getValues(x),
                            periodVarNum, iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers,
                            absorptionChillers);
                } else {
                    UserResult userResult = new UserResult(user.getUserId(), cplex.getStatus().toString());
                    userResult.setMinCost(Double.MAX_VALUE);
                    microgridResult.put(user.getUserId(), userResult);
                }
            } else {
                UserResult userResult = new UserResult(user.getUserId(), cplex.getStatus().toString());
                userResult.setMinCost(Double.MAX_VALUE);
                microgridResult.put(user.getUserId(), userResult);
            }

            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    public void mgCenDistDemandResp() {
        microgridResult = new HashMap<>(microgrid.getUsers().size());
        peakShaveRatios = new HashMap<>(microgridResult.size());
        offers = new ArrayList<>(microgridResult.size());
        for (User user : microgrid.getUsers().values()) {
//            cenDistDemandResp(user);
            cenDistDemandRespIL(user);
        }
    }

    /**
     * 集中-分布式需求响应
     */
    public void cenDistDemandResp(User user) {
        String userId = user.getUserId();
        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
        List<AirCon> airCons = user.getAirCons();
        List<Converter> converters = user.getConverters();
        List<GasBoiler> gasBoilers = user.getGasBoilers();
        List<GasTurbine> gasTurbines = user.getGasTurbines();
        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
        List<Storage> storages = user.getStorages();
        SteamLoad steamLoad = user.getSteamLoad();
        Photovoltaic photovoltaic = user.getPhotovoltaic();
        double[] acLoad = user.acLoad;
        double[] dcLoad = user.dcLoad;
        double[] heatLoad = user.heatLoad;
        double[] coolingLoad = user.coolingLoad;
        double[] gatePowers = user.gatePowers;
        try {
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2;
            // 最后一个变量为削峰比例
            int varNum = periodNum * periodVarNum + 1;
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[varNum];
            // 状态变量上限
            double columnUpper[] = new double[varNum];
            // 指明变量类型
            IloNumVarType[] xt = new IloNumVarType[varNum];
            // 约束方程系数
            double[][] coeff = new double[(1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    periodNum +
                    peakShaveTime.length +
                    3 * periodNum][varNum];

            // 设置变量上下限
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[j * periodVarNum + i] = 0;
                    columnUpper[j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + converters.size() + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;
            }
            columnLower[varNum - 1] = 0;
            columnUpper[varNum - 1] = Double.MAX_VALUE;
            xt[varNum - 1] = IloNumVarType.Float;

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    objValue[j * periodVarNum + i] = iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    objValue[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = gasTurbines.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = storages.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + i] += storages.get(i).getCbw() * t;    // 折旧成本
                    objValue[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                }

                handledVarNum += 2 * storages.size();
                handledVarNum += 2 * converters.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = elecPrices[j] * t; // 购电成本
                // 购买削峰量成本
                if (peakShaveTime[j] == 1) {
                    objValue[j * periodVarNum + handledVarNum] += clearingPrice * t;
                }

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = airCons.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = gasBoilers.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += absorptionChillers.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = steamPrices[j] * t; // 购热成本
            }
            cplex.addMinimize(cplex.scalProd(x, objValue));

            //记录数组中存储元素的个数
            int coeffNum = 0;
            for (int j = 0; j < periodNum; j++) {
                // 交流母线电功率平衡约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                }
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = - 1; // 变流器AC-DC交流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                }
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = - iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = - iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), acLoad[j]);
                coeffNum += 1;

                // 直流母线电功率平衡约束
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = - 1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                }
                for (int i = 0; i < storages.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                }
                double periodDcLoad = dcLoad[j];
                periodDcLoad -= photovoltaic.getPower()[j];
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), periodDcLoad);
                coeffNum += 1;

                // 向电网购电功率与电网输入电功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1; // 电网输入电功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 向园区购热功率与园区输入热功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = - 1; // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 燃气轮机启停功率约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMinP();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气轮机爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMaxRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMinRampRate());
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉启停功率约束
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMaxH();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMinH();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                    }
                }

                // 电池储能爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                    }
                }
            }

            // 电池储能容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < storages.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * storages.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * storages.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 电池储能日电量累积约束
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * storages.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 冰蓄冷耗电功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = 1; // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxP());
                    coeffNum += 1;
                }
            }

            // 冰蓄冷容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * iceStorageAcs.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * iceStorageAcs.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 冰蓄冷日累积冰量约束
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * iceStorageAcs.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 关口功率约束
            for (int j = 0; j < periodNum; j++) {
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                coeffNum += 1;
            }

            // 等比例削峰约束
            for (int j = 0; j < periodNum; j++) {
                if (peakShaveTime[j] == 1) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    coeff[coeffNum][varNum - 1] = peakShavePowers.get(userId)[j];
                    cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), selfOptResult.get(userId).getPin()[j]);
                    coeffNum += 1;
                }
            }

            // 热功率约束（不考虑空间热负荷）
            for (int j = 0; j < periodNum; j++) {
                // 总热量约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                            gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = - 1;   // 吸收式制冷机耗热功率
                }
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                coeffNum += 1;

                // 中品味热约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j]);
                coeffNum += 1;
            }

            // 冷功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                    coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + i] = airCons.get(i).getEffac() * airCons.get(i).getEERc();   // 空调制冷功率
                }
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), coolingLoad[j]);
                coeffNum += 1;
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double minCost = cplex.getObjValue();
                    // 加上光伏运行成本
                    for (int j = 0; j < periodNum; j++) {
                        minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];
                    }
                    // 加上购买削峰量成本
                    for (int j = 0; j < periodNum; j++) {
                        if (peakShaveTime[j] == 1) {
                            minCost -= clearingPrice * t * (selfOptResult.get(userId).getPin()[j] - peakShavePowers.get(userId)[j]);
                        }
                    }
                    double[] result = cplex.getValues(x);
                    createUserResult(userId, cplex.getStatus().toString(), minCost, result, periodVarNum,
                            iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers, absorptionChillers);
                    peakShaveRatios.put(userId, result[varNum - 1]);
                    // 削峰成本
                    double peakShaveCost = minCost;
                    for (int j = 0; j < periodNum; j++) {
                        if (peakShaveTime[j] == 1) {
                            peakShaveCost -= clearingPrice * t * (peakShavePowers.get(userId)[j] - (selfOptResult.get(userId).getPin()[j] -
                                    result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()]));
                        }
                    }
                    // 按超出指定削峰量的部分计算平均成本
                    double price = 0;
                    if (result[varNum - 1] - 1 > 1e-6) {
                        price = (peakShaveCost - demandRespResult.get(userId).getMinCost()) / ((result[varNum - 1] - 1) * peakShaveCapRatios.get(userId) * totalPeakShaveCap);
                    }
                    offers.add(new Offer(userId, price, result[varNum - 1], peakShaveCapRatios.get(userId)));
                } else {
//                    UserResult userResult = new UserResult(userId, cplex.getStatus().toString());
//                    microgridResult.put(userId, userResult);
                }
            }
            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    /**
     * 带可中断负荷的集中-分布式需求响应
     */
    public void cenDistDemandRespIL(User user) {
        int peakShaveTimeNum = 0;
        for (int i = 0; i < periodNum; i++) {
            if (peakShaveTime[i] == 1) {
                peakShaveTimeNum++;
            }
        }
        String userId = user.getUserId();
        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
        List<AirCon> airCons = user.getAirCons();
        List<Converter> converters = user.getConverters();
        List<GasBoiler> gasBoilers = user.getGasBoilers();
        List<GasTurbine> gasTurbines = user.getGasTurbines();
        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
        List<Storage> storages = user.getStorages();
        SteamLoad steamLoad = user.getSteamLoad();
        Photovoltaic photovoltaic = user.getPhotovoltaic();
        double[] acLoad = user.acLoad;
        double[] dcLoad = user.dcLoad;
        double[] heatLoad = user.heatLoad;
        double[] coolingLoad = user.coolingLoad;
        double[] gatePowers = user.gatePowers;
        InterruptibleLoad interruptibleLoad = user.getInterruptibleLoad();
        try {
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2;
            // 增加削峰比例变量，中断负荷量
            int varNum = periodNum * periodVarNum + 1 + peakShaveTimeNum;
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[varNum];
            // 状态变量上限
            double columnUpper[] = new double[varNum];
            // 指明变量类型
            IloNumVarType[] xt = new IloNumVarType[varNum];
            // 约束方程系数
            double[][] coeff = new double[(1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    periodNum +
                    peakShaveTime.length +
                    3 * periodNum][varNum];

            // 设置变量上下限
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[j * periodVarNum + i] = 0;
                    columnUpper[j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + converters.size() + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;
            }
            columnLower[periodNum * periodVarNum] = 0;
            columnUpper[periodNum * periodVarNum] = Double.MAX_VALUE;
            xt[periodNum * periodVarNum] = IloNumVarType.Float;

            for (int j = 0; j < peakShaveTimeNum; j++) {
                columnLower[periodNum * periodVarNum + 1 + j] = 0;
                columnUpper[periodNum * periodVarNum + 1 + j] = interruptibleLoad.maxP;
                xt[periodNum * periodVarNum + 1 + j] = IloNumVarType.Float;
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    objValue[j * periodVarNum + i] = iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    objValue[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = gasTurbines.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = storages.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + i] += storages.get(i).getCbw() * t;    // 折旧成本
                    objValue[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                }

                handledVarNum += 2 * storages.size();
                handledVarNum += 2 * converters.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = elecPrices[j] * t; // 购电成本
                // 购买削峰量成本
                if (peakShaveTime[j] == 1) {
                    objValue[j * periodVarNum + handledVarNum] += clearingPrice * t;
                }

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = airCons.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = gasBoilers.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += absorptionChillers.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = steamPrices[j] * t; // 购热成本
            }
            IloNumExpr obj = cplex.numExpr();
            for (int j = 0; j < peakShaveTimeNum; j++) {
                objValue[periodNum * periodVarNum + 1 + j] = interruptibleLoad.getB() * t; // 中断负荷成本
                IloNumExpr p = cplex.prod(x[periodNum * periodVarNum + 1 + j], x[periodNum * periodVarNum + 1 + j]);
                obj = cplex.sum(obj, cplex.prod(interruptibleLoad.getA() * t * t, p));
            }
            cplex.addMinimize(cplex.sum(obj, cplex.scalProd(x, objValue)));

            //记录数组中存储元素的个数
            int coeffNum = 0;
            for (int j = 0; j < periodNum; j++) {
                // 交流母线电功率平衡约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                }
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = - 1; // 变流器AC-DC交流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                }
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = - iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = - iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                }
                if (peakShaveTime[j] == 1) {
                    coeff[coeffNum][periodNum * periodVarNum + 1 + j - 13] = 1;   // 中断负荷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), acLoad[j]);
                coeffNum += 1;

                // 直流母线电功率平衡约束
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = - 1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                }
                for (int i = 0; i < storages.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                }
                double periodDcLoad = dcLoad[j];
                periodDcLoad -= photovoltaic.getPower()[j];
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), periodDcLoad);
                coeffNum += 1;

                // 向电网购电功率与电网输入电功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1; // 电网输入电功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 向园区购热功率与园区输入热功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = - 1; // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 燃气轮机启停功率约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMinP();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气轮机爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMaxRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMinRampRate());
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉启停功率约束
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMaxH();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMinH();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                    }
                }

                // 电池储能爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                    }
                }
            }

            // 电池储能容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < storages.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * storages.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * storages.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 电池储能日电量累积约束
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * storages.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 冰蓄冷耗电功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = 1; // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxP());
                    coeffNum += 1;
                }
            }

            // 冰蓄冷容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * iceStorageAcs.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * iceStorageAcs.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 冰蓄冷日累积冰量约束
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * iceStorageAcs.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 关口功率约束
            for (int j = 0; j < periodNum; j++) {
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                coeffNum += 1;
            }

            // 等比例削峰约束
            for (int j = 0; j < periodNum; j++) {
                if (peakShaveTime[j] == 1) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    coeff[coeffNum][periodNum * periodVarNum] = peakShavePowers.get(userId)[j];
                    cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), selfOptResult.get(userId).getPin()[j]);
                    coeffNum += 1;
                }
            }

            // 热功率约束（不考虑空间热负荷）
            for (int j = 0; j < periodNum; j++) {
                // 总热量约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                            gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = - 1;   // 吸收式制冷机耗热功率
                }
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                coeffNum += 1;

                // 中品味热约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j]);
                coeffNum += 1;
            }

            // 冷功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                    coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + i] = airCons.get(i).getEffac() * airCons.get(i).getEERc();   // 空调制冷功率
                }
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), coolingLoad[j]);
                coeffNum += 1;
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double minCost = cplex.getObjValue();
                    // 加上光伏运行成本
                    for (int j = 0; j < periodNum; j++) {
                        minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];
                    }
                    // 加上购买削峰量成本
                    for (int j = 0; j < periodNum; j++) {
                        if (peakShaveTime[j] == 1) {
                            minCost -= clearingPrice * t * (selfOptResult.get(userId).getPin()[j] - peakShavePowers.get(userId)[j]);
                        }
                    }
                    double[] result = cplex.getValues(x);
                    createUserResult(userId, cplex.getStatus().toString(), minCost, result, periodVarNum,
                            iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers, absorptionChillers);
                    peakShaveRatios.put(userId, result[periodNum * periodVarNum]);
                    // 削峰成本
                    double peakShaveCost = minCost;
                    for (int j = 0; j < periodNum; j++) {
                        if (peakShaveTime[j] == 1) {
                            peakShaveCost -= clearingPrice * t * (peakShavePowers.get(userId)[j] - (selfOptResult.get(userId).getPin()[j] -
                                    result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()]));
                        }
                    }
                    // 按超出指定削峰量的部分计算平均成本
                    double price = 0;
                    if (result[periodNum * periodVarNum] - 1 > 1e-6) {
                        price = (peakShaveCost - demandRespResult.get(userId).getMinCost()) / ((result[periodNum * periodVarNum] - 1) * peakShaveCapRatios.get(userId) * totalPeakShaveCap);
                    }
                    offers.add(new Offer(userId, price, result[periodNum * periodVarNum], peakShaveCapRatios.get(userId)));
                } else {
//                    UserResult userResult = new UserResult(userId, cplex.getStatus().toString());
//                    microgridResult.put(userId, userResult);
                }
            }
            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    public void calPeakShavePowers(double[] parkGatePower, double[] parkPeakShavePower) {
        Map<String, User> users = microgrid.getUsers();
        peakShavePowers = new HashMap<>(users.size());
        peakShaveCapRatios = new HashMap<>(users.size());
        totalPeakShaveCap = 0;
        double totalBasicCap = 0;
        for (String userId : users.keySet()) {
            totalBasicCap += users.get(userId).getBasicCap();
        }
        double[] availCap = new double[periodNum];  // 总可用容量
        for (int i = 0; i < periodNum; i++) {
            if (peakShaveTime[i] == 1) {
                totalPeakShaveCap += parkPeakShavePower[i];
            }
            availCap[i] = parkGatePower[i] - parkPeakShavePower[i];
        }
        for (String userId : users.keySet()) {
            double[] shaveGatePower = new double[periodNum];
            peakShavePowers.put(userId, new double[periodNum]);
            double peakShaveCapRatio = 0;
            double[] purP = selfOptResult.get(userId).getPurP();
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    peakShavePowers.get(userId)[i] = purP[i] - availCap[i] * users.get(userId).getBasicCap() / totalBasicCap;
                    peakShaveCapRatio += peakShavePowers.get(userId)[i] / totalPeakShaveCap;
                    shaveGatePower[i] = availCap[i] * users.get(userId).getBasicCap() / totalBasicCap;
                } else {
                    shaveGatePower[i] = users.get(userId).getGatePowers()[i];
                }
            }
            peakShaveCapRatios.put(userId, peakShaveCapRatio);
            shaveGatePowers.put(userId, shaveGatePower);
        }
        totalPeakShaveCap *= t;
        timeShaveCapRatios = new ArrayList<>();
        double[] totalTimeShaveCap = new double[periodNum];
        for (String userId : users.keySet()) {
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    totalTimeShaveCap[i] += peakShavePowers.get(userId)[i];
                }
            }
        }
        for (int i = 0; i < periodNum; i++) {
            if (peakShaveTime[i] == 1) {
                Map<String, Double> timeShaveCapRatio = new HashMap<>();
                for (String userId : users.keySet()) {
                    timeShaveCapRatio.put(userId, peakShavePowers.get(userId)[i] / totalTimeShaveCap[i]);
                }
                timeShaveCapRatios.add(timeShaveCapRatio);
            }
        }
    }

    /**
     * 集中式需求响应
     */
    public void cenIDR() {
        Map<String, User> users = microgrid.getUsers();
        Map<String, Integer> userStarts = new HashMap<>(users.size());
        Map<String, Integer> periodVarNums = new HashMap<>(users.size());
        Map<String, Integer> userVarNums = new HashMap<>(users.size());
        int[] userStartsArray = new int[users.size()];
        int varNum = 0;
        Map<String, Integer> coeffNums = new HashMap<>(users.size());
        Map<String, Integer> coeffStarts = new HashMap<>(users.size());
        int[] coeffStartsArray = new int[users.size()];
        int coeffSumNum = 0;
        int count = 0;
        for (User user : users.values()) {
            List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
            List<AirCon> airCons = user.getAirCons();
            List<Converter> converters = user.getConverters();
            List<GasBoiler> gasBoilers = user.getGasBoilers();
            List<GasTurbine> gasTurbines = user.getGasTurbines();
            List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
            List<Storage> storages = user.getStorages();
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2;
            periodVarNums.put(user.getUserId(), periodVarNum);
            userVarNums.put(user.getUserId(), periodNum * periodVarNum);
            varNum += periodNum * periodVarNum;

            int coeffNum = (1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    4 * periodNum;
            coeffSumNum += coeffNum;
            coeffNums.put(user.getUserId(), coeffNum);

            if (count == 0) {
                userStartsArray[count] = periodNum * periodVarNum;
                coeffStartsArray[count] = coeffNum;
            } else {
                userStartsArray[count] = userStartsArray[count - 1] + periodNum * periodVarNum;
                coeffStartsArray[count] = coeffStartsArray[count - 1] + coeffNum;
            }
            count++;
        }
        for (int i = 0; i < users.size() - 1; i++) {
            userStartsArray[users.size() - 1 - i] = userStartsArray[users.size() - 1 - i - 1];
            coeffStartsArray[users.size() - 1 - i] = coeffStartsArray[users.size() - 1 - i - 1];
        }
        userStartsArray[0] = 0;
        coeffStartsArray[0] = 0;

        count = 0;
        for (User user : users.values()) {
            userStarts.put(user.getUserId(), userStartsArray[count]);
            coeffStarts.put(user.getUserId(), coeffStartsArray[count]);
            count++;
        }

        for (int i = 0; i < peakShaveTime.length; i++) {
            if (peakShaveTime[i] == 1) {
                coeffSumNum++;
            }
        }

        // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double[] columnLower = new double[varNum];
        // 状态变量上限
        double[] columnUpper = new double[varNum];
        // 指明变量类型
        IloNumVarType[] xt = new IloNumVarType[varNum];
        // 约束方程系数
        double[][] coeff = new double[coeffSumNum][varNum];

        for (User user : users.values()) {
            int periodVarNum = periodVarNums.get(user.getUserId());
            int userStart = userStarts.get(user.getUserId());

            List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
            List<AirCon> airCons = user.getAirCons();
            List<Converter> converters = user.getConverters();
            List<GasBoiler> gasBoilers = user.getGasBoilers();
            List<GasTurbine> gasTurbines = user.getGasTurbines();
            List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
            List<Storage> storages = user.getStorages();

            // 设置变量上下限
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[userStart + j * periodVarNum + i] = 0;
                    columnUpper[userStart + j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[userStart + j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[userStart + j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[userStart + j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = 1;
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[userStart + j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[userStart + j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[userStart + j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[userStart + j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = -Double.MAX_VALUE;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[userStart + j * periodVarNum + handledVarNum + converters.size() + i] = -Double.MAX_VALUE;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[userStart + j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[userStart + j * periodVarNum + handledVarNum] = -Double.MAX_VALUE;
                columnUpper[userStart + j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[userStart + j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[userStart + j * periodVarNum + handledVarNum] = 0;
                columnUpper[userStart + j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[userStart + j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = 1;
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[userStart + j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[userStart + j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[userStart + j * periodVarNum + handledVarNum] = -Double.MAX_VALUE;
                columnUpper[userStart + j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[userStart + j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[userStart + j * periodVarNum + handledVarNum] = 0;
                columnUpper[userStart + j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[userStart + j * periodVarNum + handledVarNum] = IloNumVarType.Float;
            }
        }

        try {
            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            for (User user : users.values()) {
                int periodVarNum = periodVarNums.get(user.getUserId());
                int userStart = userStarts.get(user.getUserId());

                List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
                List<AirCon> airCons = user.getAirCons();
                List<Converter> converters = user.getConverters();
                List<GasBoiler> gasBoilers = user.getGasBoilers();
                List<GasTurbine> gasTurbines = user.getGasTurbines();
                List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
                List<Storage> storages = user.getStorages();

                for (int j = 0; j < periodNum; j++) {
                    int handledVarNum = 0;
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        objValue[userStart + j * periodVarNum + i] = iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                        objValue[userStart + j * periodVarNum + iceStorageAcs.size() + i] = 0;
                        objValue[userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    }

                    handledVarNum += 3 * iceStorageAcs.size();
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        objValue[userStart + j * periodVarNum + handledVarNum + i] = 0;
                        objValue[userStart + j * periodVarNum + handledVarNum + gasTurbines.size() + i] = gasTurbines.get(i).getCss();  // 启停成本
                        objValue[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getCoper() * t;   // 运维成本
                        objValue[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                    }

                    handledVarNum += 3 * gasTurbines.size();
                    for (int i = 0; i < storages.size(); i++) {
                        objValue[userStart + j * periodVarNum + handledVarNum + i] = storages.get(i).getCoper() * t;   // 运维成本
                        objValue[userStart + j * periodVarNum + handledVarNum + i] += storages.get(i).getCbw() * t;    // 折旧成本
                        objValue[userStart + j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                    }

                    handledVarNum += 2 * storages.size();
                    handledVarNum += 2 * converters.size();
                    handledVarNum += 1;
                    objValue[userStart + j * periodVarNum + handledVarNum] = elecPrices[j] * t; // 购电成本

                    handledVarNum += 1;
                    for (int i = 0; i < airCons.size(); i++) {
                        objValue[userStart + j * periodVarNum + handledVarNum + i] = airCons.get(i).getCoper() * t;   // 运维成本
                    }

                    handledVarNum += airCons.size();
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        objValue[userStart + j * periodVarNum + handledVarNum + i] = 0;
                        objValue[userStart + j * periodVarNum + handledVarNum + gasBoilers.size() + i] = gasBoilers.get(i).getCss();  // 启停成本
                        objValue[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getCoper() * t;   // 运维成本
                        objValue[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                    }

                    handledVarNum += 3 * gasBoilers.size();
                    for (int i = 0; i < absorptionChillers.size(); i++) {
                        objValue[userStart + j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getCoper() * t;   // 运维成本
                    }

                    handledVarNum += absorptionChillers.size();
                    handledVarNum += 1;
                    objValue[userStart + j * periodVarNum + handledVarNum] = steamPrices[j] * t; // 购热成本
                }
            }
            cplex.addMinimize(cplex.scalProd(x, objValue));

            for (User user : users.values()) {
                int periodVarNum = periodVarNums.get(user.getUserId());
                int userStart = userStarts.get(user.getUserId());
                int coeffStart = coeffStarts.get(user.getUserId());

                List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
                List<AirCon> airCons = user.getAirCons();
                List<Converter> converters = user.getConverters();
                List<GasBoiler> gasBoilers = user.getGasBoilers();
                List<GasTurbine> gasTurbines = user.getGasTurbines();
                List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
                List<Storage> storages = user.getStorages();
                SteamLoad steamLoad = user.getSteamLoad();
                Photovoltaic photovoltaic = user.getPhotovoltaic();
                double[] acLoad = user.acLoad;
                double[] dcLoad = user.dcLoad;
                double[] heatLoad = user.heatLoad;
                double[] coolingLoad = user.coolingLoad;
                double[] gatePowers = user.gatePowers;
                //记录数组中存储元素的个数
                int coeffNum = 0;
                for (int j = 0; j < periodNum; j++) {
                    // 交流母线电功率平衡约束
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    }
                    for (int i = 0; i < converters.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = -1; // 变流器AC-DC交流侧功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                    }
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + i] = -iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + iceStorageAcs.size() + i] = -iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                    }
                    for (int i = 0; i < airCons.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                    }
                    cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), acLoad[j]);
                    coeffNum += 1;

                    // 直流母线电功率平衡约束
                    for (int i = 0; i < converters.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = -1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                    }
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = -1; // 储能充电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                    }
                    double periodDcLoad = dcLoad[j];
                    periodDcLoad -= photovoltaic.getPower()[j];
                    cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), periodDcLoad);
                    coeffNum += 1;

                    // 向电网购电功率与电网输入电功率关系约束
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = -1; // 电网输入电功率
                    cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                    coeffNum += 1;

                    // 向园区购热功率与园区输入热功率关系约束
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = -1; // 园区输入热功率
                    cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                    coeffNum += 1;

                    // 燃气轮机启停功率约束
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = -gasTurbines.get(i).getMaxP();
                        cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = -gasTurbines.get(i).getMinP();
                        cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                        coeffNum += 1;
                    }

                    // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                    if (j == 0) {
                        for (int i = 0; i < gasTurbines.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = -1; // 燃气轮机状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), - gasTurbines.get(i).getInitState());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gasTurbines.get(i).getInitState());
                            coeffNum += 1;
                        }
                    } else {
                        for (int i = 0; i < gasTurbines.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = -1; // 燃气轮机状态
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = -1; // 燃气轮机上一时刻状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                            coeffNum += 1;
                        }
                    }

                    // 燃气轮机爬坡率约束
                    if (j > 0) {
                        for (int i = 0; i < gasTurbines.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = -1;   // 上一时刻燃气轮机产电功率
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gasTurbines.get(i).getMaxRampRate());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = -1;   // 上一时刻燃气轮机产电功率
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gasTurbines.get(i).getMinRampRate());
                            coeffNum += 1;
                        }
                    }

                    // 燃气锅炉启停功率约束
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + i] = -gasBoilers.get(i).getMaxH();
                        cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + i] = -gasBoilers.get(i).getMinH();
                        cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                        coeffNum += 1;
                    }

                    // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                    if (j == 0) {
                        for (int i = 0; i < gasBoilers.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = -1; // 燃气锅炉状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), -gasBoilers.get(i).getInitState());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gasBoilers.get(i).getInitState());
                            coeffNum += 1;
                        }
                    } else {
                        for (int i = 0; i < gasBoilers.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = -1; // 燃气锅炉状态
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = -1; // 燃气锅炉上一时刻状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                            coeffNum += 1;
                        }
                    }

                    // 燃气锅炉爬坡率约束
                    if (j > 0) {
                        for (int i = 0; i < gasBoilers.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = -1;   // 上一时刻燃气锅炉产热功率
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gasBoilers.get(i).getRampRate());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = -1;   // 上一时刻燃气锅炉产热功率
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), -gasBoilers.get(i).getRampRate());
                            coeffNum += 1;
                        }
                    }

                    // 电池储能爬坡率约束
                    if (j > 0) {
                        for (int i = 0; i < storages.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = -1;   // 上一时刻储能充电功率
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = -1;   // 上一时刻储能充电功率
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), -storages.get(i).getYin() * storages.get(i).getMaxPIn());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -1;   // 上一时刻储能放电功率
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -1;   // 上一时刻储能放电功率
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                            coeffNum += 1;
                        }
                    }
                }

                // 电池储能容量约束
                for (int j = 0; j < periodNum; j++) {
                    if (j < periodNum - 1) {
                        for (int i = 0; i < storages.size(); i++) {
                            if (j > 0) {
                                coeff[coeffStart + coeffNum] = coeff[coeffStart + coeffNum - 2 * storages.size()];
                                coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum + 1] = coeff[coeffStart + coeffNum + 1 - 2 * storages.size()];
                                coeff[coeffStart + coeffNum + 1][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum + 1][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            }
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -t / storages.get(i).getEffOut(); // 储能放电量
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -t / storages.get(i).getEffOut(); // 储能放电量
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                            coeffNum += 1;
                        }
                    } else {
                        // 电池储能日电量累积约束
                        for (int i = 0; i < storages.size(); i++) {
                            coeff[coeffStart + coeffNum] = coeff[coeffStart + coeffNum + i - 2 * storages.size()];
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -t / storages.get(i).getEffOut(); // 储能放电量
                            cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                            coeffNum += 1;
                        }
                    }
                }

                // 冰蓄冷耗电功率约束
                for (int j = 0; j < periodNum; j++) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + i] = 1; // 制冷机耗电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), iceStorageAcs.get(i).getMaxP());
                        coeffNum += 1;
                    }
                }

                // 冰蓄冷容量约束
                for (int j = 0; j < periodNum; j++) {
                    if (j < periodNum - 1) {
                        for (int i = 0; i < iceStorageAcs.size(); i++) {
                            if (j > 0) {
                                coeff[coeffStart + coeffNum] = coeff[coeffStart + coeffNum - 2 * iceStorageAcs.size()];
                                coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum + 1] = coeff[coeffStart + coeffNum + 1 - 2 * iceStorageAcs.size()];
                                coeff[coeffStart + coeffNum + 1][userStart + (j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum + 1][userStart + (j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            }
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = -t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = -t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                            coeffNum += 1;
                        }
                    } else {
                        // 冰蓄冷日累积冰量约束
                        for (int i = 0; i < iceStorageAcs.size(); i++) {
                            coeff[coeffStart + coeffNum] = coeff[coeffStart + coeffNum + i - 2 * iceStorageAcs.size()];
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = -t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                            cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                            coeffNum += 1;
                        }
                    }
                }

                // 关口功率约束
                for (int j = 0; j < periodNum; j++) {
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                }

                // 热功率约束（不考虑空间热负荷）
                for (int j = 0; j < periodNum; j++) {
                    // 总热量约束
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                                gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                    }
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                    }
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                    for (int i = 0; i < absorptionChillers.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = -1;   // 吸收式制冷机耗热功率
                    }
                    cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                    coeffNum += 1;

                    // 中品味热约束
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                    }
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                    }
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                    cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), steamLoad.getDemand()[j]);
                    coeffNum += 1;
                }

                // 冷功率约束
                for (int j = 0; j < periodNum; j++) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                    }
                    for (int i = 0; i < airCons.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + i] = airCons.get(i).getEffac() * airCons.get(i).getEERc();   // 空调制冷功率
                    }
                    for (int i = 0; i < absorptionChillers.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                    }
                    cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), coolingLoad[j]);
                    coeffNum += 1;
                }
            }

            // 削峰约束
            int coeffStart = coeffSumNum;
            for (int i = 0; i < peakShaveTime.length; i++) {
                if (peakShaveTime[i] == 1) {
                    coeffStart--;
                }
            }

            int coeffNum = 0;
            for (int i = 0; i < peakShaveTime.length; i++) {
                if (peakShaveTime[i] == 1) {
                    for (User user : users.values()) {
                        int periodVarNum = periodVarNums.get(user.getUserId());
                        int userStart = userStarts.get(user.getUserId());

                        List<Converter> converters = user.getConverters();
                        List<GasTurbine> gasTurbines = user.getGasTurbines();
                        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
                        List<Storage> storages = user.getStorages();

                        coeff[coeffStart + coeffNum][userStart + i * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    }
                    cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gatePowerSum[i]);
                    coeffNum += 1;
                }
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double minCost = cplex.getObjValue();
                    double[] results = cplex.getValues(x);
                    count = 0;
                    for (User user : users.values()) {
                        int periodVarNum = periodVarNums.get(user.getUserId());
                        int userVarNum = userVarNums.get(user.getUserId());
                        int userStart = userStarts.get(user.getUserId());

                        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
                        List<AirCon> airCons = user.getAirCons();
                        List<Converter> converters = user.getConverters();
                        List<GasBoiler> gasBoilers = user.getGasBoilers();
                        List<GasTurbine> gasTurbines = user.getGasTurbines();
                        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
                        List<Storage> storages = user.getStorages();
                        Photovoltaic photovoltaic = user.getPhotovoltaic();

                        // 每个用户的成本
                        double userMinCost = 0;
                        for (int j = 0; j < periodNum; j++) {
                            // 加上光伏运行成本
                            minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];
                            userMinCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];

                            int handledVarNum = 0;
                            for (int i = 0; i < iceStorageAcs.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + i] * iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                            }

                            handledVarNum += 3 * iceStorageAcs.size();
                            for (int i = 0; i < gasTurbines.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + gasTurbines.size() + i] * gasTurbines.get(i).getCss();  // 启停成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] * gasTurbines.get(i).getCoper() * t;   // 运维成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] * gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                            }

                            handledVarNum += 3 * gasTurbines.size();
                            for (int i = 0; i < storages.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + i] * storages.get(i).getCoper() * t;   // 运维成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + i] * storages.get(i).getCbw() * t;    // 折旧成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + storages.size() + i] * storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                            }

                            handledVarNum += 2 * storages.size();
                            handledVarNum += 2 * converters.size();
                            handledVarNum += 1;
                            userMinCost += results[userStart + j * periodVarNum + handledVarNum] * elecPrices[j] * t; // 购电成本

                            handledVarNum += 1;
                            for (int i = 0; i < airCons.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + i] * airCons.get(i).getCoper() * t;   // 运维成本
                            }

                            handledVarNum += airCons.size();
                            for (int i = 0; i < gasBoilers.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + gasBoilers.size() + i] * gasBoilers.get(i).getCss();  // 启停成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] * gasBoilers.get(i).getCoper() * t;   // 运维成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] * gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                            }

                            handledVarNum += 3 * gasBoilers.size();
                            for (int i = 0; i < absorptionChillers.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + i] * absorptionChillers.get(i).getCoper() * t;   // 运维成本
                            }

                            handledVarNum += absorptionChillers.size();
                            handledVarNum += 1;
                            userMinCost += results[userStart + j * periodVarNum + handledVarNum] * steamPrices[j] * t; // 购热成本
                        }

                        double[] result = new double[userVarNum];
                        if (count < users.size() - 1) {
                            for (int i = userStartsArray[count]; i < userStartsArray[count + 1]; i++) {
                                result[i - userStartsArray[count]] = results[i];
                            }
                        } else {
                            for (int i = userStartsArray[count]; i < varNum; i++) {
                                result[i - userStartsArray[count]] = results[i];
                            }
                        }

                        createUserResult(user.getUserId(), cplex.getStatus().toString(), userMinCost, result,
                                periodVarNum, iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers,
                                absorptionChillers);
                        count++;
                    }
                    System.out.println("总最小成本\t" + minCost);
                } else {
                    for (User user : users.values()) {
                        UserResult userResult = new UserResult(user.getUserId(), cplex.getStatus().toString());
                        userResult.setMinCost(Double.MAX_VALUE);
                        microgridResult.put(user.getUserId(), userResult);
                    }
                }
            } else {
                for (User user : users.values()) {
                    UserResult userResult = new UserResult(user.getUserId(), cplex.getStatus().toString());
                    userResult.setMinCost(Double.MAX_VALUE);
                    microgridResult.put(user.getUserId(), userResult);
                }
            }

            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    public void mgDistIDR() {
        microgridResult = new HashMap<>(microgrid.getUsers().size());
        peakShaveRatios = new HashMap<>(microgridResult.size());
        offers = new ArrayList<>(microgridResult.size());
        for (User user : microgrid.getUsers().values()) {
//            distIDRIL(user);
            distIDRILNew(user);
        }
    }

    /**
     * 分布式需求响应
     */
    public void distIDR(User user) {
        String userId = user.getUserId();
        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
        List<AirCon> airCons = user.getAirCons();
        List<Converter> converters = user.getConverters();
        List<GasBoiler> gasBoilers = user.getGasBoilers();
        List<GasTurbine> gasTurbines = user.getGasTurbines();
        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
        List<Storage> storages = user.getStorages();
        SteamLoad steamLoad = user.getSteamLoad();
        Photovoltaic photovoltaic = user.getPhotovoltaic();
        double[] acLoad = user.acLoad;
        double[] dcLoad = user.dcLoad;
        double[] heatLoad = user.heatLoad;
        double[] coolingLoad = user.coolingLoad;
        double[] gatePowers = user.gatePowers;
        double[] mc = mcs.get(userId);
        try {
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2;
            int varNum = periodNum * periodVarNum;
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[varNum];
            // 状态变量上限
            double columnUpper[] = new double[varNum];
            // 指明变量类型
            IloNumVarType[] xt = new IloNumVarType[varNum];
            // 约束方程系数
            double[][] coeff = new double[(1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    periodNum +
                    3 * periodNum][varNum];

            // 设置变量上下限
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[j * periodVarNum + i] = 0;
                    columnUpper[j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + converters.size() + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    objValue[j * periodVarNum + i] = iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    objValue[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = gasTurbines.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = storages.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + i] += storages.get(i).getCbw() * t;    // 折旧成本
                    objValue[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                }

                handledVarNum += 2 * storages.size();
                handledVarNum += 2 * converters.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = elecPrices[j] * t; // 购电成本
                // 购买削峰量成本
                if (peakShaveTime[j] == 1) {
                    objValue[j * periodVarNum + handledVarNum] += mc[j] * t;
                }

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = airCons.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = gasBoilers.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += absorptionChillers.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = steamPrices[j] * t; // 购热成本
            }
            cplex.addMinimize(cplex.scalProd(x, objValue));

            //记录数组中存储元素的个数
            int coeffNum = 0;
            for (int j = 0; j < periodNum; j++) {
                // 交流母线电功率平衡约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                }
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = - 1; // 变流器AC-DC交流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                }
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = - iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = - iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), acLoad[j]);
                coeffNum += 1;

                // 直流母线电功率平衡约束
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = - 1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                }
                for (int i = 0; i < storages.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                }
                double periodDcLoad = dcLoad[j];
                periodDcLoad -= photovoltaic.getPower()[j];
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), periodDcLoad);
                coeffNum += 1;

                // 向电网购电功率与电网输入电功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1; // 电网输入电功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 向园区购热功率与园区输入热功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = - 1; // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 燃气轮机启停功率约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMinP();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气轮机爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMaxRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMinRampRate());
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉启停功率约束
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMaxH();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMinH();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                    }
                }

                // 电池储能爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                    }
                }
            }

            // 电池储能容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < storages.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * storages.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * storages.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 电池储能日电量累积约束
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * storages.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 冰蓄冷耗电功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = 1; // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxP());
                    coeffNum += 1;
                }
            }

            // 冰蓄冷容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * iceStorageAcs.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * iceStorageAcs.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 冰蓄冷日累积冰量约束
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * iceStorageAcs.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 关口功率约束
            for (int j = 0; j < periodNum; j++) {
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                coeffNum += 1;
            }

            // 热功率约束（不考虑空间热负荷）
            for (int j = 0; j < periodNum; j++) {
                // 总热量约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                            gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = - 1;   // 吸收式制冷机耗热功率
                }
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                coeffNum += 1;

                // 中品味热约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j]);
                coeffNum += 1;
            }

            // 冷功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                    coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + i] = airCons.get(i).getEffac() * airCons.get(i).getEERc();   // 空调制冷功率
                }
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), coolingLoad[j]);
                coeffNum += 1;
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double minCost = cplex.getObjValue();
                    // 加上光伏运行成本
                    for (int j = 0; j < periodNum; j++) {
                        minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];
                    }
                    // 加上购买削峰量成本
                    for (int j = 0; j < periodNum; j++) {
                        if (peakShaveTime[j] == 1) {
                            minCost -= mc[j] * t * (selfOptResult.get(userId).getPin()[j] - peakShavePowers.get(userId)[j]);
                        }
                    }
                    double[] result = cplex.getValues(x);
                    createUserResult(userId, cplex.getStatus().toString(), minCost, result, periodVarNum,
                            iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers, absorptionChillers);

                    double[] peakShaveCap = new double[periodNum];
                    for (int j = 0; j < periodNum; j++) {
                        if (peakShaveTime[j] == 1) {
                            peakShaveCap[j] = selfOptResult.get(userId).getPin()[j] -
                                    result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()];
                        }
                    }
                    peakShaveCaps.put(userId, peakShaveCap);
                } else {
                    UserResult userResult = new UserResult(userId, cplex.getStatus().toString());
                    microgridResult.put(userId, userResult);
                }
            }
            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    /**
     * 带可中断负荷的集中式需求响应
     */
    public void cenIDRIL() {
        int peakShaveTimeNum = 0;
        for (int i = 0; i < periodNum; i++) {
            if (peakShaveTime[i] == 1) {
                peakShaveTimeNum++;
            }
        }
        Map<String, User> users = microgrid.getUsers();
        Map<String, Integer> userStarts = new HashMap<>(users.size());
        Map<String, Integer> periodVarNums = new HashMap<>(users.size());
        Map<String, Integer> userVarNums = new HashMap<>(users.size());
        int[] userStartsArray = new int[users.size()];
        int varNum = 0;
        Map<String, Integer> coeffNums = new HashMap<>(users.size());
        Map<String, Integer> coeffStarts = new HashMap<>(users.size());
        int[] coeffStartsArray = new int[users.size()];
        int coeffSumNum = 0;
        int count = 0;
        for (User user : users.values()) {
            List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
            List<AirCon> airCons = user.getAirCons();
            List<Converter> converters = user.getConverters();
            List<GasBoiler> gasBoilers = user.getGasBoilers();
            List<GasTurbine> gasTurbines = user.getGasTurbines();
            List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
            List<Storage> storages = user.getStorages();
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2;
            periodVarNums.put(user.getUserId(), periodVarNum);
            userVarNums.put(user.getUserId(), periodNum * periodVarNum + peakShaveTimeNum);
            varNum += periodNum * periodVarNum + peakShaveTimeNum;

            int coeffNum = (1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    4 * periodNum;
            coeffSumNum += coeffNum;
            coeffNums.put(user.getUserId(), coeffNum);

            if (count == 0) {
                userStartsArray[count] = periodNum * periodVarNum + peakShaveTimeNum;
                coeffStartsArray[count] = coeffNum;
            } else {
                userStartsArray[count] = userStartsArray[count - 1] + periodNum * periodVarNum + peakShaveTimeNum;
                coeffStartsArray[count] = coeffStartsArray[count - 1] + coeffNum;
            }
            count++;
        }
        for (int i = 0; i < users.size() - 1; i++) {
            userStartsArray[users.size() - 1 - i] = userStartsArray[users.size() - 1 - i - 1];
            coeffStartsArray[users.size() - 1 - i] = coeffStartsArray[users.size() - 1 - i - 1];
        }
        userStartsArray[0] = 0;
        coeffStartsArray[0] = 0;

        count = 0;
        for (User user : users.values()) {
            userStarts.put(user.getUserId(), userStartsArray[count]);
            coeffStarts.put(user.getUserId(), coeffStartsArray[count]);
            count++;
        }

        for (int i = 0; i < peakShaveTime.length; i++) {
            if (peakShaveTime[i] == 1) {
                coeffSumNum++;
            }
        }

        // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double[] columnLower = new double[varNum];
        // 状态变量上限
        double[] columnUpper = new double[varNum];
        // 指明变量类型
        IloNumVarType[] xt = new IloNumVarType[varNum];
        // 约束方程系数
        double[][] coeff = new double[coeffSumNum][varNum];

        for (User user : users.values()) {
            int periodVarNum = periodVarNums.get(user.getUserId());
            int userStart = userStarts.get(user.getUserId());

            List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
            List<AirCon> airCons = user.getAirCons();
            List<Converter> converters = user.getConverters();
            List<GasBoiler> gasBoilers = user.getGasBoilers();
            List<GasTurbine> gasTurbines = user.getGasTurbines();
            List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
            List<Storage> storages = user.getStorages();
            InterruptibleLoad interruptibleLoad = user.getInterruptibleLoad();

            // 设置变量上下限
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[userStart + j * periodVarNum + i] = 0;
                    columnUpper[userStart + j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[userStart + j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[userStart + j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[userStart + j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = 1;
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[userStart + j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[userStart + j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[userStart + j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[userStart + j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = -Double.MAX_VALUE;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[userStart + j * periodVarNum + handledVarNum + converters.size() + i] = -Double.MAX_VALUE;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[userStart + j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[userStart + j * periodVarNum + handledVarNum] = -Double.MAX_VALUE;
                columnUpper[userStart + j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[userStart + j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[userStart + j * periodVarNum + handledVarNum] = 0;
                columnUpper[userStart + j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[userStart + j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = 1;
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[userStart + j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[userStart + j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[userStart + j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[userStart + j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[userStart + j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[userStart + j * periodVarNum + handledVarNum] = -Double.MAX_VALUE;
                columnUpper[userStart + j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[userStart + j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[userStart + j * periodVarNum + handledVarNum] = 0;
                columnUpper[userStart + j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[userStart + j * periodVarNum + handledVarNum] = IloNumVarType.Float;
            }
            for (int j = 0; j < peakShaveTimeNum; j++) {
                columnLower[userStart + periodNum * periodVarNum + j] = 0;
                columnUpper[userStart + periodNum * periodVarNum + j] = interruptibleLoad.maxP;
                xt[userStart + periodNum * periodVarNum + j] = IloNumVarType.Float;
            }
        }

        try {
            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            IloNumExpr obj = cplex.numExpr();
            for (User user : users.values()) {
                int periodVarNum = periodVarNums.get(user.getUserId());
                int userStart = userStarts.get(user.getUserId());

                List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
                List<AirCon> airCons = user.getAirCons();
                List<Converter> converters = user.getConverters();
                List<GasBoiler> gasBoilers = user.getGasBoilers();
                List<GasTurbine> gasTurbines = user.getGasTurbines();
                List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
                List<Storage> storages = user.getStorages();
                InterruptibleLoad interruptibleLoad = user.interruptibleLoad;

                for (int j = 0; j < periodNum; j++) {
                    int handledVarNum = 0;
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        objValue[userStart + j * periodVarNum + i] = iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                        objValue[userStart + j * periodVarNum + iceStorageAcs.size() + i] = 0;
                        objValue[userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    }

                    handledVarNum += 3 * iceStorageAcs.size();
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        objValue[userStart + j * periodVarNum + handledVarNum + i] = 0;
                        objValue[userStart + j * periodVarNum + handledVarNum + gasTurbines.size() + i] = gasTurbines.get(i).getCss();  // 启停成本
                        objValue[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getCoper() * t;   // 运维成本
                        objValue[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                    }

                    handledVarNum += 3 * gasTurbines.size();
                    for (int i = 0; i < storages.size(); i++) {
                        objValue[userStart + j * periodVarNum + handledVarNum + i] = storages.get(i).getCoper() * t;   // 运维成本
                        objValue[userStart + j * periodVarNum + handledVarNum + i] += storages.get(i).getCbw() * t;    // 折旧成本
                        objValue[userStart + j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                    }

                    handledVarNum += 2 * storages.size();
                    handledVarNum += 2 * converters.size();
                    handledVarNum += 1;
                    objValue[userStart + j * periodVarNum + handledVarNum] = elecPrices[j] * t; // 购电成本

                    handledVarNum += 1;
                    for (int i = 0; i < airCons.size(); i++) {
                        objValue[userStart + j * periodVarNum + handledVarNum + i] = airCons.get(i).getCoper() * t;   // 运维成本
                    }

                    handledVarNum += airCons.size();
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        objValue[userStart + j * periodVarNum + handledVarNum + i] = 0;
                        objValue[userStart + j * periodVarNum + handledVarNum + gasBoilers.size() + i] = gasBoilers.get(i).getCss();  // 启停成本
                        objValue[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getCoper() * t;   // 运维成本
                        objValue[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                    }

                    handledVarNum += 3 * gasBoilers.size();
                    for (int i = 0; i < absorptionChillers.size(); i++) {
                        objValue[userStart + j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getCoper() * t;   // 运维成本
                    }

                    handledVarNum += absorptionChillers.size();
                    handledVarNum += 1;
                    objValue[userStart + j * periodVarNum + handledVarNum] = steamPrices[j] * t; // 购热成本
                }
                for (int j = 0; j < peakShaveTimeNum; j++) {
                    objValue[userStart + periodNum * periodVarNum + j] = interruptibleLoad.getB() * t; // 中断负荷成本
                    IloNumExpr p = cplex.prod(x[userStart + periodNum * periodVarNum + j], x[userStart + periodNum * periodVarNum + j]);
                    obj = cplex.sum(obj, cplex.prod(interruptibleLoad.getA() * t * t, p));
                }
            }
            cplex.addMinimize(cplex.sum(obj, cplex.scalProd(x, objValue)));

            for (User user : users.values()) {
                int periodVarNum = periodVarNums.get(user.getUserId());
                int userStart = userStarts.get(user.getUserId());
                int coeffStart = coeffStarts.get(user.getUserId());

                List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
                List<AirCon> airCons = user.getAirCons();
                List<Converter> converters = user.getConverters();
                List<GasBoiler> gasBoilers = user.getGasBoilers();
                List<GasTurbine> gasTurbines = user.getGasTurbines();
                List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
                List<Storage> storages = user.getStorages();
                SteamLoad steamLoad = user.getSteamLoad();
                Photovoltaic photovoltaic = user.getPhotovoltaic();
                double[] acLoad = user.acLoad;
                double[] dcLoad = user.dcLoad;
                double[] heatLoad = user.heatLoad;
                double[] coolingLoad = user.coolingLoad;
                double[] gatePowers = user.gatePowers;
                //记录数组中存储元素的个数
                int coeffNum = 0;
                for (int j = 0; j < periodNum; j++) {
                    // 交流母线电功率平衡约束
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    }
                    for (int i = 0; i < converters.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = -1; // 变流器AC-DC交流侧功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                    }
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + i] = -iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + iceStorageAcs.size() + i] = -iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                    }
                    for (int i = 0; i < airCons.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                    }
                    if (peakShaveTime[j] == 1) {
                        coeff[coeffStart + coeffNum][userStart + periodNum * periodVarNum + j - 13] = 1;   // 中断负荷功率
                    }
                    cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), acLoad[j]);
                    coeffNum += 1;

                    // 直流母线电功率平衡约束
                    for (int i = 0; i < converters.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = -1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                    }
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = -1; // 储能充电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                    }
                    double periodDcLoad = dcLoad[j];
                    periodDcLoad -= photovoltaic.getPower()[j];
                    cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), periodDcLoad);
                    coeffNum += 1;

                    // 向电网购电功率与电网输入电功率关系约束
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = -1; // 电网输入电功率
                    cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                    coeffNum += 1;

                    // 向园区购热功率与园区输入热功率关系约束
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = -1; // 园区输入热功率
                    cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                    coeffNum += 1;

                    // 燃气轮机启停功率约束
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = -gasTurbines.get(i).getMaxP();
                        cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = -gasTurbines.get(i).getMinP();
                        cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                        coeffNum += 1;
                    }

                    // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                    if (j == 0) {
                        for (int i = 0; i < gasTurbines.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = -1; // 燃气轮机状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), - gasTurbines.get(i).getInitState());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gasTurbines.get(i).getInitState());
                            coeffNum += 1;
                        }
                    } else {
                        for (int i = 0; i < gasTurbines.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = -1; // 燃气轮机状态
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = -1; // 燃气轮机上一时刻状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                            coeffNum += 1;
                        }
                    }

                    // 燃气轮机爬坡率约束
                    if (j > 0) {
                        for (int i = 0; i < gasTurbines.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = -1;   // 上一时刻燃气轮机产电功率
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gasTurbines.get(i).getMaxRampRate());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = -1;   // 上一时刻燃气轮机产电功率
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gasTurbines.get(i).getMinRampRate());
                            coeffNum += 1;
                        }
                    }

                    // 燃气锅炉启停功率约束
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + i] = -gasBoilers.get(i).getMaxH();
                        cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + i] = -gasBoilers.get(i).getMinH();
                        cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                        coeffNum += 1;
                    }

                    // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                    if (j == 0) {
                        for (int i = 0; i < gasBoilers.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = -1; // 燃气锅炉状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), -gasBoilers.get(i).getInitState());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gasBoilers.get(i).getInitState());
                            coeffNum += 1;
                        }
                    } else {
                        for (int i = 0; i < gasBoilers.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = -1; // 燃气锅炉状态
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                    2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = -1; // 燃气锅炉上一时刻状态
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), 0);
                            coeffNum += 1;
                        }
                    }

                    // 燃气锅炉爬坡率约束
                    if (j > 0) {
                        for (int i = 0; i < gasBoilers.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = -1;   // 上一时刻燃气锅炉产热功率
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gasBoilers.get(i).getRampRate());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = -1;   // 上一时刻燃气锅炉产热功率
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), -gasBoilers.get(i).getRampRate());
                            coeffNum += 1;
                        }
                    }

                    // 电池储能爬坡率约束
                    if (j > 0) {
                        for (int i = 0; i < storages.size(); i++) {
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = -1;   // 上一时刻储能充电功率
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = -1;   // 上一时刻储能充电功率
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), -storages.get(i).getYin() * storages.get(i).getMaxPIn());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -1;   // 上一时刻储能放电功率
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -1;   // 上一时刻储能放电功率
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                            coeffNum += 1;
                        }
                    }
                }

                // 电池储能容量约束
                for (int j = 0; j < periodNum; j++) {
                    if (j < periodNum - 1) {
                        for (int i = 0; i < storages.size(); i++) {
                            if (j > 0) {
                                coeff[coeffStart + coeffNum] = coeff[coeffStart + coeffNum - 2 * storages.size()];
                                coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum + 1] = coeff[coeffStart + coeffNum + 1 - 2 * storages.size()];
                                coeff[coeffStart + coeffNum + 1][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum + 1][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            }
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -t / storages.get(i).getEffOut(); // 储能放电量
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -t / storages.get(i).getEffOut(); // 储能放电量
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                            coeffNum += 1;
                        }
                    } else {
                        // 电池储能日电量累积约束
                        for (int i = 0; i < storages.size(); i++) {
                            coeff[coeffStart + coeffNum] = coeff[coeffStart + coeffNum + i - 2 * storages.size()];
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = -t / storages.get(i).getEffOut(); // 储能放电量
                            cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                            coeffNum += 1;
                        }
                    }
                }

                // 冰蓄冷耗电功率约束
                for (int j = 0; j < periodNum; j++) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + i] = 1; // 制冷机耗电功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), iceStorageAcs.get(i).getMaxP());
                        coeffNum += 1;
                    }
                }

                // 冰蓄冷容量约束
                for (int j = 0; j < periodNum; j++) {
                    if (j < periodNum - 1) {
                        for (int i = 0; i < iceStorageAcs.size(); i++) {
                            if (j > 0) {
                                coeff[coeffStart + coeffNum] = coeff[coeffStart + coeffNum - 2 * iceStorageAcs.size()];
                                coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum + 1] = coeff[coeffStart + coeffNum + 1 - 2 * iceStorageAcs.size()];
                                coeff[coeffStart + coeffNum + 1][userStart + (j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                                coeff[coeffStart + coeffNum + 1][userStart + (j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            }
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = -t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                            cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                            coeffNum += 1;
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = -t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                            cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                            coeffNum += 1;
                        }
                    } else {
                        // 冰蓄冷日累积冰量约束
                        for (int i = 0; i < iceStorageAcs.size(); i++) {
                            coeff[coeffStart + coeffNum] = coeff[coeffStart + coeffNum + i - 2 * iceStorageAcs.size()];
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffStart + coeffNum][userStart + (j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                            coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = -t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                            cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                            coeffNum += 1;
                        }
                    }
                }

                // 关口功率约束
                for (int j = 0; j < periodNum; j++) {
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gatePowers[j]);
                    coeffNum += 1;
                }

                // 热功率约束（不考虑空间热负荷）
                for (int j = 0; j < periodNum; j++) {
                    // 总热量约束
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                                gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                    }
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                    }
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                    for (int i = 0; i < absorptionChillers.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = -1;   // 吸收式制冷机耗热功率
                    }
                    cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                    coeffNum += 1;

                    // 中品味热约束
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                    }
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                    }
                    coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                    cplex.addGe(cplex.scalProd(x, coeff[coeffStart + coeffNum]), steamLoad.getDemand()[j]);
                    coeffNum += 1;
                }

                // 冷功率约束
                for (int j = 0; j < periodNum; j++) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                    }
                    for (int i = 0; i < airCons.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + i] = airCons.get(i).getEffac() * airCons.get(i).getEERc();   // 空调制冷功率
                    }
                    for (int i = 0; i < absorptionChillers.size(); i++) {
                        coeff[coeffStart + coeffNum][userStart + j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                    }
                    cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), coolingLoad[j]);
                    coeffNum += 1;
                }
            }

            // 削峰约束
            int coeffStart = coeffSumNum;
            for (int i = 0; i < peakShaveTime.length; i++) {
                if (peakShaveTime[i] == 1) {
                    coeffStart--;
                }
            }

            int coeffNum = 0;
            for (int i = 0; i < peakShaveTime.length; i++) {
                if (peakShaveTime[i] == 1) {
                    for (User user : users.values()) {
                        int periodVarNum = periodVarNums.get(user.getUserId());
                        int userStart = userStarts.get(user.getUserId());

                        List<Converter> converters = user.getConverters();
                        List<GasTurbine> gasTurbines = user.getGasTurbines();
                        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
                        List<Storage> storages = user.getStorages();

                        coeff[coeffStart + coeffNum][userStart + i * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                    }
                    cplex.addEq(cplex.scalProd(x, coeff[coeffStart + coeffNum]), gatePowerSum[i]);
                    coeffNum += 1;
                }
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double minCost = cplex.getObjValue();
                    double[] results = cplex.getValues(x);
                    count = 0;
                    for (User user : users.values()) {
                        int periodVarNum = periodVarNums.get(user.getUserId());
                        int userVarNum = userVarNums.get(user.getUserId());
                        int userStart = userStarts.get(user.getUserId());

                        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
                        List<AirCon> airCons = user.getAirCons();
                        List<Converter> converters = user.getConverters();
                        List<GasBoiler> gasBoilers = user.getGasBoilers();
                        List<GasTurbine> gasTurbines = user.getGasTurbines();
                        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
                        List<Storage> storages = user.getStorages();
                        Photovoltaic photovoltaic = user.getPhotovoltaic();
                        InterruptibleLoad interruptibleLoad = user.getInterruptibleLoad();

                        // 每个用户的成本
                        double userMinCost = 0;
                        for (int j = 0; j < periodNum; j++) {
                            // 加上光伏运行成本
                            minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];
                            userMinCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];

                            int handledVarNum = 0;
                            for (int i = 0; i < iceStorageAcs.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + i] * iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                            }

                            handledVarNum += 3 * iceStorageAcs.size();
                            for (int i = 0; i < gasTurbines.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + gasTurbines.size() + i] * gasTurbines.get(i).getCss();  // 启停成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] * gasTurbines.get(i).getCoper() * t;   // 运维成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] * gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                            }

                            handledVarNum += 3 * gasTurbines.size();
                            for (int i = 0; i < storages.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + i] * storages.get(i).getCoper() * t;   // 运维成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + i] * storages.get(i).getCbw() * t;    // 折旧成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + storages.size() + i] * storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                            }

                            handledVarNum += 2 * storages.size();
                            handledVarNum += 2 * converters.size();
                            handledVarNum += 1;
                            userMinCost += results[userStart + j * periodVarNum + handledVarNum] * elecPrices[j] * t; // 购电成本

                            handledVarNum += 1;
                            for (int i = 0; i < airCons.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + i] * airCons.get(i).getCoper() * t;   // 运维成本
                            }

                            handledVarNum += airCons.size();
                            for (int i = 0; i < gasBoilers.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + gasBoilers.size() + i] * gasBoilers.get(i).getCss();  // 启停成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] * gasBoilers.get(i).getCoper() * t;   // 运维成本
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] * gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                            }

                            handledVarNum += 3 * gasBoilers.size();
                            for (int i = 0; i < absorptionChillers.size(); i++) {
                                userMinCost += results[userStart + j * periodVarNum + handledVarNum + i] * absorptionChillers.get(i).getCoper() * t;   // 运维成本
                            }

                            handledVarNum += absorptionChillers.size();
                            handledVarNum += 1;
                            userMinCost += results[userStart + j * periodVarNum + handledVarNum] * steamPrices[j] * t; // 购热成本
                        }
                        for (int j = 0; j < peakShaveTimeNum; j++) {
                            userMinCost += results[userStart + periodNum * periodVarNum + j] * t *
                                    (interruptibleLoad.getA() * results[userStart + periodNum * periodVarNum + j] * t + interruptibleLoad.getB()); // 中断负荷成本
                            System.out.println("中断负荷量：" + results[userStart + periodNum * periodVarNum + j]);
                            double mgc = 2 * interruptibleLoad.getA() * results[userStart + periodNum * periodVarNum + j] * t + interruptibleLoad.getB() - 0.7014;
                            System.out.println("边际成本：" + mgc);
                        }

                        double[] result = new double[userVarNum];
                        if (count < users.size() - 1) {
                            for (int i = userStartsArray[count]; i < userStartsArray[count + 1]; i++) {
                                result[i - userStartsArray[count]] = results[i];
                            }
                        } else {
                            for (int i = userStartsArray[count]; i < varNum; i++) {
                                result[i - userStartsArray[count]] = results[i];
                            }
                        }

                        createUserResult(user.getUserId(), cplex.getStatus().toString(), userMinCost, result,
                                periodVarNum, iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers,
                                absorptionChillers);
                        count++;
                    }
                    System.out.println("总最小成本\t" + minCost);
                } else {
                    for (User user : users.values()) {
                        UserResult userResult = new UserResult(user.getUserId(), cplex.getStatus().toString());
                        userResult.setMinCost(Double.MAX_VALUE);
                        microgridResult.put(user.getUserId(), userResult);
                    }
                }
            } else {
                for (User user : users.values()) {
                    UserResult userResult = new UserResult(user.getUserId(), cplex.getStatus().toString());
                    userResult.setMinCost(Double.MAX_VALUE);
                    microgridResult.put(user.getUserId(), userResult);
                }
            }

            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    /**
     * 带可中断负荷的分布式需求响应，两部制电价确定应削峰量
     */
    public void distIDRIL(User user) {
        int peakShaveTimeNum = 0;
        for (int i = 0; i < periodNum; i++) {
            if (peakShaveTime[i] == 1) {
                peakShaveTimeNum++;
            }
        }
        String userId = user.getUserId();
        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
        List<AirCon> airCons = user.getAirCons();
        List<Converter> converters = user.getConverters();
        List<GasBoiler> gasBoilers = user.getGasBoilers();
        List<GasTurbine> gasTurbines = user.getGasTurbines();
        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
        List<Storage> storages = user.getStorages();
        SteamLoad steamLoad = user.getSteamLoad();
        Photovoltaic photovoltaic = user.getPhotovoltaic();
        double[] acLoad = user.acLoad;
        double[] dcLoad = user.dcLoad;
        double[] heatLoad = user.heatLoad;
        double[] coolingLoad = user.coolingLoad;
        double[] gatePowers = user.gatePowers;
        double[] mc = mcs.get(userId);
        InterruptibleLoad interruptibleLoad = user.getInterruptibleLoad();
        try {
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2;
            int varNum = periodNum * periodVarNum + peakShaveTimeNum;
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[varNum];
            // 状态变量上限
            double columnUpper[] = new double[varNum];
            // 指明变量类型
            IloNumVarType[] xt = new IloNumVarType[varNum];
            // 约束方程系数
            double[][] coeff = new double[(1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    periodNum +
                    3 * periodNum][varNum];

            // 设置变量上下限
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[j * periodVarNum + i] = 0;
                    columnUpper[j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + converters.size() + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;
            }
            for (int j = 0; j < peakShaveTimeNum; j++) {
                columnLower[periodNum * periodVarNum + j] = 0;
                columnUpper[periodNum * periodVarNum + j] = interruptibleLoad.maxP;
                xt[periodNum * periodVarNum + j] = IloNumVarType.Float;
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    objValue[j * periodVarNum + i] = iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    objValue[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = gasTurbines.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = storages.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + i] += storages.get(i).getCbw() * t;    // 折旧成本
                    objValue[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                }

                handledVarNum += 2 * storages.size();
                handledVarNum += 2 * converters.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = elecPrices[j] * t; // 购电成本
                // 购买削峰量成本
                if (peakShaveTime[j] == 1) {
                    objValue[j * periodVarNum + handledVarNum] += mc[j] * t;
                }

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = airCons.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = gasBoilers.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += absorptionChillers.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = steamPrices[j] * t; // 购热成本
            }
            IloNumExpr obj = cplex.numExpr();
            for (int j = 0; j < peakShaveTimeNum; j++) {
                objValue[periodNum * periodVarNum + j] = interruptibleLoad.getB() * t; // 中断负荷成本
                IloNumExpr p = cplex.prod(x[periodNum * periodVarNum + j], x[periodNum * periodVarNum + j]);
                obj = cplex.sum(obj, cplex.prod(interruptibleLoad.getA() * t * t, p));
            }
            cplex.addMinimize(cplex.sum(obj, cplex.scalProd(x, objValue)));

            //记录数组中存储元素的个数
            int coeffNum = 0;
            for (int j = 0; j < periodNum; j++) {
                // 交流母线电功率平衡约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                }
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = - 1; // 变流器AC-DC交流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                }
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = - iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = - iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                }
                if (peakShaveTime[j] == 1) {
                    coeff[coeffNum][periodNum * periodVarNum + j - 13] = 1;   // 中断负荷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), acLoad[j]);
                coeffNum += 1;

                // 直流母线电功率平衡约束
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = - 1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                }
                for (int i = 0; i < storages.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                }
                double periodDcLoad = dcLoad[j];
                periodDcLoad -= photovoltaic.getPower()[j];
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), periodDcLoad);
                coeffNum += 1;

                // 向电网购电功率与电网输入电功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1; // 电网输入电功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 向园区购热功率与园区输入热功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = - 1; // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 燃气轮机启停功率约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMinP();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气轮机爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMaxRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMinRampRate());
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉启停功率约束
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMaxH();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMinH();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                    }
                }

                // 电池储能爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                    }
                }
            }

            // 电池储能容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < storages.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * storages.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * storages.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 电池储能日电量累积约束
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * storages.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 冰蓄冷耗电功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = 1; // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxP());
                    coeffNum += 1;
                }
            }

            // 冰蓄冷容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * iceStorageAcs.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * iceStorageAcs.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 冰蓄冷日累积冰量约束
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * iceStorageAcs.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 关口功率约束
            for (int j = 0; j < periodNum; j++) {
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                coeffNum += 1;
            }

            // 热功率约束（不考虑空间热负荷）
            for (int j = 0; j < periodNum; j++) {
                // 总热量约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                            gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = - 1;   // 吸收式制冷机耗热功率
                }
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                coeffNum += 1;

                // 中品味热约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j]);
                coeffNum += 1;
            }

            // 冷功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                    coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + i] = airCons.get(i).getEffac() * airCons.get(i).getEERc();   // 空调制冷功率
                }
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), coolingLoad[j]);
                coeffNum += 1;
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double minCost = cplex.getObjValue();
                    // 加上光伏运行成本
                    for (int j = 0; j < periodNum; j++) {
                        minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];
                    }
                    // 加上购买削峰量成本
                    for (int j = 0; j < periodNum; j++) {
                        if (peakShaveTime[j] == 1) {
                            minCost -= mc[j] * t * (selfOptResult.get(userId).getPin()[j] - peakShavePowers.get(userId)[j]);
                        }
                    }
                    double[] result = cplex.getValues(x);
                    // 减去购买削峰量成本
//                    for (int j = 0; j < periodNum; j++) {
//                        if (peakShaveTime[j] == 1) {
//                            minCost += mc[j] * t * (selfOptResult.get(userId).getPin()[j] - peakShavePowers.get(userId)[j] -
//                                    result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1]);
//                        }
//                    }
                    createUserResult(userId, cplex.getStatus().toString(), minCost, result, periodVarNum,
                            iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers, absorptionChillers);

                    double[] peakShaveCap = new double[periodNum];
                    for (int j = 0; j < periodNum; j++) {
                        if (peakShaveTime[j] == 1) {
                            peakShaveCap[j] = selfOptResult.get(userId).getPin()[j] -
                                    result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()];
                        }
                    }
                    peakShaveCaps.put(userId, peakShaveCap);
                } else {
                    UserResult userResult = new UserResult(userId, cplex.getStatus().toString());
                    microgridResult.put(userId, userResult);
                }
            }
            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    /**
     * 带可中断负荷的分布式需求响应，应削峰量为0
     */
    public void distIDRILNew(User user) {
        int peakShaveTimeNum = 0;
        for (int i = 0; i < periodNum; i++) {
            if (peakShaveTime[i] == 1) {
                peakShaveTimeNum++;
            }
        }
        String userId = user.getUserId();
        List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
        List<AirCon> airCons = user.getAirCons();
        List<Converter> converters = user.getConverters();
        List<GasBoiler> gasBoilers = user.getGasBoilers();
        List<GasTurbine> gasTurbines = user.getGasTurbines();
        List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
        List<Storage> storages = user.getStorages();
        SteamLoad steamLoad = user.getSteamLoad();
        Photovoltaic photovoltaic = user.getPhotovoltaic();
        double[] acLoad = user.acLoad;
        double[] dcLoad = user.dcLoad;
        double[] heatLoad = user.heatLoad;
        double[] coolingLoad = user.coolingLoad;
        double[] gatePowers = user.gatePowers;
        double[] mc = mcs.get(userId);
        InterruptibleLoad interruptibleLoad = user.getInterruptibleLoad();
        try {
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2;
            int varNum = periodNum * periodVarNum + peakShaveTimeNum;
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[varNum];
            // 状态变量上限
            double columnUpper[] = new double[varNum];
            // 指明变量类型
            IloNumVarType[] xt = new IloNumVarType[varNum];
            // 约束方程系数
            double[][] coeff = new double[(1 + 1 + 1 + 1) * periodNum + (2 + 2) * gasTurbines.size() * periodNum +
                    2 * gasTurbines.size() * (periodNum - 1) + (2 + 2) * gasBoilers.size() * periodNum +
                    2 * gasBoilers.size() * (periodNum - 1) + 4 * storages.size() * (periodNum - 1) +
                    2 * storages.size() * (periodNum - 1) + storages.size() +
                    iceStorageAcs.size() * periodNum +
                    2 * iceStorageAcs.size() * (periodNum - 1) +
                    iceStorageAcs.size() +
                    periodNum +
                    3 * periodNum][varNum];

            // 设置变量上下限
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[j * periodVarNum + i] = 0;
                    columnUpper[j * periodVarNum + i] = iceStorageAcs.get(i).getMaxP();
                    xt[j * periodVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + iceStorageAcs.size() + i] = iceStorageAcs.get(i).getMaxPice();
                    xt[j * periodVarNum + iceStorageAcs.size() + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                    columnUpper[j * periodVarNum + 2 * iceStorageAcs.size() + i] = iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getMaxPmelt();
                    xt[j * periodVarNum + 2 * iceStorageAcs.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = storages.get(i).getMaxPIn();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + storages.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getMaxPOut();
                    xt[j * periodVarNum + handledVarNum + storages.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * storages.size();
                for (int i = 0; i < converters.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                    columnLower[j * periodVarNum + handledVarNum + converters.size() + i] = - Double.MAX_VALUE;
                    columnUpper[j * periodVarNum + handledVarNum + converters.size() + i] = Double.MAX_VALUE;
                    xt[j * periodVarNum + handledVarNum + converters.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 2 * converters.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMinP();
                    columnUpper[j * periodVarNum + handledVarNum + i] = airCons.get(i).getMaxP();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + i] = 1;
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = 1;
                    xt[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = IloNumVarType.Bool;
                    columnLower[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = 0;
                    columnUpper[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = IloNumVarType.Float;
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    columnLower[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMinH();
                    columnUpper[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getMaxH();
                    xt[j * periodVarNum + handledVarNum + i] = IloNumVarType.Float;
                }

                handledVarNum += absorptionChillers.size();
                columnLower[j * periodVarNum + handledVarNum] = - Double.MAX_VALUE;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;

                handledVarNum += 1;
                columnLower[j * periodVarNum + handledVarNum] = 0;
                columnUpper[j * periodVarNum + handledVarNum] = Double.MAX_VALUE;
                xt[j * periodVarNum + handledVarNum] = IloNumVarType.Float;
            }
            for (int j = 0; j < peakShaveTimeNum; j++) {
                columnLower[periodNum * periodVarNum + j] = 0;
                columnUpper[periodNum * periodVarNum + j] = interruptibleLoad.maxP;
                xt[periodNum * periodVarNum + j] = IloNumVarType.Float;
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            for (int j = 0; j < periodNum; j++) {
                int handledVarNum = 0;
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    objValue[j * periodVarNum + i] = iceStorageAcs.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + iceStorageAcs.size() + i] = 0;
                    objValue[j * periodVarNum + 2 * iceStorageAcs.size() + i] = 0;
                }

                handledVarNum += 3 * iceStorageAcs.size();
                for (int i = 0; i < gasTurbines.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasTurbines.size() + i] = gasTurbines.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] += gasPrices[j] / gasTurbines.get(i).getEffe() * t;    // 燃料成本
                }

                handledVarNum += 3 * gasTurbines.size();
                for (int i = 0; i < storages.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = storages.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + i] += storages.get(i).getCbw() * t;    // 折旧成本
                    objValue[j * periodVarNum + handledVarNum + storages.size() + i] = storages.get(i).getCoper() / storages.get(i).effOut * t;  // 运维成本
                }

                handledVarNum += 2 * storages.size();
                handledVarNum += 2 * converters.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = elecPrices[j] * t; // 购电成本
                // 削峰成本（去掉常数项）
                if (peakShaveTime[j] == 1) {
                    objValue[j * periodVarNum + handledVarNum] += mc[j] * t;
                }

                handledVarNum += 1;
                for (int i = 0; i < airCons.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = airCons.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += airCons.size();
                for (int i = 0; i < gasBoilers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = 0;
                    objValue[j * periodVarNum + handledVarNum + gasBoilers.size() + i] = gasBoilers.get(i).getCss();  // 启停成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getCoper() * t;   // 运维成本
                    objValue[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] += gasPrices[j] / gasBoilers.get(i).getEffgb() * t; // 燃料成本
                }

                handledVarNum += 3 * gasBoilers.size();
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = absorptionChillers.get(i).getCoper() * t;   // 运维成本
                }

                handledVarNum += absorptionChillers.size();
                handledVarNum += 1;
                objValue[j * periodVarNum + handledVarNum] = steamPrices[j] * t; // 购热成本
            }
            IloNumExpr obj = cplex.numExpr();
            for (int j = 0; j < peakShaveTimeNum; j++) {
                objValue[periodNum * periodVarNum + j] = interruptibleLoad.getB() * t; // 中断负荷成本
                IloNumExpr p = cplex.prod(x[periodNum * periodVarNum + j], x[periodNum * periodVarNum + j]);
                obj = cplex.sum(obj, cplex.prod(interruptibleLoad.getA() * t * t, p));
            }
            cplex.addMinimize(cplex.sum(obj, cplex.scalProd(x, objValue)));

            //记录数组中存储元素的个数
            int coeffNum = 0;
            for (int j = 0; j < periodNum; j++) {
                // 交流母线电功率平衡约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 电网输入电功率
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                }
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = - 1; // 变流器AC-DC交流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                }
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = - iceStorageAcs.get(i).getConsumCoef(); // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = - iceStorageAcs.get(i).getConsumCoef();  // 蓄冰槽耗电功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + i] = - airCons.get(i).getConsumCoef();   // 中央空调耗电功率
                }
                if (peakShaveTime[j] == 1) {
                    coeff[coeffNum][periodNum * periodVarNum + j - 13] = 1;   // 中断负荷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), acLoad[j]);
                coeffNum += 1;

                // 直流母线电功率平衡约束
                for (int i = 0; i < converters.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = - 1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                }
                for (int i = 0; i < storages.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1; // 储能充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                }
                double periodDcLoad = dcLoad[j];
                periodDcLoad -= photovoltaic.getPower()[j];
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), periodDcLoad);
                coeffNum += 1;

                // 向电网购电功率与电网输入电功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 1] = 1; // 向电网购电功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = - 1; // 电网输入电功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 向园区购热功率与园区输入热功率关系约束
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1] = 1; // 向园区购热功率
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = - 1; // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                coeffNum += 1;

                // 燃气轮机启停功率约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - gasTurbines.get(i).getMinP();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气轮机状态变化的变量与燃气轮机启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + gasTurbines.size() + i] = 1; // 燃气轮机状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + i] = 1; // 燃气轮机状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + i] = - 1; // 燃气轮机上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气轮机爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMaxRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = - 1;   // 上一时刻燃气轮机产电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasTurbines.get(i).getMinRampRate());
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉启停功率约束
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMaxH();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + i] = - gasBoilers.get(i).getMinH();
                    cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                }

                // 表示燃气锅炉状态变化的变量与燃气锅炉启停状态关系约束
                if (j == 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getInitState());
                        coeffNum += 1;
                    }
                } else {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + gasBoilers.size() + i] = 1; // 燃气锅炉状态变化
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = 1; // 燃气锅炉状态
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                                2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i] = - 1; // 燃气锅炉上一时刻状态
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), 0);
                        coeffNum += 1;
                    }
                }

                // 燃气锅炉爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < gasBoilers.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1; // 燃气锅炉产热功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = - 1;   // 上一时刻燃气锅炉产热功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - gasBoilers.get(i).getRampRate());
                        coeffNum += 1;
                    }
                }

                // 电池储能爬坡率约束
                if (j > 0) {
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYin() * storages.get(i).getMaxPIn());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), - storages.get(i).getYout() * storages.get(i).getMaxPOut());
                        coeffNum += 1;
                    }
                }
            }

            // 电池储能容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < storages.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * storages.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * storages.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMaxS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getMinS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 电池储能日电量累积约束
                    for (int i = 0; i < storages.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * storages.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] *= 1 - storages.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = t * storages.get(i).getEffIn(); // 储能充电量
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - t / storages.get(i).getEffOut(); // 储能放电量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getEndS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 冰蓄冷耗电功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = 1; // 制冷机耗电功率
                    coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = 1; // 蓄冰槽耗电功率
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxP());
                    coeffNum += 1;
                }
            }

            // 冰蓄冷容量约束
            for (int j = 0; j < periodNum; j++) {
                if (j < periodNum - 1) {
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        if (j > 0) {
                            coeff[coeffNum] = coeff[coeffNum - 2 * iceStorageAcs.size()];
                            coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1] = coeff[coeffNum + 1 - 2 * iceStorageAcs.size()];
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                            coeff[coeffNum + 1][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        }
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMaxS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getMinS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                } else {
                    // 冰蓄冷日累积冰量约束
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffNum] = coeff[coeffNum + i - 2 * iceStorageAcs.size()];
                        coeff[coeffNum][(j - 1) * periodVarNum + iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][(j - 1) * periodVarNum + 2 * iceStorageAcs.size() + i] *= 1 - iceStorageAcs.get(i).getLossCoef();
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = t * iceStorageAcs.get(i).getEERice() * iceStorageAcs.get(i).getEffice(); // 冰蓄冷蓄冰量
                        coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = - t / iceStorageAcs.get(i).getEffmelt(); // 冰蓄冷融冰量
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), iceStorageAcs.get(i).getInitS() - (1 - iceStorageAcs.get(i).getLossCoef()) * iceStorageAcs.get(i).getInitS());
                        coeffNum += 1;
                    }
                }
            }

            // 关口功率约束
            for (int j = 0; j < periodNum; j++) {
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;    // 电网输入电功率
                cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePowers[j]);
                coeffNum += 1;
            }

            // 热功率约束（不考虑空间热负荷）
            for (int j = 0; j < periodNum; j++) {
                // 总热量约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) /
                            gasTurbines.get(i).getEffe() * (gasTurbines.get(i).getEffhm() + gasTurbines.get(i).getEffhl());   // 燃气轮机中品味热和低品位热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = - 1;   // 吸收式制冷机耗热功率
                }
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j] * (1 - steamLoad.getEffh()) + heatLoad[j]);
                coeffNum += 1;

                // 中品味热约束
                for (int i = 0; i < gasTurbines.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = (1 - gasTurbines.get(i).getEffe()) / gasTurbines.get(i).getEffe() * gasTurbines.get(i).getEffhm();   // 燃气轮机中品味热
                }
                for (int i = 0; i < gasBoilers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i] = 1;   // 燃气锅炉产热功率
                }
                coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()] = 1;   // 园区输入热功率
                cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), steamLoad.getDemand()[j]);
                coeffNum += 1;
            }

            // 冷功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + i] = iceStorageAcs.get(i).getEffref() * iceStorageAcs.get(i).getEERc();   // 制冷机制冷功率
                    coeff[coeffNum][j * periodVarNum + 2 * iceStorageAcs.size() + i] = 1;   // 蓄冰槽制冷功率
                }
                for (int i = 0; i < airCons.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + i] = airCons.get(i).getEffac() * airCons.get(i).getEERc();   // 空调制冷功率
                }
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
                }
                cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), coolingLoad[j]);
                coeffNum += 1;
            }

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double minCost = cplex.getObjValue();
                    // 加上光伏运行成本
                    for (int j = 0; j < periodNum; j++) {
                        minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];
                    }
                    // 加上削峰成本常数项
                    for (int j = 0; j < periodNum; j++) {
                        if (peakShaveTime[j] == 1) {
                            minCost -= mc[j] * t * selfOptResult.get(userId).getPin()[j];
                        }
                    }
                    double[] result = cplex.getValues(x);
                    createUserResult(userId, cplex.getStatus().toString(), minCost, result, periodVarNum,
                            iceStorageAcs, gasTurbines, storages, converters, airCons, gasBoilers, absorptionChillers);

                    double[] peakShaveCap = new double[periodNum];
                    for (int j = 0; j < periodNum; j++) {
                        if (peakShaveTime[j] == 1) {
                            peakShaveCap[j] = selfOptResult.get(userId).getPin()[j] -
                                    result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()];
                        }
                    }
                    peakShaveCaps.put(userId, peakShaveCap);
                } else {
                    UserResult userResult = new UserResult(userId, cplex.getStatus().toString());
                    microgridResult.put(userId, userResult);
                }
            }
            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    public void setSelfOptResult(Map<String, UserResult> selfOptResult) {
        this.selfOptResult = selfOptResult;
    }

    public void setDemandRespResult(Map<String, UserResult> demandRespResult) {
        this.demandRespResult = demandRespResult;
    }

    public int[] getPeakShaveTime() {
        return peakShaveTime;
    }

    public void setPeakShaveTime(int[] peakShaveTime) {
        this.peakShaveTime = peakShaveTime;
    }

    public double getClearingPrice() {
        return clearingPrice;
    }

    public void setClearingPrice(double clearingPrice) {
        this.clearingPrice = clearingPrice;
    }

    public Map<String, double[]> getPeakShavePowers() {
        return peakShavePowers;
    }

    public void setPeakShavePowers(Map<String, double[]> peakShavePowers) {
        this.peakShavePowers = peakShavePowers;
    }

    public double getTotalPeakShaveCap() {
        return totalPeakShaveCap;
    }

    public Map<String, Double> getPeakShaveRatios() {
        return peakShaveRatios;
    }

    public List<Offer> getOffers() {
        return offers;
    }

    public Map<String, double[]> getShaveGatePowers() {
        return shaveGatePowers;
    }

    public List<Map<String, Double>> getTimeShaveCapRatios() {
        return timeShaveCapRatios;
    }

    public Map<String, double[]> getMcs() {
        return mcs;
    }

    public void setMcs(Map<String, double[]> mcs) {
        this.mcs = mcs;
    }

    public Map<String, double[]> getPeakShaveCaps() {
        return peakShaveCaps;
    }

    public void setGatePowerSum(double[] gatePowerSum) {
        this.gatePowerSum = gatePowerSum;
    }
}
