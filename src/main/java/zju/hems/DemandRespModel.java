package zju.hems;

import ilog.concert.IloException;
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

    public DemandRespModel(Microgrid microgrid, int periodNum, double[] elecPrices, double[] gasPrices, double[] steamPrices) {
        super(microgrid, periodNum, elecPrices, gasPrices, steamPrices);
    }

    public void mgOrigDemandResp() {
        microgridResult = new HashMap<>(microgrid.getUsers().size());
        for (User user : microgrid.getUsers().values()) {
            origDemandResp(user);
        }
    }

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
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getInitS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
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
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 3 + airCons.size() + 3 * gasBoilers.size()] = - 1;   // 吸收式制冷机耗热功率
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
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i] = absorptionChillers.get(i).getIc();   // 吸收式制冷机供冷功率
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

    public void mgDemandResp() {
        microgridResult = new HashMap<>(microgrid.getUsers().size());
        for (User user : microgrid.getUsers().values()) {
            demandResp(user);
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
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getInitS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
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
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size()] = - 1;   // 吸收式制冷机耗热功率
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
            cenDistDemandResp(user);
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
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getInitS() - (1 - storages.get(i).getLossCoef()) * storages.get(i).getInitS());
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
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size()] = - 1;   // 吸收式制冷机耗热功率
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
}
