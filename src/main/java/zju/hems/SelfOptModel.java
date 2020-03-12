package zju.hems;

import ilog.concert.IloException;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;

import java.util.*;

/**
 * 自趋优模型
 * @author Xu Chengsi
 * @date 2019/6/20
 */
public class SelfOptModel {

    Microgrid microgrid;
    int periodNum; // 时段数
    double t;   // 单位时段长度
    double[] elecPrices;    // 电价
    double[] gasPrices;    // 天然气价格
    double[] steamPrices;    // 园区CHP蒸汽价格
    double[] chargePrices;    // 充电桩充电价格

    Map<String, UserResult> microgridResult;

    public SelfOptModel(Microgrid microgrid, int periodNum, double t, double[] elecPrices, double[] gasPrices,
                        double[] steamPrices) {
        this.microgrid = microgrid;
        this.periodNum = periodNum;
        this.t = t;
        this.elecPrices = elecPrices;
        this.gasPrices = gasPrices;
        this.steamPrices = steamPrices;
    }

    public SelfOptModel(Microgrid microgrid, int periodNum, double t, double[] elecPrices, double[] gasPrices,
                        double[] steamPrices, double[] chargePrices) {
        this(microgrid, periodNum, t, elecPrices, gasPrices, steamPrices);
        this.chargePrices = chargePrices;
    }

    public void mgSelfOpt() {
        microgridResult = new HashMap<>(microgrid.getUsers().size());
        for (User user : microgrid.getUsers().values()) {
            userSelfOpt(user);
//            chargingPileOpt(user);
        }
    }

    public void userSelfOpt(User user) {
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

    public void chargingPileOpt(User user) {
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
            // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
            // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
            // 电网输入电功率，向电网购电功率，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，
            // 燃气锅炉产热功率，吸收式制冷机耗热功率，园区输入热功率，向园区购热功率，充电桩启停状态，充电桩功率
            int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
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
                    4 * periodNum +
                    2 * chargingPiles.size() * periodNum + chargingPiles.size()][varNum];

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

                handledVarNum += 1;
                handledVarNum += chargingPiles.size();
                for (int i = 0; i < chargingPiles.size(); i++) {
                    objValue[j * periodVarNum + handledVarNum + i] = - chargePrices[j] * t;   // 充电桩收益
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
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = -1; // 变流器AC-DC交流侧功率
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
                for (int i = 0; i < chargingPiles.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
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

            // 充电桩启停功率约束
            for (int j = 0; j < periodNum; j++) {
                for (int i = 0; i < chargingPiles.size(); i++) {
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                            chargingPiles.size() + i] = 1; // 充电桩充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() +
                            absorptionChillers.size() + 2 + i] = - chargingPiles.get(i).getMaxP();
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), 0);
                    coeffNum += 1;
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                            chargingPiles.size() + i] = 1; // 充电桩充电功率
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                            2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() +
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
                                2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                                chargingPiles.size() + i] = t; // 充电桩充电量
                    }
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), chargingPile.getSm());
                } else if (chargingPile.getMode() == 2) {
                    // 对于处于按时间充电模式的充电桩，在设定的时间过后，充电桩停止充电
                    for (int j = 0; j < periodNum; j++) {
                        if (j >= chargingPile.getTe()) {
                            coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                                    chargingPiles.size() + i] = 1; // 充电桩充电功率
                        }
                        cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), 0);
                    }
                } else if (chargingPile.getMode() == 3) {
                    // 对于处于按金额充电模式的充电桩，充电桩消费金额不超过设定金额
                    for (int j = 0; j < periodNum; j++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
                                chargingPiles.size() + i] = chargePrices[j] * t; // 充电费用
                    }
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), chargingPile.getM());
                } else if (chargingPile.getMode() == 4) {
                    // 对于处于按电量充电模式的充电桩，充电量不超过设定的电量
                    for (int j = 0; j < periodNum; j++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 2 +
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
                    double minCost = cplex.getObjValue();
                    // 加上光伏运行成本
                    for (int j = 0; j < periodNum; j++) {
                        minCost += photovoltaic.getCoper() * photovoltaic.getPower()[j];
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
                                 List<AbsorptionChiller> absorptionChillers) {
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
                airConsP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + i];
            }
            for (int i = 0; i < gasBoilers.size(); i++) {
                gasBoilersState.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i];
                gasBoilersH.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                        2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i];
            }
            for (int i = 0; i < absorptionChillers.size(); i++) {
                absorptionChillersH.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                        2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i];
            }
            Hin[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()];
            purH[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1];
        }
        UserResult userResult = new UserResult(userId, status, minCost,
                frigesP, iceTanksP, iceTanksQ, gasTurbinesState, gasTurbinesP, storagesP, convertersP, Pin,
                purP, airConsP, gasBoilersState, gasBoilersH, absorptionChillersH, Hin, purH);
        microgridResult.put(userId, userResult);
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
                airConsP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + i];
            }
            for (int i = 0; i < gasBoilers.size(); i++) {
                gasBoilersState.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + i];
                gasBoilersH.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                        2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 2 * gasBoilers.size() + i];
            }
            for (int i = 0; i < absorptionChillers.size(); i++) {
                absorptionChillersH.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                        2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + i];
            }
            Hin[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size()];
            purH[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                    2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1];
            for (int i = 0; i < chargingPiles.size(); i++) {
                chargingPilesState.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                        2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() +
                        absorptionChillers.size() + 2 + i];
                chargingPilesP.get(i)[j] = result[j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() +
                        2 * storages.size() + 2 * converters.size() + 2 + airCons.size() + 3 * gasBoilers.size() +
                        absorptionChillers.size() + 2 + chargingPiles.size() + i];
            }
        }
        UserResult userResult = new UserResult(userId, status, minCost,
                frigesP, iceTanksP, iceTanksQ, gasTurbinesState, gasTurbinesP, storagesP, convertersP, Pin, purP,
                airConsP, gasBoilersState, gasBoilersH, absorptionChillersH, Hin, purH, chargingPilesState, chargingPilesP);
        microgridResult.put(userId, userResult);
    }

    public Microgrid getMicrogrid() {
        return microgrid;
    }

    public Map<String, UserResult> getMicrogridResult() {
        return microgridResult;
    }
}
