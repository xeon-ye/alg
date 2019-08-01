package zju.hems;

import ilog.concert.IloException;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;

import java.util.List;
import java.util.Map;

public class SelfOptModel {

    Microgrid microgrid;
    int periodNum = 96; // 一天的时段数
    double t;   // 单位时段长度
    double[] acLoad;    // 交流负荷
    double[] dcLoad;    // 直流负荷
    double[] elecPrices;    // 电价
    double[] gasPrices;    // 天然气价格
    double[] steamPrices;    // 园区CHP蒸汽价格
    Map<String, double[]> gatePower;   // 用户关口功率

    public SelfOptModel(Microgrid microgrid, int periodNum, double[] acLoad, double[] dcLoad, double[] elecPrices,
                        double[] gasPrices, double[] steamPrices, Map<String, double[]> gatePower) {
        this.microgrid = microgrid;
        this.periodNum = periodNum;
        t = 24 / periodNum;
        this.acLoad = acLoad;
        this.dcLoad = dcLoad;
        this.elecPrices = elecPrices;
        this.gasPrices = gasPrices;
        this.steamPrices = steamPrices;
        this.gatePower = gatePower;
    }

    public void doSelfOpt() {
        for (User user : microgrid.getUsers().values()) {
            List<AbsorptionChiller> absorptionChillers = user.getAbsorptionChillers();
            List<AirCon> airCons = user.getAirCons();
            List<Converter> converters = user.getConverters();
            List<GasBoiler> gasBoilers = user.getGasBoilers();
            List<GasTurbine> gasTurbines = user.getGasTurbines();
            List<IceStorageAc> iceStorageAcs = user.getIceStorageAcs();
            List<Storage> storages = user.getStorages();
            List<Photovoltaic> photovoltaics = user.getPhotovoltaics();
            List<SteamLoad> steamLoads = user.getSteamLoads();
            try {
                // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率(Q)，燃气轮机启停状态，表示燃气轮机状态变化的变量，
                // 燃气轮机产电功率，储能充电功率(外部)，储能放电功率(外部)，变流器AC-DC交流侧功率，变流器DC-AC交流侧功率，
                // 向电网购电功率，中央空调耗电功率，燃气锅炉启停状态，表示燃气锅炉状态变化的变量，燃气锅炉产热功率，
                // 吸收式制冷机耗热功率，向园区购热功率
                int periodVarNum = 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                        2 * converters.size() + 1 + airCons.size() + 3 * gasBoilers.size() + absorptionChillers.size() + 1;
                int varNum = periodNum * periodVarNum;
                // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
                double columnLower[] = new double[varNum];
                // 状态变量上限
                double columnUpper[] = new double[varNum];
                // 指明变量类型
                IloNumVarType[] xt = new IloNumVarType[varNum];
                // 约束方程系数
                double[][] coeff = new double[(1 + 1) * periodNum][varNum];

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
                        columnLower[j * periodVarNum + handledVarNum + 2 * gasTurbines.size() + i] = gasTurbines.get(i).getMinP();
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
                    xt[j * periodVarNum + handledVarNum] =IloNumVarType.Float;

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
                        columnLower[j * periodVarNum + handledVarNum + 2 * gasBoilers.size() + i] = gasBoilers.get(i).getMinH();
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
                    xt[j * periodVarNum + handledVarNum] =IloNumVarType.Float;
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
                    objValue[j * periodVarNum + handledVarNum] = steamPrices[j] * t; // 购热成本
                }
                cplex.addMinimize(cplex.scalProd(x, objValue));

                //记录数组中存储元素的个数
                int coeffNum = 0;
                for (int j = 0; j < periodNum; j++) {
                    // 交流母线电功率平衡约束
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;   // 购电功率
                    for (int i = 0; i < gasTurbines.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 2 * gasTurbines.size() + i] = 1; // 燃气轮机产电功率
                    }
                    for (int i = 0; i < converters.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = - 1; // 变流器AC-DC交流侧功率
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = 1;   // 变流器DC-AC交流侧功率
                    }
                    for (int i = 0; i < iceStorageAcs.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + i] = - iceStorageAcs.get(i).consumCoef; // 制冷机耗电功率
                        coeff[coeffNum][j * periodVarNum + iceStorageAcs.size() + i] = - iceStorageAcs.get(i).consumCoef;  // 蓄冰槽耗电功率
                    }
                    for (int i = 0; i < airCons.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() +
                                2 * converters.size() + 1 + i] = - airCons.get(i).consumCoef;   // 中央空调耗电功率
                    }
                    cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), acLoad[j]);
                    coeffNum += 1;

                    // 直流母线电功率平衡约束
                    for (int i = 0; i < converters.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + i] = 1 / converters.get(i).getEffad(); // 变流器AC-DC直流侧功率
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + converters.size() + i] = - 1 / converters.get(i).getEffda();   // 变流器DC-AC直流侧功率
                    }
                    for (int i = 0; i < converters.size(); i++) {
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1; // 储能充电功率
                        coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1;   // 储能放电功率
                    }
                    double periodDcLoad = dcLoad[j];
                    for (int i = 0; i < photovoltaics.size(); i++) {
                        periodDcLoad -= photovoltaics.get(i).getPower()[j];
                    }
                    cplex.addEq(cplex.scalProd(x, coeff[coeffNum]), periodVarNum);
                    coeffNum += 1;

                    // 电池储能爬坡率约束
                    if (j > 0) {
                        for (int i = 0; i < storages.size(); i++) {
                            coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                            cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYin() * storages.get(i).getMaxPIn());
                            coeffNum += 1;
                            coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = 1; // 储能充电功率
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + i] = - 1;   // 上一时刻储能充电功率
                            cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), -storages.get(i).getYin() * storages.get(i).getMaxPIn());
                            coeffNum += 1;
                            coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                            cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), storages.get(i).getYout() * storages.get(i).getMaxPOut());
                            coeffNum += 1;
                            coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = 1; // 储能放电功率
                            coeff[coeffNum][(j - 1) * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + storages.size() + i] = - 1;   // 上一时刻储能放电功率
                            cplex.addGe(cplex.scalProd(x, coeff[coeffNum]), -storages.get(i).getYout() * storages.get(i).getMaxPOut());
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
                    coeff[coeffNum][j * periodVarNum + 3 * iceStorageAcs.size() + 3 * gasTurbines.size() + 2 * storages.size() + 2 * converters.size()] = 1;
                    cplex.addLe(cplex.scalProd(x, coeff[coeffNum]), gatePower.get(user.getUserId())[j]);
                    coeffNum += 1;
                }

                // 热功率约束（空调不考虑制热）

                cplex.end();
            } catch (IloException e) {
                System.err.println("Concert exception caught: " + e);
            }
        }
    }
}
