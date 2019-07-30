package zju.hems;

import ilog.concert.IloException;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;

import java.util.List;

public class SelfOptModel {

    Microgrid microgrid;
    int periodNum = 96; // 一天的时段数
    double t;   // 单位时段长度
    double[] elecPrices;    // 电价
    double[] gasPrices;    // 天然气价格
    double[] steamPrices;    // 园区CHP蒸汽价格

    public SelfOptModel(Microgrid microgrid) {
        this.microgrid = microgrid;
    }

    public void doSelfOpt() {
        for (User user : microgrid.getUsers()) {
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
                // 变量：制冷机耗电功率，蓄冰槽耗电功率，蓄冰槽制冷功率，燃气轮机产电功率，储能充电功率，
                // 变流器交流侧消耗功率，向电网购电功率，中央空调耗电功率，燃气锅炉产热功率，吸收式制冷机耗热功率
                int varNum = 3 * iceStorageAcs.size() + gasTurbines.size() + storages.size() + converters.size()
                        + periodNum + airCons.size() + gasBoilers.size() + absorptionChillers.size();
                // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
                double columnLower[] = new double[varNum];
                // 状态变量上限
                double columnUpper[] = new double[varNum];
                // 指明变量类型
                IloNumVarType[] xt = new IloNumVarType[varNum];
                // 约束方程系数
                double[][] coeff = new double[1][varNum];
                //记录数组中存储元素的个数
                int coeffNum = 0;

                for (int i = 0; i < iceStorageAcs.size(); i++) {
                    columnLower[i] = 0;
                    columnUpper[i] = 1;
                    xt[i] = IloNumVarType.Bool;
                }

                IloCplex cplex = new IloCplex(); // creat a model
                // 变量
                IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
                // 目标函数
//            IloNumExpr obj = cplex.numExpr();
                double[] objValue = new double[varNum];
                for (int i = 0; i < absorptionChillers.size(); i++) {
                    objValue[i] = 0;
                }
                for (int i = 0; i < airCons.size(); i++) {
                    objValue[absorptionChillers.size() + i] += 0;
                }
                cplex.addMinimize(cplex.scalProd(x, objValue));

                cplex.end();
            } catch (IloException e) {
                System.err.println("Concert exception caught: " + e);
            }
        }
    }
}
