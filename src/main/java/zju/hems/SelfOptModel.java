package zju.hems;

import ilog.concert.IloException;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;

import java.util.List;

import static java.lang.Math.*;

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
            List<Photovoltaic> photovoltaics = user.getPhotovoltaics();
            List<SteamLoad> steamLoads = user.getSteamLoads();
            List<Storage> storages = user.getStorages();
            try {
                // 变量：吸收式制冷机耗热功率，中央空调
                int varNum = absorptionChillers.size() + airCons.size() + periodNum;
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

                IloCplex cplex = new IloCplex(); // creat a model
                // 变量
                IloNumVar[] x = cplex.numVarArray(columnLower.length, columnLower, columnUpper, xt);
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
