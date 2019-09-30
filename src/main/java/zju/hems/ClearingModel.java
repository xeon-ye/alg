package zju.hems;

import ilog.concert.IloException;
import ilog.concert.IloNumVar;
import ilog.concert.IloNumVarType;
import ilog.cplex.IloCplex;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClearingModel {

    List<Offer> offers; // 报价
    Map<String, Double> timeShaveCapRatios;   // 用户应削峰容量占总削峰容量的比例
    double maxPrice;    // 最大出清价格
    String status;  // 求解状态
    Map<String, Double> bidRatios;  // 中标削峰比例
    double clearingPrice;   // 出清价格

    public ClearingModel(List<Offer> offers, double maxPrice, Map<String, Double> timeShaveCapRatios) {
        this.offers = offers;
        this.maxPrice = maxPrice;
        this.timeShaveCapRatios = timeShaveCapRatios;
    }

    public void clearing() {
        bidRatios = new HashMap<>(offers.size());
        try {
            // 变量：中标削峰比例
            int varNum = offers.size();
            // 状态变量下限, column里元素的个数等于矩阵C里系数的个数
            double columnLower[] = new double[varNum];
            // 状态变量上限
            double columnUpper[] = new double[varNum];
            // 指明变量类型
            IloNumVarType[] xt = new IloNumVarType[varNum];
            // 约束方程系数
            double[] coeff = new double[varNum];

            // 设置变量上下限和类型
            for (int i = 0; i < offers.size(); i++) {
                columnLower[i] = 0;
                columnUpper[i] = offers.get(i).getMaxPeakShaveRatio();
                xt[i] = IloNumVarType.Float;
            }

            IloCplex cplex = new IloCplex(); // creat a model
            // 变量
            IloNumVar[] x = cplex.numVarArray(varNum, columnLower, columnUpper, xt);
            // 目标函数
            double[] objValue = new double[varNum];
            for (int i = 0; i < offers.size(); i++) {
                Offer offer = offers.get(i);
                objValue[i] = offer.getPrice() * timeShaveCapRatios.get(offer.getUserId()) - maxPrice * timeShaveCapRatios.get(offer.getUserId());
            }
            cplex.addMinimize(cplex.scalProd(x, objValue));

            for (int i = 0; i < offers.size(); i++) {
                coeff[i] = timeShaveCapRatios.get(offers.get(i).getUserId());
            }
            cplex.addLe(cplex.scalProd(x, coeff), 1);

            if (cplex.solve()) {
                cplex.output().println("Solution status = " + cplex.getStatus());
                cplex.output().println("Solution value = " + cplex.getObjValue());
                status = cplex.getStatus().toString();
                if (cplex.getStatus() == IloCplex.Status.Optimal) {
                    double[] result = cplex.getValues(x);
                    clearingPrice = 0;
                    for (int i = 0; i < offers.size(); i++) {
                        Offer offer = offers.get(i);
                        bidRatios.put(offer.getUserId(), result[i]);
                        if (result[i] > 1 && offer.getPrice() > clearingPrice) {
                            clearingPrice = offer.getPrice();
                        }
                    }
                }
            }

            cplex.end();
        } catch (IloException e) {
            System.err.println("Concert exception caught: " + e);
        }
    }

    public void setOffers(List<Offer> offers) {
        this.offers = offers;
    }

    public void setMaxPrice(double maxPrice) {
        this.maxPrice = maxPrice;
    }

    public String getStatus() {
        return status;
    }

    public double getClearingPrice() {
        return clearingPrice;
    }

    public Map<String, Double> getBidRatios() {
        return bidRatios;
    }
}
