package zju.hems;

import java.util.List;

/**
 * 用户自趋优计算结果.
 * @author Xu Chengsi
 * @date 2019/8/7
 */
public class UserResult {

    String userId;

    String status;  // 求解状态
    double minCost; // 目标函数值

    List<double[]> frigesP; // 制冷机耗电功率
    List<double[]> iceTanksP;   // 蓄冰槽耗电功率
    List<double[]> iceTanksQ;   // 蓄冰槽制冷功率(Q)
    List<double[]> gasTurbinesState; // 燃气轮机启停状态
    List<double[]> gasTurbinesP;    // 燃气轮机产电功率
    List<double[]> storagesP;   // 储能充电功率(外部)
    List<double[]> convertersP;  // 变流器AC-DC交流侧功率
    double[] Pin;   // 电网输入功率
    double[] purP; // 向电网购电功率
    List<double[]> airConsP;    // 中央空调耗电功率
    List<double[]> gasBoilersState; // 燃气锅炉启停状态
    List<double[]> gasBoilersH; // 燃气锅炉产热功率
    List<double[]> absorptionChillersH; // 吸收式制冷机耗热功率
    double[] Hin;   // 园区输入热功率
    double[] purH; // 向园区购热功率
    List<double[]> chargingPilesState; // 充电桩启停状态
    List<double[]> chargingPilesP;    // 充电桩耗电功率

    public UserResult(String userId, String status) {
        this.userId = userId;
        this.status = status;
    }

    public UserResult(String userId, String status, double minCost, List<double[]> frigesP, List<double[]> iceTanksP,
                      List<double[]> iceTanksQ, List<double[]> gasTurbinesState, List<double[]> gasTurbinesP,
                      List<double[]> storagesP, List<double[]> convertersP, double[] Pin, double[] purP,
                      List<double[]> airConsP, List<double[]> gasBoilersState, List<double[]> gasBoilersH,
                      List<double[]> absorptionChillersH, double[] Hin, double[] purH) {
        this.userId = userId;
        this.status = status;
        this.minCost = minCost;
        this.frigesP = frigesP;
        this.iceTanksP = iceTanksP;
        this.iceTanksQ = iceTanksQ;
        this.gasTurbinesState = gasTurbinesState;
        this.gasTurbinesP = gasTurbinesP;
        this.storagesP = storagesP;
        this.convertersP = convertersP;
        this.Pin = Pin;
        this.purP = purP;
        this.airConsP = airConsP;
        this.gasBoilersState = gasBoilersState;
        this.gasBoilersH = gasBoilersH;
        this.absorptionChillersH = absorptionChillersH;
        this.Hin = Hin;
        this.purH = purH;
    }

    public UserResult(String userId, String status, double minCost, List<double[]> frigesP, List<double[]> iceTanksP,
                      List<double[]> iceTanksQ, List<double[]> gasTurbinesState, List<double[]> gasTurbinesP,
                      List<double[]> storagesP, List<double[]> convertersP, double[] Pin, double[] purP,
                      List<double[]> airConsP, List<double[]> gasBoilersState, List<double[]> gasBoilersH,
                      List<double[]> absorptionChillersH, double[] Hin, double[] purH,
                      List<double[]> chargingPilesState, List<double[]> chargingPilesP) {
        this(userId, status, minCost, frigesP, iceTanksP, iceTanksQ, gasTurbinesState, gasTurbinesP, storagesP,
                convertersP, Pin, purP, airConsP, gasBoilersState, gasBoilersH, absorptionChillersH, Hin, purH);
        this.chargingPilesState = chargingPilesState;
        this.chargingPilesP = chargingPilesP;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public double getMinCost() {
        return minCost;
    }

    public void setMinCost(double minCost) {
        this.minCost = minCost;
    }

    public List<double[]> getFrigesP() {
        return frigesP;
    }

    public void setFrigesP(List<double[]> frigesP) {
        this.frigesP = frigesP;
    }

    public List<double[]> getIceTanksP() {
        return iceTanksP;
    }

    public void setIceTanksP(List<double[]> iceTanksP) {
        this.iceTanksP = iceTanksP;
    }

    public List<double[]> getIceTanksQ() {
        return iceTanksQ;
    }

    public void setIceTanksQ(List<double[]> iceTanksQ) {
        this.iceTanksQ = iceTanksQ;
    }

    public List<double[]> getGasTurbinesState() {
        return gasTurbinesState;
    }

    public void setGasTurbinesState(List<double[]> gasTurbinesState) {
        this.gasTurbinesState = gasTurbinesState;
    }

    public List<double[]> getGasTurbinesP() {
        return gasTurbinesP;
    }

    public void setGasTurbinesP(List<double[]> gasTurbinesP) {
        this.gasTurbinesP = gasTurbinesP;
    }

    public List<double[]> getStoragesP() {
        return storagesP;
    }

    public void setStoragesP(List<double[]> storagesP) {
        this.storagesP = storagesP;
    }

    public List<double[]> getConvertersP() {
        return convertersP;
    }

    public void setConvertersP(List<double[]> convertersP) {
        this.convertersP = convertersP;
    }

    public double[] getPin() {
        return Pin;
    }

    public void setPin(double[] pin) {
        Pin = pin;
    }

    public double[] getPurP() {
        return purP;
    }

    public void setPurP(double[] purP) {
        this.purP = purP;
    }

    public List<double[]> getAirConsP() {
        return airConsP;
    }

    public void setAirConsP(List<double[]> airConsP) {
        this.airConsP = airConsP;
    }

    public List<double[]> getGasBoilersState() {
        return gasBoilersState;
    }

    public void setGasBoilersState(List<double[]> gasBoilersState) {
        this.gasBoilersState = gasBoilersState;
    }

    public List<double[]> getGasBoilersH() {
        return gasBoilersH;
    }

    public void setGasBoilersH(List<double[]> gasBoilersH) {
        this.gasBoilersH = gasBoilersH;
    }

    public List<double[]> getAbsorptionChillersH() {
        return absorptionChillersH;
    }

    public void setAbsorptionChillersH(List<double[]> absorptionChillersH) {
        this.absorptionChillersH = absorptionChillersH;
    }

    public double[] getHin() {
        return Hin;
    }

    public void setHin(double[] hin) {
        Hin = hin;
    }

    public double[] getPurH() {
        return purH;
    }

    public void setPurH(double[] purH) {
        this.purH = purH;
    }

    public List<double[]> getChargingPilesState() {
        return chargingPilesState;
    }

    public void setChargingPilesState(List<double[]> chargingPilesState) {
        this.chargingPilesState = chargingPilesState;
    }

    public List<double[]> getChargingPilesP() {
        return chargingPilesP;
    }

    public void setChargingPilesP(List<double[]> chargingPilesP) {
        this.chargingPilesP = chargingPilesP;
    }
}
