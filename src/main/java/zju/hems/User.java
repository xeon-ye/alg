package zju.hems;

import java.util.List;
import java.util.Map;

/**
 * 用户.
 * @author Xu Chengsi
 * @date 2019/7/29
 */
public class User {

    String userId;
    List<AbsorptionChiller> absorptionChillers;
    List<AirCon> airCons;
    List<Converter> converters;
    List<GasBoiler> gasBoilers;
    List<GasTurbine> gasTurbines;
    List<IceStorageAc> iceStorageAcs;
    List<Storage> storages;
    double basicCap;    // 基本电费对应的容量
    SteamLoad steamLoad;
    Photovoltaic photovoltaic;
    double[] acLoad;    // 交流负荷
    double[] dcLoad;    // 直流负荷
    double[] heatLoad;    // 热水负荷
    double[] coolingLoad;   // 冷负荷
    double[] gatePowers;   // 用户关口功率
    InterruptibleLoad interruptibleLoad;    // 可中断负荷

    WindPower windPower;

    public User(String userId, List<AbsorptionChiller> absorptionChillerList, List<AirCon> airCons, List<Converter> converters,
                 List<GasBoiler> gasBoilers, List<GasTurbine> gasTurbines, List<IceStorageAc> iceStorageAcs,
                List<Storage> storages, double basicCap, SteamLoad steamLoad, Photovoltaic photovoltaic, double[] acLoad,
                double[] dcLoad, double[] heatLoad, double[] coolingLoad, double[] gatePowers) {
        this.userId = userId;
        this.absorptionChillers = absorptionChillerList;
        this.airCons = airCons;
        this.converters = converters;
        this.gasBoilers = gasBoilers;
        this.gasTurbines = gasTurbines;
        this.iceStorageAcs = iceStorageAcs;
        this.storages = storages;
        this.basicCap = basicCap;
        this.steamLoad = steamLoad;
        this.photovoltaic = photovoltaic;
        this.acLoad = acLoad;
        this.dcLoad = dcLoad;
        this.heatLoad = heatLoad;
        this.coolingLoad = coolingLoad;
        this.gatePowers = gatePowers;
    }

    public User(String userId, List<AbsorptionChiller> absorptionChillerList, List<AirCon> airCons,
                List<Converter> converters, List<GasBoiler> gasBoilers, List<GasTurbine> gasTurbines,
                List<IceStorageAc> iceStorageAcs, List<Storage> storages, double basicCap) {
        this.userId = userId;
        this.absorptionChillers = absorptionChillerList;
        this.airCons = airCons;
        this.converters = converters;
        this.gasBoilers = gasBoilers;
        this.gasTurbines = gasTurbines;
        this.iceStorageAcs = iceStorageAcs;
        this.storages = storages;
        this.basicCap = basicCap;
    }

    public User(String userId, List<AbsorptionChiller> absorptionChillerList, List<AirCon> airCons,
                List<Converter> converters, List<GasBoiler> gasBoilers, List<GasTurbine> gasTurbines,
                List<IceStorageAc> iceStorageAcs, List<Storage> storages, double basicCap, InterruptibleLoad interruptibleLoad) {
        this.userId = userId;
        this.absorptionChillers = absorptionChillerList;
        this.airCons = airCons;
        this.converters = converters;
        this.gasBoilers = gasBoilers;
        this.gasTurbines = gasTurbines;
        this.iceStorageAcs = iceStorageAcs;
        this.storages = storages;
        this.basicCap = basicCap;
        this.interruptibleLoad = interruptibleLoad;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public List<AbsorptionChiller> getAbsorptionChillers() {
        return absorptionChillers;
    }

    public void setAbsorptionChillers(List<AbsorptionChiller> absorptionChillers) {
        this.absorptionChillers = absorptionChillers;
    }

    public List<AirCon> getAirCons() {
        return airCons;
    }

    public void setAirCons(List<AirCon> airCons) {
        this.airCons = airCons;
    }

    public List<Converter> getConverters() {
        return converters;
    }

    public void setConverters(List<Converter> converters) {
        this.converters = converters;
    }

    public List<GasBoiler> getGasBoilers() {
        return gasBoilers;
    }

    public void setGasBoilers(List<GasBoiler> gasBoilers) {
        this.gasBoilers = gasBoilers;
    }

    public List<GasTurbine> getGasTurbines() {
        return gasTurbines;
    }

    public void setGasTurbines(List<GasTurbine> gasTurbines) {
        this.gasTurbines = gasTurbines;
    }

    public List<IceStorageAc> getIceStorageAcs() {
        return iceStorageAcs;
    }

    public void setIceStorageAcs(List<IceStorageAc> iceStorageAcs) {
        this.iceStorageAcs = iceStorageAcs;
    }

    public List<Storage> getStorages() {
        return storages;
    }

    public void setStorages(List<Storage> storages) {
        this.storages = storages;
    }

    public double getBasicCap() {
        return basicCap;
    }

    public void setBasicCap(double basicCap) {
        this.basicCap = basicCap;
    }

    public SteamLoad getSteamLoad() {
        return steamLoad;
    }

    public void setSteamLoad(SteamLoad steamLoad) {
        this.steamLoad = steamLoad;
    }

    public Photovoltaic getPhotovoltaic() {
        return photovoltaic;
    }

    public void setPhotovoltaic(Photovoltaic photovoltaic) {
        this.photovoltaic = photovoltaic;
    }

    public double[] getAcLoad() {
        return acLoad;
    }

    public void setAcLoad(double[] acLoad) {
        this.acLoad = acLoad;
    }

    public double[] getDcLoad() {
        return dcLoad;
    }

    public void setDcLoad(double[] dcLoad) {
        this.dcLoad = dcLoad;
    }

    public double[] getHeatLoad() {
        return heatLoad;
    }

    public void setHeatLoad(double[] heatLoad) {
        this.heatLoad = heatLoad;
    }

    public double[] getCoolingLoad() {
        return coolingLoad;
    }

    public void setCoolingLoad(double[] coolingLoad) {
        this.coolingLoad = coolingLoad;
    }

    public double[] getGatePowers() {
        return gatePowers;
    }

    public void setGatePowers(double[] gatePowers) {
        this.gatePowers = gatePowers;
    }

    public InterruptibleLoad getInterruptibleLoad() {
        return interruptibleLoad;
    }

    public void setInterruptibleLoad(InterruptibleLoad interruptibleLoad) {
        this.interruptibleLoad = interruptibleLoad;
    }

    public WindPower getWindPower() {
        return windPower;
    }

    public void setWindPower(WindPower windPower) {
        this.windPower = windPower;
    }
}
