package zju.hems;

import java.util.List;

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
    List<Photovoltaic> photovoltaics;
    List<SteamLoad> steamLoads;
    List<Storage> storages;

    public User(String userId, List<AbsorptionChiller> absorptionChillerList, List<AirCon> airCons, List<Converter> converters,
                 List<GasBoiler> gasBoilers, List<GasTurbine> gasTurbines, List<IceStorageAc> iceStorageAcs,
                 List<Photovoltaic> photovoltaics, List<SteamLoad> steamLoads, List<Storage> storages) {
        this.userId = userId;
        this.absorptionChillers = absorptionChillerList;
        this.airCons = airCons;
        this.converters = converters;
        this.gasBoilers = gasBoilers;
        this.gasTurbines = gasTurbines;
        this.iceStorageAcs = iceStorageAcs;
        this.photovoltaics = photovoltaics;
        this.steamLoads = steamLoads;
        this.storages = storages;
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

    public List<Photovoltaic> getPhotovoltaics() {
        return photovoltaics;
    }

    public void setPhotovoltaics(List<Photovoltaic> photovoltaics) {
        this.photovoltaics = photovoltaics;
    }

    public List<SteamLoad> getSteamLoads() {
        return steamLoads;
    }

    public void setSteamLoads(List<SteamLoad> steamLoads) {
        this.steamLoads = steamLoads;
    }

    public List<Storage> getStorages() {
        return storages;
    }

    public void setStorages(List<Storage> storages) {
        this.storages = storages;
    }
}
