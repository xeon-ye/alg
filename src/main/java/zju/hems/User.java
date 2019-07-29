package zju.hems;

import java.util.List;

public class User {

    List<AbsorptionChiller> absorptionChillerList;
    List<AirCon> airCons;
    List<Converter> converters;
    List<GasBoiler> gasBoilers;
    List<GasTurbine> gasTurbines;
    List<IceStorageAc> iceStorageAcs;
    List<Photovoltaic> photovoltaics;
    List<SteamLoad> steamLoads;
    List<Storage> storages;

    public User(List<AbsorptionChiller> absorptionChillerList, List<AirCon> airCons, List<Converter> converters,
                 List<GasBoiler> gasBoilers, List<GasTurbine> gasTurbines, List<IceStorageAc> iceStorageAcs,
                 List<Photovoltaic> photovoltaics, List<SteamLoad> steamLoads, List<Storage> storages) {
        this.absorptionChillerList = absorptionChillerList;
        this.airCons = airCons;
        this.converters = converters;
        this.gasBoilers = gasBoilers;
        this.gasTurbines = gasTurbines;
        this.iceStorageAcs = iceStorageAcs;
        this.photovoltaics = photovoltaics;
        this.steamLoads = steamLoads;
        this.storages = storages;
    }

    public List<AbsorptionChiller> getAbsorptionChillerList() {
        return absorptionChillerList;
    }

    public void setAbsorptionChillerList(List<AbsorptionChiller> absorptionChillerList) {
        this.absorptionChillerList = absorptionChillerList;
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
