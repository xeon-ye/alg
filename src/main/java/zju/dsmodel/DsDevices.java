package zju.dsmodel;

import zju.devmodel.MapObject;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-9
 */
public class DsDevices implements Serializable {
    List<MapObject> spotLoads;
    List<MapObject> distributedLoads;
    List<MapObject> feeders;
    List<MapObject> transformers;
    List<MapObject> regulators;
    List<MapObject> shuntCapacitors;
    List<MapObject> switches;
    List<MapObject> dispersedGens;

    public void initialLists() {
        spotLoads = new ArrayList<MapObject>();
        distributedLoads = new ArrayList<MapObject>();
        feeders = new ArrayList<MapObject>();
        transformers = new ArrayList<MapObject>();
        regulators = new ArrayList<MapObject>();
        shuntCapacitors = new ArrayList<MapObject>();
        switches = new ArrayList<MapObject>();
        dispersedGens = new ArrayList<MapObject>();
    }

    public List<MapObject> getSpotLoads() {
        return spotLoads;
    }

    public void setSpotLoads(List<MapObject> spotLoads) {
        this.spotLoads = spotLoads;
    }

    public List<MapObject> getFeeders() {
        return feeders;
    }

    public void setFeeders(List<MapObject> feeders) {
        this.feeders = feeders;
    }

    public List<MapObject> getTransformers() {
        return transformers;
    }

    public void setTransformers(List<MapObject> transformers) {
        this.transformers = transformers;
    }

    public List<MapObject> getRegulators() {
        return regulators;
    }

    public void setRegulators(List<MapObject> regulators) {
        this.regulators = regulators;
    }

    public List<MapObject> getShuntCapacitors() {
        return shuntCapacitors;
    }

    public void setShuntCapacitors(List<MapObject> shuntCapacitors) {
        this.shuntCapacitors = shuntCapacitors;
    }

    public List<MapObject> getSwitches() {
        return switches;
    }

    public void setSwitches(List<MapObject> switches) {
        this.switches = switches;
    }

    public List<MapObject> getDistributedLoads() {
        return distributedLoads;
    }

    public void setDistributedLoads(List<MapObject> distributedLoads) {
        this.distributedLoads = distributedLoads;
    }

    public List<MapObject> getDispersedGens() {
        return dispersedGens;
    }

    public void setDispersedGens(List<MapObject> dispersedGens) {
        this.dispersedGens = dispersedGens;
    }

    public Object clone() {
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream out = new ObjectOutputStream(baos);
            out.writeObject(this);
            out.close();
            ByteArrayInputStream bin = new ByteArrayInputStream(baos.toByteArray());
            ObjectInputStream in = new ObjectInputStream(bin);
            Object clone = in.readObject();
            in.close();
            return (clone);
        } catch (Exception e) {
            throw new InternalError(e.toString());
        }
    }
}
