package zju.ieeeformat;

import java.util.List;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-11-13
 */
public class SubareaModel {
    IEEEDataIsland originalIsland;
    List<IEEEDataIsland> islands;
    List<BranchData> ties;
    Map<BusData, List<BranchData>> boundaryBus2tieline;
    Map<BranchData, IEEEDataIsland[]> tieline2island;
    Map<BranchData, BusData[]> tieline2boundaryBus;
    Map<IEEEDataIsland, List<BusData>> island2boundaryBus;
    Map<IEEEDataIsland, List<BusData>> island2externalBus;

    Map<BranchData, BranchData> new2original;
    Map<BranchData, BranchData> original2new;

    public static String TYPE_BOUNDARY = "boundary";
    public static String TYPE_EXTERNAL = "external";

    public IEEEDataIsland getOriginalIsland() {
        return originalIsland;
    }

    public void setOriginalIsland(IEEEDataIsland originalIsland) {
        this.originalIsland = originalIsland;
    }

    public List<IEEEDataIsland> getIslands() {
        return islands;
    }

    public void setIslands(List<IEEEDataIsland> islands) {
        this.islands = islands;
    }

    public List<BranchData> getTies() {
        return ties;
    }

    public void setTies(List<BranchData> ties) {
        this.ties = ties;
    }

    public Map<BusData, List<BranchData>> getBoundaryBus2tieline() {
        return boundaryBus2tieline;
    }

    public void setBoundaryBus2tieline(Map<BusData, List<BranchData>> boundaryBus2tieline) {
        this.boundaryBus2tieline = boundaryBus2tieline;
    }

    public Map<BranchData, IEEEDataIsland[]> getTieline2island() {
        return tieline2island;
    }

    public void setTieline2island(Map<BranchData, IEEEDataIsland[]> tieline2island) {
        this.tieline2island = tieline2island;
    }

    public Map<BranchData, BusData[]> getTieline2boundaryBus() {
        return tieline2boundaryBus;
    }

    public void setTieline2boundaryBus(Map<BranchData, BusData[]> tieline2boundaryBus) {
        this.tieline2boundaryBus = tieline2boundaryBus;
    }

    public Map<IEEEDataIsland, List<BusData>> getIsland2boundaryBus() {
        return island2boundaryBus;
    }

    public void setIsland2boundaryBus(Map<IEEEDataIsland, List<BusData>> island2boundaryBus) {
        this.island2boundaryBus = island2boundaryBus;
    }

    public Map<BranchData, BranchData> getNew2original() {
        return new2original;
    }

    public void setNew2original(Map<BranchData, BranchData> new2original) {
        this.new2original = new2original;
    }

    public Map<BranchData, BranchData> getOriginal2new() {
        return original2new;
    }

    public void setOriginal2new(Map<BranchData, BranchData> original2new) {
        this.original2new = original2new;
    }

    public Map<IEEEDataIsland, List<BusData>> getIsland2externalBus() {
        return island2externalBus;
    }

    public void setIsland2externalBus(Map<IEEEDataIsland, List<BusData>> island2externalBus) {
        this.island2externalBus = island2externalBus;
    }
}
