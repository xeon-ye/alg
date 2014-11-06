package zju.pf;

import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink;

import java.io.Serializable;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-17
 */
public class PfResultInfo implements Serializable {

    IEEEDataIsland island;

    ASparseMatrixLink[] admittance;

    Map<Integer, Double> busV;

    Map<Integer, Double> busTheta;

    Map<Integer, Double> busPGen;

    Map<Integer, Double> busQGen;

    Map<Integer, Double> busPLoad;

    Map<Integer, Double> busQLoad;

    Map<Integer, Double> busP;

    Map<Integer, Double> busQ;

    Map<Integer, Double> branchPLoss;

    Map<Integer, Double> branchQLoss;

    double genPCapacity;

    double genQCapacity;

    double genPTotal;

    double genQTotal;

    double loadPTotal;

    double loadQTotal;

    double linePLossTotal;

    double lineQLossTotal;

    double transformerPLossTotal;

    double transformerQLossTotal;

    double pLossTotal;

    double qLossTotal;

    public IEEEDataIsland getIsland() {
        return island;
    }

    public void setIsland(IEEEDataIsland island) {
        this.island = island;
    }

    public ASparseMatrixLink[] getAdmittance() {
        return admittance;
    }

    public void setAdmittance(ASparseMatrixLink[] admittance) {
        this.admittance = admittance;
    }

    public Map<Integer, Double> getBusV() {
        return busV;
    }

    public void setBusV(Map<Integer, Double> busV) {
        this.busV = busV;
    }

    public Map<Integer, Double> getBusTheta() {
        return busTheta;
    }

    public void setBusTheta(Map<Integer, Double> busTheta) {
        this.busTheta = busTheta;
    }

    public Map<Integer, Double> getBusP() {
        return busP;
    }

    public void setBusP(Map<Integer, Double> busP) {
        this.busP = busP;
    }

    public Map<Integer, Double> getBusQ() {
        return busQ;
    }

    public void setBusQ(Map<Integer, Double> busQ) {
        this.busQ = busQ;
    }

    public Map<Integer, Double> getBusPGen() {
        return busPGen;
    }

    public void setBusPGen(Map<Integer, Double> busPGen) {
        this.busPGen = busPGen;
    }

    public Map<Integer, Double> getBusQGen() {
        return busQGen;
    }

    public void setBusQGen(Map<Integer, Double> busQGen) {
        this.busQGen = busQGen;
    }

    public Map<Integer, Double> getBusPLoad() {
        return busPLoad;
    }

    public void setBusPLoad(Map<Integer, Double> busPLoad) {
        this.busPLoad = busPLoad;
    }

    public Map<Integer, Double> getBusQLoad() {
        return busQLoad;
    }

    public void setBusQLoad(Map<Integer, Double> busQLoad) {
        this.busQLoad = busQLoad;
    }

    public Map<Integer, Double> getBranchPLoss() {
        return branchPLoss;
    }

    public void setBranchPLoss(Map<Integer, Double> branchPLoss) {
        this.branchPLoss = branchPLoss;
    }

    public Map<Integer, Double> getBranchQLoss() {
        return branchQLoss;
    }

    public void setBranchQLoss(Map<Integer, Double> branchQLoss) {
        this.branchQLoss = branchQLoss;
    }

    public double getGenPCapacity() {
        return genPCapacity;
    }

    public void setGenPCapacity(double genPCapacity) {
        this.genPCapacity = genPCapacity;
    }

    public double getGenQCapacity() {
        return genQCapacity;
    }

    public void setGenQCapacity(double genQCapacity) {
        this.genQCapacity = genQCapacity;
    }

    public double getGenPTotal() {
        return genPTotal;
    }

    public void setGenPTotal(double genPTotal) {
        this.genPTotal = genPTotal;
    }

    public double getGenQTotal() {
        return genQTotal;
    }

    public void setGenQTotal(double genQTotal) {
        this.genQTotal = genQTotal;
    }

    public double getLoadPTotal() {
        return loadPTotal;
    }

    public void setLoadPTotal(double loadPTotal) {
        this.loadPTotal = loadPTotal;
    }

    public double getLoadQTotal() {
        return loadQTotal;
    }

    public void setLoadQTotal(double loadQTotal) {
        this.loadQTotal = loadQTotal;
    }

    public double getLinePLossTotal() {
        return linePLossTotal;
    }

    public void setLinePLossTotal(double linePLossTotal) {
        this.linePLossTotal = linePLossTotal;
    }

    public double getLineQLossTotal() {
        return lineQLossTotal;
    }

    public void setLineQLossTotal(double lineQLossTotal) {
        this.lineQLossTotal = lineQLossTotal;
    }

    public double getTransformerPLossTotal() {
        return transformerPLossTotal;
    }

    public void setTransformerPLossTotal(double transformerPLossTotal) {
        this.transformerPLossTotal = transformerPLossTotal;
    }

    public double getTransformerQLossTotal() {
        return transformerQLossTotal;
    }

    public void setTransformerQLossTotal(double transformerQLossTotal) {
        this.transformerQLossTotal = transformerQLossTotal;
    }

    public double getPLossTotal() {
        return pLossTotal;
    }

    public void setPLossTotal(double pLossTotal) {
        this.pLossTotal = pLossTotal;
    }

    public double getQLossTotal() {
        return qLossTotal;
    }

    public void setQLossTotal(double qLossTotal) {
        this.qLossTotal = qLossTotal;
    }

    public String toString() {
        if(island != null)
            return island.toString();
        else
            return super.toString();
    }
}
