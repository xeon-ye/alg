package zju.opf.result;

import zju.ieeeformat.IEEEDataIsland;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-1-14
 */
public class OpfResult implements Serializable {
    IEEEDataIsland ieeeResult;
    boolean isConverged;
    double optimizedObjectiveValue = -999999;
    double originalObjectiveValue = -999999;
    String warnInfo = "";
    String errorInfo = "";
    List<PilotNodeInfo> pilotNodes = new ArrayList<PilotNodeInfo>();
    List<OutLimitNodeInfo> outLimitNodes = new ArrayList<OutLimitNodeInfo>();
    List<OutLimitBranchInfo> outLimitBranches = new ArrayList<OutLimitBranchInfo>();
    List<GeneratorPCommand> generatorPCommands = new ArrayList<GeneratorPCommand>();
    List<GeneratorQCommand> generatorQCommands = new ArrayList<GeneratorQCommand>();
    List<CompensatorCommand> compensatorCommands = new ArrayList<CompensatorCommand>();
    List<TransformerCommand> transformerCommands = new ArrayList<TransformerCommand>();
    List<BusVoltageCommand> busVoltageCommands = new ArrayList<BusVoltageCommand>();

    public IEEEDataIsland getIeeeResult() {
        return ieeeResult;
    }

    public void setIeeeResult(IEEEDataIsland ieeeResult) {
        this.ieeeResult = ieeeResult;
    }

    public boolean isConverged() {
        return isConverged;
    }

    public void setConverged(boolean converged) {
        isConverged = converged;
    }

    public double getOptimizedObjectiveValue() {
        return optimizedObjectiveValue;
    }

    public void setOptimizedObjectiveValue(double optimizedObjectiveValue) {
        this.optimizedObjectiveValue = optimizedObjectiveValue;
    }

    public double getOriginalObjectiveValue() {
        return originalObjectiveValue;
    }

    public void setOriginalObjectiveValue(double originalObjectiveValue) {
        this.originalObjectiveValue = originalObjectiveValue;
    }

    public String getWarnInfo() {
        return warnInfo;
    }

    public void setWarnInfo(String warnInfo) {
        this.warnInfo = warnInfo;
    }

    public String getErrorInfo() {
        return errorInfo;
    }

    public void setErrorInfo(String errorInfo) {
        this.errorInfo = errorInfo;
    }

    public List<PilotNodeInfo> getPilotNodes() {
        return pilotNodes;
    }

    public void setPilotNodes(List<PilotNodeInfo> pilotNodes) {
        this.pilotNodes = pilotNodes;
    }

    public List<OutLimitNodeInfo> getOutLimitNodes() {
        return outLimitNodes;
    }

    public void setOutLimitNodes(List<OutLimitNodeInfo> outLimitNodes) {
        this.outLimitNodes = outLimitNodes;
    }

    public List<OutLimitBranchInfo> getOutLimitBranches() {
        return outLimitBranches;
    }

    public void setOutLimitBranches(List<OutLimitBranchInfo> outLimitBranches) {
        this.outLimitBranches = outLimitBranches;
    }

    public List<GeneratorPCommand> getGeneratorPCommands() {
        return generatorPCommands;
    }

    public void setGeneratorPCommands(List<GeneratorPCommand> generatorPCommands) {
        this.generatorPCommands = generatorPCommands;
    }

    public List<GeneratorQCommand> getGeneratorQCommands() {
        return generatorQCommands;
    }

    public void setGeneratorQCommands(List<GeneratorQCommand> generatorQCommands) {
        this.generatorQCommands = generatorQCommands;
    }

    public List<CompensatorCommand> getCompensatorCommands() {
        return compensatorCommands;
    }

    public void setCompensatorCommands(List<CompensatorCommand> compensatorCommands) {
        this.compensatorCommands = compensatorCommands;
    }

    public List<TransformerCommand> getTransformerCommands() {
        return transformerCommands;
    }

    public void setTransformerCommands(List<TransformerCommand> transformerCommands) {
        this.transformerCommands = transformerCommands;
    }

    public List<BusVoltageCommand> getBusVoltageCommands() {
        return busVoltageCommands;
    }

    public void setBusVoltageCommands(List<BusVoltageCommand> busVoltageCommands) {
        this.busVoltageCommands = busVoltageCommands;
    }
}
