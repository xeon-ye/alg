package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-11-12
 */
public class BusOutput implements Serializable {
    private String busName;
    private double baseKv;
    private int voltageAmpl;
    private int voltageAngle;
    private int frequency;
    private int loadP;
    private int loadQ;
    private int group;
    private int phaseAVoltageAmpl;
    private int phaseAVoltageAngle;
    private int phaseBVoltageAmpl;
    private int phaseBVoltageAngle;
    private int phaseCVoltageAmpl;
    private int phaseCVoltageAngle;
    private int phaseNegaVoltageAmpl;
    private int phaseNegaVoltageAngle;
    private int phaseZeroVoltageAmpl;
    private int phaseZeroVoltageAngle;

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("B").append("  ");
        str.append(DataOutputFormat.format.getFormatStr(busName, "8"));
        str.append(BpaFileRwUtil.getFormatStr(baseKv, "4.3")).append("  ");//the bpa manual is 4.0
        str.append(BpaFileRwUtil.getFormatStr(voltageAmpl, "1"));
        str.append(BpaFileRwUtil.getFormatStr(voltageAngle, "1")).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(frequency, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(loadP, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(loadQ, "1")).append("    ");
        str.append(BpaFileRwUtil.getFormatStr(group, "3")).append("   ");
        str.append(BpaFileRwUtil.getFormatStr(phaseAVoltageAmpl, "1"));
        str.append(BpaFileRwUtil.getFormatStr(phaseAVoltageAngle, "1"));
        str.append(BpaFileRwUtil.getFormatStr(phaseBVoltageAmpl, "1"));
        str.append(BpaFileRwUtil.getFormatStr(phaseBVoltageAngle, "1"));
        str.append(BpaFileRwUtil.getFormatStr(phaseCVoltageAmpl, "1"));
        str.append(BpaFileRwUtil.getFormatStr(phaseCVoltageAngle, "1"));
        str.append(BpaFileRwUtil.getFormatStr(phaseNegaVoltageAmpl, "1"));
        str.append(BpaFileRwUtil.getFormatStr(phaseNegaVoltageAngle, "1"));
        str.append(DataOutputFormat.format.getFormatStr(phaseZeroVoltageAmpl, "1"));
        str.append(BpaFileRwUtil.getFormatStr(phaseZeroVoltageAngle, "1"));//the bpa manual is 4.0
        return str.toString();
    }

    public String getBusName() {
        return busName;
    }

    public void setBusName(String busName) {
        this.busName = busName;
    }

    public double getBaseKv() {
        return baseKv;
    }

    public void setBaseKv(double baseKv) {
        this.baseKv = baseKv;
    }

    public int getVoltageAmpl() {
        return voltageAmpl;
    }

    public void setVoltageAmpl(int voltageAmpl) {
        this.voltageAmpl = voltageAmpl;
    }

    public int getVoltageAngle() {
        return voltageAngle;
    }

    public void setVoltageAngle(int voltageAngle) {
        this.voltageAngle = voltageAngle;
    }

    public int getFrequency() {
        return frequency;
    }

    public void setFrequency(int frequency) {
        this.frequency = frequency;
    }

    public int getLoadP() {
        return loadP;
    }

    public void setLoadP(int loadP) {
        this.loadP = loadP;
    }

    public int getLoadQ() {
        return loadQ;
    }

    public void setLoadQ(int loadQ) {
        this.loadQ = loadQ;
    }

    public int getGroup() {
        return group;
    }

    public void setGroup(int group) {
        this.group = group;
    }

    public int getPhaseAVoltageAmpl() {
        return phaseAVoltageAmpl;
    }

    public void setPhaseAVoltageAmpl(int phaseAVoltageAmpl) {
        this.phaseAVoltageAmpl = phaseAVoltageAmpl;
    }

    public int getPhaseAVoltageAngle() {
        return phaseAVoltageAngle;
    }

    public void setPhaseAVoltageAngle(int phaseAVoltageAngle) {
        this.phaseAVoltageAngle = phaseAVoltageAngle;
    }

    public int getPhaseBVoltageAmpl() {
        return phaseBVoltageAmpl;
    }

    public void setPhaseBVoltageAmpl(int phaseBVoltageAmpl) {
        this.phaseBVoltageAmpl = phaseBVoltageAmpl;
    }

    public int getPhaseBVoltageAngle() {
        return phaseBVoltageAngle;
    }

    public void setPhaseBVoltageAngle(int phaseBVoltageAngle) {
        this.phaseBVoltageAngle = phaseBVoltageAngle;
    }

    public int getPhaseCVoltageAmpl() {
        return phaseCVoltageAmpl;
    }

    public void setPhaseCVoltageAmpl(int phaseCVoltageAmpl) {
        this.phaseCVoltageAmpl = phaseCVoltageAmpl;
    }

    public int getPhaseCVoltageAngle() {
        return phaseCVoltageAngle;
    }

    public void setPhaseCVoltageAngle(int phaseCVoltageAngle) {
        this.phaseCVoltageAngle = phaseCVoltageAngle;
    }

    public int getPhaseNegaVoltageAmpl() {
        return phaseNegaVoltageAmpl;
    }

    public void setPhaseNegaVoltageAmpl(int phaseNegaVoltageAmpl) {
        this.phaseNegaVoltageAmpl = phaseNegaVoltageAmpl;
    }

    public int getPhaseNegaVoltageAngle() {
        return phaseNegaVoltageAngle;
    }

    public void setPhaseNegaVoltageAngle(int phaseNegaVoltageAngle) {
        this.phaseNegaVoltageAngle = phaseNegaVoltageAngle;
    }

    public int getPhaseZeroVoltageAmpl() {
        return phaseZeroVoltageAmpl;
    }

    public void setPhaseZeroVoltageAmpl(int phaseZeroVoltageAmpl) {
        this.phaseZeroVoltageAmpl = phaseZeroVoltageAmpl;
    }

    public int getPhaseZeroVoltageAngle() {
        return phaseZeroVoltageAngle;
    }

    public void setPhaseZeroVoltageAngle(int phaseZeroVoltageAngle) {
        this.phaseZeroVoltageAngle = phaseZeroVoltageAngle;
    }
}
