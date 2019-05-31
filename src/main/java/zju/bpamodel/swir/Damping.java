package zju.bpamodel.swir;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-5-31
 */
public class Damping {
    private String busName1 = "";
    private double baseKv1;
    private String busName2 = "";
    private double baseKv2;
    private String variableName = "";
    private double oscillationAmp1;
    private double oscillationFreq1;
    private double attenuationCoef1;
    private double dampingRatio1;
    private double oscillationAmp2;
    private double oscillationFreq2;
    private double attenuationCoef2;
    private double dampingRatio2;

    public String getBusName1() {
        return busName1;
    }

    public void setBusName1(String busName1) {
        this.busName1 = busName1;
    }

    public double getBaseKv1() {
        return baseKv1;
    }

    public void setBaseKv1(double baseKv1) {
        this.baseKv1 = baseKv1;
    }

    public String getBusName2() {
        return busName2;
    }

    public void setBusName2(String busName2) {
        this.busName2 = busName2;
    }

    public void setBaseKv2(double baseKv2) {
        this.baseKv2 = baseKv2;
    }

    public double getBaseKv2() {
        return baseKv2;
    }

    public String getVariableName() {
        return variableName;
    }

    public void setVariableName(String variableName) {
        this.variableName = variableName;
    }

    public double getOscillationAmp1() {
        return oscillationAmp1;
    }

    public void setOscillationAmp1(double oscillationAmp1) {
        this.oscillationAmp1 = oscillationAmp1;
    }

    public double getOscillationFreq1() {
        return oscillationFreq1;
    }

    public void setOscillationFreq1(double oscillationFreq1) {
        this.oscillationFreq1 = oscillationFreq1;
    }

    public double getAttenuationCoef1() {
        return attenuationCoef1;
    }

    public void setAttenuationCoef1(double attenuationCoef1) {
        this.attenuationCoef1 = attenuationCoef1;
    }

    public double getDampingRatio1() {
        return dampingRatio1;
    }

    public void setDampingRatio1(double dampingRatio1) {
        this.dampingRatio1 = dampingRatio1;
    }

    public double getOscillationAmp2() {
        return oscillationAmp2;
    }

    public void setOscillationAmp2(double oscillationAmp2) {
        this.oscillationAmp2 = oscillationAmp2;
    }

    public double getOscillationFreq2() {
        return oscillationFreq2;
    }

    public void setOscillationFreq2(double oscillationFreq2) {
        this.oscillationFreq2 = oscillationFreq2;
    }

    public double getAttenuationCoef2() {
        return attenuationCoef2;
    }

    public void setAttenuationCoef2(double attenuationCoef2) {
        this.attenuationCoef2 = attenuationCoef2;
    }

    public double getDampingRatio2() {
        return dampingRatio2;
    }

    public void setDampingRatio2(double dampingRatio2) {
        this.dampingRatio2 = dampingRatio2;
    }
}
