package zju.bpamodel.swir;

public class GenOneStepData {
    private double time;
    private double relativeAngle;
    private double freqDeviation;
    private double fieldVoltage;
    private double mechPower;
    private double elecPower;
    private double regulatorOutput;
    private double ReactivePower;
    private double fieldCurrent;

    public double getTime() {
        return time;
    }

    public void setTime(double time) {
        this.time = time;
    }

    public double getRelativeAngle() {
        return relativeAngle;
    }

    public void setRelativeAngle(double relativeAngle) {
        this.relativeAngle = relativeAngle;
    }

    public double getFreqDeviation() {
        return freqDeviation;
    }

    public void setFreqDeviation(double freqDeviation) {
        this.freqDeviation = freqDeviation;
    }

    public double getFieldVoltage() {
        return fieldVoltage;
    }

    public void setFieldVoltage(double fieldVoltage) {
        this.fieldVoltage = fieldVoltage;
    }

    public double getMechPower() {
        return mechPower;
    }

    public void setMechPower(double mechPower) {
        this.mechPower = mechPower;
    }

    public double getElecPower() {
        return elecPower;
    }

    public void setElecPower(double elecPower) {
        this.elecPower = elecPower;
    }

    public double getRegulatorOutput() {
        return regulatorOutput;
    }

    public void setRegulatorOutput(double regulatorOutput) {
        this.regulatorOutput = regulatorOutput;
    }

    public double getReactivePower() {
        return ReactivePower;
    }

    public void setReactivePower(double reactivePower) {
        ReactivePower = reactivePower;
    }

    public double getFieldCurrent() {
        return fieldCurrent;
    }

    public void setFieldCurrent(double fieldCurrent) {
        this.fieldCurrent = fieldCurrent;
    }
}
