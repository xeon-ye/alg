package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-29
 */
public class GenOutput implements Serializable {
    private String zone = "";
    private String genBusName = "";
    private double baseKv;
    private char genId = ' ';
    private int angle;
    private int velocityDeviation;
    private int fieldVolt;
    private int fluxLinageEpq;
    private int maniFieldSat;
    private int mechPower;
    private int electricalPower;
    private int exciterSat;
    private int regulatorOutput;
    private int acceleratingPower;
    private int genMvar;
    private int exciterSupSig;
    private int dampingTorque;
    private int fieldCurrent;
    private String refGenBusName = "";
    private double refGenBaseKv;
    private String refGenId = "";
    private int group;
    private int otherVar;

    public static GenOutput createOutput(String content) {
        GenOutput genOutput = new GenOutput();
        genOutput.parseString(content);
        return genOutput;
    }

    public void parseString(String content) {
        try {
            parseString(content, "GBK");
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }

    public void parseString(String content, String charset) throws UnsupportedEncodingException {
        byte[] src = charset != null ? content.getBytes(charset) : content.getBytes();

        boolean isZone = true;
        for (int i = 5; i < 17; i++) {
            if (src[i] != ' ')
                isZone = false;
        }
        if (isZone) {
            if (charset != null)
                zone = new String(BpaFileRwUtil.getTarget(src, 3, 5), charset).trim();
            else
                zone = new String(BpaFileRwUtil.getTarget(src, 3, 5)).trim();
        } else {
            if (charset != null)
                genBusName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
            else
                genBusName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
        }
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim());
        genId = (char) src[16];
        angle = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 19, 20)).trim());
        velocityDeviation = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 22, 23)).trim());
        fieldVolt = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 25, 26)).trim());
        fluxLinageEpq = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 28, 29)).trim());
        maniFieldSat = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 31, 32)).trim());
        mechPower = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 34, 35)).trim());
        electricalPower = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 37, 38)).trim());
        exciterSat = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 40, 41)).trim());
        regulatorOutput = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 43, 44)).trim());
        acceleratingPower = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 46, 47)).trim());
        genMvar = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 49, 50)).trim());
        exciterSupSig = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 52, 53)).trim());
        dampingTorque = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 55, 56)).trim());
        fieldCurrent = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 58, 59)).trim());
        if (charset != null)
            refGenBusName = new String(BpaFileRwUtil.getTarget(src, 62, 70), charset).trim();
        else
            refGenBusName = new String(BpaFileRwUtil.getTarget(src, 62, 70)).trim();
        refGenBaseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 70, 74)).trim());
        if (charset != null)
            refGenId = new String(BpaFileRwUtil.getTarget(src, 74, 75), charset).trim();
        else
            refGenId = new String(BpaFileRwUtil.getTarget(src, 74, 75)).trim();

        group = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 75, 78)).trim());
        otherVar = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 79, 80)).trim());
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("G").append("  ");
        if (!zone.equals("")) {
            str.append(DataOutputFormat.format.getFormatStr(zone, "2"));
            str.append(DataOutputFormat.format.getFormatStr(" ", "6"));
        } else
            str.append(DataOutputFormat.format.getFormatStr(genBusName, "8"));
        str.append(BpaFileRwUtil.getFormatStr(baseKv, "4.3")).append(" ");//the bpa manual is 4.0
        str.append(genId).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(angle, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(velocityDeviation, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(fieldVolt, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(fluxLinageEpq, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(maniFieldSat, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(mechPower, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(electricalPower, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(exciterSat, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(regulatorOutput, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(acceleratingPower, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(genMvar, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(exciterSupSig, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(dampingTorque, "1")).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(fieldCurrent, "1")).append("   ");
        str.append(DataOutputFormat.format.getFormatStr(refGenBusName, "8"));
        str.append(BpaFileRwUtil.getFormatStr(refGenBaseKv, "4.3"));//the bpa manual is 4.0
        str.append(DataOutputFormat.format.getFormatStr(refGenId, "1"));
        str.append(BpaFileRwUtil.getFormatStr(group, "3")).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(otherVar, "1"));
        return str.toString();
    }

    public String getZone() {
        return zone;
    }

    public void setZone(String zone) {
        this.zone = zone;
    }

    public String getGenBusName() {
        return genBusName;
    }

    public void setGenBusName(String genBusName) {
        this.genBusName = genBusName;
    }

    public double getBaseKv() {
        return baseKv;
    }

    public void setBaseKv(double baseKv) {
        this.baseKv = baseKv;
    }

    public char getGenId() {
        return genId;
    }

    public void setGenId(char genId) {
        this.genId = genId;
    }

    public int getAngle() {
        return angle;
    }

    public void setAngle(int angle) {
        this.angle = angle;
    }

    public int getVelocityDeviation() {
        return velocityDeviation;
    }

    public void setVelocityDeviation(int velocityDeviation) {
        this.velocityDeviation = velocityDeviation;
    }

    public int getFieldVolt() {
        return fieldVolt;
    }

    public void setFieldVolt(int fieldVolt) {
        this.fieldVolt = fieldVolt;
    }

    public int getFluxLinageEpq() {
        return fluxLinageEpq;
    }

    public void setFluxLinageEpq(int fluxLinageEpq) {
        this.fluxLinageEpq = fluxLinageEpq;
    }

    public int getManiFieldSat() {
        return maniFieldSat;
    }

    public void setManiFieldSat(int maniFieldSat) {
        this.maniFieldSat = maniFieldSat;
    }

    public int getMechPower() {
        return mechPower;
    }

    public void setMechPower(int mechPower) {
        this.mechPower = mechPower;
    }

    public int getElectricalPower() {
        return electricalPower;
    }

    public void setElectricalPower(int electricalPower) {
        this.electricalPower = electricalPower;
    }

    public int getExciterSat() {
        return exciterSat;
    }

    public void setExciterSat(int exciterSat) {
        this.exciterSat = exciterSat;
    }

    public int getRegulatorOutput() {
        return regulatorOutput;
    }

    public void setRegulatorOutput(int regulatorOutput) {
        this.regulatorOutput = regulatorOutput;
    }

    public int getAcceleratingPower() {
        return acceleratingPower;
    }

    public void setAcceleratingPower(int acceleratingPower) {
        this.acceleratingPower = acceleratingPower;
    }

    public int getGenMvar() {
        return genMvar;
    }

    public void setGenMvar(int genMvar) {
        this.genMvar = genMvar;
    }

    public int getExciterSupSig() {
        return exciterSupSig;
    }

    public void setExciterSupSig(int exciterSupSig) {
        this.exciterSupSig = exciterSupSig;
    }

    public int getDampingTorque() {
        return dampingTorque;
    }

    public void setDampingTorque(int dampingTorque) {
        this.dampingTorque = dampingTorque;
    }

    public int getFieldCurrent() {
        return fieldCurrent;
    }

    public void setFieldCurrent(int fieldCurrent) {
        this.fieldCurrent = fieldCurrent;
    }

    public String getRefGenBusName() {
        return refGenBusName;
    }

    public void setRefGenBusName(String refGenBusName) {
        this.refGenBusName = refGenBusName;
    }

    public double getRefGenBaseKv() {
        return refGenBaseKv;
    }

    public void setRefGenBaseKv(double refGenBaseKv) {
        this.refGenBaseKv = refGenBaseKv;
    }

    public String getRefGenId() {
        return refGenId;
    }

    public void setRefGenId(String refGenId) {
        this.refGenId = refGenId;
    }

    public int getGroup() {
        return group;
    }

    public void setGroup(int group) {
        this.group = group;
    }

    public int getOtherVar() {
        return otherVar;
    }

    public void setOtherVar(int otherVar) {
        this.otherVar = otherVar;
    }
}
