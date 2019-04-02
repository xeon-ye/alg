package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-3-24
 */
public class Servo implements Serializable {
    private char type;
    private char subType;
    private String genName;
    private double baseKv;
    private char generatorCode = ' ';
    private double pe;
    private double tc;
    private double to;
    private double closeVel;
    private double openVel;
    private double maxPower;
    private double minPower;
    private double t1;
    private double kp;
    private double kd;
    private double ki;
    private double maxIntg;
    private double minIntg;
    private double maxPID;
    private double minPID;

    public static Servo createServo(String content) {
        Servo servo = new Servo();
        servo.parseString(content);
        return servo;
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
        type = (char) src[0];
        subType = (char) src[1];
        if(charset != null)
            genName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
        else
            genName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim(), "4.3");
        generatorCode = (char) src[15];
        pe = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 22)).trim(), "6.2");
        tc = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 22, 26)).trim(), "4.2");
        to = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 26, 30)).trim(), "4.2");
        closeVel = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 30, 34)).trim(), "4.2");
        openVel = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 34, 38)).trim(), "4.2");
        maxPower = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 38, 42)).trim(), "4.2");
        minPower = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 42, 46)).trim(), "4.2");
        t1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 46, 50)).trim(), "4.2");
        kp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 50, 54)).trim(), "4.2");
        kd = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 54, 58)).trim(), "4.2");
        ki = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 58, 62)).trim(), "4.2");
        maxIntg = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 66)).trim(), "4.2");
        minIntg = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 66, 70)).trim(), "4.2");
        maxPID = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 70, 74)).trim(), "4.2");
        minPID = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 74, 78)).trim(), "4.2");
    }

    public String toString() {
        String strLine = "";
        strLine += type;
        strLine += subType;
        strLine += " ";
        strLine += DataOutputFormat.format.getFormatStr(genName, "8");
        strLine += BpaFileRwUtil.getFormatStr(getBaseKv(), "4.3");
        strLine += generatorCode;
        strLine += BpaFileRwUtil.getFormatStr(pe, "6.2");
        strLine += BpaFileRwUtil.getFormatStr(tc, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(to, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(closeVel, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(openVel, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(maxPower, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(minPower, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(t1, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(kp, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(kd, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(ki, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(maxIntg, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(minIntg, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(maxPID, "4.2");
        strLine += BpaFileRwUtil.getFormatStr(minPID, "4.2");
        return strLine;
    }

    public char getType() {
        return type;
    }

    public void setType(char type) {
        this.type = type;
    }

    public char getSubType() {
        return subType;
    }

    public void setSubType(char subType) {
        this.subType = subType;
    }

    public String getGenName() {
        return genName;
    }

    public void setGenName(String genName) {
        this.genName = genName;
    }

    public double getBaseKv() {
        return baseKv;
    }

    public void setBaseKv(double baseKv) {
        this.baseKv = baseKv;
    }

    public char getGeneratorCode() {
        return generatorCode;
    }

    public void setGeneratorCode(char generatorCode) {
        this.generatorCode = generatorCode;
    }

    public double getPe() {
        return pe;
    }

    public void setPe(double pe) {
        this.pe = pe;
    }

    public double getTc() {
        return tc;
    }

    public void setTc(double tc) {
        this.tc = tc;
    }

    public double getTo() {
        return to;
    }

    public void setTo(double to) {
        this.to = to;
    }

    public double getCloseVel() {
        return closeVel;
    }

    public void setCloseVel(double closeVel) {
        this.closeVel = closeVel;
    }

    public double getOpenVel() {
        return openVel;
    }

    public void setOpenVel(double openVel) {
        this.openVel = openVel;
    }

    public double getMaxPower() {
        return maxPower;
    }

    public void setMaxPower(double maxPower) {
        this.maxPower = maxPower;
    }

    public double getMinPower() {
        return minPower;
    }

    public void setMinPower(double minPower) {
        this.minPower = minPower;
    }

    public double getT1() {
        return t1;
    }

    public void setT1(double t1) {
        this.t1 = t1;
    }

    public double getKp() {
        return kp;
    }

    public void setKp(double kp) {
        this.kp = kp;
    }

    public double getKd() {
        return kd;
    }

    public void setKd(double kd) {
        this.kd = kd;
    }

    public double getKi() {
        return ki;
    }

    public void setKi(double ki) {
        this.ki = ki;
    }

    public double getMaxIntg() {
        return maxIntg;
    }

    public void setMaxIntg(double maxIntg) {
        this.maxIntg = maxIntg;
    }

    public double getMinIntg() {
        return minIntg;
    }

    public void setMinIntg(double minIntg) {
        this.minIntg = minIntg;
    }

    public double getMaxPID() {
        return maxPID;
    }

    public void setMaxPID(double maxPID) {
        this.maxPID = maxPID;
    }

    public double getMinPID() {
        return minPID;
    }

    public void setMinPID(double minPID) {
        this.minPID = minPID;
    }
}
