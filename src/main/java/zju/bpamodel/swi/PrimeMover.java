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
public class PrimeMover implements Serializable {
    private char type;
    private char subType;
    private String busName;
    private double baseKv;
    private char generatorCode = ' ';
    private double maxPower;
    private double r;
    private double tg;
    private double tp;
    private double td;
    private double tw2;
    private double closeVel;
    private double openVel;
    private double dd;
    private double deadZone;

    public static PrimeMover createPrimeMover(String content) {
        PrimeMover primeMover = new PrimeMover();
        primeMover.parseString(content);
        return primeMover;
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
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
        else
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim());
        generatorCode = (char) src[15];
        if (type == 'G') {
            maxPower = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 22)).trim(), "6.1");
            r = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 22, 27)).trim(), "5.3");
            tg = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 27, 32)).trim(), "5.3");
            tp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 32, 37)).trim(), "5.3");
            td = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 37, 42)).trim(), "5.3");
            tw2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 42, 47)).trim(), "5.3");
            closeVel = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 47, 52)).trim(), "5.3");
            openVel = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 52, 57)).trim(), "5.3");
            dd = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 57, 62)).trim(), "5.3");
            deadZone = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 68)).trim(), "6.5");
        } else {
            if (subType == 'W') {
                tw2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 31, 36)).trim(), "5.3");
            }
        }
    }

    public String toString() {
        String strLine = "";
        strLine += type;
        strLine += subType;
        strLine += " ";
        strLine += DataOutputFormat.format.getFormatStr(getBusName(), "8");
        strLine += BpaFileRwUtil.getFormatStr(getBaseKv(), "4.1");// the bpa model is 4.0
        strLine += generatorCode;

        strLine += BpaFileRwUtil.getFormatStr(maxPower, "6.1");
        strLine += BpaFileRwUtil.getFormatStr(r, "5.3");
        strLine += BpaFileRwUtil.getFormatStr(tg, "5.3");
        strLine += BpaFileRwUtil.getFormatStr(tp, "5.3");
        strLine += BpaFileRwUtil.getFormatStr(td, "5.3");
        strLine += BpaFileRwUtil.getFormatStr(tw2, "5.3");
        strLine += BpaFileRwUtil.getFormatStr(closeVel, "5.3");
        strLine += BpaFileRwUtil.getFormatStr(openVel, "5.3");
        strLine += BpaFileRwUtil.getFormatStr(dd, "5.3");
        strLine += BpaFileRwUtil.getFormatStr(deadZone, "6.5");
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

    public char getGeneratorCode() {
        return generatorCode;
    }

    public void setGeneratorCode(char generatorCode) {
        this.generatorCode = generatorCode;
    }

    public double getMaxPower() {
        return maxPower;
    }

    public void setMaxPower(double maxPower) {
        this.maxPower = maxPower;
    }

    public double getR() {
        return r;
    }

    public void setR(double r) {
        this.r = r;
    }

    public double getTg() {
        return tg;
    }

    public void setTg(double tg) {
        this.tg = tg;
    }

    public double getTp() {
        return tp;
    }

    public void setTp(double tp) {
        this.tp = tp;
    }

    public double getTd() {
        return td;
    }

    public void setTd(double td) {
        this.td = td;
    }

    public double getTw2() {
        return tw2;
    }

    public void setTw2(double tw2) {
        this.tw2 = tw2;
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

    public double getDd() {
        return dd;
    }

    public void setDd(double dd) {
        this.dd = dd;
    }

    public double getDeadZone() {
        return deadZone;
    }

    public void setDeadZone(double deadZone) {
        this.deadZone = deadZone;
    }
}
