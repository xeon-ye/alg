package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-3-18
 */
public class InductionMotor implements Serializable {
    private char type = 'M';
    private char subType = ' ';
    private String zone;//MJ
    private String areaName;//MK
    private String busName = "";//ML
    private double baseKv;//ML
    private char id;//ML
    private double tJ;
    private double powerPercent;
    private double loadRate;
    private double minPower;
    private double rs;
    private double xs;
    private double xm;
    private double rr;
    private double xr;
    private double vi;
    private double ti;
    private double a;
    private double b;
    private int im;

    public static InductionMotor createInductionMotor(String content) {
        InductionMotor inductionMotor = new InductionMotor();
        inductionMotor.parseString(content);
        return inductionMotor;
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
        if (subType == 'J') {
            if (charset != null)
                zone = new String(BpaFileRwUtil.getTarget(src, 3, 5), charset).trim();
            else
                zone = new String(BpaFileRwUtil.getTarget(src, 3, 5)).trim();
        } else if (subType == 'K') {
            if (charset != null)
                areaName = new String(BpaFileRwUtil.getTarget(src, 3, 13), charset).trim();
            else
                areaName = new String(BpaFileRwUtil.getTarget(src, 3, 13)).trim();
        } else if (subType == 'L') {
            if(charset != null)
                busName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
            else
                busName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
            baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim(), "4.3");
            id = (char) src[15];
        }
        tJ = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 22)).trim(), "6.4");
        powerPercent = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 22, 25)).trim(), "3.3");
        loadRate = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 29)).trim(), "4.4");
        minPower = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 29, 32)).trim(), "3.0");
        rs = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 32, 37)).trim(), "5.4");
        xs = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 37, 42)).trim(), "5.4");
        xm = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 42, 47)).trim(), "5.4");
        rr = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 47, 52)).trim(), "5.4");
        xr = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 52, 57)).trim(), "5.4");
        vi = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 57, 60)).trim(), "3.2");
        ti = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 60, 64)).trim(), "4.2");
        a = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 64, 69)).trim(), "5.4");
        b = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 69, 74)).trim(), "5.4");
        im = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 79, 80)).trim());
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type).append(subType).append(" ");
        if (subType == 'J') {
            str.append(DataOutputFormat.format.getFormatStr(zone, "2")).append("           ");
        } else if (subType == 'K') {
            str.append(DataOutputFormat.format.getFormatStr(areaName, "10")).append("   ");
        } else if (subType == 'L') {
            str.append(DataOutputFormat.format.getFormatStr(busName, "8"));
            str.append(BpaFileRwUtil.getFormatStr(baseKv, "4.3"));//the bpa manual is 4.0
            str.append(id);
        }
        str.append(BpaFileRwUtil.getFormatStr(tJ, "6.4"));
        str.append(BpaFileRwUtil.getFormatStr(powerPercent, "3.3"));
        str.append(BpaFileRwUtil.getFormatStr(loadRate, "4.4"));
        str.append(BpaFileRwUtil.getFormatStr(minPower, "3.0"));
        str.append(BpaFileRwUtil.getFormatStr(rs, "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(xs, "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(xm, "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(rr, "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(xr, "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(vi, "3.2"));
        str.append(BpaFileRwUtil.getFormatStr(ti, "4.2"));
        str.append(BpaFileRwUtil.getFormatStr(a, "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(b, "5.4")).append("     ");
        str.append(im);
        return str.toString();
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

    public String getZone() {
        return zone;
    }

    public void setZone(String zone) {
        this.zone = zone;
    }

    public String getAreaName() {
        return areaName;
    }

    public void setAreaName(String areaName) {
        this.areaName = areaName;
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

    public char getId() {
        return id;
    }

    public void setId(char id) {
        this.id = id;
    }

    public double gettJ() {
        return tJ;
    }

    public void settJ(double tJ) {
        this.tJ = tJ;
    }

    public double getPowerPercent() {
        return powerPercent;
    }

    public void setPowerPercent(double powerPercent) {
        this.powerPercent = powerPercent;
    }

    public double getLoadRate() {
        return loadRate;
    }

    public void setLoadRate(double loadRate) {
        this.loadRate = loadRate;
    }

    public double getMinPower() {
        return minPower;
    }

    public void setMinPower(double minPower) {
        this.minPower = minPower;
    }

    public double getRs() {
        return rs;
    }

    public void setRs(double rs) {
        this.rs = rs;
    }

    public double getXs() {
        return xs;
    }

    public void setXs(double xs) {
        this.xs = xs;
    }

    public double getXm() {
        return xm;
    }

    public void setXm(double xm) {
        this.xm = xm;
    }

    public double getRr() {
        return rr;
    }

    public void setRr(double rr) {
        this.rr = rr;
    }

    public double getXr() {
        return xr;
    }

    public void setXr(double xr) {
        this.xr = xr;
    }

    public double getVi() {
        return vi;
    }

    public void setVi(double vi) {
        this.vi = vi;
    }

    public double getTi() {
        return ti;
    }

    public void setTi(double ti) {
        this.ti = ti;
    }

    public double getA() {
        return a;
    }

    public void setA(double a) {
        this.a = a;
    }

    public double getB() {
        return b;
    }

    public void setB(double b) {
        this.b = b;
    }

    public int getIm() {
        return im;
    }

    public void setIm(int im) {
        this.im = im;
    }
}
