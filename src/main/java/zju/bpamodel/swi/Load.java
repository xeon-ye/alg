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
public class Load implements Serializable {
    private char type = 'L';//L
    private char subType = ' ';//A,B
    private char chgCode = ' ';
    private String busName = "";
    private double baseKv;
    private String zone;
    private String areaName;
    private double p1;
    private double q1;
    private double p2;
    private double q2;
    private double p3;
    private double q3;
    private double p4;
    private double q4;
    private double ldp;
    private double ldq;

    public static Load createLoad(String content) {
        Load load = new Load();
        load.parseString(content);
        return load;
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
        chgCode = (char) src[2];
        if(charset != null)
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
        else
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim(), "4.3");
        if(charset != null)
            zone = new String(BpaFileRwUtil.getTarget(src, 15, 17), charset).trim();
        else
            zone = new String(BpaFileRwUtil.getTarget(src, 15, 17)).trim();
        if(charset != null)
            areaName = new String(BpaFileRwUtil.getTarget(src, 17, 27), charset).trim();
        else
            areaName = new String(BpaFileRwUtil.getTarget(src, 17, 27)).trim();
        p1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 27, 32)).trim(), "5.3");
        q1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 32, 37)).trim(), "5.3");
        p2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 37, 42)).trim(), "5.3");
        q2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 42, 47)).trim(), "5.3");
        p3 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 47, 52)).trim(), "5.3");
        q3 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 52, 57)).trim(), "5.3");
        p4 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 57, 62)).trim(), "5.3");
        q4 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 67)).trim(), "5.3");
        ldp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 67, 72)).trim(), "5.3");
        ldq = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 72, 77)).trim(), "5.3");
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type).append(subType).append(chgCode);
        str.append(DataOutputFormat.format.getFormatStr(busName, "8"));
        str.append(BpaFileRwUtil.getFormatStr(baseKv, "4.3"));//the bpa manual is 4.0
        str.append(DataOutputFormat.format.getFormatStr(zone, "2"));
        str.append(DataOutputFormat.format.getFormatStr(areaName, "10"));
        str.append(BpaFileRwUtil.getFormatStr(p1, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(q1, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(p2, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(q2, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(p3, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(q3, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(p4, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(q4, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(ldp, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(ldq, "5.3"));
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

    public char getChgCode() {
        return chgCode;
    }

    public void setChgCode(char chgCode) {
        this.chgCode = chgCode;
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

    public double getP1() {
        return p1;
    }

    public void setP1(double p1) {
        this.p1 = p1;
    }

    public double getQ1() {
        return q1;
    }

    public void setQ1(double q1) {
        this.q1 = q1;
    }

    public double getP2() {
        return p2;
    }

    public void setP2(double p2) {
        this.p2 = p2;
    }

    public double getQ2() {
        return q2;
    }

    public void setQ2(double q2) {
        this.q2 = q2;
    }

    public double getP3() {
        return p3;
    }

    public void setP3(double p3) {
        this.p3 = p3;
    }

    public double getQ3() {
        return q3;
    }

    public void setQ3(double q3) {
        this.q3 = q3;
    }

    public double getP4() {
        return p4;
    }

    public void setP4(double p4) {
        this.p4 = p4;
    }

    public double getQ4() {
        return q4;
    }

    public void setQ4(double q4) {
        this.q4 = q4;
    }

    public double getLdp() {
        return ldp;
    }

    public void setLdp(double ldp) {
        this.ldp = ldp;
    }

    public double getLdq() {
        return ldq;
    }

    public void setLdq(double ldq) {
        this.ldq = ldq;
    }
}
