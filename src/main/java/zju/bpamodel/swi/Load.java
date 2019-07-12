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
    private char type;
    private char subType = ' ';//A,B,L
    private char chgCode = ' ';
    private String busName;
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

    // InductionMotor
    private char id;
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
    private char s;
    private int im;

    // LI
    private String total;
    private double dp;
    private double dq;
    private double dt = 60;
    private double tend;
    private char specCode;

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
        if (type == 'L') {
            if (subType == 'A' || subType == 'B') {
                chgCode = (char) src[2];
                if (charset != null)
                    busName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
                else
                    busName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
                baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim());
                if (charset != null)
                    zone = new String(BpaFileRwUtil.getTarget(src, 15, 17), charset).trim();
                else
                    zone = new String(BpaFileRwUtil.getTarget(src, 15, 17)).trim();
                if (charset != null)
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
            } else if (subType == 'I') {
                if ((new String(BpaFileRwUtil.getTarget(src, 3, 13)).trim()).isEmpty()) {
                    if (charset != null)
                        zone = new String(BpaFileRwUtil.getTarget(src, 13, 15), charset).trim();
                    else
                        zone = new String(BpaFileRwUtil.getTarget(src, 13, 15)).trim();
                } else if ((new String(BpaFileRwUtil.getTarget(src, 8, 15)).trim()).isEmpty()) {
                    total = new String(BpaFileRwUtil.getTarget(src, 3, 8)).trim();
                } else if ((new String(BpaFileRwUtil.getTarget(src, 13, 15)).trim()).isEmpty()) {
                    if (charset != null)
                        areaName = new String(BpaFileRwUtil.getTarget(src, 3, 13), charset).trim();
                    else
                        areaName = new String(BpaFileRwUtil.getTarget(src, 3, 13)).trim();
                } else {
                    if (charset != null)
                        busName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
                    else
                        busName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
                    baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim());
                }
                dp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 22)).trim(), "6.2");
                dq = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 22, 28)).trim(), "6.2");
                dt = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 28, 33)).trim(), "5.1");
                tend = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 33, 38)).trim(), "5.1");
                specCode = (char) src[39];
            }
        } else if (type == 'M') {
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
                baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim());
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
            s = (char) src[78];
            im = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 79, 80)).trim());
        }
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type).append(subType);
        if (type == 'L') {
            if (subType == 'A' || subType == 'B') {
                str.append(chgCode);
                str.append(DataOutputFormat.format.getFormatStr(busName, "8L"));
                str.append(BpaFileRwUtil.getFormatStr(baseKv, "4.1"));//the bpa manual is 4.0
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
            } else if (subType == 'I') {
                str.append(" ");
                if (busName != null) {
                    str.append(DataOutputFormat.format.getFormatStr(busName, "8L"));
                    str.append(BpaFileRwUtil.getFormatStr(baseKv, "4.3"));//the bpa manual is 4.0
                } else if (zone != null) {
                    str.append("          ").append(DataOutputFormat.format.getFormatStr(busName, "2L"));
                } else if (areaName != null) {
                    str.append(DataOutputFormat.format.getFormatStr(areaName, "10L")).append("  ");
                } else {
                    str.append(total).append("       ");
                }
                str.append(" ");
                str.append(BpaFileRwUtil.getFormatStr(dp, "6.2"));
                str.append(BpaFileRwUtil.getFormatStr(dq, "6.2"));
                str.append(BpaFileRwUtil.getFormatStr(dt, "5.1"));
                str.append(BpaFileRwUtil.getFormatStr(tend, "5.1")).append(" ");
                str.append(specCode);
            }
        }  else if (type == 'M') {
            str.append(" ");
            if (subType == 'J') {
                str.append(DataOutputFormat.format.getFormatStr(zone, "2")).append("           ");
            } else if (subType == 'K') {
                str.append(DataOutputFormat.format.getFormatStr(areaName, "10")).append("   ");
            } else if (subType == 'L') {
                str.append(DataOutputFormat.format.getFormatStr(busName, "8L"));
                str.append(BpaFileRwUtil.getFormatStr(baseKv, "4.1"));//the bpa manual is 4.0
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
            str.append(BpaFileRwUtil.getFormatStr(b, "5.4")).append("    ");
            str.append(s);
            str.append(im);
        }
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

    public char getS() {
        return s;
    }

    public void setS(char s) {
        this.s = s;
    }

    public int getIm() {
        return im;
    }

    public void setIm(int im) {
        this.im = im;
    }

    public String getTotal() {
        return total;
    }

    public void setTotal(String total) {
        this.total = total;
    }

    public double getDp() {
        return dp;
    }

    public void setDp(double dp) {
        this.dp = dp;
    }

    public double getDq() {
        return dq;
    }

    public void setDq(double dq) {
        this.dq = dq;
    }

    public double getDt() {
        return dt;
    }

    public void setDt(double dt) {
        this.dt = dt;
    }

    public double getTend() {
        return tend;
    }

    public void setTend(double tend) {
        this.tend = tend;
    }

    public char getSpecCode() {
        return specCode;
    }

    public void setSpecCode(char specCode) {
        this.specCode = specCode;
    }
}
