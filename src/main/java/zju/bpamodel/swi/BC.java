package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-3-28
 */
public class BC implements Serializable {
    private String type = "BC";
    private String busName = "";
    private double baseKv;
    private char id = ' ';
    private double pPercent;
    private int ipCon;
    private double tma;
    private double ta1;
    private double ta;
    private double kpa;
    private double kia;
    private double tsa;
    private double c;
    private double dcBaseKv;
    private double k;
    private double mva;
    private double kover;
    private int converterNum;

    public static BC createBC(String content) {
        BC bc = new BC();
        bc.parseString(content);
        return bc;
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
        type = new String(BpaFileRwUtil.getTarget(src, 0, 2)).trim();
        if(charset != null)
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
        else
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim(), "4.3");
        id = (char) src[15];
        pPercent = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 19)).trim(), "3.0");
        ipCon = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 19, 20)).trim());
        tma = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 20, 25)).trim(), "5.0");
        ta1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 30)).trim(), "5.0");
        ta = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 30, 35)).trim(),"5.0");
        kpa = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 35, 40)).trim(), "5.0");
        kia = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 40, 45)).trim(), "5.0");
        tsa = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 45, 50)).trim(), "5.0");
        c = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 50, 55)).trim(), "5.0");
        dcBaseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 55, 60)).trim(), "5.3");// the bpa model is 5.0
        k = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 60, 65)).trim(), "5.0");
        mva = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 65, 70)).trim(), "5.0");
        kover = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 70, 75)).trim(), "5.0");
        converterNum = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 75, 80)).trim());
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(getBusName(), "8"));
        str.append(BpaFileRwUtil.getFormatStr(getBaseKv(), "4.3"));
        str.append(getId());
        str.append(BpaFileRwUtil.getFormatStr(pPercent, "3.0"));
        str.append(ipCon);
        str.append(BpaFileRwUtil.getFormatStr(tma, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(ta1, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(ta, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(kpa, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(kia, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(tsa, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(c, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(dcBaseKv, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(k, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(mva, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(kover, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(converterNum, "5"));
        return str.toString();
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
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

    public double getpPercent() {
        return pPercent;
    }

    public void setpPercent(double pPercent) {
        this.pPercent = pPercent;
    }

    public int getIpCon() {
        return ipCon;
    }

    public void setIpCon(int ipCon) {
        this.ipCon = ipCon;
    }

    public double getTma() {
        return tma;
    }

    public void setTma(double tma) {
        this.tma = tma;
    }

    public double getTa1() {
        return ta1;
    }

    public void setTa1(double ta1) {
        this.ta1 = ta1;
    }

    public double getTa() {
        return ta;
    }

    public void setTa(double ta) {
        this.ta = ta;
    }

    public double getKpa() {
        return kpa;
    }

    public void setKpa(double kpa) {
        this.kpa = kpa;
    }

    public double getKia() {
        return kia;
    }

    public void setKia(double kia) {
        this.kia = kia;
    }

    public double getTsa() {
        return tsa;
    }

    public void setTsa(double tsa) {
        this.tsa = tsa;
    }

    public double getC() {
        return c;
    }

    public void setC(double c) {
        this.c = c;
    }

    public double getDcBaseKv() {
        return dcBaseKv;
    }

    public void setDcBaseKv(double dcBaseKv) {
        this.dcBaseKv = dcBaseKv;
    }

    public double getK() {
        return k;
    }

    public void setK(double k) {
        this.k = k;
    }

    public double getMva() {
        return mva;
    }

    public void setMva(double mva) {
        this.mva = mva;
    }

    public double getKover() {
        return kover;
    }

    public void setKover(double kover) {
        this.kover = kover;
    }

    public int getConverterNum() {
        return converterNum;
    }

    public void setConverterNum(int converterNum) {
        this.converterNum = converterNum;
    }
}
