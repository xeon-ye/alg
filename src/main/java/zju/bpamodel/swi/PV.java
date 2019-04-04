package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-3-27
 */
public class PV implements Serializable {
    private String type = "PV";
    private String busName = "";
    private double baseKv;
    private char id = ' ';
    private double t;
    private double s;
    private double uoc;
    private double isc;
    private double um;
    private double im;
    private int n1;
    private int n2;

    public static PV createPV(String content) {
        PV pv = new PV();
        pv.parseString(content);
        return pv;
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
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim());
        id = (char) src[15];
        t = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 20, 25)).trim(), "5.2");//the bpa model is 5.0
        s = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 30)).trim(), "5.2");//the bpa model is 5.0
        uoc = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 30, 35)).trim(), "5.2");//the bpa model is 5.0
        isc = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 35, 40)).trim(), "5.2");//the bpa model is 5.0
        um = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 40, 45)).trim(),"5.2");//the bpa model is 5.0
        im = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 45, 50)).trim(), "5.2");//the bpa model is 5.0
        n1 = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 50, 55)).trim());
        n2 = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 55, 60)).trim());
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(getBusName(), "8L"));
        str.append(BpaFileRwUtil.getFormatStr(getBaseKv(), "4.1"));// the bpa model is 4.0
        str.append(getId()).append("    ");
        str.append(BpaFileRwUtil.getFormatStr(t, "5.2"));
        str.append(BpaFileRwUtil.getFormatStr(s, "5.2"));
        str.append(BpaFileRwUtil.getFormatStr(uoc, "5.2"));
        str.append(BpaFileRwUtil.getFormatStr(isc, "5.2"));
        str.append(BpaFileRwUtil.getFormatStr(um, "5.2"));
        str.append(BpaFileRwUtil.getFormatStr(im, "5.2"));
        str.append(BpaFileRwUtil.getFormatStr(n1, "5"));
        str.append(BpaFileRwUtil.getFormatStr(n2, "5"));
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

    public double getT() {
        return t;
    }

    public void setT(double t) {
        this.t = t;
    }

    public double getS() {
        return s;
    }

    public void setS(double s) {
        this.s = s;
    }

    public double getUoc() {
        return uoc;
    }

    public void setUoc(double uoc) {
        this.uoc = uoc;
    }

    public double getIsc() {
        return isc;
    }

    public void setIsc(double isc) {
        this.isc = isc;
    }

    public double getUm() {
        return um;
    }

    public void setUm(double um) {
        this.um = um;
    }

    public double getIm() {
        return im;
    }

    public void setIm(double im) {
        this.im = im;
    }

    public int getN1() {
        return n1;
    }

    public void setN1(int n1) {
        this.n1 = n1;
    }

    public int getN2() {
        return n2;
    }

    public void setN2(int n2) {
        this.n2 = n2;
    }
}
