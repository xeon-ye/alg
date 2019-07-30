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
public class F0Card implements Serializable {
    private String type = "F0";
    private int ig;
    private int ia;
    private String genName1;
    private double genBaseKv1;
    private char id1;
    private String genName2;
    private double genBaseKv2;
    private char id2;
    private double amax;
    private double amin;
    private int iv;
    private String busName1;
    private double busBaseKv1;
    private int iF;
    private String busName2;
    private double busBaseKv2;

    public static F0Card createF0(String content) {
        F0Card f0Card = new F0Card();
        f0Card.parseString(content);
        return f0Card;
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
        ig = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 4, 5)).trim());
        ia = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 7, 8)).trim());
        if (charset != null)
            genName1 = new String(BpaFileRwUtil.getTarget(src, 9, 17), charset).trim();
        else
            genName1 = new String(BpaFileRwUtil.getTarget(src, 9, 17)).trim();
        genBaseKv1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 17, 21)).trim());
        id1 = (char) src[21];
        if (charset != null)
            genName2 = new String(BpaFileRwUtil.getTarget(src, 23, 31), charset).trim();
        else
            genName2 = new String(BpaFileRwUtil.getTarget(src, 23, 31)).trim();
        genBaseKv2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 31, 35)).trim());
        id2 = (char) src[35];
        amax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 37, 42)).trim(), "3.0");
        amin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 42, 47)).trim());
        iv = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 49, 50)).trim());
        if (charset != null)
            busName1 = new String(BpaFileRwUtil.getTarget(src, 51, 59), charset).trim();
        else
            busName1 = new String(BpaFileRwUtil.getTarget(src, 51, 59)).trim();
        busBaseKv1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 59, 63)).trim());
        iF = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 65, 66)).trim());
        if (charset != null)
            busName2 = new String(BpaFileRwUtil.getTarget(src, 67, 75), charset).trim();
        else
            busName2 = new String(BpaFileRwUtil.getTarget(src, 67, 75)).trim();
        busBaseKv2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 75, 80)).trim());
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type).append("  ");
        str.append(ig).append("  ");
        str.append(ia).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(genName1, "8L"));
        str.append(BpaFileRwUtil.getFormatStr(genBaseKv1, "4.1"));// the bpa model is 4.0
        str.append(id1).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(genName2, "8L"));
        str.append(BpaFileRwUtil.getFormatStr(genBaseKv2, "4.1"));// the bpa model is 4.0
        str.append(id2).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(amax, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(amin, "5.0")).append("  ");
        str.append(iv).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(busName1, "8L"));
        str.append(BpaFileRwUtil.getFormatStr(busBaseKv1, "4.1")).append("  ");// the bpa model is 4.0
        str.append(iF).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(busName2, "8L"));
        str.append(BpaFileRwUtil.getFormatStr(busBaseKv2, "4.1")).append(" ");// the bpa model is 4.0
        return str.toString();
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public int getIg() {
        return ig;
    }

    public void setIg(int ig) {
        this.ig = ig;
    }

    public int getIa() {
        return ia;
    }

    public void setIa(int ia) {
        this.ia = ia;
    }

    public String getGenName1() {
        return genName1;
    }

    public void setGenName1(String genName1) {
        this.genName1 = genName1;
    }

    public double getGenBaseKv1() {
        return genBaseKv1;
    }

    public void setGenBaseKv1(double genBaseKv1) {
        this.genBaseKv1 = genBaseKv1;
    }

    public char getId1() {
        return id1;
    }

    public void setId1(char id1) {
        this.id1 = id1;
    }

    public String getGenName2() {
        return genName2;
    }

    public void setGenName2(String genName2) {
        this.genName2 = genName2;
    }

    public double getGenBaseKv2() {
        return genBaseKv2;
    }

    public void setGenBaseKv2(double genBaseKv2) {
        this.genBaseKv2 = genBaseKv2;
    }

    public char getId2() {
        return id2;
    }

    public void setId2(char id2) {
        this.id2 = id2;
    }

    public double getAmax() {
        return amax;
    }

    public void setAmax(double amax) {
        this.amax = amax;
    }

    public double getAmin() {
        return amin;
    }

    public void setAmin(double amin) {
        this.amin = amin;
    }

    public int getIv() {
        return iv;
    }

    public void setIv(int iv) {
        this.iv = iv;
    }

    public String getBusName1() {
        return busName1;
    }

    public void setBusName1(String busName1) {
        this.busName1 = busName1;
    }

    public double getBusBaseKv1() {
        return busBaseKv1;
    }

    public void setBusBaseKv1(double busBaseKv1) {
        this.busBaseKv1 = busBaseKv1;
    }

    public int getiF() {
        return iF;
    }

    public void setiF(int iF) {
        this.iF = iF;
    }

    public String getBusName2() {
        return busName2;
    }

    public void setBusName2(String busName2) {
        this.busName2 = busName2;
    }

    public double getBusBaseKv2() {
        return busBaseKv2;
    }

    public void setBusBaseKv2(double busBaseKv2) {
        this.busBaseKv2 = busBaseKv2;
    }
}
