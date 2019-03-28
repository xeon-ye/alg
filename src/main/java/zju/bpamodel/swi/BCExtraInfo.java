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
public class BCExtraInfo implements Serializable {
    private String type = "BC+";
    private String busName = "";
    private double baseKv;
    private char id = ' ';
    private double qPercent;
    private int rpCon;
    private double tmb;
    private double tb1;
    private double tb;
    private double kpb;
    private double kib;
    private double tsb;
    private double kd;

    public static BCExtraInfo createBCExtraInfo(String content) {
        BCExtraInfo bcExtraInfo = new BCExtraInfo();
        bcExtraInfo.parseString(content);
        return bcExtraInfo;
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
        type = new String(BpaFileRwUtil.getTarget(src, 0, 3)).trim();
        if(charset != null)
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
        else
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim(), "4.0");
        id = (char) src[15];
        qPercent = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 19)).trim(), "3.0");
        rpCon = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 19, 20)).trim());
        tmb = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 20, 25)).trim(), "5.0");
        tb1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 30)).trim(), "5.0");
        tb = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 30, 35)).trim(),"5.0");
        kpb = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 35, 40)).trim(), "5.0");
        kib = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 40, 45)).trim(), "5.0");
        tsb = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 45, 50)).trim(), "5.0");
        kd = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 50, 55)).trim(), "5.3");
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type);
        str.append(DataOutputFormat.format.getFormatStr(getBusName(), "8"));
        str.append(BpaFileRwUtil.getFormatStr(getBaseKv(), "4.0"));
        str.append(getId());
        str.append(BpaFileRwUtil.getFormatStr(qPercent, "3.0"));
        str.append(rpCon);
        str.append(BpaFileRwUtil.getFormatStr(tmb, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(tb1, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(tb, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(kpb, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(kib, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(tsb, "5.0"));
        str.append(BpaFileRwUtil.getFormatStr(kd, "5.3"));
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

    public double getqPercent() {
        return qPercent;
    }

    public void setqPercent(double qPercent) {
        this.qPercent = qPercent;
    }

    public int getRpCon() {
        return rpCon;
    }

    public void setRpCon(int rpCon) {
        this.rpCon = rpCon;
    }

    public double getTmb() {
        return tmb;
    }

    public void setTmb(double tmb) {
        this.tmb = tmb;
    }

    public double getTb1() {
        return tb1;
    }

    public void setTb1(double tb1) {
        this.tb1 = tb1;
    }

    public double getTb() {
        return tb;
    }

    public void setTb(double tb) {
        this.tb = tb;
    }

    public double getKpb() {
        return kpb;
    }

    public void setKpb(double kpb) {
        this.kpb = kpb;
    }

    public double getKib() {
        return kib;
    }

    public void setKib(double kib) {
        this.kib = kib;
    }

    public double getTsb() {
        return tsb;
    }

    public void setTsb(double tsb) {
        this.tsb = tsb;
    }

    public double getKd() {
        return kd;
    }

    public void setKd(double kd) {
        this.kd = kd;
    }
}
