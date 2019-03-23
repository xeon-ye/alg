package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-3-23
 */
public class PSS implements Serializable {
    private char type = 'S';
    private char subType = ' ';
    private String genName = "";
    private double baseKv;
    private char id;
    private double kqv;
    private double tqv;
    private double kqs;
    private double tqs;
    private double tq;
    private double tq1;
    private double tpq1;
    private double tq2;
    private double tpq2;
    private double tq3;
    private double tpq3;
    private double maxVs;
    private double cutoffV;
    private double slowV;
    private String remoteBusName = "";
    private double remoteBaseKv;
    private double kqsBaseCap;

    // SI卡
    private double trw;
    private double t5;
    private double t6;
    private double t7;
    private double kr;
    private double trp;
    private double tw;
    private double tw1;
    private double tw2;
    private double ks;
    private double t9;
    private double t10;
    private double t12;
    private int inp;

    public static PSS createPSS(String content) {
        PSS pss = new PSS();
        pss.parseString(content);
        return pss;
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
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim(), "4.0");
        id = (char) src[15];
        if (subType == 'F' || subType == 'P' || subType == 'S' || subType == 'G') {
            kqv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 20)).trim(), "4.3");
            tqv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 20, 23)).trim(), "3.3");
            kqs = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 23, 27)).trim(), "4.3");
            tqs = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 27, 30)).trim(), "3.3");
            tq = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 30, 34)).trim(), "4.2");
            tq1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 34, 38)).trim(), "4.3");
            tpq1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 38, 42)).trim(), "4.3");
            tq2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 42, 46)).trim(), "4.3");
            tpq2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 46, 50)).trim(), "4.3");
            tq3 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 50, 54)).trim(), "4.3");
            tpq3 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 54, 58)).trim(), "4.3");
            maxVs = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 58, 62)).trim(), "4.3");
            cutoffV = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 66)).trim(), "4.3");
            slowV = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 66, 68)).trim(), "2.2");
            if (charset != null)
                remoteBusName = new String(BpaFileRwUtil.getTarget(src, 68, 76), charset).trim();
            else
                remoteBusName = new String(BpaFileRwUtil.getTarget(src, 68, 76)).trim();
            if (subType == 'P' || subType == 'G')
                kqsBaseCap = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 76, 80)).trim(), "4.0");
            else
                remoteBaseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 76, 80)).trim(), "4.0");
        } else if (subType == 'I') {
            trw = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 20)).trim(), "4.4");
            t5 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 20, 25)).trim(), "5.3");
            t6 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 30)).trim(), "5.3");
            t7 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 30, 35)).trim(), "5.3");
            kr = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 35, 41)).trim(), "6.4");
            trp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 41, 45)).trim(), "4.4");
            tw = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 45, 50)).trim(), "5.3");
            tw1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 50, 55)).trim(), "5.3");
            tw2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 55, 60)).trim(), "5.3");
            ks = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 60, 64)).trim(), "4.2");
            t9 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 64, 69)).trim(), "5.3");
            t10 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 69, 74)).trim(), "5.3");
            t12 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 74, 79)).trim(), "5.3");
            inp = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 79, 80)).trim());
        }
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type).append(subType).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(genName, "8"));
        str.append(BpaFileRwUtil.getFormatStr(baseKv, "4.0"));
        str.append(id);
        if (subType == 'F' || subType == 'P' || subType == 'S' || subType == 'G') {
            str.append(BpaFileRwUtil.getFormatStr(kqv, "4.3"));
            str.append(BpaFileRwUtil.getFormatStr(tqv, "3.3"));
            str.append(BpaFileRwUtil.getFormatStr(kqs, "4.3"));
            str.append(BpaFileRwUtil.getFormatStr(tqs, "3.3"));
            str.append(BpaFileRwUtil.getFormatStr(tq, "4.2"));
            str.append(BpaFileRwUtil.getFormatStr(tq1, "4.3"));
            str.append(BpaFileRwUtil.getFormatStr(tpq1, "4.3"));
            str.append(BpaFileRwUtil.getFormatStr(tq2, "4.3"));
            str.append(BpaFileRwUtil.getFormatStr(tpq2, "4.3"));
            str.append(BpaFileRwUtil.getFormatStr(tq3, "4.3"));
            str.append(BpaFileRwUtil.getFormatStr(tpq3, "4.3"));
            str.append(BpaFileRwUtil.getFormatStr(maxVs, "4.3"));
            str.append(BpaFileRwUtil.getFormatStr(cutoffV, "4.3"));
            str.append(BpaFileRwUtil.getFormatStr(slowV, "2.2"));
            str.append(DataOutputFormat.format.getFormatStr(remoteBusName, "8"));
            if (subType == 'P' || subType == 'G')
                str.append(BpaFileRwUtil.getFormatStr(kqsBaseCap, "4.0"));
            else
                str.append(BpaFileRwUtil.getFormatStr(remoteBaseKv, "4.0"));
        } else if (subType == 'I') {
            str.append(BpaFileRwUtil.getFormatStr(trw, "4.4"));
            str.append(BpaFileRwUtil.getFormatStr(t5, "5.3"));
            str.append(BpaFileRwUtil.getFormatStr(t6, "5.3"));
            str.append(BpaFileRwUtil.getFormatStr(t7, "5.3"));
            str.append(BpaFileRwUtil.getFormatStr(kr, "6.4"));
            str.append(BpaFileRwUtil.getFormatStr(trp, "4.4"));
            str.append(BpaFileRwUtil.getFormatStr(tw, "5.3"));
            str.append(BpaFileRwUtil.getFormatStr(tw1, "5.3"));
            str.append(BpaFileRwUtil.getFormatStr(tw2, "5.3"));
            str.append(BpaFileRwUtil.getFormatStr(ks, "4.2"));
            str.append(BpaFileRwUtil.getFormatStr(t9, "5.3"));
            str.append(BpaFileRwUtil.getFormatStr(t10, "5.3"));
            str.append(BpaFileRwUtil.getFormatStr(t12, "5.3"));
            str.append(inp);
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

    public String getBusName() {
        return genName;
    }

    public void setBusName(String busName) {
        this.genName = busName;
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

    public double getKqv() {
        return kqv;
    }

    public void setKqv(double kqv) {
        this.kqv = kqv;
    }

    public double getTqv() {
        return tqv;
    }

    public void setTqv(double tqv) {
        this.tqv = tqv;
    }

    public double getKqs() {
        return kqs;
    }

    public void setKqs(double kqs) {
        this.kqs = kqs;
    }

    public double getTqs() {
        return tqs;
    }

    public void setTqs(double tqs) {
        this.tqs = tqs;
    }

    public double getTq() {
        return tq;
    }

    public void setTq(double tq) {
        this.tq = tq;
    }

    public double getTq1() {
        return tq1;
    }

    public void setTq1(double tq1) {
        this.tq1 = tq1;
    }

    public double getTpq1() {
        return tpq1;
    }

    public void setTpq1(double tpq1) {
        this.tpq1 = tpq1;
    }

    public double getTq2() {
        return tq2;
    }

    public void setTq2(double tq2) {
        this.tq2 = tq2;
    }

    public double getTpq2() {
        return tpq2;
    }

    public void setTpq2(double tpq2) {
        this.tpq2 = tpq2;
    }

    public double getTq3() {
        return tq3;
    }

    public void setTq3(double tq3) {
        this.tq3 = tq3;
    }

    public double getTpq3() {
        return tpq3;
    }

    public void setTpq3(double tpq3) {
        this.tpq3 = tpq3;
    }

    public double getMaxVs() {
        return maxVs;
    }

    public void setMaxVs(double maxVs) {
        this.maxVs = maxVs;
    }

    public double getCutoffV() {
        return cutoffV;
    }

    public void setCutoffV(double cutoffV) {
        this.cutoffV = cutoffV;
    }

    public double getSlowV() {
        return slowV;
    }

    public void setSlowV(double slowV) {
        this.slowV = slowV;
    }

    public String getRemoteBusName() {
        return remoteBusName;
    }

    public void setRemoteBusName(String remoteBusName) {
        this.remoteBusName = remoteBusName;
    }

    public double getRemoteBaseKv() {
        return remoteBaseKv;
    }

    public void setRemoteBaseKv(double remoteBaseKv) {
        this.remoteBaseKv = remoteBaseKv;
    }

    public double getKqsBaseCap() {
        return kqsBaseCap;
    }

    public void setKqsBaseCap(double kqsBaseCap) {
        this.kqsBaseCap = kqsBaseCap;
    }

    public double getTrw() {
        return trw;
    }

    public void setTrw(double trw) {
        this.trw = trw;
    }

    public double getT5() {
        return t5;
    }

    public void setT5(double t5) {
        this.t5 = t5;
    }

    public double getT6() {
        return t6;
    }

    public void setT6(double t6) {
        this.t6 = t6;
    }

    public double getT7() {
        return t7;
    }

    public void setT7(double t7) {
        this.t7 = t7;
    }

    public double getKr() {
        return kr;
    }

    public void setKr(double kr) {
        this.kr = kr;
    }

    public double getTrp() {
        return trp;
    }

    public void setTrp(double trp) {
        this.trp = trp;
    }

    public double getTw() {
        return tw;
    }

    public void setTw(double tw) {
        this.tw = tw;
    }

    public double getTw1() {
        return tw1;
    }

    public void setTw1(double tw1) {
        this.tw1 = tw1;
    }

    public double getTw2() {
        return tw2;
    }

    public void setTw2(double tw2) {
        this.tw2 = tw2;
    }

    public double getKs() {
        return ks;
    }

    public void setKs(double ks) {
        this.ks = ks;
    }

    public double getT9() {
        return t9;
    }

    public void setT9(double t9) {
        this.t9 = t9;
    }

    public double getT10() {
        return t10;
    }

    public void setT10(double t10) {
        this.t10 = t10;
    }

    public double getT12() {
        return t12;
    }

    public void setT12(double t12) {
        this.t12 = t12;
    }

    public int getInp() {
        return inp;
    }

    public void setInp(int inp) {
        this.inp = inp;
    }
}
