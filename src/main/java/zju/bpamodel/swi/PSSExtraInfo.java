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
public class PSSExtraInfo implements Serializable {
    private String type;
    private String genName = "";
    private double baseKv;
    private char id;
    // SI+卡
    private double kp;
    private double t1;
    private double t2;
    private double t13;
    private double t14;
    private double t3;
    private double t4;
    private double maxVs;
    private double minVs;
    private int ib;
    private String busName;
    private double busBaseKv;
    private double xq;
    private double kMVA;

    public static PSSExtraInfo createPSSExtraInfo(String content) {
        PSSExtraInfo pssExtraInfo = new PSSExtraInfo();
        pssExtraInfo.parseString(content);
        return pssExtraInfo;
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
            genName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
        else
            genName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim(), "4.3");
        id = (char) src[15];
        kp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 21)).trim(), "5.3");
        t1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 21, 26)).trim(), "5.3");
        t2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 26, 31)).trim(), "5.3");
        t13 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 31, 36)).trim(), "5.3");
        t14 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 36, 41)).trim(), "5.3");
        t3 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 41, 46)).trim(), "5.3");
        t4 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 46, 51)).trim(), "5.3");
        maxVs = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 51, 57)).trim(), "6.4");
        minVs = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 57, 63)).trim(), "6.4");
        ib = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 63, 64)).trim());
        if(charset != null)
            busName = new String(BpaFileRwUtil.getTarget(src, 64, 72), charset).trim();
        else
            busName = new String(BpaFileRwUtil.getTarget(src, 64, 72)).trim();
        if (ib == 1) {
            busBaseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 72, 76)).trim(), "4.3");
        } else if (ib == 2) {
            xq = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 72, 76)).trim(), "4.3");
        }
        kMVA = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 76, 80)).trim(), "4.0");
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type);
        str.append(DataOutputFormat.format.getFormatStr(genName, "8"));
        str.append(BpaFileRwUtil.getFormatStr(baseKv, "4.3"));
        str.append(id);
        str.append(BpaFileRwUtil.getFormatStr(kp, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(t1, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(t2, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(t13, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(t14, "5.2"));
        str.append(BpaFileRwUtil.getFormatStr(t3, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(t4, "5.3"));
        str.append(BpaFileRwUtil.getFormatStr(maxVs, "6.4"));
        str.append(BpaFileRwUtil.getFormatStr(minVs, "6.4"));
        str.append(ib);
        if (busName == null) {
            busName = "";
        }
        str.append(DataOutputFormat.format.getFormatStr(busName, "8"));
        if (ib == 1) {
            str.append(BpaFileRwUtil.getFormatStr(busBaseKv, "4.3"));
        } else {
            str.append(BpaFileRwUtil.getFormatStr(xq, "4.3"));  // ib == 2
        }
        str.append(BpaFileRwUtil.getFormatStr(kMVA, "4.0"));
        return str.toString();
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
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

    public char getId() {
        return id;
    }

    public void setId(char id) {
        this.id = id;
    }

    public double getKp() {
        return kp;
    }

    public void setKp(double kp) {
        this.kp = kp;
    }

    public double getT1() {
        return t1;
    }

    public void setT1(double t1) {
        this.t1 = t1;
    }

    public double getT2() {
        return t2;
    }

    public void setT2(double t2) {
        this.t2 = t2;
    }

    public double getT13() {
        return t13;
    }

    public void setT13(double t13) {
        this.t13 = t13;
    }

    public double getT14() {
        return t14;
    }

    public void setT14(double t14) {
        this.t14 = t14;
    }

    public double getT3() {
        return t3;
    }

    public void setT3(double t3) {
        this.t3 = t3;
    }

    public double getT4() {
        return t4;
    }

    public void setT4(double t4) {
        this.t4 = t4;
    }

    public double getMaxVs() {
        return maxVs;
    }

    public void setMaxVs(double maxVs) {
        this.maxVs = maxVs;
    }

    public double getMinVs() {
        return minVs;
    }

    public void setMinVs(double minVs) {
        this.minVs = minVs;
    }

    public int getIb() {
        return ib;
    }

    public void setIb(int ib) {
        this.ib = ib;
    }

    public String getBusName() {
        return genName;
    }

    public void setBusName(String busName) {
        this.genName = busName;
    }

    public double getBusBaseKv() {
        return busBaseKv;
    }

    public void setBusBaseKv(double busBaseKv) {
        this.busBaseKv = busBaseKv;
    }

    public double getXq() {
        return xq;
    }

    public void setXq(double xq) {
        this.xq = xq;
    }

    public double getkMVA() {
        return kMVA;
    }

    public void setkMVA(double kMVA) {
        this.kMVA = kMVA;
    }
}
