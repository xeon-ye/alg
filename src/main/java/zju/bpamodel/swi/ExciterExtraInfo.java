package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-11-6
 */
public class ExciterExtraInfo implements Serializable {
    private String type;//F+,F#
    private String busName;
    private double baseKv;
    private char generatorCode = ' ';
    private double vamax;
    private double vamin;
    private double vaimax;
    private double vaimin;
    private double kb;
    private double t5;
    private double ke;
    private double te;
    private double se1;
    private double se2;
    private double vrmax;
    private double vrmin;
    private double kc;
    private double kd;
    private double kli;
    private double vlir;
    private double efdmax;

    public static ExciterExtraInfo createExciterExtraInfo(String content) {
        ExciterExtraInfo generator = new ExciterExtraInfo();
        generator.parseString(content);
        return generator;
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
        if(charset != null)
            type = new String(BpaFileRwUtil.getTarget(src, 0, 2), charset).trim();
        else
            type = new String(BpaFileRwUtil.getTarget(src, 0, 2)).trim();
        if(charset != null)
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
        else
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim());
        generatorCode = (char) src[15];
        if (type.equals("F+")) {
            vamax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 21)).trim(), "5.3");
            vamin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 21, 26)).trim(), "5.3");
            kb = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 26, 30)).trim(), "4.2");
            t5 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 30, 34)).trim(), "4.2");
            ke = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 34, 38)).trim(), "4.2");
            te = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 38, 42)).trim(), "4.2");
            se1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 42, 47)).trim(), "5.4");
            se2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 47, 52)).trim(), "5.4");
            vrmax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 52, 56)).trim(), "4.2");
            vrmin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 56, 60)).trim(), "4.2");
            kc = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 60, 64)).trim(), "4.2");
            kd = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 64, 68)).trim(), "4.2");
            kli = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 68, 72)).trim(), "4.2");
            vlir = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 72, 76)).trim(), "4.2");
            efdmax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 76, 80)).trim(), "4.2");
        } else if (type.equals("F#")) {
            vaimax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 58, 63)).trim(), "5.0");
            vaimin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 63, 68)).trim(), "5.0");
        }
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(getBusName(), "8L"));
        str.append(BpaFileRwUtil.getFormatStr(getBaseKv(), "4.3"));//the bpa manual is 4.0
        str.append(generatorCode);
        if (type.equals("F+")) {
            str.append(BpaFileRwUtil.getFormatStr(vamax, "5.3"));
            str.append(BpaFileRwUtil.getFormatStr(vamin, "5.3"));
            str.append(BpaFileRwUtil.getFormatStr(kb, "4.2"));
            str.append(BpaFileRwUtil.getFormatStr(t5, "4.3"));//the bpa manual is 4.2
            str.append(BpaFileRwUtil.getFormatStr(ke, "4.2"));
            str.append(BpaFileRwUtil.getFormatStr(te, "4.2"));
            str.append(BpaFileRwUtil.getFormatStr(se1, "5.4"));
            str.append(BpaFileRwUtil.getFormatStr(se2, "5.4"));
            str.append(BpaFileRwUtil.getFormatStr(vrmax, "4.2"));
            str.append(BpaFileRwUtil.getFormatStr(vrmin, "4.2"));
            str.append(BpaFileRwUtil.getFormatStr(kc, "4.3"));//the bpa manual is 4.2
            str.append(BpaFileRwUtil.getFormatStr(kd, "4.2"));
            str.append(BpaFileRwUtil.getFormatStr(kli, "4.2"));
            str.append(BpaFileRwUtil.getFormatStr(vlir, "4.2"));
            str.append(BpaFileRwUtil.getFormatStr(efdmax, "4.2"));
        } else if (type.equals("F#")) {
            for (int i = 0; i < 42; i++)
                str.append(" ");
            str.append(BpaFileRwUtil.getFormatStr(vaimax, "5.4")); //the bpa manual is 5.0
            str.append(BpaFileRwUtil.getFormatStr(vaimin, "5.4")); //the bpa manual is 5.0
        }
        return str.toString();
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public double getVaimax() {
        return vaimax;
    }

    public void setVaimax(double vaimax) {
        this.vaimax = vaimax;
    }

    public double getVaimin() {
        return vaimin;
    }

    public void setVaimin(double vaimin) {
        this.vaimin = vaimin;
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

    public double getVamax() {
        return vamax;
    }

    public void setVamax(double vamax) {
        this.vamax = vamax;
    }

    public double getVamin() {
        return vamin;
    }

    public void setVamin(double vamin) {
        this.vamin = vamin;
    }

    public double getKb() {
        return kb;
    }

    public void setKb(double kb) {
        this.kb = kb;
    }

    public double getT5() {
        return t5;
    }

    public void setT5(double t5) {
        this.t5 = t5;
    }

    public double getKe() {
        return ke;
    }

    public void setKe(double ke) {
        this.ke = ke;
    }

    public double getTe() {
        return te;
    }

    public void setTe(double te) {
        this.te = te;
    }

    public double getSe1() {
        return se1;
    }

    public void setSe1(double se1) {
        this.se1 = se1;
    }

    public double getSe2() {
        return se2;
    }

    public void setSe2(double se2) {
        this.se2 = se2;
    }

    public double getVrmax() {
        return vrmax;
    }

    public void setVrmax(double vrmax) {
        this.vrmax = vrmax;
    }

    public double getVrmin() {
        return vrmin;
    }

    public void setVrmin(double vrmin) {
        this.vrmin = vrmin;
    }

    public double getKc() {
        return kc;
    }

    public void setKc(double kc) {
        this.kc = kc;
    }

    public double getKd() {
        return kd;
    }

    public void setKd(double kd) {
        this.kd = kd;
    }

    public double getKli() {
        return kli;
    }

    public void setKli(double kli) {
        this.kli = kli;
    }

    public double getVlir() {
        return vlir;
    }

    public void setVlir(double vlir) {
        this.vlir = vlir;
    }

    public double getEfdmax() {
        return efdmax;
    }

    public void setEfdmax(double efdmax) {
        this.efdmax = efdmax;
    }
}
