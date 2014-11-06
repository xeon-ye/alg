package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-12
 */
public class Exciter implements Serializable {
    private char type;//E,F
    private char subType;
    private String busName;
    private double baseKv;
    private char generatorCode = ' ';
    private double rc;
    private double xc;
    private double tr;
    private double vimax;
    private double vimin;
    private double tb;
    private double tc;
    private double ka;
    private double kv;
    private double ta;
    private double trh;
    private double vrmax;
    private double vamax;
    private double vrmin;
    private double vamin;
    private double ke;
    private double kj;
    private double te;
    private double kf;
    private double tf;
    private double kh;
    private double k;
    private double t1;
    private double t2;
    private double t3;
    private double t4;
    private double ta1;
    private double vrminmult;
    private double ki;
    private double kp;
    private double se75max;
    private double semax;
    private double efdmin;
    private double vbmax;
    private double efdmax;
    private double xl;
    private double tf1;

    public static Exciter createExciter(String content) {
        Exciter exciter = new Exciter();
        exciter.parseString(content);
        return exciter;
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
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
        else
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim());
        generatorCode = (char) src[15];
        if (type == 'E') {
            tr = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 20)).trim(), "4.3");
            switch (subType) {
                case 'E':
                    kv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 20, 25)).trim(), "5.2");
                    trh = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 29)).trim(), "4.2");
                    break;
                default:
                    ka = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 20, 25)).trim(), "5.2");
                    ta = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 29)).trim(), "4.2");
                    break;
            }
            ta1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 29, 33)).trim(), "4.3");
            switch (subType) {
                case 'D':
                case 'J':
                    vrmin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 33, 37)).trim(), "4.2");
                    break;
                default:
                    vrminmult = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 33, 37)).trim(), "4.2");
                    break;
            }
            ke = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 37, 41)).trim(), "4.3");
            te = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 41, 45)).trim(), "4.3");
            switch (subType) {
                case 'D':
                    ki = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 45, 49)).trim(), "4.3");
                    kp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 49, 53)).trim(), "4.3");
                    break;
                default:
                    se75max = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 45, 49)).trim(), "4.3");
                    semax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 49, 53)).trim(), "4.3");
                    break;
            }
            efdmin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 53, 58)).trim(), "5.3");
            switch (subType) {
                case 'D':
                    vbmax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 58, 62)).trim(), "4.3");
                    break;
                default:
                    efdmax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 58, 62)).trim(), "4.3");
                    break;
            }
            kf = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 66)).trim(), "4.3");
            tf = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 66, 70)).trim(), "4.3");
            if (subType == 'D') {
                xl = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 70, 75)).trim(), "5.4");
                tf1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 75, 80)).trim(), "5.4");
            }
        } else if (subType >= 'M' && subType <= 'V') {
            rc = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 20)).trim(), "4.3");
            xc = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 20, 24)).trim(), "4.3");
            tr = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 24, 29)).trim(), "5.3");
            k = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 29, 34)).trim(), "5.3");
            kv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 34, 37)).trim());
            t1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 37, 42)).trim(), "5.3");
            t2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 42, 47)).trim(), "5.3");
            t3 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 47, 52)).trim(), "5.3");
            t4 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 52, 57)).trim(), "5.3");
            ka = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 57, 62)).trim(), "5.3");
            ta = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 67)).trim(), "5.3");
            kf = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 67, 72)).trim(), "5.3");
            tf = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 72, 76)).trim(), "4.3");
            kh = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 76, 80)).trim(), "4.2");
        } else {
            rc = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 21)).trim(), "5.4");
            xc = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 21, 26)).trim(), "5.4");
            tr = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 26, 31)).trim(), "5.4");
            switch (subType) {
                case 'G':
                case 'K':
                case 'L':
                    vimax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 31, 36)).trim(), "5.3");
                    vimin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 36, 41)).trim(), "5.3");
                    break;
                case 'F':
                    vamax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 31, 36)).trim(), "5.3");
                    vamin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 36, 41)).trim(), "5.3");
                    break;
            }
            tb = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 41, 46)).trim(), "5.3");
            tc = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 46, 51)).trim(), "5.3");
            switch (subType) {
                case 'E':
                    kv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 51, 56)).trim(), "5.2");
                    trh = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 56, 61)).trim(), "5.3");
                    break;
                default:
                    ka = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 51, 56)).trim(), "5.2");
                    ta = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 56, 61)).trim(), "5.3");
                    break;
            }
            switch (subType) {
                case 'H':
                    vamax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 61, 66)).trim(), "5.3");
                    vamin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 66, 71)).trim(), "5.3");
                    break;
                default:
                    vrmax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 61, 66)).trim(), "5.3");
                    vrmin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 66, 71)).trim(), "5.3");
                    break;
            }
            switch (subType) {
                case 'H':
                    kj = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 71, 76)).trim(), "5.3");
                    break;
                default:
                    ke = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 71, 76)).trim(), "5.3");
                    break;
            }
            te = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 76, 80)).trim(), "4.3");
        }
    }

    public String toString() {
        String strLine = "";
        strLine += type;
        strLine += subType;
        strLine += " ";
        strLine += DataOutputFormat.format.getFormatStr(getBusName(), "8");
        strLine += BpaFileRwUtil.getFormatStr(getBaseKv(), "4.3");//the bpa manual is 4.0
        strLine += generatorCode;
        if (type == 'E') {
            strLine += BpaFileRwUtil.getFormatStr(tr, "4.3");
            switch (subType) {
                case 'E':
                    strLine += BpaFileRwUtil.getFormatStr(kv, "5.2");
                    strLine += BpaFileRwUtil.getFormatStr(trh, "4.2");
                    break;
                default:
                    strLine += BpaFileRwUtil.getFormatStr(ka, "5.2");
                    strLine += BpaFileRwUtil.getFormatStr(ta, "4.3");//the bpa manual is 4.2
                    break;
            }
            strLine += BpaFileRwUtil.getFormatStr(ta1, "4.3");
            switch (subType) {
                case 'D':
                case 'J':
                    strLine += BpaFileRwUtil.getFormatStr(vrmin, "4.2");
                    break;
                default:
                    strLine += BpaFileRwUtil.getFormatStr(vrminmult, "4.2");
                    break;
            }
            strLine += BpaFileRwUtil.getFormatStr(ke, "4.3");
            strLine += BpaFileRwUtil.getFormatStr(te, "4.3");
            switch (subType) {
                case 'D':
                    strLine += BpaFileRwUtil.getFormatStr(ki, "4.3");
                    strLine += BpaFileRwUtil.getFormatStr(kp, "4.3");
                    break;
                default:
                    strLine += BpaFileRwUtil.getFormatStr(se75max, "4.3");
                    strLine += BpaFileRwUtil.getFormatStr(semax, "4.3");
                    break;
            }
            strLine += BpaFileRwUtil.getFormatStr(efdmin, "5.3");
            switch (subType) {
                case 'D':
                    strLine += BpaFileRwUtil.getFormatStr(vbmax, "4.3");
                    break;
                default:
                    strLine += BpaFileRwUtil.getFormatStr(efdmax, "4.3");
                    break;
            }
            strLine += BpaFileRwUtil.getFormatStr(kf, "4.3");
            strLine += BpaFileRwUtil.getFormatStr(tf, "4.3");
            if (subType == 'D') {
                strLine += BpaFileRwUtil.getFormatStr(xl, "5.4");
                strLine += BpaFileRwUtil.getFormatStr(tf1, "5.4");
            }
        } else if (type == 'F' && subType >= 'M' && subType <= 'V') {
            strLine += BpaFileRwUtil.getFormatStr(rc, "4.3");
            strLine += BpaFileRwUtil.getFormatStr(xc, "4.3");
            strLine += BpaFileRwUtil.getFormatStr(tr, "5.4");//the bpa manual is 5.3
            strLine += BpaFileRwUtil.getFormatStr(k, "5.3");
            strLine += BpaFileRwUtil.getFormatStr(kv, "3.0");
            strLine += BpaFileRwUtil.getFormatStr(t1, "5.3");
            strLine += BpaFileRwUtil.getFormatStr(t2, "5.3");
            strLine += BpaFileRwUtil.getFormatStr(t3, "5.3");
            strLine += BpaFileRwUtil.getFormatStr(t4, "5.3");
            strLine += BpaFileRwUtil.getFormatStr(ka, "5.3");
            strLine += BpaFileRwUtil.getFormatStr(ta, "5.4");//the bpa manual is 5.3
            strLine += BpaFileRwUtil.getFormatStr(kf, "5.3");
            strLine += BpaFileRwUtil.getFormatStr(tf, "4.3");
            strLine += BpaFileRwUtil.getFormatStr(kh, "4.2");
        } else if (type == 'F') {
            strLine += BpaFileRwUtil.getFormatStr(rc, "5.4");
            strLine += BpaFileRwUtil.getFormatStr(xc, "5.4");
            strLine += BpaFileRwUtil.getFormatStr(tr, "5.4");
            switch (subType) {
                case 'G':
                case 'K':
                case 'L':
                    strLine += BpaFileRwUtil.getFormatStr(vimax, "5.3");
                    strLine += BpaFileRwUtil.getFormatStr(vimin, "5.3");
                    break;
                case 'F':
                    strLine += BpaFileRwUtil.getFormatStr(vamax, "5.3");
                    strLine += BpaFileRwUtil.getFormatStr(vamin, "5.3");
                    break;
                default:
                    strLine += DataOutputFormat.format.getFormatStr("", "5");
                    strLine += DataOutputFormat.format.getFormatStr("", "5");
            }
            strLine += BpaFileRwUtil.getFormatStr(tb, "5.3");
            strLine += BpaFileRwUtil.getFormatStr(tc, "5.3");
            switch (subType) {
                case 'E':
                    strLine += BpaFileRwUtil.getFormatStr(kv, "5.2");
                    strLine += BpaFileRwUtil.getFormatStr(trh, "5.3");
                    break;
                default:
                    strLine += BpaFileRwUtil.getFormatStr(ka, "5.2");
                    strLine += BpaFileRwUtil.getFormatStr(ta, "5.3");
                    break;
            }
            switch (subType) {
                case 'H':
                    strLine += BpaFileRwUtil.getFormatStr(vamax, "5.3");
                    strLine += BpaFileRwUtil.getFormatStr(vamin, "5.3");
                    break;
                default:
                    strLine += BpaFileRwUtil.getFormatStr(vrmax, "5.3");
                    strLine += BpaFileRwUtil.getFormatStr(vrmin, "5.3");
                    break;
            }
            switch (subType) {
                case 'H':
                    strLine += BpaFileRwUtil.getFormatStr(kj, "5.3");
                    break;
                default:
                    strLine += BpaFileRwUtil.getFormatStr(ke, "5.3");
                    break;
            }
            strLine += BpaFileRwUtil.getFormatStr(te, "4.3");
        }
        return strLine;
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

    public double getRc() {
        return rc;
    }

    public void setRc(double rc) {
        this.rc = rc;
    }

    public double getXc() {
        return xc;
    }

    public void setXc(double xc) {
        this.xc = xc;
    }

    public double getTr() {
        return tr;
    }

    public void setTr(double tr) {
        this.tr = tr;
    }

    public double getVimax() {
        return vimax;
    }

    public void setVimax(double vimax) {
        this.vimax = vimax;
    }

    public double getVamax() {
        return vamax;
    }

    public void setVamax(double vamax) {
        this.vamax = vamax;
    }

    public double getVrmin() {
        return vrmin;
    }

    public void setVrmin(double vrmin) {
        this.vrmin = vrmin;
    }

    public double getVimin() {
        return vimin;
    }

    public void setVimin(double vimin) {
        this.vimin = vimin;
    }

    public double getVamin() {
        return vamin;
    }

    public void setVamin(double vamin) {
        this.vamin = vamin;
    }

    public double getKe() {
        return ke;
    }

    public void setKe(double ke) {
        this.ke = ke;
    }

    public double getKj() {
        return kj;
    }

    public void setKj(double kj) {
        this.kj = kj;
    }

    public double getTe() {
        return te;
    }

    public void setTe(double te) {
        this.te = te;
    }

    public double getTb() {
        return tb;
    }

    public void setTb(double tb) {
        this.tb = tb;
    }

    public double getTc() {
        return tc;
    }

    public void setTc(double tc) {
        this.tc = tc;
    }

    public double getKa() {
        return ka;
    }

    public void setKa(double ka) {
        this.ka = ka;
    }

    public double getKv() {
        return kv;
    }

    public void setKv(double kv) {
        this.kv = kv;
    }

    public double getTa() {
        return ta;
    }

    public void setTa(double ta) {
        this.ta = ta;
    }

    public double getTrh() {
        return trh;
    }

    public void setTrh(double trh) {
        this.trh = trh;
    }

    public double getVrmax() {
        return vrmax;
    }

    public void setVrmax(double vrmax) {
        this.vrmax = vrmax;
    }

    public double getKf() {
        return kf;
    }

    public void setKf(double kf) {
        this.kf = kf;
    }

    public double getTf() {
        return tf;
    }

    public void setTf(double tf) {
        this.tf = tf;
    }

    public double getKh() {
        return kh;
    }

    public void setKh(double kh) {
        this.kh = kh;
    }

    public double getK() {
        return k;
    }

    public void setK(double k) {
        this.k = k;
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

    public double getTa1() {
        return ta1;
    }

    public void setTa1(double ta1) {
        this.ta1 = ta1;
    }

    public double getVrminmult() {
        return vrminmult;
    }

    public void setVrminmult(double vrminmult) {
        this.vrminmult = vrminmult;
    }

    public double getKi() {
        return ki;
    }

    public void setKi(double ki) {
        this.ki = ki;
    }

    public double getKp() {
        return kp;
    }

    public void setKp(double kp) {
        this.kp = kp;
    }

    public double getSe75max() {
        return se75max;
    }

    public void setSe75max(double se75max) {
        this.se75max = se75max;
    }

    public double getSemax() {
        return semax;
    }

    public void setSemax(double semax) {
        this.semax = semax;
    }

    public double getEfdmin() {
        return efdmin;
    }

    public void setEfdmin(double efdmin) {
        this.efdmin = efdmin;
    }

    public double getVbmax() {
        return vbmax;
    }

    public void setVbmax(double vbmax) {
        this.vbmax = vbmax;
    }

    public double getEfdmax() {
        return efdmax;
    }

    public void setEfdmax(double efdmax) {
        this.efdmax = efdmax;
    }

    public double getXl() {
        return xl;
    }

    public void setXl(double xl) {
        this.xl = xl;
    }

    public double getTf1() {
        return tf1;
    }

    public void setTf1(double tf1) {
        this.tf1 = tf1;
    }
}
