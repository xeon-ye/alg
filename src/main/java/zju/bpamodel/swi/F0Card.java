package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-3-18
 */
public class F0Card implements Serializable {
    private String type = "F0";
    private double t;
    private double dt;
    private double endT;
    private double dtc;
    private int istp;
    private double toli;
    private int ilim;
    private double delAng;
    private int dc;
    private double dmp;
    private double frqBse;
    private int lovtex;
    private int imblok;
    private int mfdep;
    private int igslim;
    private int lsolqit;
    private int noAngLim;
    private int infBus;
    private int noPp;
    private int noDq;
    private int noSat;
    private int noGv;
    private int ieqpc;
    private int noEx;
    private int mftomg;
    private int noSc;
    private int mgtomf;
    private int noLoad;

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
        t = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 4, 7)).trim(), "3.0");
        dt = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 8, 11)).trim());
        endT = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 12, 17)).trim(), "5.0");
        dtc = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 18, 21)).trim(), "3.1");
        istp = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 22, 25)).trim());
        toli = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 26, 31)).trim(), "5.5");
        ilim = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 32, 35)).trim());
        delAng = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 36, 40)).trim(), "4.4");
        dc = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 41, 43)).trim());
        dmp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 44, 47)).trim(), "3.3");
        frqBse = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 54, 56)).trim(), "2.0");
        lovtex = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 57, 58)).trim());
        imblok = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 59, 60)).trim());
        mfdep = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 63, 64)).trim());
        igslim = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 64, 65)).trim());
        lsolqit = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 65, 66)).trim());
        noAngLim = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 67, 68)).trim());
        infBus = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 69, 70)).trim());
        noPp = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 70, 71)).trim());
        noDq = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 71, 72)).trim());
        noSat = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 72, 73)).trim());
        noGv = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 73, 74)).trim());
        ieqpc = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 74, 75)).trim());
        noEx = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 75, 76)).trim());
        mftomg = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 76, 77)).trim());
        noSc = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 77, 78)).trim());
        mgtomf = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 78, 79)).trim());
        noLoad = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 79, 80)).trim());
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type).append("  ");
        str.append(BpaFileRwUtil.getFormatStr(t, "3.0")).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(dt, "3.1")).append(" ");//the bpa manual is 3.0
        str.append(BpaFileRwUtil.getFormatStr(endT, "5.0")).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(dtc, "3.1")).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(istp, "3")).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(toli, "5.5")).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(ilim, "3")).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(delAng, "4.4")).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(dc, "2")).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(dmp, "3.3")).append("       ");
        str.append(BpaFileRwUtil.getFormatStr(frqBse, "2.0")).append(" ");
        str.append(lovtex).append(" ");
        str.append(imblok).append("   ");
        str.append(mfdep);
        str.append(igslim);
        str.append(lsolqit).append(" ");
        str.append(noAngLim).append(" ");
        str.append(infBus);
        str.append(noPp);
        str.append(noDq);
        str.append(noSat);
        str.append(noGv);
        str.append(ieqpc);
        str.append(noEx);
        str.append(mftomg);
        str.append(noSc);
        str.append(mgtomf);
        str.append(noLoad);
        return str.toString();
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public double getT() {
        return t;
    }

    public void setT(double t) {
        this.t = t;
    }

    public double getDt() {
        return dt;
    }

    public void setDt(double dt) {
        this.dt = dt;
    }

    public double getEndT() {
        return endT;
    }

    public void setEndT(double endT) {
        this.endT = endT;
    }

    public double getDtc() {
        return dtc;
    }

    public void setDtc(double dtc) {
        this.dtc = dtc;
    }

    public int getIstp() {
        return istp;
    }

    public void setIstp(int istp) {
        this.istp = istp;
    }

    public double getToli() {
        return toli;
    }

    public void setToli(double toli) {
        this.toli = toli;
    }

    public int getIlim() {
        return ilim;
    }

    public void setIlim(int ilim) {
        this.ilim = ilim;
    }

    public double getDelAng() {
        return delAng;
    }

    public void setDelAng(double delAng) {
        this.delAng = delAng;
    }

    public int getDc() {
        return dc;
    }

    public void setDc(int dc) {
        this.dc = dc;
    }

    public double getDmp() {
        return dmp;
    }

    public void setDmp(double dmp) {
        this.dmp = dmp;
    }

    public double getFrqBse() {
        return frqBse;
    }

    public void setFrqBse(double frqBse) {
        this.frqBse = frqBse;
    }

    public int getLovtex() {
        return lovtex;
    }

    public void setLovtex(int lovtex) {
        this.lovtex = lovtex;
    }

    public int getImblok() {
        return imblok;
    }

    public void setImblok(int imblok) {
        this.imblok = imblok;
    }

    public int getMfdep() {
        return mfdep;
    }

    public void setMfdep(int mfdep) {
        this.mfdep = mfdep;
    }

    public int getIgslim() {
        return igslim;
    }

    public void setIgslim(int igslim) {
        this.igslim = igslim;
    }

    public int getLsolqit() {
        return lsolqit;
    }

    public void setLsolqit(int lsolqit) {
        this.lsolqit = lsolqit;
    }

    public int getNoAngLim() {
        return noAngLim;
    }

    public void setNoAngLim(int noAngLim) {
        this.noAngLim = noAngLim;
    }

    public int getInfBus() {
        return infBus;
    }

    public void setInfBus(int infBus) {
        this.infBus = infBus;
    }

    public int getNoPp() {
        return noPp;
    }

    public void setNoPp(int noPp) {
        this.noPp = noPp;
    }

    public int getNoDq() {
        return noDq;
    }

    public void setNoDq(int noDq) {
        this.noDq = noDq;
    }

    public int getNoSat() {
        return noSat;
    }

    public void setNoSat(int noSat) {
        this.noSat = noSat;
    }

    public int getNoGv() {
        return noGv;
    }

    public void setNoGv(int noGv) {
        this.noGv = noGv;
    }

    public int getIeqpc() {
        return ieqpc;
    }

    public void setIeqpc(int ieqpc) {
        this.ieqpc = ieqpc;
    }

    public int getNoEx() {
        return noEx;
    }

    public void setNoEx(int noEx) {
        this.noEx = noEx;
    }

    public int getMftomg() {
        return mftomg;
    }

    public void setMftomg(int mftomg) {
        this.mftomg = mftomg;
    }

    public int getNoSc() {
        return noSc;
    }

    public void setNoSc(int noSc) {
        this.noSc = noSc;
    }

    public int getMgtomf() {
        return mgtomf;
    }

    public void setMgtomf(int mgtomf) {
        this.mgtomf = mgtomf;
    }

    public int getNoLoad() {
        return noLoad;
    }

    public void setNoLoad(int noLoad) {
        this.noLoad = noLoad;
    }
}
