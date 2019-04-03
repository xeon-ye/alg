package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-3-24
 */
public class Governor implements Serializable {
    private char type;
    private char subType;
    private String genName;
    private double baseKv;
    private char generatorCode = ' ';
    private double kw;
    private double tr;
    private double minusDb1;
    private double db1;
    private double kp;
    private double kd;
    private double ki;
    private double td;
    private double maxIntg;
    private double minIntg;
    private double maxPID;
    private double minPID;
    private double delt;
    private double maxDb;
    private double minDb;

    public static Governor createGovernor(String content) {
        Governor governor = new Governor();
        governor.parseString(content);
        return governor;
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
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim());
        generatorCode = (char) src[15];
        kw = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 21)).trim(), "5.0");
        tr = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 21, 25)).trim(), "4.4");
        minusDb1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 29)).trim(), "4.4");
        db1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 29, 33)).trim(), "4.4");
        kp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 33, 38)).trim(), "5.0");
        kd = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 38, 43)).trim(), "5.0");
        ki = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 43, 48)).trim(), "5.0");
        td = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 48, 52)).trim(), "4.4");
        maxIntg = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 52, 56)).trim(), "4.4");
        minIntg = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 56, 60)).trim(), "4.4");
        maxPID = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 60, 64)).trim(), "4.4");
        minPID = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 64, 68)).trim(), "4.4");
        delt = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 68, 72)).trim(), "4.4");
        maxDb = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 72, 76)).trim(), "4.4");
        minDb = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 76, 80)).trim(), "4.4");
    }

    public String toString() {
        String strLine = "";
        strLine += type;
        strLine += subType;
        strLine += " ";
        strLine += DataOutputFormat.format.getFormatStr(genName, "8");
        strLine += BpaFileRwUtil.getFormatStr(getBaseKv(), "4.1");// the bpa model is 4.0
        strLine += generatorCode;
        strLine += BpaFileRwUtil.getFormatStr(kw, "5.0");
        strLine += BpaFileRwUtil.getFormatStr(tr, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(minusDb1, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(db1, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(kp, "5.0");
        strLine += BpaFileRwUtil.getFormatStr(kd, "5.0");
        strLine += BpaFileRwUtil.getFormatStr(ki, "5.0");
        strLine += BpaFileRwUtil.getFormatStr(td, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(maxIntg, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(minIntg, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(maxPID, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(minPID, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(delt, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(maxDb, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(minDb, "4.4");
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

    public char getGeneratorCode() {
        return generatorCode;
    }

    public void setGeneratorCode(char generatorCode) {
        this.generatorCode = generatorCode;
    }

    public double getKw() {
        return kw;
    }

    public void setKw(double kw) {
        this.kw = kw;
    }

    public double getTr() {
        return tr;
    }

    public void setTr(double tr) {
        this.tr = tr;
    }

    public double getMinusDb1() {
        return minusDb1;
    }

    public void setMinusDb1(double minusDb1) {
        this.minusDb1 = minusDb1;
    }

    public double getDb1() {
        return db1;
    }

    public void setDb1(double db1) {
        this.db1 = db1;
    }

    public double getKp() {
        return kp;
    }

    public void setKp(double kp) {
        this.kp = kp;
    }

    public double getKd() {
        return kd;
    }

    public void setKd(double kd) {
        this.kd = kd;
    }

    public double getKi() {
        return ki;
    }

    public void setKi(double ki) {
        this.ki = ki;
    }

    public double getTd() {
        return td;
    }

    public void setTd(double td) {
        this.td = td;
    }

    public double getMaxIntg() {
        return maxIntg;
    }

    public void setMaxIntg(double maxIntg) {
        this.maxIntg = maxIntg;
    }

    public double getMinIntg() {
        return minIntg;
    }

    public void setMinIntg(double minIntg) {
        this.minIntg = minIntg;
    }

    public double getMaxPID() {
        return maxPID;
    }

    public void setMaxPID(double maxPID) {
        this.maxPID = maxPID;
    }

    public double getMinPID() {
        return minPID;
    }

    public void setMinPID(double minPID) {
        this.minPID = minPID;
    }

    public double getDelt() {
        return delt;
    }

    public void setDelt(double delt) {
        this.delt = delt;
    }

    public double getMaxDb() {
        return maxDb;
    }

    public void setMaxDb(double maxDb) {
        this.maxDb = maxDb;
    }

    public double getMinDb() {
        return minDb;
    }

    public void setMinDb(double minDb) {
        this.minDb = minDb;
    }
}
