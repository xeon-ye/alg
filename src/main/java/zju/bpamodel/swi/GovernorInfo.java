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
public class GovernorInfo implements Serializable {
    private String type;
    private String genName;
    private double baseKv;
    private char generatorCode = ' ';
    private double delt2;
    private double tr2;
    private double ep;
    private double minusDb2;
    private double db2;
    private double maxDb2;
    private double minDb2;
    private int ityp;
    private int ityp2;

    public static GovernorInfo createGovernorInfo(String content) {
        GovernorInfo governorInfo = new GovernorInfo();
        governorInfo.parseString(content);
        return governorInfo;
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
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim(), "4.0");
        generatorCode = (char) src[15];
        delt2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 20)).trim(), "4.4");
        tr2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 20, 24)).trim(), "4.4");
        ep = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 24, 29)).trim(), "5.0");
        minusDb2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 29, 34)).trim(), "5.0");
        db2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 34, 39)).trim(), "5.0");
        maxDb2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 39, 44)).trim(), "5.0");
        minDb2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 44, 49)).trim(), "5.0");
        ityp = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 49, 50)).trim());
        ityp2 = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 50, 51)).trim());
    }

    public String toString() {
        String strLine = "";
        strLine += DataOutputFormat.format.getFormatStr(type, "3");
        strLine += " ";
        strLine += DataOutputFormat.format.getFormatStr(genName, "8");
        strLine += BpaFileRwUtil.getFormatStr(getBaseKv(), "4.0");
        strLine += generatorCode;
        strLine += BpaFileRwUtil.getFormatStr(delt2, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(tr2, "4.4");
        strLine += BpaFileRwUtil.getFormatStr(ep, "5.0");
        strLine += BpaFileRwUtil.getFormatStr(minusDb2, "5.0");
        strLine += BpaFileRwUtil.getFormatStr(db2, "5.0");
        strLine += BpaFileRwUtil.getFormatStr(maxDb2, "5.0");
        strLine += BpaFileRwUtil.getFormatStr(minDb2, "5.0");
        strLine += ityp;
        strLine += ityp2;
        return strLine;
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

    public char getGeneratorCode() {
        return generatorCode;
    }

    public void setGeneratorCode(char generatorCode) {
        this.generatorCode = generatorCode;
    }

    public double getDelt2() {
        return delt2;
    }

    public void setDelt2(double delt2) {
        this.delt2 = delt2;
    }

    public double getTr2() {
        return tr2;
    }

    public void setTr2(double tr2) {
        this.tr2 = tr2;
    }

    public double getEp() {
        return ep;
    }

    public void setEp(double ep) {
        this.ep = ep;
    }

    public double getMinusDb2() {
        return minusDb2;
    }

    public void setMinusDb2(double minusDb2) {
        this.minusDb2 = minusDb2;
    }

    public double getDb2() {
        return db2;
    }

    public void setDb2(double db2) {
        this.db2 = db2;
    }

    public double getMaxDb2() {
        return maxDb2;
    }

    public void setMaxDb2(double maxDb2) {
        this.maxDb2 = maxDb2;
    }

    public double getMinDb2() {
        return minDb2;
    }

    public void setMinDb2(double minDb2) {
        this.minDb2 = minDb2;
    }

    public int getItyp() {
        return ityp;
    }

    public void setItyp(int ityp) {
        this.ityp = ityp;
    }

    public int getItyp2() {
        return ityp2;
    }

    public void setItyp2(int ityp2) {
        this.ityp2 = ityp2;
    }
}
