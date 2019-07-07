package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-7-7
 */
public class CaseCard implements Serializable {
    private String type = "CASE";
    private String caseId;
    private int iNoPersist;
    private int ixo;
    private int dsw;
    private int iwscc;
    private double x2fac;
    private double xfact;
    private double tdodps;
    private double tqodps;
    private double tdodph;
    private double tqodph;
    private double cfacl2;

    public static CaseCard createCase(String content) {
        CaseCard caseCard = new CaseCard();
        caseCard.parseString(content);
        return caseCard;
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
        type = new String(BpaFileRwUtil.getTarget(src, 0, 4)).trim();
        if(charset != null)
            caseId = new String(BpaFileRwUtil.getTarget(src, 5, 15), charset).trim();
        else
            caseId = new String(BpaFileRwUtil.getTarget(src, 5, 15)).trim();
        iNoPersist = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 16, 17)).trim());
        ixo = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 20, 21)).trim());
        dsw = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 21, 22)).trim());
        iwscc = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 22, 23)).trim());
        x2fac = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 44, 49)).trim(), "5.5");
        xfact = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 49, 54)).trim(), "5.5");
        tdodps = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 54, 59)).trim(), "5.5");
        tqodps = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 59, 64)).trim(), "5.5");
        tdodph = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 64, 69)).trim(), "5.5");
        tqodph = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 69, 74)).trim(), "5.5");
        cfacl2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 74, 80)).trim(), "6.5");
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(caseId, "10L")).append(" ");
        str.append(iNoPersist).append("   ");
        str.append(ixo);
        str.append(dsw);
        str.append(iwscc);
        for (int i = 0; i < 21; i++) {
            str.append(" ");
        }
        str.append(BpaFileRwUtil.getFormatStr(x2fac, "5.5"));
        str.append(BpaFileRwUtil.getFormatStr(xfact, "5.5"));
        str.append(BpaFileRwUtil.getFormatStr(tdodps, "5.5"));
        str.append(BpaFileRwUtil.getFormatStr(tqodps, "5.5"));
        str.append(BpaFileRwUtil.getFormatStr(tdodph, "5.5"));
        str.append(BpaFileRwUtil.getFormatStr(tqodph, "5.5"));
        str.append(BpaFileRwUtil.getFormatStr(cfacl2, "6.5"));
        return str.toString();
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getCaseId() {
        return caseId;
    }

    public void setCaseId(String caseId) {
        this.caseId = caseId;
    }

    public int getiNoPersist() {
        return iNoPersist;
    }

    public void setiNoPersist(int iNoPersist) {
        this.iNoPersist = iNoPersist;
    }

    public int getIxo() {
        return ixo;
    }

    public void setIxo(int ixo) {
        this.ixo = ixo;
    }

    public int getDsw() {
        return dsw;
    }

    public void setDsw(int dsw) {
        this.dsw = dsw;
    }

    public int getIwscc() {
        return iwscc;
    }

    public void setIwscc(int iwscc) {
        this.iwscc = iwscc;
    }

    public double getX2fac() {
        return x2fac;
    }

    public void setX2fac(double x2fac) {
        this.x2fac = x2fac;
    }

    public double getXfact() {
        return xfact;
    }

    public void setXfact(double xfact) {
        this.xfact = xfact;
    }

    public double getTdodps() {
        return tdodps;
    }

    public void setTdodps(double tdodps) {
        this.tdodps = tdodps;
    }

    public double getTqodps() {
        return tqodps;
    }

    public void setTqodps(double tqodps) {
        this.tqodps = tqodps;
    }

    public double getTdodph() {
        return tdodph;
    }

    public void setTdodph(double tdodph) {
        this.tdodph = tdodph;
    }

    public double getTqodph() {
        return tqodph;
    }

    public void setTqodph(double tqodph) {
        this.tqodph = tqodph;
    }

    public double getCfacl2() {
        return cfacl2;
    }

    public void setCfacl2(double cfacl2) {
        this.cfacl2 = cfacl2;
    }
}
