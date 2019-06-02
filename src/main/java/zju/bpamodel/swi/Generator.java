package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-11
 */
public class Generator implements Serializable {
    private char type = 'M';//M
    private char subType = ' ';//C,F,G
    private String busName = "";
    private double baseKv;
    private char id = ' ';
    private double eMWS;
    private double pPercent;
    private double qPercent;
    private double baseMva;
    private double ra;
    private double xdp;
    private double xqp;
    private double xd;
    private double xq;
    private double tdop;
    private double tqop;
    private double xl;
    private double sg10;
    private double sg12;
    private double d;

    public static Generator createGen(String content) {
        Generator generator = new Generator();
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
        type = (char) src[0];
        subType = (char) src[1];
        if(charset != null)
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
        else
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim());
        id = (char) src[15];
        eMWS = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 22)).trim());
        pPercent = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 22, 25)).trim(), "3.2");
        qPercent = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 28)).trim(), "3.2");
        baseMva = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 28, 32)).trim(), "4.0");
        ra = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 32, 36)).trim());
        xdp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 36, 41)).trim(), "5.4");
        xqp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 41, 46)).trim(), "5.4");
        xd = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 46, 51)).trim(), "5.4");
        xq = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 51, 56)).trim(), "5.4");
        tdop = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 56, 60)).trim(), "4.2");
        tqop = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 60, 63)).trim(), "3.2");
        xl = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 63, 68)).trim(), "5.4");
        sg10 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 68, 73)).trim(), "5.4");
        sg12 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 73, 77)).trim(), "4.3");
        d = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 77, 80)).trim(), "3.2");
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append(type).append(subType).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(getBusName(), "8L"));
        str.append(BpaFileRwUtil.getFormatStr(getBaseKv(), "4.3"));//the bpa manual is 4.0
        str.append(getId());
        str.append(BpaFileRwUtil.getFormatStr(geteMWS(), "6.5"));//the bpa manual is 6.0
        str.append(BpaFileRwUtil.getFormatStr(getpPercent(), "3.2"));
        str.append(BpaFileRwUtil.getFormatStr(getqPercent(), "3.2"));
        str.append(BpaFileRwUtil.getFormatStr(getBaseMva(), "4.3"));//the bpa manual is 4.0
        str.append(BpaFileRwUtil.getFormatStr(getRa(), "4.4"));//the bpa manual is 4.4
        str.append(BpaFileRwUtil.getFormatStr(getXdp(), "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(getXqp(), "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(getXd(), "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(getXq(), "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(getTdop(), "4.2"));
        str.append(BpaFileRwUtil.getFormatStr(getTqop(), "3.2"));
        str.append(BpaFileRwUtil.getFormatStr(getXl(), "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(getSg10(), "5.4"));
        str.append(BpaFileRwUtil.getFormatStr(getSg12(), "4.3"));
        str.append(BpaFileRwUtil.getFormatStr(getD(), "3.2"));
        return str.toString();
    }

    public char getType() {
        return type;
    }

    public void setType(char type) {
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

    public double geteMWS() {
        return eMWS;
    }

    public void seteMWS(double eMWS) {
        this.eMWS = eMWS;
    }

    public double getpPercent() {
        return pPercent;
    }

    public void setpPercent(double pPercent) {
        this.pPercent = pPercent;
    }

    public double getqPercent() {
        return qPercent;
    }

    public void setqPercent(double qPercent) {
        this.qPercent = qPercent;
    }

    public double getBaseMva() {
        return baseMva;
    }

    public void setBaseMva(double baseMva) {
        this.baseMva = baseMva;
    }

    public double getRa() {
        return ra;
    }

    public void setRa(double ra) {
        this.ra = ra;
    }

    public double getXdp() {
        return xdp;
    }

    public void setXdp(double xdp) {
        this.xdp = xdp;
    }

    public double getXqp() {
        return xqp;
    }

    public void setXqp(double xqp) {
        this.xqp = xqp;
    }

    public double getXd() {
        return xd;
    }

    public void setXd(double xd) {
        this.xd = xd;
    }

    public double getXq() {
        return xq;
    }

    public void setXq(double xq) {
        this.xq = xq;
    }

    public double getTdop() {
        return tdop;
    }

    public void setTdop(double tdop) {
        this.tdop = tdop;
    }

    public double getTqop() {
        return tqop;
    }

    public void setTqop(double tqop) {
        this.tqop = tqop;
    }

    public double getXl() {
        return xl;
    }

    public void setXl(double xl) {
        this.xl = xl;
    }

    public double getSg10() {
        return sg10;
    }

    public void setSg10(double sg10) {
        this.sg10 = sg10;
    }

    public double getSg12() {
        return sg12;
    }

    public void setSg12(double sg12) {
        this.sg12 = sg12;
    }

    public double getD() {
        return d;
    }

    public void setD(double d) {
        this.d = d;
    }

    public char getSubType() {
        return subType;
    }

    public void setSubType(char subType) {
        this.subType = subType;
    }
}
