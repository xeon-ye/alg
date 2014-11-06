package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Data class describing damping winding parameters of generator
 * User: Dong Shufeng
 * Date: 12-7-28
 */
public class GeneratorDW implements Serializable {
    private String busName;
    private double baseKv;
    private char id = ' ';
    private double baseMva;
    private double powerFactor;
    private String type;
    private String owner;
    private double xdpp;
    private double xqpp;
    private double xdopp;
    private double xqopp;

    public static GeneratorDW createGenDampingWinding(String content) {
        GeneratorDW dw = new GeneratorDW();
        dw.parseString(content);
        return dw;
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
        if (charset != null)
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11), charset).trim();
        else
            busName = new String(BpaFileRwUtil.getTarget(src, 3, 11)).trim();
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 11, 15)).trim());
        id = (char) src[15];

        baseMva = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 16, 21)).trim(), "5.1");
        powerFactor = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 22, 25)).trim(), "3.2");
        if (charset != null)
            type = new String(BpaFileRwUtil.getTarget(src, 30, 32), charset).trim();
        else
            type = new String(BpaFileRwUtil.getTarget(src, 30, 32)).trim();
        if (charset != null)
            owner = new String(BpaFileRwUtil.getTarget(src, 33, 36), charset);
        else
            owner = new String(BpaFileRwUtil.getTarget(src, 33, 36));
        xdpp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 37, 42)).trim(), "5.4");
        xqpp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 42, 47)).trim(), "5.4");
        xdopp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 47, 51)).trim(), "4.4");
        xqopp = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 51, 55)).trim(), "4.4");
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("M  ");
        str.append(DataOutputFormat.format.getFormatStr(busName, "8"));
        str.append(BpaFileRwUtil.getFormatStr(baseKv, "4.3"));//the bpa manual is 4.0
        str.append(id);
        str.append(BpaFileRwUtil.getFormatStr(baseMva, "5.1")).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(powerFactor, "3.2")).append("     ");
        str.append(DataOutputFormat.format.getFormatStr(type, "2")).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(owner, "3")).append(" ");
        str.append(DataOutputFormat.format.getFormatStr(xdpp, "5.4"));
        str.append(DataOutputFormat.format.getFormatStr(xqpp, "5.4"));
        str.append(DataOutputFormat.format.getFormatStr(xdopp, "4.4"));
        str.append(DataOutputFormat.format.getFormatStr(xqopp, "4.4"));
        return str.toString();
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

    public double getBaseMva() {
        return baseMva;
    }

    public void setBaseMva(double baseMva) {
        this.baseMva = baseMva;
    }

    public double getPowerFactor() {
        return powerFactor;
    }

    public void setPowerFactor(double powerFactor) {
        this.powerFactor = powerFactor;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getOwner() {
        return owner;
    }

    public void setOwner(String owner) {
        this.owner = owner;
    }

    public double getXdpp() {
        return xdpp;
    }

    public void setXdpp(double xdpp) {
        this.xdpp = xdpp;
    }

    public double getXqpp() {
        return xqpp;
    }

    public void setXqpp(double xqpp) {
        this.xqpp = xqpp;
    }

    public double getXdopp() {
        return xdopp;
    }

    public void setXdopp(double xdopp) {
        this.xdopp = xdopp;
    }

    public double getXqopp() {
        return xqopp;
    }

    public void setXqopp(double xqopp) {
        this.xqopp = xqopp;
    }
}
