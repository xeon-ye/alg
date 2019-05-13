package zju.bpamodel.pf;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-11
 */
public class Transformer implements Serializable {
    private char subType = ' ';
    private char chgCode = ' ';
    private String owner = "";
    private String busName1 = "";
    private String busName2 = "";
    private double baseKv1;
    private double baseKv2;
    private char circuit;
    private double baseMva;
    private int linkMeterCode;
    private int shuntTransformerNum;
    private double r;
    private double x;
    private double g;
    private double b;
    private double tapKv1;
    private double tapKv2;
    private double phaseAngle;
    private String onlineDate = "";
    private String offlineDate = "";
    private String desc = "";

    public static Transformer createTransformer(String content) {
        Transformer transformer = new Transformer();
        transformer.parseString(content);
        return transformer;
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

        subType = (char) src[1];
        chgCode = (char) src[2];
        if (charset != null)
            owner = new String(BpaFileRwUtil.getTarget(src, 3, 6), charset).trim();
        else
            owner = new String(BpaFileRwUtil.getTarget(src, 3, 6)).trim();
        if (charset != null)
            busName1 = new String(BpaFileRwUtil.getTarget(src, 6, 14), charset).trim();
        else
            busName1 = new String(BpaFileRwUtil.getTarget(src, 6, 14)).trim();
        baseKv1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 14, 18)).trim());
        linkMeterCode = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 18, 19)).trim());
        if (charset != null)
            busName2 = new String(BpaFileRwUtil.getTarget(src, 19, 27), charset).trim();
        else
            busName2 = new String(BpaFileRwUtil.getTarget(src, 19, 27)).trim();
        baseKv2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 27, 31)).trim());
        circuit = (char) src[31];
        baseMva = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 33, 37)).trim());
        shuntTransformerNum = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 37, 38)).trim());
        r = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 38, 44)).trim(), "6.5");
        x = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 44, 50)).trim(), "6.5");
        g = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 50, 56)).trim(), "6.5");
        b = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 56, 62)).trim(), "6.5");
        if (subType == 'P') {
            phaseAngle = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 67)).trim(), "5.2");
        } else {
            tapKv1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 67)).trim(), "5.2");
            tapKv2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 67, 72)).trim(), "5.2");
        }
        if (charset != null)
            onlineDate = new String(BpaFileRwUtil.getTarget(src, 74, 77), charset).trim();
        else
            onlineDate = new String(BpaFileRwUtil.getTarget(src, 74, 77)).trim();
        if (charset != null)
            offlineDate = new String(BpaFileRwUtil.getTarget(src, 77, 80), charset).trim();
        else
            offlineDate = new String(BpaFileRwUtil.getTarget(src, 77, 80)).trim();

        //非标准的BPA可能有Desc数据
        if (charset != null)
            desc = new String(BpaFileRwUtil.getTarget(src, 104, src.length), charset).trim();
        else
            desc = new String(BpaFileRwUtil.getTarget(src, 104, src.length)).trim();
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("T").append(subType).append(chgCode);
        str.append(DataOutputFormat.format.getFormatStr(owner, "3"));
        str.append(DataOutputFormat.format.getFormatStr(busName1, "8"));
        str.append(BpaFileRwUtil.getFormatStr(baseKv1, "4.3")); //the bpa manual is 4.0;
        str.append(linkMeterCode);
        str.append(DataOutputFormat.format.getFormatStr(busName2, "8"));
        str.append(BpaFileRwUtil.getFormatStr(baseKv2, "4.3")); //the bpa manual is 4.0;
        str.append(circuit).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(baseMva, "4.3")); //the bpa manual is 4.0;
        str.append(shuntTransformerNum);
        str.append(BpaFileRwUtil.getFormatStr(r, "6.5"));
        str.append(BpaFileRwUtil.getFormatStr(x, "6.5"));
        str.append(BpaFileRwUtil.getFormatStr(g, "6.5"));
        str.append(BpaFileRwUtil.getFormatStr(b, "6.5"));
        if (subType == 'P') {
            str.append(BpaFileRwUtil.getFormatStr(phaseAngle, "5.2")).append("       ");
        } else {
            str.append(BpaFileRwUtil.getFormatStr(tapKv1, "5.2"));
            str.append(BpaFileRwUtil.getFormatStr(tapKv2, "5.2")).append("  ");
        }
        str.append(DataOutputFormat.format.getFormatStr(onlineDate, "3"));
        str.append(DataOutputFormat.format.getFormatStr(offlineDate, "3"));
        return str.toString();
    }

    public char getSubType() {
        return subType;
    }

    public void setSubType(char subType) {
        this.subType = subType;
    }

    public char getChgCode() {
        return chgCode;
    }

    public void setChgCode(char chgCode) {
        this.chgCode = chgCode;
    }

    public String getOwner() {
        return owner;
    }

    public void setOwner(String owner) {
        this.owner = owner;
    }

    public String getBusName1() {
        return busName1;
    }

    public void setBusName1(String busName1) {
        this.busName1 = busName1;
    }

    public String getBusName2() {
        return busName2;
    }

    public void setBusName2(String busName2) {
        this.busName2 = busName2;
    }

    public double getBaseKv1() {
        return baseKv1;
    }

    public void setBaseKv1(double baseKv1) {
        this.baseKv1 = baseKv1;
    }

    public double getBaseKv2() {
        return baseKv2;
    }

    public void setBaseKv2(double baseKv2) {
        this.baseKv2 = baseKv2;
    }

    public char getCircuit() {
        return circuit;
    }

    public void setCircuit(char circuit) {
        this.circuit = circuit;
    }

    public double getBaseMva() {
        return baseMva;
    }

    public void setBaseMva(double baseMva) {
        this.baseMva = baseMva;
    }

    public int getLinkMeterCode() {
        return linkMeterCode;
    }

    public void setLinkMeterCode(int linkMeterCode) {
        this.linkMeterCode = linkMeterCode;
    }

    public int getShuntTransformerNum() {
        return shuntTransformerNum;
    }

    public void setShuntTransformerNum(int shuntTransformerNum) {
        this.shuntTransformerNum = shuntTransformerNum;
    }

    public double getR() {
        return r;
    }

    public void setR(double r) {
        this.r = r;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getG() {
        return g;
    }

    public void setG(double g) {
        this.g = g;
    }

    public double getB() {
        return b;
    }

    public void setB(double b) {
        this.b = b;
    }

    public double getTapKv1() {
        return tapKv1;
    }

    public void setTapKv1(double tapKv1) {
        this.tapKv1 = tapKv1;
    }

    public double getTapKv2() {
        return tapKv2;
    }

    public void setTapKv2(double tapKv2) {
        this.tapKv2 = tapKv2;
    }

    public double getPhaseAngle() {
        return phaseAngle;
    }

    public void setPhaseAngle(double phaseAngle) {
        this.phaseAngle = phaseAngle;
    }

    public String getOnlineDate() {
        return onlineDate;
    }

    public void setOnlineDate(String onlineDate) {
        this.onlineDate = onlineDate;
    }

    public String getOfflineDate() {
        return offlineDate;
    }

    public void setOfflineDate(String offlineDate) {
        this.offlineDate = offlineDate;
    }

    public String getDesc() {
        return desc;
    }

    public void setDesc(String desc) {
        this.desc = desc;
    }
}
