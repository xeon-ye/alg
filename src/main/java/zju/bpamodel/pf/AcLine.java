package zju.bpamodel.pf;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-25
 */
public class AcLine implements Serializable {
    private char chgCode = ' ';
    private String owner = "";
    private int linkMeterCode;
    private String busName1;
    private String busName2;
    private double baseKv1;
    private double baseKv2;
    private char circuit = ' ';
    private double baseI;
    private int shuntLineNum;
    private double r;
    private double x;
    private double halfG;
    private double halfB;
    private double length;
    private String desc = "";
    private String onlineDate = "";
    private String offlineDate = "";

    public static AcLine createAcLine(String content) {
        AcLine acLine = new AcLine();
        acLine.parseString(content);
        return acLine;
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

        chgCode = (char) src[2];
        if(charset != null)
            owner = new String(BpaFileRwUtil.getTarget(src, 3, 6), charset).trim();
        else
            owner = new String(BpaFileRwUtil.getTarget(src, 3, 6)).trim();
        if(charset != null)
            busName1 = new String(BpaFileRwUtil.getTarget(src, 6, 14), charset).trim();
        else
            busName1 = new String(BpaFileRwUtil.getTarget(src, 6, 14)).trim();
        baseKv1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 14, 18)).trim());
        linkMeterCode = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 18, 19)).trim());
        if(charset != null)
            busName2 = new String(BpaFileRwUtil.getTarget(src, 19, 27), charset).trim();
        else
            busName2 = new String(BpaFileRwUtil.getTarget(src, 19, 27)).trim();
        baseKv2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 27, 31)).trim());
        circuit = (char) src[31];
        baseI = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 33, 37)).trim());
        shuntLineNum = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 37, 38)).trim());
        r = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 38, 44)).trim(), "6.5");
        x = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 44, 50)).trim(), "6.5");
        halfG = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 50, 56)).trim(), "6.5");
        halfB = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 56, 62)).trim(), "6.5");
        length = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 66)).trim(), "4.1");
        if(charset != null)
            desc = new String(BpaFileRwUtil.getTarget(src, 66, 74), charset).trim();
        else
            desc = new String(BpaFileRwUtil.getTarget(src, 66, 74)).trim();
        if(charset != null)
            onlineDate = new String(BpaFileRwUtil.getTarget(src, 74, 77), charset).trim();
        else
            onlineDate = new String(BpaFileRwUtil.getTarget(src, 74, 77)).trim();
        if(charset != null)
            offlineDate = new String(BpaFileRwUtil.getTarget(src, 77, 80), charset).trim();
        else
            offlineDate = new String(BpaFileRwUtil.getTarget(src, 77, 80)).trim();
    }

    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("L").append(" ").append(chgCode);
        str.append(DataOutputFormat.format.getFormatStr(owner, "3L"));
        str.append(DataOutputFormat.format.getFormatStr(busName1, "8L"));
        str.append(BpaFileRwUtil.getFormatStr(baseKv1, "4.3")); //the bpa manual is 4.0;
        str.append(linkMeterCode);
        str.append(DataOutputFormat.format.getFormatStr(busName2, "8"));
        str.append(BpaFileRwUtil.getFormatStr(baseKv2, "4.3")); //the bpa manual is 4.0;
        str.append(circuit).append(" ");
        str.append(BpaFileRwUtil.getFormatStr(baseI, "4.3")); //the bpa manual is 4.0;
        str.append(shuntLineNum);
        str.append(BpaFileRwUtil.getFormatStr(r, "6.5"));
        str.append(BpaFileRwUtil.getFormatStr(x, "6.5"));
        str.append(BpaFileRwUtil.getFormatStr(halfG, "6.5"));
        str.append(BpaFileRwUtil.getFormatStr(halfB, "6.5"));
        str.append(BpaFileRwUtil.getFormatStr(length, "4.1"));
        str.append(DataOutputFormat.format.getFormatStr(desc, "8"));
        str.append(DataOutputFormat.format.getFormatStr(onlineDate, "3"));
        str.append(DataOutputFormat.format.getFormatStr(offlineDate, "3"));
        return str.toString();
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

    public int getLinkMeterCode() {
        return linkMeterCode;
    }

    public void setLinkMeterCode(int linkMeterCode) {
        this.linkMeterCode = linkMeterCode;
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

    public double getBaseI() {
        return baseI;
    }

    public void setBaseI(double baseI) {
        this.baseI = baseI;
    }

    public int getShuntLineNum() {
        return shuntLineNum;
    }

    public void setShuntLineNum(int shuntLineNum) {
        this.shuntLineNum = shuntLineNum;
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

    public double getHalfG() {
        return halfG;
    }

    public void setHalfG(double halfG) {
        this.halfG = halfG;
    }

    public double getHalfB() {
        return halfB;
    }

    public void setHalfB(double halfB) {
        this.halfB = halfB;
    }

    public double getLength() {
        return length;
    }

    public void setLength(double length) {
        this.length = length;
    }

    public String getDesc() {
        return desc;
    }

    public void setDesc(String desc) {
        this.desc = desc;
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
}
