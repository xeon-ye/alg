package zju.bpamodel.pf;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-5-13
 */
public class PowerExchange implements Serializable {
    private char type;
    private char subType = ' ';
    private char chgCode = ' ';
    private String areaName;
    private String areaBusName;
    private double areaBusBaseKv;
    private double exchangePower;
    private String zoneName = "";

    // I卡
    private String area1Name;
    private String area2Name;

    public static PowerExchange createPowerExchange(String content) {
        PowerExchange powerExchange = new PowerExchange();
        powerExchange.parseString(content);
        return powerExchange;
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
        if (type == 'A') {
            subType = (char) src[1];
            chgCode = (char) src[2];
            if (charset != null)
                areaName = new String(BpaFileRwUtil.getTarget(src, 3, 13), charset).trim();
            else
                areaName = new String(BpaFileRwUtil.getTarget(src, 3, 13)).trim();
            if (subType == 'C' || subType == ' ') {
                if (charset != null)
                    areaBusName = new String(BpaFileRwUtil.getTarget(src, 13, 21), charset).trim();
                else
                    areaBusName = new String(BpaFileRwUtil.getTarget(src, 13, 21)).trim();
                areaBusBaseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 21, 25)).trim(), "4.0");
                exchangePower = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 26, 34)).trim(), "8.0");
                if (charset != null)
                    zoneName = new String(BpaFileRwUtil.getTarget(src, 35, 94), charset).trim();
                else
                    zoneName = new String(BpaFileRwUtil.getTarget(src, 35, 94)).trim();
            } else if (subType == 'O') {
                if (charset != null)
                    zoneName = new String(BpaFileRwUtil.getTarget(src, 14, 73), charset).trim();
                else
                    zoneName = new String(BpaFileRwUtil.getTarget(src, 14, 73)).trim();
            }
        } else if (type == 'I') {
            chgCode = (char) src[2];
            if (charset != null)
                area1Name = new String(BpaFileRwUtil.getTarget(src, 3, 13), charset).trim();
            else
                area1Name = new String(BpaFileRwUtil.getTarget(src, 3, 13)).trim();
            if (charset != null)
                area2Name = new String(BpaFileRwUtil.getTarget(src, 14, 24), charset).trim();
            else
                area2Name = new String(BpaFileRwUtil.getTarget(src, 14, 24)).trim();
            exchangePower = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 26, 34)).trim(), "8.0");
        }
    }

    public String toString() {
        StringBuilder strLine = new StringBuilder();
        strLine.append(type);
        if (type == 'A') {
            strLine.append(subType).append(chgCode);
            strLine.append(DataOutputFormat.format.getFormatStr(areaName, "10L"));
            if (subType == 'C' || subType == ' ') {
                strLine.append(DataOutputFormat.format.getFormatStr(areaBusName, "8L"));
                strLine.append(BpaFileRwUtil.getFormatStr(areaBusBaseKv, "4.0")).append(" ");
                strLine.append(BpaFileRwUtil.getFormatStr(exchangePower, "8.0")).append(" ");
                strLine.append(DataOutputFormat.format.getFormatStr(zoneName, "60L"));
            } else if (subType == 'O') {
                strLine.append(" ").append(DataOutputFormat.format.getFormatStr(zoneName, "59L"));
            }
        } else if (type == 'I') {
            strLine.append(" ").append(chgCode);
            strLine.append(DataOutputFormat.format.getFormatStr(area1Name, "10L")).append(" ");
            strLine.append(DataOutputFormat.format.getFormatStr(area2Name, "10L")).append("  ");
            strLine.append(BpaFileRwUtil.getFormatStr(exchangePower, "8.0"));
        }
        return strLine.toString();
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

    public char getChgCode() {
        return chgCode;
    }

    public void setChgCode(char chgCode) {
        this.chgCode = chgCode;
    }

    public String getAreaName() {
        return areaName;
    }

    public void setAreaName(String areaName) {
        this.areaName = areaName;
    }

    public String getAreaBusName() {
        return areaBusName;
    }

    public void setAreaBusName(String areaBusName) {
        this.areaBusName = areaBusName;
    }

    public double getAreaBusBaseKv() {
        return areaBusBaseKv;
    }

    public void setAreaBusBaseKv(double areaBusBaseKv) {
        this.areaBusBaseKv = areaBusBaseKv;
    }

    public double getExchangePower() {
        return exchangePower;
    }

    public void setExchangePower(double exchangePower) {
        this.exchangePower = exchangePower;
    }

    public String getZoneName() {
        return zoneName;
    }

    public void setZoneName(String zoneName) {
        this.zoneName = zoneName;
    }

    public String getArea1Name() {
        return area1Name;
    }

    public void setArea1Name(String area1Name) {
        this.area1Name = area1Name;
    }

    public String getArea2Name() {
        return area2Name;
    }

    public void setArea2Name(String area2Name) {
        this.area2Name = area2Name;
    }
}
