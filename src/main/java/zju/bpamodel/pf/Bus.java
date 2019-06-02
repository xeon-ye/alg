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
public class Bus implements Serializable {
    private char subType;
    private char chgCode = ' ';
    private String owner = "";
    private String name;
    private double baseKv;
    private String zoneName = "";
    private double loadMw;
    private double loadMvar;
    private double shuntMw;
    private double shuntMvar;
    private double genMwMax;
    private double genMw;
    private double genMvarShed;
    private double genMvarMax;
    private double genMvarMin;
    private double genMvar;
    private double vAmplMax;
    private double vAmplDesired;
    private double vAmplMin;
    private double slackBusVAngle;
    //The following three is used for controlling remote bus
    private String remoteCtrlBusName = "";
    private double remoteCtrlBusBaseKv;
    private double genMvarPercent;

    public static Bus createBus(String content) {
        Bus bus = new Bus();
        bus.parseString(content);
        return bus;
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
        if(charset != null)
            owner = new String(BpaFileRwUtil.getTarget(src, 3, 6), charset).trim();
        else
            owner = new String(BpaFileRwUtil.getTarget(src, 3, 6)).trim();
        if(charset != null)
            name = new String(BpaFileRwUtil.getTarget(src, 6, 14), charset).trim();
        else
            name = new String(BpaFileRwUtil.getTarget(src, 6, 14)).trim();
        baseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 14, 18)).trim());
        if(charset != null)
            zoneName = new String(BpaFileRwUtil.getTarget(src, 18, 20), charset).trim();
        else
            zoneName = new String(BpaFileRwUtil.getTarget(src, 18, 20)).trim();
        loadMw = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 20, 25)).trim());
        loadMvar = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 30)).trim());
        if (subType != 'X') {
            shuntMw = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 30, 34)).trim());
            shuntMvar = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 34, 38)).trim());
        }
        genMwMax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 38, 42)).trim());
        genMw = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 42, 47)).trim());
        if (subType == ' ' || subType == 'C' || subType == 'T' || subType == 'V') {
            genMvarShed = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 47, 52)).trim());
        } else
            genMvarMax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 47, 52)).trim());
        genMvarMin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 52, 57)).trim());
        //The following two
        vAmplDesired = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 57, 61)).trim(), "4.3");
        vAmplMax = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 57, 61)).trim(), "4.3");
        if (subType == 'S') {
            slackBusVAngle = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 61, 65)).trim());
        } else
            vAmplMin = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 61, 65)).trim());
        if (subType == 'G' || subType == 'X') {
            if(charset != null)
                remoteCtrlBusName = new String(BpaFileRwUtil.getTarget(src, 65, 73), charset).trim();
            else
                remoteCtrlBusName = new String(BpaFileRwUtil.getTarget(src, 65, 73)).trim();
            remoteCtrlBusBaseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 73, 77)).trim());
            genMvarPercent = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 77, 80)).trim());
        }
    }

    public String toString() {
        StringBuilder strLine = new StringBuilder();
        strLine.append("B").append(subType).append(chgCode);
        strLine.append(DataOutputFormat.format.getFormatStr(owner, "3L"));
        strLine.append(DataOutputFormat.format.getFormatStr(name, "8L"));
        strLine.append(BpaFileRwUtil.getFormatStr(getBaseKv(), "4.3"));//the bpa manual is 4.0
        strLine.append(DataOutputFormat.format.getFormatStr(getZoneName(), "2"));
        strLine.append(BpaFileRwUtil.getFormatStr(getLoadMw(), "5.4"));//the bpa manual is 5.0
        strLine.append(BpaFileRwUtil.getFormatStr(getLoadMvar(), "5.4"));//the bpa manual is 5.0
        if (subType != 'X') {
            strLine.append(BpaFileRwUtil.getFormatStr(getShuntMw(), "4.3"));//the bpa manual is 4.0
            strLine.append(BpaFileRwUtil.getFormatStr(getShuntMvar(), "4.3"));//the bpa manual is 4.0
        } else
            strLine.append(DataOutputFormat.format.getFormatStr("", "8"));
        strLine.append(BpaFileRwUtil.getFormatStr(getGenMwMax(), "4.3"));//the bpa manual is 4.0
        strLine.append(BpaFileRwUtil.getFormatStr(getGenMw(), "5.4"));//the bpa manual is 5.0
        if (subType == ' ' || subType == 'C' || subType == 'T' || subType == 'V') {
            strLine.append(BpaFileRwUtil.getFormatStr(getGenMvarShed(), "5.4"));//the bpa manual is 5.0
        } else
            strLine.append(BpaFileRwUtil.getFormatStr(getGenMvarMax(), "5.4"));//the bpa manual is 5.0
        strLine.append(BpaFileRwUtil.getFormatStr(getGenMvarMin(), "5.4"));//the bpa manual is 5.0
        if (getvAmplDesired() > 0) //may be not right.
            strLine.append(BpaFileRwUtil.getFormatStr(getvAmplDesired(), "4.3"));
        else
            strLine.append(BpaFileRwUtil.getFormatStr(getvAmplMax(), "4.3"));
        if (subType == 'S') {
            strLine.append(BpaFileRwUtil.getFormatStr(getSlackBusVAngle(), "4.3"));
        } else
            strLine.append(BpaFileRwUtil.getFormatStr(getvAmplMin(), "4.3"));
        if (subType == 'G' || subType == 'X') {
            strLine.append(DataOutputFormat.format.getFormatStr(getRemoteCtrlBusName(), "8"));
            strLine.append(BpaFileRwUtil.getFormatStr(getRemoteCtrlBusBaseKv(), "4.3"));//the bpa manual is 4.0;
            strLine.append(BpaFileRwUtil.getFormatStr(getGenMvarPercent(), "3.2"));//the bpa manual is 3.0;
        }
        return strLine.toString();
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

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public double getBaseKv() {
        return baseKv;
    }

    public void setBaseKv(double baseKv) {
        this.baseKv = baseKv;
    }

    public String getZoneName() {
        return zoneName;
    }

    public void setZoneName(String zoneName) {
        this.zoneName = zoneName;
    }

    public double getLoadMw() {
        return loadMw;
    }

    public void setLoadMw(double loadMw) {
        this.loadMw = loadMw;
    }

    public double getLoadMvar() {
        return loadMvar;
    }

    public void setLoadMvar(double loadMvar) {
        this.loadMvar = loadMvar;
    }

    public double getShuntMw() {
        return shuntMw;
    }

    public void setShuntMw(double shuntMw) {
        this.shuntMw = shuntMw;
    }

    public double getShuntMvar() {
        return shuntMvar;
    }

    public void setShuntMvar(double shuntMvar) {
        this.shuntMvar = shuntMvar;
    }

    public double getGenMwMax() {
        return genMwMax;
    }

    public void setGenMwMax(double genMwMax) {
        this.genMwMax = genMwMax;
    }

    public double getGenMw() {
        return genMw;
    }

    public void setGenMw(double genMw) {
        this.genMw = genMw;
    }

    public double getGenMvarShed() {
        return genMvarShed;
    }

    public void setGenMvarShed(double genMvarShed) {
        this.genMvarShed = genMvarShed;
    }

    public double getGenMvarMax() {
        return genMvarMax;
    }

    public void setGenMvarMax(double genMvarMax) {
        this.genMvarMax = genMvarMax;
    }

    public double getGenMvarMin() {
        return genMvarMin;
    }

    public void setGenMvarMin(double genMvarMin) {
        this.genMvarMin = genMvarMin;
    }

    public double getGenMvar() {
        return genMvar;
    }

    public void setGenMvar(double genMvar) {
        this.genMvar = genMvar;
    }

    public double getvAmplMax() {
        return vAmplMax;
    }

    public void setvAmplMax(double vAmplMax) {
        this.vAmplMax = vAmplMax;
    }

    public double getvAmplDesired() {
        return vAmplDesired;
    }

    public void setvAmplDesired(double vAmplDesired) {
        this.vAmplDesired = vAmplDesired;
    }

    public double getvAmplMin() {
        return vAmplMin;
    }

    public void setvAmplMin(double vAmplMin) {
        this.vAmplMin = vAmplMin;
    }

    public double getSlackBusVAngle() {
        return slackBusVAngle;
    }

    public void setSlackBusVAngle(double slackBusVAngle) {
        this.slackBusVAngle = slackBusVAngle;
    }

    public String getRemoteCtrlBusName() {
        return remoteCtrlBusName;
    }

    public void setRemoteCtrlBusName(String remoteCtrlBusName) {
        this.remoteCtrlBusName = remoteCtrlBusName;
    }

    public double getRemoteCtrlBusBaseKv() {
        return remoteCtrlBusBaseKv;
    }

    public void setRemoteCtrlBusBaseKv(double remoteCtrlBusBaseKv) {
        this.remoteCtrlBusBaseKv = remoteCtrlBusBaseKv;
    }

    public double getGenMvarPercent() {
        return genMvarPercent;
    }

    public void setGenMvarPercent(double genMvarPercent) {
        this.genMvarPercent = genMvarPercent;
    }
}
