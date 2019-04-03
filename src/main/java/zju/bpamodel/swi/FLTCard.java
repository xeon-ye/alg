package zju.bpamodel.swi;

import zju.bpamodel.BpaFileRwUtil;
import zju.ieeeformat.DataOutputFormat;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-29
 */
public class FLTCard implements Serializable {
    private String type;
    private String busAName;
    private double busABaseKv;
    private String busBName;
    private double busBBaseKv;
    private char circuitId;
    private int fltType;
    private int phase;
    private int side;//1,2
    private double tcyc0;
    private double tcyc1;
    private double tcyc2;
    private double posPercent;
    private double faultR;
    private double faultX;
    private double tcyc11;
    private double tcyc21;
    private double tcyc12;
    private double tcyc22;

    public static FLTCard createFault(String content) {
        FLTCard fault = new FLTCard();
        fault.parseString(content);
        return fault;
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
            busAName = new String(BpaFileRwUtil.getTarget(src, 4, 12), charset).trim();
        else
            busAName = new String(BpaFileRwUtil.getTarget(src, 4, 12)).trim();
        busABaseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 12, 16)).trim());
        if(charset != null)
            busBName = new String(BpaFileRwUtil.getTarget(src, 17, 25), charset).trim();
        else
            busBName = new String(BpaFileRwUtil.getTarget(src, 17, 25)).trim();
        busBBaseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25, 29)).trim());
        circuitId = (char) src[29];
        fltType = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 31, 33)).trim());
        side = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 36, 37)).trim());
        tcyc0 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 38, 42)).trim(), "4.0");
        tcyc1 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 42, 46)).trim(), "4.0");
        tcyc2 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 46, 50)).trim(), "4.0");
        posPercent = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 50, 52)).trim(), "2.0");
        faultR = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 52, 57)).trim(), "5.0");
        faultX = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 57, 62)).trim(), "5.0");
        if (fltType == 2) {
            phase = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 34, 35)).trim());
            tcyc11 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 66)).trim(), "4.0");
            tcyc21 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 66, 70)).trim(), "4.0");
        } else if (fltType == 3) {
            phase = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 34, 35)).trim());
            tcyc11 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 66)).trim(), "4.0");
            tcyc21 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 66, 70)).trim(), "4.0");
            tcyc11 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 62, 66)).trim(), "4.0");
            tcyc21 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 66, 70)).trim(), "4.0");
            tcyc12 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 70, 74)).trim(), "4.0");
            tcyc22 = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 74, 78)).trim(), "4.0");
        }
    }

    public String toString() {
        StringBuilder strLine = new StringBuilder();
        strLine.append("FLT").append(" ");
        strLine.append(DataOutputFormat.format.getFormatStr(getBusAName(), "8"));
        strLine.append(BpaFileRwUtil.getFormatStr(getBusABaseKv(), "4.1")).append(" ");// the bpa model is 4.0
        strLine.append(DataOutputFormat.format.getFormatStr(getBusBName(), "8"));
        strLine.append(BpaFileRwUtil.getFormatStr(getBusBBaseKv(), "4.1"));// the bpa model is 4.0
        strLine.append(circuitId).append(" ");
        strLine.append(BpaFileRwUtil.getFormatStr(fltType, "2")).append(" ");
        if (fltType == 1) {
            strLine.append("  ").append(side).append(" ");
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc0, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc1, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc2, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(posPercent, "2.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(getFaultR(), "5.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(getFaultX(), "5.0"));
        } else if (fltType == 2) {
            strLine.append(phase).append(" ");
            strLine.append(side).append(" ");
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc0, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc1, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc2, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(posPercent, "2.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(getFaultR(), "5.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(getFaultX(), "5.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc11, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc21, "4.0"));
        } else if (fltType == 3) {
            strLine.append(phase).append(" ");
            strLine.append(side).append(" ");
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc0, "4.0")).append("  ");
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc1, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc2, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(posPercent, "2.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(getFaultR(), "5.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(getFaultX(), "5.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc11, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc21, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc12, "4.0"));
            strLine.append(BpaFileRwUtil.getFormatStr(tcyc22, "4.0"));
        }
        return strLine.toString();
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getBusAName() {
        return busAName;
    }

    public void setBusAName(String busAName) {
        this.busAName = busAName;
    }

    public double getBusABaseKv() {
        return busABaseKv;
    }

    public void setBusABaseKv(double busABaseKv) {
        this.busABaseKv = busABaseKv;
    }

    public String getBusBName() {
        return busBName;
    }

    public void setBusBName(String busBName) {
        this.busBName = busBName;
    }

    public double getBusBBaseKv() {
        return busBBaseKv;
    }

    public void setBusBBaseKv(double busBBaseKv) {
        this.busBBaseKv = busBBaseKv;
    }

    public char getCircuitId() {
        return circuitId;
    }

    public void setCircuitId(char circuitId) {
        this.circuitId = circuitId;
    }

    public int getFltType() {
        return fltType;
    }

    public void setFltType(int fltType) {
        this.fltType = fltType;
    }

    public int getPhase() {
        return phase;
    }

    public void setPhase(int phase) {
        this.phase = phase;
    }

    public int getSide() {
        return side;
    }

    public void setSide(int side) {
        this.side = side;
    }

    public double getTcyc0() {
        return tcyc0;
    }

    public void setTcyc0(double tcyc0) {
        this.tcyc0 = tcyc0;
    }

    public double getTcyc1() {
        return tcyc1;
    }

    public void setTcyc1(double tcyc1) {
        this.tcyc1 = tcyc1;
    }

    public double getTcyc2() {
        return tcyc2;
    }

    public void setTcyc2(double tcyc2) {
        this.tcyc2 = tcyc2;
    }

    public double getPosPercent() {
        return posPercent;
    }

    public void setPosPercent(double posPercent) {
        this.posPercent = posPercent;
    }

    public double getFaultR() {
        return faultR;
    }

    public void setFaultR(double faultR) {
        this.faultR = faultR;
    }

    public double getFaultX() {
        return faultX;
    }

    public void setFaultX(double faultX) {
        this.faultX = faultX;
    }

    public double getTcyc11() {
        return tcyc11;
    }

    public void setTcyc11(double tcyc11) {
        this.tcyc11 = tcyc11;
    }

    public double getTcyc12() {
        return tcyc12;
    }

    public void setTcyc12(double tcyc12) {
        this.tcyc12 = tcyc12;
    }

    public double getTcyc21() {
        return tcyc21;
    }

    public void setTcyc21(double tcyc21) {
        this.tcyc21 = tcyc21;
    }

    public double getTcyc22() {
        return tcyc22;
    }

    public void setTcyc22(double tcyc22) {
        this.tcyc22 = tcyc22;
    }
}
