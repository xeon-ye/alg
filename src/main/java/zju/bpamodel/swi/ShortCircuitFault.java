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
public class ShortCircuitFault implements Serializable {
    private char busASign = ' ';
    private String busAName;
    private double busABaseKv;
    private char busBSign = ' ';
    private String busBName;
    private double busBBaseKv;
    private char parallelBranchCode;
    private int mode;//1,2,3
    private double startCycle;
    private double faultR;
    private double faultX;
    private double posPercent;

    public static ShortCircuitFault createFault(String content) {
        ShortCircuitFault fault = new ShortCircuitFault();
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

        busASign = (char) src[3];
        if(charset != null)
            busAName = new String(BpaFileRwUtil.getTarget(src, 4, 12), charset).trim();
        else
            busAName = new String(BpaFileRwUtil.getTarget(src, 4, 12)).trim();
        busABaseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 12, 16)).trim());
        busBSign = (char) src[17];
        if(charset != null)
            busBName = new String(BpaFileRwUtil.getTarget(src, 18, 26), charset).trim();
        else
            busBName = new String(BpaFileRwUtil.getTarget(src, 18, 26)).trim();
        busBBaseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 26, 30)).trim());
        parallelBranchCode = (char) src[31];
        mode = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 35, 37)).trim());
        startCycle = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 39, 45)).trim());
        faultR = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 45, 51)).trim());
        faultX = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 51, 57)).trim());
        if (mode == 3)
            posPercent = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 57, 63)).trim());
    }

    public String toString() {
        StringBuilder strLine = new StringBuilder();
        strLine.append("LS").append(" ");
        strLine.append(getBusASign());
        strLine.append(DataOutputFormat.format.getFormatStr(getBusAName(), "8L"));
        strLine.append(BpaFileRwUtil.getFormatStr(getBusABaseKv(), "4.3"));//the bpa manual is 4.0
        strLine.append(" ").append(getBusBSign());
        strLine.append(DataOutputFormat.format.getFormatStr(getBusBName(), "8L"));
        strLine.append(BpaFileRwUtil.getFormatStr(getBusBBaseKv(), "4.3"));//the bpa manual is 4.0
        strLine.append(" ").append(parallelBranchCode).append("   ");
        strLine.append(BpaFileRwUtil.getFormatStr(getMode(), "2")).append("  ");
        strLine.append(BpaFileRwUtil.getFormatStr(getStartCycle(), "6.5"));//the bpa manual is 6.0
        strLine.append(BpaFileRwUtil.getFormatStr(getFaultR(), "6.5"));//the bpa manual is 6.0
        strLine.append(BpaFileRwUtil.getFormatStr(getFaultX(), "6.5"));//the bpa manual is 6.0
        if (mode == 3 || mode == -3)
            strLine.append(BpaFileRwUtil.getFormatStr(getPosPercent(), "6.5"));//the bpa manual is 6.0
        return strLine.toString();
    }

    public char getBusASign() {
        return busASign;
    }

    public void setBusASign(char busASign) {
        this.busASign = busASign;
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

    public char getBusBSign() {
        return busBSign;
    }

    public void setBusBSign(char busBSign) {
        this.busBSign = busBSign;
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

    public char getParallelBranchCode() {
        return parallelBranchCode;
    }

    public void setParallelBranchCode(char parallelBranchCode) {
        this.parallelBranchCode = parallelBranchCode;
    }

    public int getMode() {
        return mode;
    }

    public void setMode(int mode) {
        this.mode = mode;
    }

    public double getStartCycle() {
        return startCycle;
    }

    public void setStartCycle(double startCycle) {
        this.startCycle = startCycle;
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

    public double getPosPercent() {
        return posPercent;
    }

    public void setPosPercent(double posPercent) {
        this.posPercent = posPercent;
    }
}
