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
public class GenOutputCtrl implements Serializable {
    private int nDot;
    private String refBusName = "";
    private double refBusBaseKv;
    private String refBusCode = "";
    private int[] classification = new int[3];// 1: generator angle; 2: slip frequency; 3: exciter voltage
    private double[] max = new double[3];
    private double[] min = new double[3];

    public static GenOutputCtrl createOutput(String content) {
        GenOutputCtrl genOutCtrl = new GenOutputCtrl();
        genOutCtrl.parseString(content);
        return genOutCtrl;
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

        nDot = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 4, 5)).trim());
        if(charset != null)
            refBusName = new String(BpaFileRwUtil.getTarget(src, 6, 14), charset).trim();
        else
            refBusName = new String(BpaFileRwUtil.getTarget(src, 6, 14)).trim();
        refBusBaseKv = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 14, 18)).trim());
        if(charset != null)
            refBusCode = new String(BpaFileRwUtil.getTarget(src, 19, 20), charset).trim();
        else
            refBusCode = new String(BpaFileRwUtil.getTarget(src, 19, 20)).trim();
        int n = (src.length - 20) % 18 == 0 ? (src.length - 20) / 18 : (src.length - 20) / 18 + 1;
        for (int i = 0; i < 3; i++) {
            classification[i] = BpaFileRwUtil.parseInt(new String(BpaFileRwUtil.getTarget(src, 21 + i * 18, 22 + +i * 18)).trim());
            max[i] = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 25 + i * 18, 31 + i * 18)).trim());
            min[i] = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 32 + i * 18, 37 + i * 18)).trim());
        }
    }

    public String toString() {
        StringBuilder strLine = new StringBuilder();
        strLine.append("GH").append("  ");
        strLine.append(getnDot() > 0 ? getnDot() : " ").append(" ");
        strLine.append(DataOutputFormat.format.getFormatStr(getRefBusName(), "8"));
        strLine.append(BpaFileRwUtil.getFormatStr(getRefBusBaseKv(), "4.3")).append(" ");//the bpa manual is 4.0
        strLine.append(DataOutputFormat.format.getFormatStr(getRefBusCode(), "1")).append(" ");
        for (int i = 0; i < classification.length; i++) {
            if (classification[i] > 0) {
                strLine.append(classification[i]).append("   ");
                strLine.append(BpaFileRwUtil.getFormatStr(max[i], "6.5")).append(" ");//the bpa manual is 6.0
                strLine.append(BpaFileRwUtil.getFormatStr(min[i], "6.5")).append(" ");//the bpa manual is 6.0
            } else
                break;
        }
        return strLine.toString();
    }

    public int getnDot() {
        return nDot;
    }

    public void setnDot(int nDot) {
        this.nDot = nDot;
    }

    public String getRefBusName() {
        return refBusName;
    }

    public void setRefBusName(String refBusName) {
        this.refBusName = refBusName;
    }

    public double getRefBusBaseKv() {
        return refBusBaseKv;
    }

    public void setRefBusBaseKv(double refBusBaseKv) {
        this.refBusBaseKv = refBusBaseKv;
    }

    public String getRefBusCode() {
        return refBusCode;
    }

    public void setRefBusCode(String refBusCode) {
        this.refBusCode = refBusCode;
    }

    public int[] getClassification() {
        return classification;
    }

    public void setClassification(int[] classification) {
        this.classification = classification;
    }

    public double[] getMax() {
        return max;
    }

    public void setMax(double[] max) {
        this.max = max;
    }

    public double[] getMin() {
        return min;
    }

    public void setMin(double[] min) {
        this.min = min;
    }
}
