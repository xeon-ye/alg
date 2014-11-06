package zju.bpamodel.pf;

import zju.bpamodel.BpaFileRwUtil;

import java.io.Serializable;
import java.io.UnsupportedEncodingException;

public class DataModifyInfo implements Serializable {
    private String zone;
    private double loadPFactor;
    private double loadQFactor;
    private double genPFactor;
    private double genQFactor;

    public static DataModifyInfo createDataModifyInfo(String content) {
        DataModifyInfo dataModifyInfo = new DataModifyInfo();
        dataModifyInfo.parseString(content);
        return dataModifyInfo;
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

        zone = new String(BpaFileRwUtil.getTarget(src, 3, 5)).trim();
        if (new String(BpaFileRwUtil.getTarget(src, 9, 14)).trim().equals("")) {
            loadPFactor = 1.0;
        } else {
            loadPFactor = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 9, 14)).trim());
        }
        if (new String(BpaFileRwUtil.getTarget(src, 15, 20)).trim().equals("")) {
            loadQFactor = loadPFactor;
        } else {
            loadQFactor = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 15, 20)).trim());
        }
        if (new String(BpaFileRwUtil.getTarget(src, 21, 26)).trim().equals("")) {
            genPFactor = 1.0;
        } else {
            genPFactor = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 21, 26)).trim());
        }
        if (new String(BpaFileRwUtil.getTarget(src, 27, 32)).trim().equals("")) {
            genQFactor = genPFactor;
        } else {
            genQFactor = BpaFileRwUtil.parseDouble(new String(BpaFileRwUtil.getTarget(src, 27, 32)).trim());
        }
    }

    public String getZone() {
        return zone;
    }

    public void setZone(String zone) {
        this.zone = zone;
    }

    public double getLoadPFactor() {
        return loadPFactor;
    }

    public void setLoadPFactor(double loadPFactor) {
        this.loadPFactor = loadPFactor;
    }

    public double getLoadQFactor() {
        return loadQFactor;
    }

    public void setLoadQFactor(double loadQFactor) {
        this.loadQFactor = loadQFactor;
    }

    public double getGenPFactor() {
        return genPFactor;
    }

    public void setGenPFactor(double genPFactor) {
        this.genPFactor = genPFactor;
    }

    public double getGenQFactor() {
        return genQFactor;
    }

    public void setGenQFactor(double genQFactor) {
        this.genQFactor = genQFactor;
    }
}

