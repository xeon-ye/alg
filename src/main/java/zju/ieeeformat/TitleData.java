package zju.ieeeformat;

import java.io.Serializable;

/**
 * Class TitleData
 * <p> title data in ieee common data format</P>
 * Copyright (c) Dong Shufeng
 *
 * @author Dong Shufeng
 * @version 1.0
 * @Date: 2006-8-29
 */
public class TitleData implements Serializable, Cloneable {
    private String date = ""; //format: DD/MM/YY

    private String originatorName = "";

    private double mvaBase;

    private int year;

    private char season;

    private String caseIdentification = "";

    public String getDate() {
        return date;
    }

    /**
     * @param date format: DD/MM/YY
     */
    public void setDate(String date) {
        this.date = date;
    }

    public String getOriginatorName() {
        return originatorName;
    }

    public void setOriginatorName(String originatorName) {
        this.originatorName = originatorName;
    }

    public double getMvaBase() {
        return mvaBase;
    }

    public void setMvaBase(double mvaBase) {
        this.mvaBase = mvaBase;
    }

    public int getYear() {
        return year;
    }

    public void setYear(int year) {
        this.year = year;
    }

    public char getSeason() {
        return season;
    }

    public void setSeason(char season) {
        this.season = season;
    }

    public String getCaseIdentification() {
        return caseIdentification;
    }

    public void setCaseIdentification(String caseIdentification) {
        this.caseIdentification = caseIdentification;
    }

    @SuppressWarnings({"CloneDoesntDeclareCloneNotSupportedException"})
    public TitleData clone() {
        try {
            return (TitleData) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
        }
        return null;
    }
}
