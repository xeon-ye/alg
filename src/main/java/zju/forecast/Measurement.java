package zju.forecast;

import java.util.Date;

public class Measurement {
    private Date dtime;
    private double val;

    public Date getDtime() {
        return dtime;
    }

    public void setDtime(Date dtime) {
        this.dtime = dtime;
    }

    public double getVal() {
        return val;
    }

    public void setVal(double val) {
        this.val = val;
    }
}
