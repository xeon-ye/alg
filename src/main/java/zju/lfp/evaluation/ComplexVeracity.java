package zju.lfp.evaluation;

import java.util.Calendar;
import java.util.List;

/**
 * Created by luojun
 * User: luojun
 * Date: 2007-5-16
 * Time: 11:43:21
 */
public class ComplexVeracity {
    private Calendar date;

    //平均准确率
    private double avgVeracity;
    //最大点准确率
    private double maxVeracity;
    //最小点准确率
    private double minVeracity;
    //极值准确率
    private double extVeracity;
    //综合准确率
    private double intVeracity;


    public ComplexVeracity() {
        this.date = null;
        this.avgVeracity = 0;
        this.maxVeracity = 0;
        this.minVeracity = 0;
        this.extVeracity = 0;
        this.intVeracity = 0;
    }

    // 注意顺序
    public ComplexVeracity(double avgVeracity, double maxVeracity, double minVeracity, double extVeracity, double intVeracity) {
        this.date = null;
        this.avgVeracity = avgVeracity;
        this.maxVeracity = maxVeracity;
        this.minVeracity = minVeracity;
        this.extVeracity = extVeracity;
        this.intVeracity = intVeracity;
    }

    // 注意顺序
    public ComplexVeracity(Calendar date, double avgVeracity, double maxVeracity, double minVeracity, double extVeracity, double intVeracity) {
        this.date = date;
        this.avgVeracity = avgVeracity;
        this.maxVeracity = maxVeracity;
        this.minVeracity = minVeracity;
        this.extVeracity = extVeracity;
        this.intVeracity = intVeracity;
    }

    //==================================

    public Calendar getDate() {
        return date;
    }

    public void setDate(Calendar date) {
        this.date = date;
    }

    public double getMaxVeracity() {
        return maxVeracity;
    }

    public void setMaxVeracity(double maxVeracity) {
        this.maxVeracity = maxVeracity;
    }

    public double getMinVeracity() {
        return minVeracity;
    }

    public void setMinVeracity(double minVeracity) {
        this.minVeracity = minVeracity;
    }

    public double getAvgVeracity() {
        return avgVeracity;
    }

    public void setAvgVeracity(double avgVeracity) {
        this.avgVeracity = avgVeracity;
    }

    public double getExtVeracity() {
        return extVeracity;
    }

    public void setExtVeracity(double extVeracity) {
        this.extVeracity = extVeracity;
    }

    public double getIntVeracity() {
        return intVeracity;
    }

    public void setIntVeracity(double intVeracity) {
        this.intVeracity = intVeracity;
    }

    public static ComplexVeracity calAvgComplexVeracity(List<ComplexVeracity> cvList) {
        if (cvList == null || cvList.size() == 0) return null;
        double varDayPrecision = 0;
        double varMaxPrecision = 0;
        double varMinPrecision = 0;
        double varExtremPrecision = 0;
        double varIntPrecision = 0;
        int count = 0;
        for (ComplexVeracity aCvList : cvList) {
            if (aCvList.getAvgVeracity() != SeriesEvaluation.SPECIAL_VERACITY
                    && !Double.isNaN(aCvList.getAvgVeracity())) {
                varDayPrecision += aCvList.getAvgVeracity();
                varMaxPrecision += aCvList.getMaxVeracity();
                varMinPrecision += aCvList.getMinVeracity();
                varExtremPrecision += aCvList.getExtVeracity();
                varIntPrecision += aCvList.getIntVeracity();
                count++;
            }
        }
        if(count == 0) return null;
        varDayPrecision /= count;
        varMaxPrecision /= count;
        varMinPrecision /= count;
        varExtremPrecision /= count;
        varIntPrecision /= count;
        ComplexVeracity avgCV = new ComplexVeracity();
        avgCV.setDate(cvList.get(0).getDate());
        avgCV.setAvgVeracity(varDayPrecision);
        avgCV.setMaxVeracity(varMaxPrecision);
        avgCV.setMinVeracity(varMinPrecision);
        avgCV.setExtVeracity(varExtremPrecision);
        avgCV.setIntVeracity(varIntPrecision);
        return avgCV;
    }
}
