package zju.hems;

/**
 * 报价类
 * @author Xu Chengsi
 * @date 2019/8/15
 */
public class Offer {
    String userId;  // 用户id
    double price;   // 报价
    double maxPeakShaveRatio;    // 上报的削峰量占应削峰量比例
    double peakShaveCapRatio;   // 用户应削峰容量占总削峰容量的比例

    public Offer(String userId, double price, double maxPeakShaveRatio, double peakShaveCapRatio) {
        this.userId = userId;
        this.price = price;
        this.maxPeakShaveRatio = maxPeakShaveRatio;
        this.peakShaveCapRatio = peakShaveCapRatio;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public double getPrice() {
        return price;
    }

    public void setPrice(double price) {
        this.price = price;
    }

    public double getMaxPeakShaveRatio() {
        return maxPeakShaveRatio;
    }

    public void setMaxPeakShaveRatio(double maxPeakShaveRatio) {
        this.maxPeakShaveRatio = maxPeakShaveRatio;
    }

    public double getPeakShaveCapRatio() {
        return peakShaveCapRatio;
    }

    public void setPeakShaveCapRatio(double peakShaveCapRatio) {
        this.peakShaveCapRatio = peakShaveCapRatio;
    }
}
