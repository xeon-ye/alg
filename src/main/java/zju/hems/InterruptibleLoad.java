package zju.hems;

public class InterruptibleLoad {
    double a;
    double b;
    double maxP;

    public InterruptibleLoad(double a, double b, double maxP) {
        this.a = a;
        this.b = b;
        this.maxP = maxP;
    }

    public double getA() {
        return a;
    }

    public void setA(double a) {
        this.a = a;
    }

    public double getB() {
        return b;
    }

    public void setB(double b) {
        this.b = b;
    }

    public double getMaxP() {
        return maxP;
    }

    public void setMaxP(double maxP) {
        this.maxP = maxP;
    }
}
