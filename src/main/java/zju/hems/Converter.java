package zju.hems;

/**
 * 变流器
 */
public class Converter {

    double Effad;   // 交流转换为直流的转换效率
    double Effda;   // 直流转换为交流的转换效率

    public Converter(double Effad, double Effda) {
        this.Effad = Effad;
        this.Effda = Effda;
    }

    public double getEffad() {
        return Effad;
    }

    public void setEffad(double effad) {
        Effad = effad;
    }

    public double getEffda() {
        return Effda;
    }

    public void setEffda(double effda) {
        Effda = effda;
    }
}
