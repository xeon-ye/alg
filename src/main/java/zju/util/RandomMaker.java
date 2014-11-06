package zju.util;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-12
 */
public class RandomMaker {

    public static double randomNorm(double miu, double sigma) {
        double n = 12;
        double x;
        //do {
        x = 0;
        for (int i = 0; i < n; i++)
            x = x + (Math.random());
        x = (x - n / 2) / (Math.sqrt(n / 12));
        x = miu + x * sigma;
        //} while (x < -10e-6 );
        return x;
    }

    public static double randomPoisson(double lamda) {
        double x = 0, b = 1, c = Math.exp(-lamda), u;
        do {
            u = Math.random();
            b *= u;
            if (b >= c)
                x++;
        } while (b >= c);
        return x;
    }
}
