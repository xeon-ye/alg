package zju.util;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-9-19
 */
public class MathUtil {

    public static void trans_rect2polar(double[] toTrans) {
        double a = toTrans[0];
        double b = toTrans[1];
        toTrans[0] = Math.sqrt(a * a + b * b);
        toTrans[1] = Math.atan2(b, a);
    }

    public static void trans_polar2rect(double[] toTrans) {
        double a = toTrans[0];
        double b = toTrans[1];
        toTrans[0] = a * Math.cos(b);
        toTrans[1] = a * Math.sin(b);
    }

    /**
     * @param toTrans source value
     * @param result  target array to store the result
     */
    public static void trans_polar2rect(double[] toTrans, double[] result) {
        double a = toTrans[0];
        double b = toTrans[1];
        result[0] = a * Math.cos(b);
        result[1] = a * Math.sin(b);
    }

    /**
     * @param toTrans the rectangular coordinates values
     */
    public static void trans_rect2polar(double[][] toTrans) {
        for (int i = 0; i < 3; i++) {
            double a = toTrans[i][0];
            double b = toTrans[i][1];
            toTrans[i][0] = Math.sqrt(a * a + b * b);
            toTrans[i][1] = Math.atan2(b, a);
        }
    }

    /**
     * conversion of polar coordinates (v theta) to rectangular coordinates (x,?0?2y)
     *
     * @param toTrans the polar coordinates values
     */
    public static void trans_polar2rect(double[][] toTrans) {
        for (int i = 0; i < 3; i++) {
            double a = toTrans[i][0];
            double b = toTrans[i][1];
            toTrans[i][0] = a * Math.cos(b);
            toTrans[i][1] = a * Math.sin(b);
        }
    }

    /**
     * conversion of polar coordinates (r,?0?2theta) to rectangular coordinates (x,?0?2y)
     *
     * @param toTrans the polar coordinates values
     * @param result  storage of lineImpedance
     */
    public static void trans_polar2rect(double[][] toTrans, double[][] result) {
        for (int i = 0; i < 3; i++) {
            result[i] = new double[2];
            double a = toTrans[i][0];
            double b = toTrans[i][1];
            result[i][0] = a * Math.cos(b);
            result[i][1] = a * Math.sin(b);
        }
    }

    /**
     * @param sorted the array to be searched, must be sorted, and leftist is smallest.
     * @param aim    the value to find in the sorted array
     * @param isLeft if true then search leftist value, if false search rightist
     * @return index as a search result correspond to value aim and option isLeft.
     */
    public static int binarySearchInt(int[] sorted, int aim, boolean isLeft) {
        int left = 0;
        int right = sorted.length - 1;
        int temp;

        while (left < right) {
            if (isLeft) {
                temp = (left + right) / 2;
                if (aim > sorted[temp]) {
                    left = temp + 1;
                } else {
                    right = temp;
                }
            } else {
                temp = (left + right) / 2 + 1;
                if (aim < sorted[temp]) {
                    right = temp - 1;
                } else {
                    left = temp;
                }
            }
        }
        if (sorted[left] == aim) {
            return left;
        } else {
            return -1;
        }
    }

    public static int binarySearchInt(int[] sorted, int aim) {
        return binarySearchInt(sorted, aim, true);
    }

    //added by Mikloyan, for degree transformation
    public static void trans_rect2polar_deg(double[][] toTrans) {
        for (int i = 0; i < 3; i++) {
            double a = toTrans[i][0];
            double b = toTrans[i][1];
            toTrans[i][0] = Math.sqrt(a * a + b * b);
            toTrans[i][1] = Math.atan2(b, a) * 180. / Math.PI;
        }
    }

    //added by mikoyan, for degree transformation.
    public static void trans_polar2rect_deg(double[][] toTrans) {
        for (int i = 0; i < 3; i++) {
            double a = toTrans[i][0];
            double b = toTrans[i][1];
            toTrans[i][0] = a * Math.cos(b * Math.PI / 180.0);
            toTrans[i][1] = a * Math.sin(b * Math.PI / 180.0);
        }
    }
}
