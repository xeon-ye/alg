package zju.bpamodel;

import zju.ieeeformat.DataOutputFormat;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-11
 */
public class BpaFileRwUtil {
    private static final double NULL_DOUBLE = 0.0;
    private static final int NULL_INT = 0;
    private static final DataOutputFormat format = new DataOutputFormat();

    public static byte[] getTarget(byte[] src, int start, int end) {
        if (end > src.length)
            end = src.length;
        if(end < start)
            return "".getBytes();
        byte[] target = new byte[end - start];
        System.arraycopy(src, start, target, 0, end - start);
        return target;
    }

    public static double parseDouble(String src) {
        if (src.length() < 1)
            return NULL_DOUBLE;
        if(src.equals("."))
            return 0.0;
        if(src.startsWith("+"))
            src = src.substring(1);
        if(src.startsWith("-0") && (src.length() > 2 && src.charAt(2) != '.'))//note this
            src = "-0." + src.substring(1);
        if(src.equals(".") || src.equals("-."))
            return 0.0;
        return Double.parseDouble(src);
    }

    public static double parseDouble(String src, String fd) {
        if (src.length() < 1)
            return NULL_DOUBLE;
        if(src.startsWith("+"))
            src = src.substring(1);
        if(src.equals(".") || src.equals("-."))
            return 0.0;
        if(src.indexOf('.') != -1)
            return Double.parseDouble(src);
        int[] t = getDecimals(fd);
        if(src.length() >= t[1] && t[1] >= 0) {
            String s = src.substring(0, src.length() - t[1]) + "." + src.substring(src.length() - t[1]);
            return Double.parseDouble(s);
        } else
            return Double.parseDouble(src);
    }

    public static int parseInt(String src) {
        if (src.length() < 1)
            return NULL_INT;
        if(src.equals("."))
            return 0;
        return Integer.parseInt(src);
    }

    public static String getFormatStr(double d, String fd) {
        if(d == NULL_DOUBLE) {
            return format.format("", fd);
        } else {
            int[] t = getDecimals(fd);
            String s = format.format(d, (t[0] + 1) + "." + t[1]);
            int i = 1;
            if (s.charAt(0) == ' ') {
                return s.substring(1);
            } else {
                int index = s.indexOf(".");
                if(index != -1 && s.length() - index - 1 == t[1]) {
                    if(s.charAt(s.length() - 1) == '0')
                        return s.substring(0, s.length() - 1);
                    else
                        return s.substring(0, index) + s.substring(index + 1);
                } else
                    return s.substring(0, s.length() - 1);
            }
        }
    }

    public static int[] getDecimals(String fd) {
        int i = fd.indexOf('.', 0);
        int width = 0, decimals = 0;
        if (i > -1) {
            decimals = Integer.parseInt(fd.substring(i + 1));
            fd = fd.substring(0, i);
        }
        if (fd.length() > 0)
            width = Integer.parseInt(fd);
        return new int[]{width, decimals};
    }

    public static String getFormatStr(int d, String fd) {
        if(d == NULL_INT) {
            return format.format("", fd);
        } else {
            return format.format(d, fd);
        }
    }
}
