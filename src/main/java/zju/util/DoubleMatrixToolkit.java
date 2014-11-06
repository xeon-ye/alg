package zju.util;

/**
 * Created by IntelliJ IDEA.
 * User: MIkoyan
 * Date: 2010-7-9
 * Time: 8:04:33
 */
public class DoubleMatrixToolkit {
    public static boolean isDoubleMatrixEqual(double[][] a, double[][] b, double eps) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                if (Math.abs(a[i][j] - b[i][j]) > eps) {
                    return false;
                }
            }
        }
        return true;
    }

    public static boolean isDoubleMatrixEqual(double[][] a, double[][] b) {
        return isDoubleMatrixEqual(a, b, 1e-6);
    }

    public static double[][] cloneDoubleMatrix(double[][] a) {
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++)
            System.arraycopy(a[i], 0, c[i], 0, a[0].length);
        return c;
    }

    public static double[][] negDoubleMatrix(double[][] a) {
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++)
                c[i][j] = -a[i][j];
        }
        return c;
    }

    public static double[][] transposeDoubleMatrix(double[][] a) {
        double[][] c = new double[a[0].length][a.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++)
                c[j][i] = -a[i][j];
        }
        return c;
    }

    public static void selfAdd(double[][] a, double[][] b) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++)
                a[i][j] += b[i][j];
        }
    }

    public static double[][] add(double[][] a, double[][] b) {
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++)
                c[i][j] = a[i][j] + b[i][j];
        }
        return c;
    }

    public static double[][] sub(double[][] a, double[][] b) {
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++)
                c[i][j] = a[i][j] - b[i][j];
        }
        return c;
    }

    public static void selfSub(double[][] a, double[][] b) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++)
                a[i][j] = a[i][j] - b[i][j];
        }
    }

    public static double[][] mul(double[][] a, double[][] b) {
        if (a[0].length != b.length) {
            return null;
        }
        double[][] c = new double[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++)
                for (int k = 0; k < a[0].length; k++)
                    c[i][j] += a[i][k] * b[k][j];
        }
        return c;
    }

    public static double[][] complexMul(double[][] a, double[][] b) {
        if (a[0].length != b.length * 2) {
            return null;
        }
        double[][] c;
        double[][] realA = submatDoubleMatrix(a, indIncr(a.length), indIncr(a[0].length / 2));
        double[][] imagA = submatDoubleMatrix(a, indIncr(a.length), indIncr(a[0].length / 2, a[0].length));
        double[][] realB = submatDoubleMatrix(b, indIncr(b.length), indIncr(b[0].length / 2));
        double[][] imagB = submatDoubleMatrix(b, indIncr(b.length), indIncr(b[0].length / 2, b[0].length));
        double[][] realC = sub(mul(realA, realB),
                mul(imagA, imagB));
        double[][] imagC = add(mul(realA, imagB),
                mul(imagA, realB));
        c = mergeDoubleMatrixByCol(realC, imagC);
        return c;
    }

    public static void selfMul(double[][] a, double b) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                a[i][j] *= b;
            }
        }
    }

    public static double[][] mul(double[][] a, double b) {
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] * b;
            }
        }
        return c;
    }

    public static double[][] mergeDoubleMatrixByRow(double[][] a, double[][] b) {
        if (a[0].length != b[0].length) {
            return null;
        }
        double[][] c = new double[a.length + b.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            System.arraycopy(a[i], 0, c[i], 0, a[0].length);
        }
        for (int i = 0; i < b.length; i++) {
            System.arraycopy(b[i], 0, c[i + a.length], 0, b[0].length);
        }
        return c;
    }

    public static double[][] mergeDoubleMatrixByCol(double[][] a, double[][] b) {
        if (a.length != b.length) {
            return null;
        }
        double[][] c = new double[a.length][a[0].length + b[0].length];
        for (int i = 0; i < a.length; i++) {
            System.arraycopy(a[i], 0, c[i], 0, a[0].length);
            System.arraycopy(b[i], 0, c[i], a[0].length, b[0].length);
        }
        return c;
    }

    public static double[][] submatDoubleMatrix(double[][] a, int[] row, int[] col) {
        double[][] b = new double[row.length][col.length];
        for (int i = 0; i < row.length; i++) {
            if (row[i] >= a.length) {
                return null;
            }
            for (int j = 0; j < col.length; j++) {
                if (col[j] >= a[0].length) {
                    return null;
                }
                b[i][j] = a[row[i]][col[j]];
            }
        }
        return b;
    }

    public static int[] indIncr(int n) {
        int[] c = new int[n];
        for (int i = 0; i < n; i++) {
            c[i] = i;
        }
        return c;
    }

    public static int[] indIncr(int start, int end) {
        int[] c = new int[Math.abs(start - end)];
        for (int i = 0; i < Math.abs(start - end); i++) {
            c[i] = start + (start < end ? i : -i);
        }
        return c;
    }

    public static double[] incr(double head, double tail, int n) {
        double[] c = new double[n];
        double step = (tail - head) / n;
        for (int i = 0; i < n; i++) {
            c[i] = head + step * i;
        }
        return c;
    }

    public static double[][] eye(int m, int n) {
        double[][] c = new double[m][n];
        for (int i = 0; i < m && i < n; i++) {
            c[i][i] = 1.0;
        }
        return c;
    }

    public static double[][] eye(int n) {
        double[][] c = new double[n][n];
        for (int i = 0; i < n; i++) {
            c[i][i] = 1.0;
        }
        return c;
    }

    public static double[][] zero(int m, int n) {
        return new double[m][n];
    }

    public static double[][] zero(int n) {
        return new double[n][n];
    }

    public static double[][] diag(double[] a) {
        double[][] c = new double[a.length][a.length];
        for (int i = 0; i < a.length; i++) {
            c[i][i] = a[i];
        }
        return c;
    }

    public static double[] diag(double[][] a) {
        int n = a.length < a[0].length ? a.length : a[0].length;
        double[] diagonal = new double[n];
        for (int i = 0; i < n; i++) {
            diagonal[i] = a[i][i];
        }
        return diagonal;
    }

    public static double[][] cart2pol_Deg(double[][] cart) {
        double[][] pol = new double[cart.length][2];
        for (int i = 0; i < cart.length; i++) {
            pol[i][0] = Math.sqrt(cart[i][0] * cart[i][0] + cart[i][1] * cart[i][1]);
            pol[i][1] = Math.atan2(cart[i][1], cart[i][0]) * 180.0 / Math.PI;
        }
        return pol;
    }

    public static double[][] pol2cart_Deg(double[][] pol) {
        double[][] cart = new double[pol.length][2];
        for (int i = 0; i < pol.length; i++) {
            cart[i][0] = pol[i][0] * Math.cos(pol[i][1] * Math.PI / 180.0);
            cart[i][1] = pol[i][0] * Math.sin(pol[i][1] * Math.PI / 180.0);
        }
        return cart;
    }

    public static void makeZero(double[][] a) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[i].length; j++)
                a[i][j] = 0.0;
        }
    }
}
