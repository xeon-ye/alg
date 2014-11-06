package zju.util;

import zju.matrix.ASparseMatrixLink;
import zju.matrix.AVector;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2009-3-14
 */
public class MatrixUtil {
    public static void formIJ(ASparseMatrixLink[] matrixes, int[] iRow, int[] jCol, int count, int rowStart) {
        for (ASparseMatrixLink H : matrixes) {
            formIJ(H, iRow, jCol, count, rowStart);
            count += H.getVA().size();
            rowStart += H.getM();
        }
    }

    public static int formIJ(ASparseMatrixLink matrix, int[] iRow, int[] jCol, int count, int rowStart) {
        for (int row = 0; row < matrix.getM(); row++) {
            int k = matrix.getIA()[row];
            while (k != -1) {
                int col = matrix.getJA().get(k);
                iRow[count] = row + rowStart;
                jCol[count++] = col;
                k = matrix.getLINK().get(k);
            }
        }
        return count;
    }

    public static int formIJ(ASparseMatrixLink matrix, int[] iRow, int[] jCol, int count, int rowStart, int columnStart) {
        for (int row = 0; row < matrix.getM(); row++) {
            int k = matrix.getIA()[row];
            while (k != -1) {
                int col = matrix.getJA().get(k);
                iRow[count] = row + rowStart;
                jCol[count++] = col + columnStart;
                k = matrix.getLINK().get(k);
            }
        }
        return count;
    }

    /**
     * @param m     matrix
     * @param iRow  row index
     * @param jCol  col index
     * @param count current index
     */
    public static void formIJWithEyes(ASparseMatrixLink m, int[] iRow, int[] jCol, int count) {
        formIJWithEyes(m, iRow, jCol, count, 0);
    }

    /**
     * @param m        matrix
     * @param iRow     row index
     * @param jCol     col index
     * @param count    current index
     * @param rowStart start of row
     */
    public static void formIJWithEyes(ASparseMatrixLink m, int[] iRow, int[] jCol, int count, int rowStart) {
        for (int row = 0; row < m.getM(); row++) {
            int k = m.getIA()[row];
            while (k != -1) {
                int col = m.getJA().get(k);
                iRow[count] = row + rowStart;
                jCol[count++] = col;
                k = m.getLINK().get(k);
            }
            iRow[count] = row + rowStart;
            jCol[count++] = m.getN() + row;
        }
    }

    public static void formValueWithEyes(ASparseMatrixLink m, double[] values, int count) {
        for (int row = 0; row < m.getM(); row++) {
            int k = m.getIA()[row];
            while (k != -1) {
                values[count++] = m.getVA().get(k);
                k = m.getLINK().get(k);
            }
            values[count++] = 1;
        }
    }

    public static void formValueWithEyes(ASparseMatrixLink m, double factor, double[] values, int count) {
        for (int row = 0; row < m.getM(); row++) {
            int k = m.getIA()[row];
            while (k != -1) {
                values[count++] = factor * m.getVA().get(k);
                k = m.getLINK().get(k);
            }
            values[count++] = 1;
        }
    }

    public static int formValue(ASparseMatrixLink m, double[] values, int count) {
        for (int row = 0; row < m.getM(); row++) {
            int k = m.getIA()[row];
            while (k != -1) {
                values[count++] = m.getVA().get(k);
                k = m.getLINK().get(k);
            }
        }
        return count;
    }

    public static void formIJ2WithEyes(ASparseMatrixLink[] matrixes, int[] iRow, int[] jCol, int count) {
        int colStart;
        for (int row = 0; row < matrixes[0].getM(); row++) {
            colStart = 0;
            for (ASparseMatrixLink H : matrixes) {
                int k = H.getIA()[row];
                while (k != -1) {
                    int col = H.getJA().get(k);
                    iRow[count] = row;
                    jCol[count++] = col + colStart;
                    k = H.getLINK().get(k);
                }
                colStart += H.getN();
            }
            iRow[count] = row;
            jCol[count++] = colStart + row;
        }
    }

    public static void formValue2WithEyes(ASparseMatrixLink[] matrixes, double[] values, int count) {
        for (int row = 0; row < matrixes[0].getM(); row++) {
            for (ASparseMatrixLink H : matrixes) {
                int k = H.getIA()[row];
                while (k != -1) {
                    values[count++] = H.getVA().get(k);
                    k = H.getLINK().get(k);
                }
            }
            values[count++] = 1;
        }
    }

    public static double multi(AVector a, AVector b) {
        if (a.getN() == b.getN()) {
            double res = 0.0;
            for (int i = 0; i < a.getN(); i++)
                res += a.getValue(i) * b.getValue(i);
            return res;
        } else
            return 0.0;
    }
}
