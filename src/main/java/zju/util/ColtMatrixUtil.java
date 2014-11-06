package zju.util;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import zju.matrix.ASparseMatrixLink;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-20
 */
public class ColtMatrixUtil {
    public static DoubleMatrix2D mergeMatrixByRow(DoubleMatrix2D[] src) {
        int rows = 0;
        int maxCol = 0;
        int nonZeroNum = 0;
        for (DoubleMatrix2D r : src) {
            rows += r.rows();
            nonZeroNum += r.cardinality();
            if (r.columns() > maxCol)
                maxCol = r.columns();
        }
        DoubleMatrix2D result = new MySparseDoubleMatrix2D(rows, maxCol, nonZeroNum, 0.2, 0.5);
        mergeMatrixByRow(src, result);
        return result;
    }

    public static void mergeMatrixByRow(DoubleMatrix2D[] src, DoubleMatrix2D target) {
        mergeMatrixByRow(src, target, 0);
    }

    public static void mergeMatrixByRow(DoubleMatrix2D[] src, DoubleMatrix2D target, int rowOffset) {
        for (DoubleMatrix2D r : src) {
            r.forEachNonZero(new MergeByRowFunction(target, rowOffset));
            rowOffset += r.rows();
        }
    }

    public static DoubleMatrix2D mergeMatrixByCol(DoubleMatrix2D[] src) {
        int cols = 0;
        int maxRow = 0;
        int nonZeroNum = 0;
        for (DoubleMatrix2D r : src) {
            cols += r.columns();
            nonZeroNum += r.cardinality();
            if (r.rows() > maxRow)
                maxRow = r.rows();
        }
        DoubleMatrix2D result = new MySparseDoubleMatrix2D(maxRow, cols, nonZeroNum, 0.2, 0.5);
        mergeMatrixByCol(src, result);
        return result;
    }

    public static void mergeMatrixByCol(DoubleMatrix2D[] src, DoubleMatrix2D target) {
        mergeMatrixByCol(src, target, 0);
    }

    public static void mergeMatrixByCol(DoubleMatrix2D[] src, DoubleMatrix2D target, int colOffset) {
        for (DoubleMatrix2D r : src) {
            r.forEachNonZero(new MergeByColFunction(target, colOffset));
            colOffset += r.columns();
        }
    }

    static class MergeByRowFunction implements IntIntDoubleFunction {
        int rowOffset;
        DoubleMatrix2D target;

        public MergeByRowFunction(DoubleMatrix2D target, int rowOffset) {
            this.target = target;
            this.rowOffset = rowOffset;
        }

        public double apply(int row, int col, double v) {
            target.setQuick(row + rowOffset, col, v);
            return v;
        }
    }

    static class MergeByColFunction implements IntIntDoubleFunction {
        int colOffset;
        DoubleMatrix2D target;

        public MergeByColFunction(DoubleMatrix2D target, int colOffset) {
            this.target = target;
            this.colOffset = colOffset;
        }

        public double apply(int row, int col, double v) {
            target.setQuick(row, col + colOffset, v);
            return v;
        }
    }

    public static AVector solveLU(ASparseMatrixLink a, AVector b) {
        DoubleMatrix2D left = trans2ColtMatrix(a);
        DoubleMatrix1D right = trans2ColtVector(b);
        if (solveLU(left, right)) {
            return trans(right, b.getN());
        } else
            return null;
    }

    public static boolean solveLU(DoubleMatrix2D a, DoubleMatrix1D b) {
        LUDecompositionQuick lud = new LUDecompositionQuick();
        lud.decompose(a);
        if (lud.isNonsingular())
            lud.solve(b);
        else
            return false;
        return true;
    }

    public static DoubleMatrix1D trans2ColtVector(AVector v) {
        DoubleMatrix1D resv = new DenseDoubleMatrix1D(v.getN());
        for (int i = 0; i < v.getN(); i++) {
            resv.setQuick(i, v.getValue(i));
        }
        return resv;
    }

    public static DoubleMatrix2D trans2ColtMatrix(ASparseMatrixLink source) {
        DoubleMatrix2D result = new SparseDoubleMatrix2D(source.getM(), source.getN());
        for (int i = 0; i < source.getM(); i++) {
            int k = source.getIA()[i];
            while (k != -1) {
                int j = source.getJA().get(k);
                double v = source.getVA().get(k);
                k = source.getLINK().get(k);
                result.setQuick(i, j, v);
            }
        }
        return result;
    }

    public static AVector trans(DoubleMatrix1D b, int n) {
        AVector res = new AVector(n);
        for (int i = 0; i < n; i++)
            res.setValue(i, b.getQuick(i));
        return res;
    }

    public static void getSluMatrixNR(DoubleMatrix2D mat, double[] a, int[] asub, int[] xa) {
        int currentRow = 0;
        for(int i = 0; i < asub.length; i++) {
            if(i >= xa[currentRow + 1])
                currentRow++;
            int col = asub[i];
            a[i] = mat.getQuick(currentRow, col);
        }
    }

    public static void getSluMatrixNC(DoubleMatrix2D mat, double[] a, int[] asub, int[] xa) {
        int currentCol = 0;
        for(int i = 0; i < asub.length; i++) {
            if(i >= xa[currentCol + 1])
                currentCol++;
            int row = asub[i];
            if(row >= mat.rows() || currentCol >= mat.columns())
                System.out.println();
            a[i] = mat.getQuick(row, currentCol);
        }
    }

    public static void printOnScreen(DoubleMatrix2D mat) {
        mat.forEachNonZero(new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int j, double v) {
                System.out.println(i + "\t" + j + "\t" + v);
                return v;
            }
        });
    }

    public static void toMyMatrix(DoubleMatrix2D source, final ASparseMatrixLink target) {
        source.forEachNonZero(new IntIntDoubleFunction() {
            @Override
            public double apply(int i, int j, double v) {
                target.setValue(i, j, v);
                return v;
            }
        });
    }
}


