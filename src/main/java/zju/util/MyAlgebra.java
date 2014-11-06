package zju.util;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

/**
 * Created by IntelliJ IDEA.
 * User: Liu Kaicheng
 * Date: 2009-12-26
 * Time: 17:33:10
 */
public class MyAlgebra extends Algebra {
    public double dotMult(DoubleMatrix2D A, DoubleMatrix2D B) {
        double a, b, value = 0;
        int i, j;
        for (i = 0; i < A.rows(); i++)
            for (j = 0; j < A.columns(); j++) {
                a = A.get(i, j);
                b = B.get(i, j);
                value += a * b;
            }
        return value;
    }

    public DoubleMatrix2D multNumber(DoubleMatrix2D A, double a) {   //B = a * A
        DoubleMatrix2D B = A.copy();
        for (int i = 0; i < A.rows(); i++)
            for (int j = 0; j < A.columns(); j++)
                B.set(i, j, a * A.get(i, j));
        return B;
    }

    public DoubleMatrix2D plus(DoubleMatrix2D A, DoubleMatrix2D B) {   //C = A + B
        DoubleMatrix2D C = A.copy();
        for (int i = 0; i < A.rows(); i++)
            for (int j = 0; j < A.columns(); j++)
                C.set(i, j, A.getQuick(i, j) + B.getQuick(i, j));
        return C;
    }

    public DoubleMatrix1D plus(DoubleMatrix1D A, DoubleMatrix1D B) {   //C = A + B
        DoubleMatrix1D C = A.copy();
        for (int i = 0; i < A.size(); i++)
            C.set(i, A.getQuick(i) + B.getQuick(i));
        return C;
    }

    public DoubleMatrix2D minus(DoubleMatrix2D A, DoubleMatrix2D B) {   //C = A - B
        DoubleMatrix2D C = A.copy();
        for (int i = 0; i < A.rows(); i++)
            for (int j = 0; j < A.columns(); j++)
                C.set(i, j, A.getQuick(i, j) - B.getQuick(i, j));
        return C;
    }

    public DoubleMatrix1D minus(DoubleMatrix1D A, DoubleMatrix1D B) {   //C = A - B
        DoubleMatrix1D C = A.copy();
        for (int i = 0; i < A.size(); i++)
            C.set(i, A.getQuick(i) - B.getQuick(i));
        return C;
    }

}
