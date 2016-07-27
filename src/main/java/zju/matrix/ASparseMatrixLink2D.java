package zju.matrix;

import org.apache.log4j.Logger;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-5
 */
public class ASparseMatrixLink2D extends ASparseMatrixLink implements Serializable {

    private static Logger log = Logger.getLogger(ASparseMatrixLink2D.class);

    private List<Integer> IA2; //非零元行号
    private List<Integer> LINK2; //link of column
    private int[] JA2; //每一列第一个非零元在VA中的位置
    private int[] NA2; //每一列非零元的个数

    public ASparseMatrixLink2D() {
    }

    public ASparseMatrixLink2D(int n) {
        this(n, n);
    }

    public ASparseMatrixLink2D(int m, int n, int nnz) {
        super(m, n, nnz);
        JA2 = new int[n];
        NA2 = new int[n];
        for (int i = 0; i < n; i++) {
            JA2[i] = -1;
            NA2[i] = 0;
        }
        IA2 = new ArrayList<>(nnz);
        LINK2 = new ArrayList<>(nnz);
    }

    public ASparseMatrixLink2D(int m, int n) {
        super(m, n);
        JA2 = new int[n];
        NA2 = new int[n];
        for (int i = 0; i < n; i++) {
            JA2[i] = -1;
            NA2[i] = 0;
        }
        IA2 = new ArrayList<>();
        LINK2 = new ArrayList<>();
    }

    public void setValue(int row, int col, double value, boolean isAddTo) {
        super.setValue(row, col, value, isAddTo);
        if (row >= m || col >= n) {
            log.error("(" + row + "," + col + ") out of range of matrix " + m + " * " + n);
            return;
        }
        int tmp = 0;
        int tmp2 = -1;
        for (int k = 0; k < NA2[col]; k++) {
            if (k == 0) tmp = JA2[col];
            else tmp = LINK2.get(tmp);
            if (IA2.get(tmp) == row)
                return;
            if (IA2.get(tmp) < row)
                tmp2 = tmp;
        }
        NA2[col]++;
        //
        int pos = LINK2.size();
        if (tmp2 > -1) {
            //int a = LINK2.get(tmp2);
            //LINK2.set(tmp2, pos);
            //LINK2.add(a);
            int a = LINK2.remove(tmp2);
            LINK2.add(tmp2, pos);
            LINK2.add(a);
        } else {
            if (NA2[col] == 1)
                LINK2.add(-1);
            else
                LINK2.add(JA2[col]);
        }
        IA2.add(row);
        if (tmp2 == -1)
            JA2[col] = pos;
    }

    public List<Integer> getIA2() {
        return IA2;
    }

    public List<Integer> getLINK2() {
        return LINK2;
    }

    public int[] getJA2() {
        return JA2;
    }

    public int[] getNA2() {
        return NA2;
    }


    public AVector getColVector(int j) {
        AVector v = new AVector(this.getM());
        int k = getJA2()[j];
        while (k != -1) {
            int i = getIA2().get(k);
            v.setValue(i, getVA().get(k));
            k = getLINK2().get(k);
        }
        return v;
    }

    public ASparseMatrixLink transpose() {
        ASparseMatrixLink2D matrix = new ASparseMatrixLink2D(getN(), getM(), getVA().size());
        for (double v : VA)
            matrix.getVA().add(v);

        System.arraycopy(IA, 0, matrix.getJA2(), 0, IA.length);
        System.arraycopy(NA, 0, matrix.getNA2(), 0, NA.length);
        for (int i : LINK) {
            matrix.getLINK2().add(i);
        }
        for (int i : JA)
            matrix.getIA2().add(i);

        for (int i : IA2)
            matrix.getJA().add(i);
        System.arraycopy(JA2, 0, matrix.getIA(), 0, JA2.length);
        System.arraycopy(NA2, 0, matrix.getNA(), 0, NA2.length);
        for (int i : LINK2)
            matrix.getLINK().add(i);
        return matrix;
    }

    public AbstractMatrix mul(AbstractMatrix m) {
        if (this.getN() != m.getM()) {
            log.error(m.getM() + "*" + m.getN() + " matrix can not multiple with " + this.getM() + "*" + this.getN() + " matrix");
            return null;
        }
        ASparseMatrixLink2D result = new ASparseMatrixLink2D(this.getM(), m.getN());
        for (int i = 0; i < getM(); i++) {
            for (int j = 0; j < m.getN(); j++) {
                AVector col = m.getColVector(j);
                int k = getIA()[i];
                double d = 0;
                while (k != -1) {
                    int c = getJA().get(k);
                    double value = getVA().get(k);
                    d += value * col.getValue(c);
                    k = getLINK().get(k);
                }
                result.setValue(i, j, d);
            }
        }
        return result;
    }

    public AbstractMatrix cloneMatrix() {
        ASparseMatrixLink2D matrix = new ASparseMatrixLink2D(getM(), getN(), getVA().size());
        for (double v : VA)
            matrix.getVA().add(v);

        System.arraycopy(IA, 0, matrix.getIA(), 0, IA.length);
        System.arraycopy(NA, 0, matrix.getNA(), 0, NA.length);
        for (int i : LINK)
            matrix.getLINK().add(i);
        for (int i : JA)
            matrix.getJA().add(i);

        for (int i : IA2)
            matrix.getIA2().add(i);
        System.arraycopy(JA2, 0, matrix.getJA2(), 0, JA2.length);
        System.arraycopy(NA2, 0, matrix.getNA2(), 0, NA2.length);
        for (int i : LINK2)
            matrix.getLINK2().add(i);
        return matrix;
    }

    public static ASparseMatrixLink2D formMatrix(ASparseMatrixLink m) {
        ASparseMatrixLink2D matrix = new ASparseMatrixLink2D(m.getM(), m.getN(), m.getVA().size());
        for (int i = 0; i < m.getM(); i++) {
            int k = m.getIA()[i];
            while (k != -1) {
                int j = m.getJA().get(k);
                matrix.setValue(i, j, m.getVA().get(k));
                k = m.getLINK().get(k);
            }
        }
        matrix.setSymmetrical(m.isSymmetrical());
        return matrix;
    }

    public void getSluStrucNC(int[] asub, int[] xa) {
        xa[0] = 0;
        for (int i = 1; i < getN() + 1; i++)
            xa[i] = xa[i - 1] + getNA2()[i - 1];

        int index = 0;
        for (int j = 0; j < getN(); j++) {
            int k = getJA2()[j];
            while (k != -1) {
                asub[index] = getIA2().get(k);
                k = getLINK2().get(k);
                index++;
            }
        }
    }

    public void getSluStrucNC(double[] a, int[] asub, int[] xa) {
        xa[0] = 0;
        for (int i = 1; i < getN() + 1; i++)
            xa[i] = xa[i - 1] + getNA2()[i - 1];

        int index = 0;
        for (int j = 0; j < getN(); j++) {
            int k = getJA2()[j];
            while (k != -1) {
                a[index] = getVA().get(k);
                asub[index] = getIA2().get(k);
                k = getLINK2().get(k);
                index++;
            }
        }
    }

    public void getSluStrucNC(double[] a) {
        int index = 0;
        for (int j = 0; j < getN(); j++) {
            int k = getJA2()[j];
            while (k != -1) {
                a[index] = getVA().get(k);
                k = getLINK2().get(k);
                index++;
            }
        }
    }
}
