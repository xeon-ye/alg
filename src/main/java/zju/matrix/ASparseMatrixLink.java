package zju.matrix;

import cern.colt.matrix.DoubleMatrix2D;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-5
 */
public class ASparseMatrixLink extends AbstractMatrix implements Serializable {
    private static final long serialVersionUID = -3831910468817616627l;
    private static Logger log = LogManager.getLogger(ASparseMatrixLink.class);
    //非零元值
    protected List<Double> VA;
    //非零元列号
    protected List<Integer> JA;
    //链表
    protected List<Integer> LINK;
    //first position of each row in VA and its length is m
    protected int[] IA;
    //total non-zero elements' num of a row
    protected int[] NA;

    public ASparseMatrixLink() {
    }

    public ASparseMatrixLink(int m, int n, int nnz) {
        super(m, n);
        IA = new int[m];
        NA = new int[m];
        for (int i = 0; i < m; i++) {
            IA[i] = -1;
            NA[i] = 0;
        }
        VA = new ArrayList<>(nnz);
        JA = new ArrayList<>(nnz);
        LINK = new ArrayList<>(nnz);
    }

    public ASparseMatrixLink(int n) {
        this(n, n);
    }

    public ASparseMatrixLink(int m, int n) {
        super(m, n);
        IA = new int[m];
        NA = new int[m];
        for (int i = 0; i < m; i++) {
            IA[i] = -1;
            NA[i] = 0;
        }
        VA = new ArrayList<>();
        JA = new ArrayList<>();
        LINK = new ArrayList<>();
    }

    public ASparseMatrixLink(int m, int n, List<Double> VA,
                             List<Integer> JA, List<Integer> LINK,
                             int[] IA, int[] NA) {
        super(m, n);
        this.VA = VA;
        this.JA = JA;
        this.LINK = LINK;
        this.IA = IA;
        this.NA = NA;
    }

    public void increase(int i, int j, double value) {
        setValue(i, j, value, true);
    }

    public void setValue(int i, int j, double value) {
        setValue(i, j, value, false);
    }

    public void setValue(int row, int col, double value, boolean isAddTo) {
        if (row >= m || col >= n) {
            log.error("(" + row + "," + col + ") out of range of matrix " + m + " * " + n);
            return;
        }
        int posInLink = 0;
        int tmp2 = -1;
        for (int k = 0; k < NA[row]; k++) {
            if (k == 0) posInLink = IA[row];
            else posInLink = LINK.get(posInLink);
            if (JA.get(posInLink) < col)
                tmp2 = posInLink;
            else if (JA.get(posInLink) == col) {
                if (!isAddTo)
                    VA.set(posInLink, value);
                else {
                    Double old = VA.get(posInLink);
                    VA.set(posInLink, old + value);
                }
                return;
            } else
                break;
        }

        NA[row]++;
        int pos = LINK.size();
        if (tmp2 > -1) {
            int a = LINK.remove(tmp2);
            LINK.add(tmp2, pos);
            LINK.add(a);
        } else {
            if (NA[row] == 1)
                LINK.add(-1);
            else
                LINK.add(IA[row]);
        }

        JA.add(col);
        VA.add(value);
        if (tmp2 == -1)
            IA[row] = pos;
    }

    public double getValue(int i, int j) {
        int k = getIA()[i];
        while (k != -1) {
            if (j == getJA().get(k))
                return getVA().get(k);
            k = getLINK().get(k);
        }
        return 0;
    }

    public boolean isExist(int i, int j) {
        int k = getIA()[i];
        while (k != -1) {
            if (j == getJA().get(k))
                return true;
            k = getLINK().get(k);
        }
        return false;
    }

    public AVector mul(AVector v) {
        if (this.getN() != v.getN()) {
            log.error("length is not equal: " + this.getN() + " != " + v.getN());
        }
        AVector result = new AVector(this.getM());
        for (int i = 0; i < getM(); i++) {
            int k = getIA()[i];
            double d = 0;
            while (k != -1) {
                int j = getJA().get(k);
                double value = getVA().get(k);
                d += value * v.getValues()[j];
                k = getLINK().get(k);
            }
            result.setValue(i, d);
        }
        return result;
    }

    public AbstractMatrix mul(AbstractMatrix m) {
        if (this.getN() != m.getM()) {
            log.error(m.getM() + "*" + m.getN() + " matrix can not multiple with " + this.getM() + "*" + this.getN() + " matrix");
            return null;
        }
        ASparseMatrixLink result = new ASparseMatrixLink(this.getM(), m.getN());
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
        ASparseMatrixLink matrix = new ASparseMatrixLink(getM(), getN());
        for (double v : VA)
            matrix.getVA().add(v);
        System.arraycopy(IA, 0, matrix.getIA(), 0, IA.length);
        System.arraycopy(NA, 0, matrix.getNA(), 0, NA.length);
        for (int i : LINK)
            matrix.getLINK().add(i);
        for (int i : JA)
            matrix.getJA().add(i);
        return matrix;
    }

    public ASparseMatrixLink inv() {
        ASparseMatrixLU lu = ASparseMatrixLU.formMatrix(this);
        lu.ldu();
        ASparseMatrixLink inv = new ASparseMatrixLink(this.getM(), this.getN());
        lu.inv(inv);
        return inv;
    }

    public ASparseMatrixLink transpose() {
        ASparseMatrixLink matrix = new ASparseMatrixLink(this.getN(), this.getM());
        for (int i = 0; i < getM(); i++) {
            int k = getIA()[i];
            while (k != -1) {
                int j = getJA().get(k);
                double v = getVA().get(k);
                matrix.setValue(j, i, v);
                k = getLINK().get(k);
            }
        }
        return matrix;
    }

    public AVector getRowVector(int i) {
        AVector v = new AVector(this.getN());
        int k = getIA()[i];
        while (k != -1) {
            int j = getJA().get(k);
            v.setValue(j, getVA().get(k));
            k = getLINK().get(k);
        }
        return v;
    }

    public AVector getColVector(int index) {
        AVector v = new AVector(this.getM());
        for (int i = 0; i < getM(); i++) {
            int k = getIA()[i];
            while (k != -1) {
                int j = getJA().get(k);
                if (j == index) {
                    v.setValue(i, getVA().get(k));
                    break;
                }
                k = getLINK().get(k);
            }
        }
        return v;
    }

    public void printOnScreen2() {
        System.out.println("========= matrix ==========");
        int j;
        for (int i = 0; i < getM(); i++) {
            int k = getIA()[i];
            j = -1;
            while (k != -1) {
                double v = getVA().get(k);
                for(int m = j + 1; m < getJA().get(k); m++)
                    System.out.print("0\t");
                System.out.print(v);
                j = getJA().get(k);
                k = getLINK().get(k);
                if(j != getN() - 1)
                    System.out.print("\t");
            }
            for(int m = j + 1; m < getN(); m++) {
                System.out.print("0");
                if(m != getN() - 1)
                    System.out.print("\t");
            }
            System.out.println();
        }
    }

    public void printOnScreen() {
        System.out.println("========= matrix ==========");
        for (int i = 0; i < getM(); i++) {
            int k = getIA()[i];
            while (k != -1) {
                int j = getJA().get(k);
                double v = getVA().get(k);
                //System.out.println("(" + i + "," + j + ")\t" + v);
                System.out.println(i + "\t" + j + "\t" + v);
                k = getLINK().get(k);
            }
        }
    }

    public void symboAnalysis() {
        for (int i = 0; i < getM() - 1; i++) {
            int k = getIA()[i];
            while (k != -1) {
                int j = getJA().get(k);
                if (i >= j) {
                    k = getLINK().get(k);
                    continue;
                }
                for (int anotherI = i + 1; anotherI < getM(); anotherI++) {
                    if (isExist(anotherI, i) && !isExist(anotherI, j))
                        setValue(anotherI, j, 0.0);
                }
                k = getLINK().get(k);
            }
        }
    }

    //
    public ASparseMatrixLink subMatrix(int rowStart, int rowEnd, int colStart, int colEnd) {
        ASparseMatrixLink result = new ASparseMatrixLink(rowEnd - rowStart + 1, colEnd - colStart + 1);
        for (int i = rowStart; i <= rowEnd; i++) {
            int k = getIA()[i];
            while (k != -1) {
                int j = getJA().get(k);
                if (j > colEnd)
                    break;
                double v = getVA().get(k);
                k = getLINK().get(k);
                if (j < colStart)
                    continue;
                result.setValue(i - rowStart, j - colStart, v);
            }
        }
        return result;
    }

    public List<Double> getVA() {
        return VA;
    }

    public List<Integer> getJA() {
        return JA;
    }

    public List<Integer> getLINK() {
        return LINK;
    }

    public int[] getIA() {
        return IA;
    }

    public int[] getNA() {
        return NA;
    }

    public DoubleMatrix2D toColtSparseMatrix() {
        DoubleMatrix2D r = new MySparseDoubleMatrix2D(m, n, getVA().size(), 0.2, 0.5);
        for (int i = 0; i < getM(); i++) {
            int k = getIA()[i];
            while (k != -1) {
                int j = getJA().get(k);
                double v = getVA().get(k);
                r.setQuick(i, j, v);
                k = getLINK().get(k);
            }
        }
        return r;
    }

    public void getSluStrucNR(int[] asub, int[] xa) {
        xa[0] = 0;
        for (int i = 1; i < getM() + 1; i++)
            xa[i] = xa[i - 1] + getNA()[i - 1];

        int index = 0;
        for (int i = 0; i < getM(); i++) {
            if (getIA()[i] != -1) {
                int k = getIA()[i];
                while (k != -1) {
                    asub[index] = getJA().get(k);
                    k = getLINK().get(k);
                    index++;
                }
            }
        }
    }

    public void toColteMatrix(DoubleMatrix2D result) {
        for (int i = 0; i < getM(); i++) {
            int k = getIA()[i];
            while (k != -1) {
                int j = getJA().get(k);
                double v = getVA().get(k);
                k = getLINK().get(k);
                result.setQuick(i, j, v);
            }
        }
    }
}
