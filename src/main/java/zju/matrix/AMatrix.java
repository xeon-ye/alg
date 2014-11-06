package zju.matrix;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-5
 */
public class AMatrix extends AbstractMatrix {
    double[][] values;

    /**
     * construct a m*n matrix
     *
     * @param m size of row
     * @param n size of column
     */
    public AMatrix(int m, int n) {
        super(m, n);
        values = new double[m][n];
    }

    public AVector mul(AVector v) {
        return null;  //todo: finish it
    }

    public AbstractMatrix mul(AbstractMatrix m) {
        return null;  //todo: finish it
    }

    public AbstractMatrix inv() {
        return null;  //todo: finish it
    }

    public AbstractMatrix transpose() {
        return null;  //todo: finish it
    }

    public AVector getRowVector(int index) {
        return null; //todo: finish it
    }

    public AVector getColVector(int index) {
        return null;  //todo: finish it
    }

    public void printOnScreen() {
        for (int i = 0; i < this.getM(); i++) {
            for (int j = 0; j < this.getN(); j++)
                System.out.print(getValue(i, j) + "\t");
            System.out.println();
        }
    }

    public void setValue(int i, int j, double v) {
        values[i][j] = v;
    }

    public double getValue(int i, int j) {
        return values[i][j];
    }

    public AbstractMatrix cloneMatrix() {
        AMatrix matrix = new AMatrix(this.getM(), this.getN());
        for (int i = 0; i < this.getM(); i++) {
            for (int j = 0; j < this.getN(); j++)
                matrix.setValue(i, j, getValue(i, j));
        }
        return matrix;
    }

    public static AMatrix formMatrix(ASparseMatrixLink m) {
        AMatrix matrix = new AMatrix(m.getM(), m.getN());
        for (int i = 0; i < m.getM(); i++) {
            int k = m.getIA()[i];
            while (k != -1) {
                int j = m.getJA().get(k);
                double v = m.getVA().get(k);
                matrix.setValue(i, j, v);
                k = m.getLINK().get(k);
            }
        }
        return matrix;
    }
}
