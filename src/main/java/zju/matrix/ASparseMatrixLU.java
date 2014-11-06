package zju.matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-6
 */
public class ASparseMatrixLU {

    List<Double> U = new ArrayList<Double>();//
    List<Integer> JU = new ArrayList<Integer>();//
    List<Integer> IU = new ArrayList<Integer>();//
    List<Double> L = new ArrayList<Double>(); //
    List<Integer> JL = new ArrayList<Integer>();//
    List<Integer> IL = new ArrayList<Integer>();//
    List<Double> D = new ArrayList<Double>();//

    int n;

    public ASparseMatrixLU(int n) {
        this.n = n;
        for (int i = 0; i < n; i++) {
            D.add(0.0);
        }
    }

    public void ldu() {
        int k, j, i;
        for (int p = 0; p < n - 1; p++) {
            for (k = IU.get(p); k <= (IU.get(p + 1) - 1); k++) {
                j = JU.get(k);
                U.set(k, U.get(k) / D.get(p));
                for (int l = JL.get(p); l <= (JL.get(p + 1) - 1); l++) {
                    i = IL.get(l);
                    if (i < j) {
                        for (int t = IU.get(i), r = 1; t <= IU.get(i + 1) - 1; t++, r++) {
                            if (JU.get(t) == j) {
                                int q = IU.get(i) - 1 + r;
                                U.set(q, U.get(q) - U.get(k) * L.get(l));
                            }
                        }
                    } else if (i > j) {
                        for (int t = JL.get(j), r = 1; t <= JL.get(j + 1) - 1; t++, r++) {
                            if (IL.get(t) == i) {
                                int q = JL.get(j) - 1 + r;
                                L.set(q, L.get(q) - U.get(k) * L.get(l));
                            }
                        }
                    } else {
                        D.set(i, D.get(i) - U.get(k) * L.get(l));
                    }
                }
            }
        }
        for (i = 0; i < n - 1; i++) {
            for (j = JL.get(i); j <= (JL.get(i + 1) - 1); j++) {
                L.set(j, L.get(j) / D.get(i));
            }
        }
    }

    public AVector solveEquation(AVector b) {
        int i, j, k;
        AVector x = new AVector(b.getN());
        AVector y = new AVector(b.getN());
        AVector z = new AVector(b.getN());
        z.assign(b);

        //todo: finis forward computation
        //for (k = 0; k < path.size(); k++) {
        //    i = path.get(k);
        //    if (i < n - 1) {
        //        for (j = JL.get(i); j <= (JL.get(i + 1) - 1); j++) {
        //            int ii = IL.get(j);
        //            z.setValue(ii, z.getValue(ii) - L.get(j) * z.getValue(path.get(k)));
        //        }
        //    }
        //}
        for (i = 0; i < n - 1; i++) {//todo
            for (j = i + 1; j < n; j++)
                z.setValue(j, z.getValue(j) - getLij(j, i) * z.getValue(i));
        }
        for (i = 0; i < n; i++) {//
            y.setValue(i, z.getValue(i) / D.get(i));
        }

        x.assign(y);//
        for (i = n - 2; i >= 0; i--) {
            for (k = (IU.get(i + 1) - 1); k >= IU.get(i); k--) {
                j = JU.get(k);
                x.setValue(i, x.getValue(i) - U.get(k) * x.getValue(j));
            }
        }
        return x;
    }

    public double getLij(int i, int j) {
        for (int k = JL.get(j); k <= (JL.get(j + 1) - 1); k++) {
            int row = IL.get(j);
            if (row == i)
                return L.get(k);
        }
        return 0.0;
    }

    public AbstractMatrix inv(AbstractMatrix target) {
        AVector vector;
        for (int i = 0; i < n; i++) {
            vector = MatrixMaker.zeros(n);
            vector.setValue(i, 1.0);
            AVector x = solveEquation(vector);
            for (int j = 0; j < n; j++) {
                if (Math.abs(x.getValue(j)) > 10e-6)
                    target.setValue(j, i, x.getValue(j));
            }
        }
        return target;
    }


    public List<Integer> getPath(int row) {
        List<Integer> path = new ArrayList<Integer>();
        int i = 0;
        path.add(row);
        while (row != (n - 1)) {
            if (IU.get(row + 1) > IU.get(row)) {
                i++;
                path.add(JU.get(IU.get(row)));
                row = path.get(i);
            }
        }
        return path;
    }

    public List<Double> getU() {
        return U;
    }

    public List<Integer> getJU() {
        return JU;
    }

    public List<Integer> getIU() {
        return IU;
    }

    public List<Double> getL() {
        return L;
    }

    public List<Integer> getJL() {
        return JL;
    }

    public List<Integer> getIL() {
        return IL;
    }

    public List<Double> getD() {
        return D;
    }

    public int getN() {
        return n;
    }

    public AbstractMatrix getUMatrix() {
        ASparseMatrixLink m = new ASparseMatrixLink(this.getN());
        for (int i = 0; i < getIU().size() - 1; i++) {
            for (int k = getIU().get(i); k < getIU().get(i + 1); k++) {
                int j = getJU().get(k);
                m.setValue(i, j, getU().get(k));
            }
        }
        for (int i = 0; i < getN(); i++)
            m.setValue(i, i, 1);
        return m;
    }

    public AbstractMatrix getLMatrix() {
        ASparseMatrixLink m = new ASparseMatrixLink(this.getN());
        for (int j = 0; j < getJL().size() - 1; j++) {
            for (int k = getJL().get(j); k < getJL().get(j + 1); k++) {
                int i = getIL().get(k);
                m.setValue(i, j, getL().get(k));
            }
        }
        for (int i = 0; i < getN(); i++)
            m.setValue(i, i, 1);
        return m;
    }

    public AbstractMatrix getDMatrix() {
        ASparseMatrixLink m = new ASparseMatrixLink(this.getN());
        for (int i = 0; i < getD().size(); i++)
            m.setValue(i, i, getD().get(i));
        return m;
    }

    public static ASparseMatrixLU formMatrix(ASparseMatrixLink link) {
        ASparseMatrixLink2D m;
        if (link instanceof ASparseMatrixLink2D)
            m = (ASparseMatrixLink2D) link;
        else
            m = ASparseMatrixLink2D.formMatrix(link);
        m.symboAnalysis();//todo
        ASparseMatrixLU matrix;
        if (m.isSymmetrical()) {
            matrix = new ASymmetricalMatrixLU(m.getM());
        } else
            matrix = new ASparseMatrixLU(m.getM());
        for (int i = 0; i < m.getM(); i++) {
            int k = m.getIA()[i];
            while (k != -1) {
                int j = m.getJA().get(k);
                double v = m.getVA().get(k);
                if (j > i) {
                    matrix.getU().add(v);
                    matrix.getJU().add(j);
                    if (matrix.getIU().size() <= i)
                        matrix.getIU().add(matrix.getU().size() - 1);
                } else {
                    matrix.getD().remove(i);
                    matrix.getD().add(i, v);
                }
                k = m.getLINK().get(k);
            }
        }
        for (int j = 0; j < m.getM(); j++) {
            int k = m.getJA2()[j];
            while (k != -1) {
                int i = m.getIA2().get(k);
                double v = m.getVA().get(k);
                if (j < i) {
                    matrix.getL().add(v);
                    matrix.getIL().add(i);
                    if (matrix.getJL().size() <= j)
                        matrix.getJL().add(matrix.getL().size() - 1);
                }
                k = m.getLINK2().get(k);
            }
        }
        int size = matrix.getIU().size();
        int max = matrix.getU().size();
        for (int i = size; i < m.getM(); i++)
            matrix.getIU().add(i, max);
        size = matrix.getJL().size();
        max = matrix.getL().size();
        for (int i = size; i < m.getM(); i++)//todo: this is just for n * n matrix
            matrix.getJL().add(i, max);

        return matrix;
    }

    public static AbstractMatrix formMatrix(ASparseMatrixLU maxtrix, AbstractMatrix m) {
        for (int i = 0; i < maxtrix.getIU().size() - 1; i++) {
            for (int k = maxtrix.getIU().get(i); k < maxtrix.getIU().get(i + 1); k++) {
                int j = maxtrix.getJU().get(k);
                m.setValue(i, j, maxtrix.getU().get(k));
            }
        }

        for (int j = 0; j < maxtrix.getJL().size() - 1; j++) {
            for (int k = maxtrix.getJL().get(j); k < maxtrix.getJL().get(j + 1); k++) {
                int i = maxtrix.getIL().get(k);
                m.setValue(i, j, maxtrix.getL().get(k));
            }
        }

        for (int i = 0; i < maxtrix.getD().size(); i++)
            m.setValue(i, i, maxtrix.getD().get(i));

        return m;
    }

    public void printOnScreen() {
        for (int i = 0; i < getIU().size() - 1; i++) {
            for (int k = getIU().get(i); k < getIU().get(i + 1); k++) {
                int j = getJU().get(k);
                System.out.println("(" + i + "," + j + ")\t" + getU().get(k));
            }
        }
        for (int j = 0; j < getJL().size() - 1; j++) {
            for (int k = getJL().get(j); k < getJL().get(j + 1); k++) {
                int i = getIL().get(k);
                System.out.println("(" + i + "," + j + ")\t" + getL().get(k));
            }
        }
        for (int i = 0; i < getD().size(); i++)
            System.out.println("(" + i + "," + i + ")\t" + getD().get(i));
    }
}
