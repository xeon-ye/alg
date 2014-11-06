package zju.matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-10
 */
public class ASymmetricalMatrixLU extends ASparseMatrixLU {

    public ASymmetricalMatrixLU(int n) {
        super(n);
    }

    public AVector solveEquation(AVector b, List<Integer> path) {
        int i, j, k;
        AVector x = new AVector(b.getN());
        AVector y = new AVector(b.getN());
        AVector z = new AVector(b.getN());
        z.assign(b);

        for (k = 0; k < path.size(); k++) {
            i = path.get(k);
            if (i < n - 1) {
                for (j = JL.get(i); j <= (JL.get(i + 1) - 1); j++) {
                    int ii = IL.get(j);
                    z.setValue(ii, z.getValue(ii) - L.get(j) * z.getValue(path.get(k)));
                }
            }
        }
        for (i = 0; i < n; i++) {
            y.setValue(i, z.getValue(i) / D.get(i));
        }

        x.assign(y);
        for (i = n - 2; i >= 0; i--) {
            for (k = (IU.get(i + 1) - 1); k >= IU.get(i); k--) {
                j = JU.get(k);
                x.setValue(i, x.getValue(i) - U.get(k) * x.getValue(j));
            }
        }
        return x;
    }

    public AbstractMatrix inv(AbstractMatrix target) {
        AVector vector;
        for (int i = 0; i < n; i++) {
            vector = MatrixMaker.zeros(n);
            vector.setValue(i, 1.0);
            List<Integer> path = getPath(i);
            AVector x = solveEquation(vector, path);
            for (int j = 0; j < n; j++) {
                if (Math.abs(x.getValue(j)) > 10e-6)
                    target.setValue(j, i, x.getValue(j));
            }
        }
        return target;
    }

    public List<Integer> getPath(int p) {
        List<Integer> path = new ArrayList<Integer>();
        int i = 0;
        path.add(p);
        while (p != (n - 1)) {
            if (IU.get(p + 1) > IU.get(p)) {
                path.add(JU.get(IU.get(p)));
                i++;
                p = path.get(i);
            } else
                break;
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
}
