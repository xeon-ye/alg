package zju.matrix;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-6
 */
public class MatrixMaker {
    public static AbstractMatrix diag(AVector v) {
        //if(thresholds1.getN() > 50) {
        ASparseMatrixLink m = new ASparseMatrixLink(v.getN());
        for (int i = 0; i < v.getN(); i++)
            m.setValue(i, i, v.getValues()[i]);
        //}
        m.setSymmetrical(true);
        return m;//todo
    }

    public static AbstractMatrix eye(int n) {
        //if(n > 50) {
        ASparseMatrixLink2D m = new ASparseMatrixLink2D(n);
        for (int i = 0; i < n; i++)
            m.setValue(i, i, 1);
        //}
        m.setSymmetrical(true);
        return m;//todo
    }

    public static AVector ones(int n) {
        AVector v = new AVector(n);
        for (int i = 0; i < n; i++)
            v.setValue(i, 1);
        return v;
    }

    public static AVector zeros(int n) {
        AVector v = new AVector(n);
        for (int i = 0; i < n; i++)
            v.setValue(i, 0);
        return v;
    }

    public static AVector cat(AVector[] vectors) {
        int n = 0;
        for (AVector vector : vectors) n += vector.getN();
        AVector v = new AVector(n);
        int index = 0;
        for (AVector vector : vectors)
            for (int j = 0; j < vector.getN(); j++) {
                v.setValue(index, vector.getValues()[j]);
                index++;
            }
        return v;
    }
}
