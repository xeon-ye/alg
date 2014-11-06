package zju.matrix;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-6
 */
public abstract class AbstractMatrix implements Serializable {

    protected int m;//row number
    protected int n;//col number
    private boolean isSymmetrical = false;

    protected AbstractMatrix() {
    }

    public AbstractMatrix(int m, int n) {
        this.m = m;
        this.n = n;
    }

    public abstract AVector mul(AVector v);

    public abstract AbstractMatrix mul(AbstractMatrix m);

    public abstract AbstractMatrix inv();

    public abstract AbstractMatrix transpose();

    public abstract AVector getRowVector(int index);

    public abstract AVector getColVector(int index);

    public abstract void printOnScreen();

    public abstract void setValue(int i, int j, double v);

    public abstract double getValue(int i, int j);

    public abstract AbstractMatrix cloneMatrix();

    public int getM() {
        return m;
    }

    public int getN() {
        return n;
    }

    public void setSymmetrical(boolean symmetrical) {
        isSymmetrical = symmetrical;
    }

    public boolean isSymmetrical() {
        return isSymmetrical;
    }
}
