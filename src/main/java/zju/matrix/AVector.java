package zju.matrix;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.Serializable;
import java.text.NumberFormat;


/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-5
 */
public class AVector implements Serializable {

    private static Logger log = LogManager.getLogger(AVector.class);

    private double[] values;

    private int n = 0;

    public AVector() {
    }

    public AVector(int n) {
        this.n = n;
        values = new double[n];
    }

    public AVector(AVector initial) {
        this(initial.getN());
        assign(initial);
    }

    public AVector(double[] initialValues) {
        this.n = initialValues.length;
        values = initialValues;
    }

    /**
     * set value to a fixed position
     *
     * @param index index starts from 0
     * @param value value of element
     */
    public void setValue(int index, double value) {
        if (index >= getN()) {
            log.error("index is out of vector's range :" + getN() + " <= " + index);
            return;
        }
        values[index] = value;
    }

    public double getValue(int index) {
        return values[index];
    }

    public double[] getValues() {
        return values;
    }

    public void setValues(double[] values) {
        this.values = values;
    }


    public int getN() {
        return n;
    }

    public void setN(int n) {
        this.n = n;
    }

    public AVector sub(AVector another) {
        AVector r = new AVector(this.getN());
        AVector.sub(this, another, r);
        return r;
    }

    public AVector add(AVector another) {
        AVector r = new AVector(this.getN());
        AVector.add(this, another, r);
        return r;
    }

    public AVector mul(double d) {
        AVector r = new AVector(this.getN());
        AVector.mul(d, this, r);
        return r;
    }

    public void subSelf(AVector another) {
        AVector.sub(this, another, this);
    }

    public void addSelf(AVector another) {
        AVector.add(this, another, this);
    }

    public void mulSelf(double d) {
        AVector.mul(d, this, this);
    }

    public static AVector sub(AVector a, AVector b) {
        AVector c = new AVector(a.getN());
        sub(a, b, c);
        return c;
    }

    public static void sub(AVector a, AVector b, AVector target) {
        if (a.getN() != b.getN()) {
            log.warn("length of vector a and b is not equal!");
            return;
        }
        for (int i = 0; i < a.getN(); i++) {
            target.setValue(i, a.getValues()[i] - b.getValues()[i]);
        }
    }

    public static AVector add(AVector a, AVector b) {
        AVector c = new AVector(a.getN());
        add(a, b, c);
        return c;
    }

    public static void add(AVector a, AVector b, AVector target) {
        if (a.getN() != b.getN()) {
            log.warn("length of vector a and b is not equal!");
            return;
        }
        for (int i = 0; i < a.getN(); i++) {
            target.setValue(i, a.getValues()[i] + b.getValues()[i]);
        }
    }

    public static void mul(double d, AVector a, AVector target) {
        for (int i = 0; i < a.getN(); i++)
            target.setValue(i, a.getValues()[i] * d);
    }

    public void assign(double[] another) {
        if (another.length != this.getN()) {
            log.error("length is not equal!");
            return;
        }
        System.arraycopy(another, 0, values, 0, n);
    }

    public void assign(AVector another) {
        assign(another.getValues());
    }

    public void printOnScreen() {
        printOnScreen(getValues());
    }

    public void printOnScreen(NumberFormat f) {
        printOnScreen(f, getValues());
    }

    public static void printOnScreen(double[] v) {
        int i = 0;
        for (; i < v.length - 1; i++)
            System.out.print(v[i] + ",");
        System.out.println(v[i]);
    }

    public static void printOnScreen(NumberFormat f, double[] v) {
        int i = 0;
        for (; i < v.length - 1; i++)
            System.out.print(f.format(v[i]) + ", ");
        System.out.println(f.format(v[i]));
    }

    public boolean isEqual(AVector x) {
        if (x == null || x.getN() != this.getN())
            return false;
        for (int i = 0; i < getN(); i++)
            if (Math.abs(getValue(i) - x.getValue(i)) > 10e-6)
                return false;
        return true;
    }

    @Override
    public AVector clone() {
        AVector clonedV = new AVector(n);
        System.arraycopy(values, 0, clonedV.getValues(), 0, n);
        return clonedV;
    }
}
