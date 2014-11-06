package zju.matrix;

import cern.colt.map.AbstractIntDoubleMap;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2010-7-20
 */
public class MySparseDoubleMatrix2D extends SparseDoubleMatrix2D {
    public MySparseDoubleMatrix2D(double[][] doubles) {
        super(doubles);
    }

    public MySparseDoubleMatrix2D(int i, int i1) {
        super(i, i1);
    }

    public MySparseDoubleMatrix2D(int i, int i1, int i2, double v, double v1) {
        super(i, i1, i2, v, v1);
    }

    protected MySparseDoubleMatrix2D(int i, int i1, AbstractIntDoubleMap abstractIntDoubleMap, int i2, int i3, int i4, int i5) {
        super(i, i1, abstractIntDoubleMap, i2, i3, i4, i5);
    }

    /**
     *  继承方法，原有方法需要判断值是否为零，这里不再判断直接设值
     * @param row     行
     * @param column 列
     * @param value  值
     */
    public void setQuick(int row, int column, double value) {
        int index = rowZero + row * rowStride + columnZero + column * columnStride;
        this.elements.put(index, value);
    }

    public void addQuick(int row, int column, double value) {
        int index = rowZero + row * rowStride + columnZero + column * columnStride;
        if(this.elements.containsKey(index))
            this.elements.put(index, this.elements.get(index) +  value);
        else
            this.elements.put(index, value);
    }

    public boolean contains(int i, int j) {
        int index = rowZero + i * rowStride + columnZero + j * columnStride;
        return elements.containsKey(index);
    }

    public AbstractIntDoubleMap getElements() {
        return elements;
    }


    public String toString() {
        return "Matrix: row:" + rows + "\tcol:" + columns;
    }
}
