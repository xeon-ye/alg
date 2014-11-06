package zju.util;

import cern.colt.function.IntIntDoubleFunction;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import junit.framework.TestCase;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.ieeeformat.IcfDataUtil;
import zju.matrix.MySparseDoubleMatrix2D;

/**
 * JacobianMakerRC Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>07/20/2010</pre>
 */
public class JacobianMakerRCTest extends TestCase {
    public JacobianMakerRCTest(String name) {
        super(name);
    }


    public void setUp() throws Exception {
    }

    public void testStandardCases() {
        doJacobianMake(IcfDataUtil.ISLAND_14.clone());
        doJacobianMake(IcfDataUtil.ISLAND_30.clone());
        doJacobianMake(IcfDataUtil.ISLAND_39.clone());
        doJacobianMake(IcfDataUtil.ISLAND_57.clone());
        doJacobianMake(IcfDataUtil.ISLAND_118.clone());
        doJacobianMake(IcfDataUtil.ISLAND_300.clone());
    }

    synchronized public void doJacobianMake(IEEEDataIsland island) {
        int n = island.getBuses().size();
        double[] x = new double[4 * n];
        NumberOptHelper numOpt = new NumberOptHelper();
        numOpt.simple(island);
        numOpt.trans(island);

        YMatrixGetter y = new YMatrixGetter(island);
        y.formYMatrix();

        for (int i = 0; i < n; i++)
            x[i] = 1.0;
        StateCalByRC.calI(y, x, x, 2 * n);
        JacobianMakerRC jacobian = new JacobianMakerRC(JacobianMakerRC.MODE_UI);
        jacobian.setY(y);
        jacobian.setUI(x);

        int m = island.getBranches().size();
        int[] pPos = new int[n];
        int[] qPos = new int[n];
        int[] vPos = new int[n];
        int[] pFromPos = new int[m];
        int[] qFromPos = new int[m];
        int[] pToPos = new int[m];
        int[] qToPos = new int[m];
        for (BusData bus : island.getBuses()) {
            pPos[bus.getBusNumber() - 1] = bus.getBusNumber();
            qPos[bus.getBusNumber() - 1] = bus.getBusNumber();
            vPos[bus.getBusNumber() - 1] = bus.getBusNumber();
        }
        for (BranchData branch : island.getBranches()) {
            pFromPos[branch.getId() - 1] = branch.getId();
            qFromPos[branch.getId() - 1] = branch.getId();
            pToPos[branch.getId() - 1] = branch.getId();
            qToPos[branch.getId() - 1] = branch.getId();
        }
        DoubleMatrix2D r1 = new MySparseDoubleMatrix2D(pPos.length, 4 * n);
        jacobian.fillJacobian_bus_p(pPos, 0, r1);
        DoubleMatrix2D r2 = new MySparseDoubleMatrix2D(qPos.length, 4 * n);
        jacobian.fillJacobian_bus_q(qPos, 0, r2);
        DoubleMatrix2D r3 = new MySparseDoubleMatrix2D(vPos.length, 4 * n);
        jacobian.fillJacobian_bus_v(vPos, 0, r3);
        //jacobian.fillJacobian_line_from_p(pFromPos);
        //jacobian.fillJacobian_line_from_q(qFromPos);
        //jacobian.fillJacobian_line_to_p(pToPos);
        //jacobian.fillJacobian_line_to_q(qToPos);
        final DoubleMatrix2D G = jacobian.getG();
        final DoubleMatrix2D B = jacobian.getB();
        final int[] nonzeroInG = new int[1];
        G.forEachNonZero(new IntIntDoubleFunction() {
            public double apply(int i, int i1, double v) {
                nonzeroInG[0]++;
                return v;
            }
        });
        final int[] nonzeroInB = new int[1];
        DoubleMatrix2D B1 = new MySparseDoubleMatrix2D(G.rows(), G.columns(), G.cardinality(), 0.2, 0.5);
        B1.assign(B);
        B1.forEachNonZero(new IntIntDoubleFunction() {
            public double apply(int i, int i1, double v) {
                nonzeroInB[0]++;
                return -v;
            }
        });
        DoubleMatrix2D I = DoubleFactory2D.sparse.identity(G.rows());
        final int[] nonzeroInI = new int[1];
        I.forEachNonZero(new IntIntDoubleFunction() {
            public double apply(int i, int i1, double v) {
                nonzeroInI[0]++;
                return -v;
            }
        });
        assertEquals(nonzeroInG[0], nonzeroInB[0]);
        assertEquals(nonzeroInI[0], G.rows());
        DoubleMatrix2D tmp1 = ColtMatrixUtil.mergeMatrixByCol(new DoubleMatrix2D[]{G, B1, I});
        DoubleMatrix2D tmp2 = ColtMatrixUtil.mergeMatrixByCol(new DoubleMatrix2D[]{B, G});
        DoubleMatrix2D tmp3 = new MySparseDoubleMatrix2D(G.rows(), 4 * G.columns(), tmp2.cardinality() + I.cardinality(), 0.2, 0.5);
        ColtMatrixUtil.mergeMatrixByCol(new DoubleMatrix2D[]{tmp2}, tmp3);
        ColtMatrixUtil.mergeMatrixByCol(new DoubleMatrix2D[]{I}, tmp3, 3 * G.columns());
        DoubleMatrix2D tmp4 = ColtMatrixUtil.mergeMatrixByRow(new DoubleMatrix2D[]{tmp1, tmp3});

        DoubleMatrix2D target = ColtMatrixUtil.mergeMatrixByRow(new DoubleMatrix2D[]{tmp4, r3, r1, r2});
        final DoubleMatrix2D tmp5 = new MySparseDoubleMatrix2D(G.rows() * 2, 4 * G.columns());
        final int colsOfJacobian = 4 * G.columns();
        G.forEachNonZero(new IntIntDoubleFunction() {
            public double apply(int row, int col, double v) {
                int[] keys = new int[4];
                keys[0] = row * colsOfJacobian + col;
                keys[1] = row * colsOfJacobian + col + G.columns();
                keys[2] = (row + G.rows()) * colsOfJacobian + col;
                keys[3] = (row + G.rows()) * colsOfJacobian + (col + B.columns());
                tmp5.setQuick(keys[0] / colsOfJacobian, keys[0] % colsOfJacobian, v);
                tmp5.setQuick(keys[1] / colsOfJacobian, keys[1] % colsOfJacobian, -B.getQuick(row, col));
                tmp5.setQuick(keys[2] / colsOfJacobian, keys[2] % colsOfJacobian, B.getQuick(row, col));
                tmp5.setQuick(keys[3] / colsOfJacobian, keys[3] % colsOfJacobian, v);
                return v;
            }
        });
        for (int i = 0; i < G.rows() + B.rows(); i++)
            tmp5.setQuick(i, G.columns() + B.columns() + i, -1.0);
        DoubleMatrix2D target2 = ColtMatrixUtil.mergeMatrixByRow(new DoubleMatrix2D[]{tmp5, r3, r1, r2});

        final int[] nonzeroNum = new int[1];
        target.forEachNonZero(new IntIntDoubleFunction() {
            public double apply(int i, int i1, double v) {
                nonzeroNum[0]++;
                return v;
            }
        });
        assertEquals(nonzeroNum[0], target.cardinality());
        double[] x2 = new double[4 * n];
        System.arraycopy(x, 0, x2, 2 * n, 2 * n);
        System.arraycopy(x, 2 * n, x2, 0, 2 * n);
        int[] iRow = new int[nonzeroNum[0]];
        int[] jCol = new int[nonzeroNum[0]];
        double[] values = new double[nonzeroNum[0]];
        System.out.println("Nonzero num in jacobian is : " + nonzeroNum[0]);
        int nele_jac = eval_jac_g(y, x2, iRow, jCol, null);
        assertEquals(nele_jac, nonzeroNum[0]);
        eval_jac_g(y, x2, iRow, jCol, values);
        for (int i = 0; i < iRow.length; i++) {
            int row = iRow[i];
            int col = jCol[i];
            int col1 = jCol[i];
            if (col >= 2 * n)
                col1 -= 2 * n;
            else
                col1 += 2 * n;
            //System.out.println("row = " + row + "\tcol=" + col1 + "\t" + values[i] + "\t" + target.getQuick(row, col1));
            assertTrue(Math.abs(values[i] - target.getQuick(row, col1)) < 1e-10);
            assertTrue(Math.abs(values[i] - target2.getQuick(row, col1)) < 1e-10);
        }
    }

    public int eval_jac_g(YMatrixGetter y, double[] x, int[] iRow, int[] jCol, double[] values) {
        int busNumber = y.getIsland().getBuses().size();
        int index = 0;
        if (values == null) {
            //freal, fimage, const part
            for (int busNo = 0; busNo < busNumber; busNo++) {
                int k = y.getAdmittance()[0].getIA()[busNo];
                while (k != -1) {
                    int j = y.getAdmittance()[0].getJA().get(k);
                    iRow[index] = busNo;    // freal/e
                    jCol[index] = j + 2 * busNumber;
                    index++;
                    iRow[index] = busNo;    // freal/f
                    jCol[index] = j + 3 * busNumber;
                    index++;
                    iRow[index] = busNo + busNumber;    // fimage/e
                    jCol[index] = j + 2 * busNumber;
                    index++;
                    iRow[index] = busNo + busNumber;    // fimage/f
                    jCol[index] = j + 3 * busNumber;
                    index++;
                    if (busNo == j) {
                        iRow[index] = busNo;    // freal/ix
                        jCol[index] = j;
                        index++;
                        iRow[index] = busNo + busNumber;    // fimage/iy
                        jCol[index] = j + busNumber;
                        index++;
                    }
                    k = y.getAdmittance()[0].getLINK().get(k);
                }
            }
            int rowIndex = 2 * busNumber;
            //measure: f(v^2)
            for (int i = 0; i < busNumber; i++) {
                iRow[index] = rowIndex;
                jCol[index] = i + 2 * busNumber;    //e
                index++;
                iRow[index] = rowIndex;
                jCol[index] = i + 3 * busNumber;   //f
                index++;
                rowIndex++;
            }
            assert rowIndex == 3 * busNumber;
            //measure:P
            for (int i = 0; i < busNumber; i++) {
                iRow[index] = rowIndex; //ix
                jCol[index] = i;
                index++;
                iRow[index] = rowIndex; //iy
                jCol[index] = i + busNumber;
                index++;
                iRow[index] = rowIndex; //e
                jCol[index] = i + 2 * busNumber;
                index++;
                iRow[index] = rowIndex; //f
                jCol[index] = i + 3 * busNumber;
                index++;
                rowIndex++;
            }
            //measure:Q
            for (int i = 0; i < busNumber; i++) {
                iRow[index] = rowIndex; //ix
                jCol[index] = i;
                index++;
                iRow[index] = rowIndex; //iy
                jCol[index] = i + busNumber;
                index++;
                iRow[index] = rowIndex; //e
                jCol[index] = i + 2 * busNumber;
                index++;
                iRow[index] = rowIndex; //f
                jCol[index] = i + 3 * busNumber;
                index++;
                rowIndex++;
            }
        } else {
            //freal, fimage, const part
            for (int busNo = 0; busNo < busNumber; busNo++) {
                int k = y.getAdmittance()[0].getIA()[busNo];
                while (k != -1) {
                    int j = y.getAdmittance()[0].getJA().get(k);
                    double gij = y.getAdmittance()[0].getVA().get(k);
                    double bij = y.getAdmittance()[1].getVA().get(k);
                    values[index] = gij;    //freal/e
                    index++;
                    values[index] = -bij;  //freal/f
                    index++;
                    values[index] = bij;    //fimage/e
                    index++;
                    values[index] = gij;    //fimage/f
                    index++;
                    if (busNo == j) {
                        values[index] = -1;    // freal/ix
                        index++;
                        values[index] = -1;    // fimage/iy
                        index++;
                    }
                    k = y.getAdmittance()[0].getLINK().get(k);
                }
            }
            //measure: f(v^2)
            for (int i = 0; i < busNumber; i++) {
                double e = x[i + 2 * busNumber];
                double f = x[i + 3 * busNumber];
                values[index] = 2 * e;
                index++;
                values[index] = 2 * f;
                index++;
            }
            //measure:P
            for (int i = 0; i < busNumber; i++) {
                double ix = x[i];
                double iy = x[i + busNumber];
                double e = x[i + 2 * busNumber];
                double f = x[i + 3 * busNumber];
                values[index] = e; //ix
                index++;
                values[index] = f; //iy
                index++;
                values[index] = ix; //e
                index++;
                values[index] = iy; //f
                index++;
            }
            //measure:Q
            for (int i = 0; i < busNumber; i++) {
                double ix = x[i];
                double iy = x[i + busNumber];
                double e = x[i + 2 * busNumber];
                double f = x[i + 3 * busNumber];
                values[index] = f; //ix
                index++;
                values[index] = -e; //iy
                index++;
                values[index] = -iy; //e
                index++;
                values[index] = ix; //f
                index++;
            }
        }
        return index;
    }
}
