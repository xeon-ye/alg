package jpscpu;

import cern.colt.matrix.DoubleMatrix2D;
import junit.framework.TestCase;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.util.ColtMatrixUtil;
import zju.util.JOFileUtil;

/**
 * LinearSolver Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>08/20/2013</pre>
 */
public class LinearSolverMTTest extends TestCase {
    public LinearSolverMTTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testCase1() {
        int m = 5, n = 5, nnz = 12;
        double a[] = new double[12];
        double b[] = new double[5];
        double s, u, p, e, r, l;
        int asub[] = new int[12];
        int xa[] = new int[6];
        //5*5矩阵如下
        /*
         1  2   3   4   5
        [s      u   u
         l  u
            l   p
                    e   u
         l  l           r]
        */
        s = 19.0;
        u = 21.0;
        p = 16.0;
        e = 5.0;
        r = 18.0;
        l = 12.0;
        a[0] = s;
        a[1] = l;
        a[2] = l;
        a[3] = u;
        a[4] = l;
        a[5] = l;
        a[6] = u;
        a[7] = p;
        a[8] = u;
        a[9] = e;
        a[10] = u;
        a[11] = r;
        asub[0] = 0;
        asub[1] = 1;
        asub[2] = 4;
        asub[3] = 1;
        asub[4] = 2;
        asub[5] = 4;
        asub[6] = 0;
        asub[7] = 2;
        asub[8] = 0;
        asub[9] = 3;
        asub[10] = 3;
        asub[11] = 4;
        xa[0] = 0;
        xa[1] = 3;
        xa[2] = 6;
        xa[3] = 8;
        xa[4] = 10;
        xa[5] = 12;
        b[0] = 1.0;
        b[1] = 1.0;
        b[2] = 1.0;
        b[3] = 1.0;
        b[4] = 1.0;

        ASparseMatrixLink2D struc = new ASparseMatrixLink2D(m, n, 12);
        struc.setValue(0, 0, s);
        struc.setValue(0, 2, u);
        struc.setValue(0, 3, u);
        struc.setValue(1, 0, l);
        struc.setValue(1, 1, u);
        struc.setValue(2, 1, l);
        struc.setValue(2, 2, p);
        struc.setValue(3, 3, e);
        struc.setValue(3, 4, u);
        struc.setValue(4, 0, l);
        struc.setValue(4, 1, l);
        struc.setValue(4, 4, r);

        DoubleMatrix2D mat = struc.toColtSparseMatrix();

        double[] a1 = new double[struc.getVA().size()];
        int[] asub1 = new int[a1.length];
        int[] xa1 = new int[n + 1];
        struc.getSluStrucNC(asub1, xa1);
        for (int i = 0; i < asub.length; i++)
            assertEquals(asub[i], asub1[i]);
        for (int i = 0; i < xa.length; i++)
            assertEquals(xa[i], xa1[i]);

        ColtMatrixUtil.getSluMatrixNC(mat, a1, asub1, xa1);
        for (int i = 0; i < a1.length; i++)
            assertEquals(a[i], a1[i]);

        LinearSolverMT solver = new LinearSolverMT(4);
        solver.setDrive(LinearSolver.SUPERLU_DRIVE_0);
        double[] sol = new double[b.length];
        System.arraycopy(b, 0, sol, 0, b.length);
        solver.solve(m, n, nnz, a, asub, xa, sol);
        assertTrue(Math.abs(-0.0313 - sol[0]) < 1e-3);
        assertTrue(Math.abs(0.0655 - sol[1]) < 1e-3);
        assertTrue(Math.abs(0.0134 - sol[2]) < 1e-3);
        assertTrue(Math.abs(0.0625 - sol[3]) < 1e-3);
        assertTrue(Math.abs(0.0327 - sol[4]) < 1e-3);

        System.arraycopy(b, 0, sol, 0, b.length);
        solver.setDrive(LinearSolver.SUPERLU_DRIVE_1);
        solver.solve(m, n, nnz, a, asub, xa, sol);
        assertTrue(Math.abs(-0.0313 - sol[0]) < 1e-3);
        assertTrue(Math.abs(0.0655 - sol[1]) < 1e-3);
        assertTrue(Math.abs(0.0134 - sol[2]) < 1e-3);
        assertTrue(Math.abs(0.0625 - sol[3]) < 1e-3);
        assertTrue(Math.abs(0.0327 - sol[4]) < 1e-3);
    }

    public void testCase2() {
        int m = 5, n = 5, nnz = 12;
        double a[] = new double[12];
        double b[] = new double[5];
        double s, u, p, e, r, l;
        int asub[] = new int[12];
        int xa[] = new int[6];
        //5*5矩阵如下
        /*
         1  2   3   4   5
        [s      u   u
         l  u
            l   p
                    e   u
         l  l           r]
        */
        s = 19.0;
        u = 21.0;
        p = 16.0;
        e = 5.0;
        r = 18.0;
        l = 12.0;
        a[0] = s;
        a[1] = l;
        a[2] = l;
        a[3] = u;
        a[4] = l;
        a[5] = l;
        a[6] = u;
        a[7] = p;
        a[8] = u;
        a[9] = e;
        a[10] = u;
        a[11] = r;
        asub[0] = 0;
        asub[1] = 1;
        asub[2] = 4;
        asub[3] = 1;
        asub[4] = 2;
        asub[5] = 4;
        asub[6] = 0;
        asub[7] = 2;
        asub[8] = 0;
        asub[9] = 3;
        asub[10] = 3;
        asub[11] = 4;
        xa[0] = 0;
        xa[1] = 3;
        xa[2] = 6;
        xa[3] = 8;
        xa[4] = 10;
        xa[5] = 12;
        b[0] = 1.0;
        b[1] = 1.0;
        b[2] = 1.0;
        b[3] = 1.0;
        b[4] = 1.0;

        ASparseMatrixLink2D struc = new ASparseMatrixLink2D(m, n, 12);
        struc.setValue(0, 0, s);
        struc.setValue(0, 2, u);
        struc.setValue(0, 3, u);
        struc.setValue(1, 0, l);
        struc.setValue(1, 1, u);
        struc.setValue(2, 1, l);
        struc.setValue(2, 2, p);
        struc.setValue(3, 3, e);
        struc.setValue(3, 4, u);
        struc.setValue(4, 0, l);
        struc.setValue(4, 1, l);
        struc.setValue(4, 4, r);

        DoubleMatrix2D mat = struc.toColtSparseMatrix();

        double[] a1 = new double[struc.getVA().size()];
        int[] asub1 = new int[a1.length];
        int[] xa1 = new int[n + 1];
        struc.getSluStrucNC(asub1, xa1);
        for (int i = 0; i < asub.length; i++)
            assertEquals(asub[i], asub1[i]);
        for (int i = 0; i < xa.length; i++)
            assertEquals(xa[i], xa1[i]);

        ColtMatrixUtil.getSluMatrixNC(mat, a1, asub1, xa1);
        for (int i = 0; i < a1.length; i++)
            assertEquals(a[i], a1[i]);

        double[] sol = new double[b.length];
        System.arraycopy(b, 0, sol, 0, b.length);
        LinearSolverMT solver = new LinearSolverMT(4);
        solver.setDrive(LinearSolverMT.SUPERLU_DRIVE_0);
        solver.solve(m, n, nnz, a, asub, xa, sol);
        assertCase5(sol);

        System.arraycopy(b, 0, sol, 0, b.length);
        solver.setDrive(LinearSolverMT.SUPERLU_DRIVE_1);
        solver.solve(m, n, nnz, a, asub, xa, sol);
        assertCase5(sol);

        System.arraycopy(b, 0, sol, 0, b.length);
        solver.solve2(struc, mat, sol);
        assertCase5(sol);

        //for(int i = 0; i < 10000000; i++) {
        System.arraycopy(b, 0, sol, 0, b.length);
        sol = solver.solve2(mat, sol);
        //}
        assertCase5(sol);
    }

    public void testSolve3() {
        ASparseMatrixLink2D bAposTwo = (ASparseMatrixLink2D) JOFileUtil.decode4Ser(this.getClass().getResourceAsStream("/matrix/BAposTwo30.dat"));
        AVector deltaQ = (AVector) JOFileUtil.decode4XML(this.getClass().getResourceAsStream("/matrix/DeltaQ30.xml"));
        //ASparseMatrixLink2D bAposTwo = (ASparseMatrixLink2D) JOFileUtil.decode4Ser(this.getClass().getResourceAsStream("/matrix/BAposTwo.dat"));
        //AVector deltaQ = (AVector) JOFileUtil.decode4XML(this.getClass().getResourceAsStream("/matrix/DeltaQ.xml"));
        LinearSolver solver = new LinearSolver();
        LinearSolver solver2 = new LinearSolver();
        double[] r1 = deltaQ.getValues().clone();
        double[] r2 = deltaQ.getValues().clone();
        solver.solve3(bAposTwo, r1);
        solver2.solve2(bAposTwo, r2, true);
        assertEqual(r1, r2);

        for (int i = 0; i < 10; i++) {
            double[] r3 = deltaQ.getValues().clone();
            double[] r4 = deltaQ.getValues().clone();
            solver.solve3(r3);
            solver2.solve2(r4);
            assertEqual(r1, r4);
            assertEqual(r1, r3);
        }
    }

    private void assertEqual(double[] r1, double[] r2) {
        for (int i = 0; i < r1.length; i++)
            assertTrue(Math.abs(r1[i] - r2[i]) < 1e-5);

    }

    private void assertCase5(double[] sol) {
        assertTrue(Math.abs(-0.0313 - sol[0]) < 1e-3);
        assertTrue(Math.abs(0.0655 - sol[1]) < 1e-3);
        assertTrue(Math.abs(0.0134 - sol[2]) < 1e-3);
        assertTrue(Math.abs(0.0625 - sol[3]) < 1e-3);
        assertTrue(Math.abs(0.0327 - sol[4]) < 1e-3);
    }

    public void testMlp() {
        double objValue[] = {1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0};
        // Lower bounds for columns
        double columnLower[] = {2.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0};
        // Upper bounds for columns
        double columnUpper[] = {2e15, 4.1, 1.0, 1.0, 4.0,
                2e15, 2e15, 4.3};
        // Lower bounds for row activities
        double rowLower[] = {2.5, -2e15, -2e15, 1.8, 3.0};
        // Upper bounds for row activities
        double rowUpper[] = {2e15, 2.1, 4.0, 5.0, 15.0};
        // Matrix stored packed
        int column[] = {0, 1, 3, 4, 7,
                1, 2,
                2, 5,
                3, 6,
                4, 7};
        double element[] = {3.0, 1.0, -2.0, -1.0, -1.0,
                2.0, 1.1,
                1.0, 1.0,
                2.8, -1.2,
                1.0, 1.9};
        int starts[] = {0, 5, 7, 9, 11, 13};
        // Integer variables (note upper bound already 1.0)
        int whichInt[] = {2, 3};
        int numberRows = rowLower.length;
        int numberColumns = columnLower.length;
        double result[] = new double[numberColumns];

        LinearSolver solver = new LinearSolver();
        solver.setDrive(LinearSolver.MLP_DRIVE_CBC);
        int status = solver.solveMlp(numberColumns, numberRows, objValue,
                columnLower, columnUpper, rowLower, rowUpper, element, column, starts, whichInt, result);

        assertTrue(status > 0);
        assertEquals(3.1, result[0]);
        assertEquals(0.0, result[1]);
        assertEquals(0.0, result[2]);
        assertEquals(1.0, result[3]);
        assertEquals(0.5, result[4]);
        assertEquals(0.0, result[5]);
        assertEquals(0.0, result[6]);
        assertEquals(4.3, result[7]);

        //测试SYMPHONTY开始
        //与CBC不同的是，SYM使用的是列主导的稀疏矩阵形式
        ASparseMatrixLink2D mat = new ASparseMatrixLink2D(5, 8);
        for(int i = 1; i < starts.length; i++) {
            for(int j = starts[i - 1]; j < starts[i]; j++)
                mat.setValue(i - 1, column[j], element[j]);
        }
        //mat.printOnScreen();
        double[] a = new double[mat.getVA().size()];
        int[] asub = new int[mat.getVA().size()];
        int[] xa = new int[mat.getN() + 1];
        mat.getSluStrucNC(a, asub, xa);
        solver.setDrive(LinearSolver.MLP_DRIVE_SYM);
        result = new double[numberColumns];
        status = solver.solveMlp(numberColumns, numberRows, objValue,
                columnLower, columnUpper, rowLower, rowUpper, a, asub, xa, whichInt, result);

        assertTrue(status > 0);
        assertEquals(3.1, result[0]);
        assertEquals(0.0, result[1]);
        assertEquals(0.0, result[2]);
        assertEquals(1.0, result[3]);
        assertEquals(0.5, result[4]);
        assertEquals(0.0, result[5]);
        assertEquals(0.0, result[6]);
        assertEquals(4.3, result[7]);
    }
}
