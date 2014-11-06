package zju.matrix;

import junit.framework.TestCase;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-6
 */
public class ASparseMatrixLUTest extends TestCase {
    public void testLDU() {
        ASparseMatrixLink2D matrix = new ASparseMatrixLink2D(4);
        matrix.setValue(0, 0, 2);
        matrix.setValue(0, 1, 7);
        matrix.setValue(0, 3, -3);
        matrix.setValue(1, 0, 5);
        matrix.setValue(1, 1, 4);
        matrix.setValue(2, 2, 5);
        matrix.setValue(3, 1, -2);
        matrix.setValue(3, 3, 6);
        matrix.symboAnalysis();
        matrix.printOnScreen();

        /**
         * resultColt expected to be:
         * L:

         * 2
         * 5    -13.5
         * 0    0   5
         * 0    -2  0   4.889

         * U:
         * 1    3.5 0   -1.5
         *      1   0   -0.555
         *           1   0
         *               1

         * D:
         * 2 -13.5 5.0 4.889
         *
         */
        ASparseMatrixLU m = ASparseMatrixLU.formMatrix(matrix);
        m.ldu();
        System.out.println("after ldu ...");
        m.printOnScreen();
        assertEquals(m.getD().get(0), 2.0);
        assertEquals(m.getD().get(1), -13.5);
        assertEquals(m.getD().get(2), 5.0);
        assertEquals(Math.abs(m.getD().get(3) - 4.888888) < 0.001, true);
    }

    public void testLDU2() {
        ASparseMatrixLink2D matrix = new ASparseMatrixLink2D(4);
        matrix.setValue(0, 0, 2);
        matrix.setValue(0, 1, -1);
        matrix.setValue(0, 3, -1);
        matrix.setValue(1, 0, -1);
        matrix.setValue(1, 1, 2);
        matrix.setValue(1, 2, -1);
        matrix.setValue(2, 1, -1);
        matrix.setValue(2, 2, 2);
        matrix.setValue(2, 3, -1);
        matrix.setValue(3, 0, -1);
        matrix.setValue(3, 2, -1);
        matrix.setValue(3, 3, 4);
        matrix.symboAnalysis();
        matrix.printOnScreen();

        /**
         * resultColt expected to be:
         * L:

         * 1
         * -0.5    1
         * 0    -0.667   1
         * -0.5    -0.333  -1   1
         * D:
         * 2 -1.5 1.333 2
         *
         */
        ASparseMatrixLU m = ASparseMatrixLU.formMatrix(matrix);
        m.ldu();
        System.out.println("after ldu ...");
        m.printOnScreen();
        assertEquals(m.getD().get(0), 2.0);
        assertEquals(m.getD().get(1), 1.5);
        assertEquals(Math.abs(m.getD().get(2) - 1.33333) < 0.001, true);
        assertEquals(Math.abs(m.getD().get(3) - 2.0) < 0.001, true);
    }

    public void testInv() {
        System.out.println("test of eye matrix begin:");
        AbstractMatrix eye = MatrixMaker.eye(4);
        ASparseMatrixLU m = ASparseMatrixLU.formMatrix((ASparseMatrixLink2D) eye);
        ASparseMatrixLink matrix = new ASparseMatrixLink(4);
        m.inv(matrix);
        m.printOnScreen();
        assertEquals(matrix.getValue(0, 0), 1.0);
        assertEquals(matrix.getValue(1, 1), 1.0);
        assertEquals(matrix.getValue(2, 2), 1.0);
        assertEquals(matrix.getValue(3, 3), 1.0);

        System.out.println("test of symmetrical matrix begin:");
        matrix = new ASparseMatrixLink2D(4);
        matrix.setValue(0, 0, 2);
        matrix.setValue(0, 1, -1);
        matrix.setValue(0, 3, -1);
        matrix.setValue(1, 0, -1);
        matrix.setValue(1, 1, 2);
        matrix.setValue(1, 2, -1);
        matrix.setValue(2, 1, -1);
        matrix.setValue(2, 2, 2);
        matrix.setValue(2, 3, -1);
        matrix.setValue(3, 0, -1);
        matrix.setValue(3, 2, -1);
        matrix.setValue(3, 3, 4);
        matrix.symboAnalysis();
        matrix.setSymmetrical(true);
        //matrix.printOnScreen();

        /**
         * resultColt expected to be:
         * L:

         * 1
         * -0.5    1
         * 0    -0.667   1
         * -0.5    -0.333  -1   1
         * D:
         * 2 -1.5 1.333 2
         *
         */
        m = ASparseMatrixLU.formMatrix(matrix);
        m.ldu();
        AbstractMatrix result = m.inv(new ASparseMatrixLink(4));
        System.out.println("inverse matrix resultColt is:");
        result.printOnScreen();
        System.out.println("check if inv operation is correct:");
        eye = result.mul(matrix);
        eye.printOnScreen();
        assertEquals(Math.abs(eye.getValue(0, 0) - 1.0) < 0.001, true);
        assertEquals(Math.abs(eye.getValue(1, 1) - 1.0) < 0.001, true);
        assertEquals(Math.abs(eye.getValue(2, 2) - 1.0) < 0.001, true);
        assertEquals(Math.abs(eye.getValue(3, 3) - 1.0) < 0.001, true);
    }
}
