package zju.matrix;

import junit.framework.TestCase;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-1-1
 */
public class ASparseMatrixLinkTest extends TestCase {

    public static ASparseMatrixLink getTestArray() {
        ASparseMatrixLink array = new ASparseMatrixLink(4);
        array.increase(0, 0, 1);
        array.increase(0, 1, 1);
        array.increase(0, 3, 1);
        array.increase(1, 0, 1);
        array.increase(1, 1, 1);
        array.increase(1, 2, 1);
        array.increase(2, 2, 1);
        array.increase(3, 1, 1);
        array.increase(3, 2, 1);
        array.increase(3, 3, 1);
        return array;
    }

    public static ASparseMatrixLink getTestArray2() {
        ASparseMatrixLink array = new ASparseMatrixLink(4);
        array.increase(1, 0, 1);
        array.increase(1, 1, 2);
        array.increase(1, 2, 3);
        array.increase(0, 0, 4);
        array.increase(2, 2, 5);
        array.increase(3, 1, 6);
        array.increase(3, 2, 7);
        array.increase(0, 1, 8);
        array.increase(0, 3, 9);
        array.increase(3, 3, 10);
        return array;
    }

    public void testInsert() {
        ASparseMatrixLink array = getTestArray();
        ASparseMatrixLink array2 = getTestArray2(); //different order in set values, JA,VA

        assertSame(array.getIA()[0], 0);
        assertSame(array.getIA()[1], 3);
        assertSame(array.getIA()[2], 6);
        assertSame(array.getIA()[3], 7);

        assertSame(array.getNA()[0], 3);
        assertSame(array.getNA()[1], 3);
        assertSame(array.getNA()[2], 1);
        assertSame(array.getNA()[3], 3);

        assertSame(array.getJA().get(0), 0);
        assertSame(array.getJA().get(1), 1);
        assertSame(array.getJA().get(2), 3);
        assertSame(array.getJA().get(3), 0);
        assertSame(array.getJA().get(4), 1);
        assertSame(array.getJA().get(5), 2);
        assertSame(array.getJA().get(6), 2);
        assertSame(array.getJA().get(7), 1);
        assertSame(array.getJA().get(8), 2);
        assertSame(array.getJA().get(9), 3);

        assertSame(array.getLINK().get(0), 1);
        assertSame(array.getLINK().get(1), 2);
        assertSame(array.getLINK().get(2), -1);
        assertSame(array.getLINK().get(3), 4);
        assertSame(array.getLINK().get(4), 5);
        assertSame(array.getLINK().get(5), -1);
        assertSame(array.getLINK().get(6), -1);
        assertSame(array.getLINK().get(7), 8);
        assertSame(array.getLINK().get(8), 9);
        assertSame(array.getLINK().get(9), -1);

        assertSame(array.getVA().get(0).intValue(), 1);
        assertSame(array.getVA().get(1).intValue(), 1);
        assertSame(array.getVA().get(2).intValue(), 1);
        assertSame(array.getVA().get(3).intValue(), 1);
        assertSame(array.getVA().get(4).intValue(), 1);
        assertSame(array.getVA().get(5).intValue(), 1);
        assertSame(array.getVA().get(6).intValue(), 1);
        assertSame(array.getVA().get(7).intValue(), 1);
        assertSame(array.getVA().get(8).intValue(), 1);
        assertSame(array.getVA().get(9).intValue(), 1);

        array.printOnScreen();
        for (int i = 0; i < array.getVA().size(); i++) {
            System.out.print(array.getVA().get(i) + "\t");
        }
        System.out.println();
        for (int i = 0; i < array.getJA().size(); i++) {
            System.out.print(array.getJA().get(i) + "\t");
        }
        System.out.println();
        for (int i = 0; i < array.getLINK().size(); i++) {
            System.out.print(array.getLINK().get(i) + "\t");
        }
        System.out.println();
        for (int i = 0; i < array.getIA().length; i++) {
            System.out.print(array.getIA()[i] + "\t");
        }
        System.out.println();
        for (int i = 0; i < array.getNA().length; i++) {
            System.out.print(array.getNA()[i] + "\t");
        }
        System.out.println();

        array2.printOnScreen();
        for (int i = 0; i < array2.getVA().size(); i++) {
            System.out.print(array2.getVA().get(i) + "\t");
        }
        System.out.println();
        for (int i = 0; i < array2.getJA().size(); i++) {
            System.out.print(array2.getJA().get(i) + "\t");
        }
        System.out.println();
        for (int i = 0; i < array2.getLINK().size(); i++) {
            System.out.print(array2.getLINK().get(i) + "\t");
        }
        System.out.println();
        for (int i = 0; i < array2.getIA().length; i++) {
            System.out.print(array2.getIA()[i] + "\t");
        }
        System.out.println();
        for (int i = 0; i < array2.getNA().length; i++) {
            System.out.print(array2.getNA()[i] + "\t");
        }
        System.out.println();

    }


    public void testSparseMatrixSet() {
        //ASparseMatrixLink2D resultColt = new ASparseMatrixLink2D(8000, 2000);
        ASparseMatrixLink result = new ASparseMatrixLink(8000, 2000);
        long start = System.currentTimeMillis();
        for (int i = 0; i < 8000; i++) {
            for (int j = 0; j < 100; j++) {
                result.setValue(i, i % 1000 + j, 1);
            }
        }
        System.out.println("time used is " + (System.currentTimeMillis() - start) + " ms");

        start = System.currentTimeMillis();
        for (int i = 0; i < 8000; i++) {
            for (int j = 0; j < 100; j++) {
                result.setValue(i, i % 1000 + j, 1);
            }
        }
        System.out.println("time used is " + (System.currentTimeMillis() - start) + " ms");
    }

}

