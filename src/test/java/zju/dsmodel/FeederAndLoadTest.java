package zju.dsmodel;

import junit.framework.TestCase;
import zju.matrix.AVector;
import zju.util.DoubleMatrixToolkit;

public class FeederAndLoadTest extends TestCase {
    public FeederAndLoadTest(String name) {
        super(name);
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void test3PLineInit() {
        Feeder line = new Feeder();
        assertNotNull(line);
    }

    //测试书《Distribution System Modeling and Analysis》中例子6.1
    public void testExample6p1() {
        Feeder line = new Feeder();
        double[][] z_real = new double[][]{{0.8667, 0.2955, 0.2907}, {0.2955, 0.8837, 0.2992}, {0.2907, 0.2992, 0.8741}};
        double[][] z_imag = new double[][]{{2.0417, 0.9502, 0.7290}, {0.9502, 1.9852, 0.8023}, {0.7290, 0.8023, 2.0172}};
        double[][] y_imag = new double[][]{{10.7409 * 1e-6, -3.4777 * 1e-6, -1.3322 * 1e-6}, {-3.4777 * 1e-6, 11.3208 * 1e-6, -2.2140 * 1e-6}, {-1.3322 * 1e-6, -2.2140 * 1e-6, 10.2104 * 1e-6}};
        line.setZ_real(z_real);
        line.setZ_imag(z_imag);
        line.setY_imag(y_imag);
        line.initialPhases();
        double[][] tailV = new double[][]{{7199.56, 0}, {-3599.78, -6235.00}, {-3599.78, 6235.00}};
        double[][] tailI = new double[][]{{250.02, -121.08}, {-229.86, -155.98}, {-20.15, 277.06}};
        double[][] headV = new double[][]{{7535.9, 206.5471}, {-3532.5, -6560.7}, {-3958.7, 6352.6}};
        double[][] headI = new double[][]{{249.9642, -120.9989}, {-229.7476, -156.0232}, {-20.1954, 276.9948}};

        double[][] calValue = new double[][]{{0, 0}, {0, 0}, {0, 0}};

        line.calHeadV(tailV, tailI, calValue);
        assertTrue(isDoubleMatrixEqual(headV, calValue, 0.5));

        line.calTailV(headV, tailI, calValue);
        assertTrue(isDoubleMatrixEqual(tailV, calValue, 0.5));

        line.calHeadI(tailV, tailI, calValue);
        assertTrue(isDoubleMatrixEqual(headI, calValue, 0.5));
    }

    //------------- test load starts --------------------------
    public void testBasicLoad() throws Exception {
        BasicLoad load = new BasicLoad();
        double[][] cs = {{2236.1 * 1e3, 26.6}, {2506 * 1e3, 28.6}, {2101.4 * 1e3, 25.3}};
        double[][] vb = {{7200., 0.}, {7200., -120.}, {7200., 120.}};
        trans_polar2rect_deg(cs);
        trans_polar2rect_deg(vb);

        double[][] expectI = {{93.2, -26.6}, {104.4, -148.6}, {87.6, 94.7}};
        trans_polar2rect_deg(expectI);
        load.setMode(DsModelCons.LOAD_Y_I);
        load.formPara(cs, 7200);
        double[][] calculatedI = {{0, 0}, {0, 0}, {0, 0}};
        load.calI(vb, calculatedI);
        DoubleMatrixToolkit.selfMul(calculatedI, 0.3);
        assertTrue(isDoubleMatrixEqual(calculatedI, expectI, 0.05));

        double[][] expectZ = {{20.7, 10.4}, {18.2, 9.9}, {22.3, 10.6}};
        expectI = new double[][]{{62.1, -26.6}, {69.6, -148.6}, {58.4, 94.7}};
        trans_polar2rect_deg(expectI);
        load.setMode(DsModelCons.LOAD_Y_Z);
        load.formPara(cs, 7200);
        load.calI(vb, calculatedI);
        assertTrue(isDoubleMatrixEqual(expectZ, load.getConstantZ(), 0.06));
        DoubleMatrixToolkit.selfMul(calculatedI, 0.2);
        assertTrue(isDoubleMatrixEqual(calculatedI, expectI, 0.05));

        DoubleMatrixToolkit.selfMul(expectZ, 3.);
        expectI = new double[][]{{54.93, -25.64}, {-55.00, -37.33}, {0.07, 62.97}};
        load.setMode(DsModelCons.LOAD_D_Z);
        load.formPara(cs, 7200);
        double[][] calculatedZ = load.getConstantZ();
        assertTrue(isDoubleMatrixEqual(expectZ, calculatedZ, 0.18));
        load.calI(vb, calculatedI);
        DoubleMatrixToolkit.selfMul(calculatedI, 0.2);
        assertTrue(isDoubleMatrixEqual(calculatedI, expectI, 0.09));

        expectI = new double[][]{{155.3, -26.6}, {174., -148.6}, {146., 94.7}};
        load.setMode(DsModelCons.LOAD_Y_PQ);
        load.formPara(cs, 7200);
        load.calI(vb, calculatedI);
        DoubleMatrixToolkit.selfMul(calculatedI, 0.5);
        trans_rect2polar_deg(calculatedI);
        assertTrue(isDoubleMatrixEqual(calculatedI, expectI, 0.09));

        //vb = new double[][]{{7200. / Math.sqrt(3), 30.}, {7200. / Math.sqrt(3), -90.}, {7200. / Math.sqrt(3), 150.}};
        //trans_polar2rect_deg(vb);
        expectI = new double[][]{{155.3, -26.6}, {174., -148.6}, {146., 94.7}};
        trans_polar2rect_deg(expectI);
        double[][] expectILine = new double[3][2];
        for (int i = 0; i < 3; i++)
            expectILine[i][0] = expectI[i][0] - expectI[(i + 2) % 3][0];
        load.setMode(DsModelCons.LOAD_D_PQ);
        load.formPara(cs, 7200);
        assertEquals(load.getConstantS(), cs);
        load.calI(vb, calculatedI);
        DoubleMatrixToolkit.selfMul(calculatedI, 0.5);
        //assertTrue(isDoubleMatrixEqual(calculatedI, expectILine, 0.01));
        //book pp.296,297
    }

    //书上266页, Example 9.2
    public void testIM() {
        InductionMachine im = new InductionMachine();
        im.setUp(0.0774, 0.1843, 0.0908, 0.1843, 4.8384);
        im.formPara(0.035);

        assertEquals(2.5029, im.getRL_12()[0], 1e-3);
        assertEquals(-0.0446, im.getRL_12()[1], 1e-4);
        assertEquals(1.9775, im.getZM_012()[1][0], 1e-3);
        assertEquals(1.3431, im.getZM_012()[1][1], 1e-3);
        assertEquals(0.1203, im.getZM_012()[2][0], 1e-3);
        assertEquals(0.3623, im.getZM_012()[2][1], 1e-3);

        double[][] VLL = {{235, 0}, {240, -117.9}, {245, 120.}};
        double[][] c_expected = {{53.15, -71}, {55.15, -175.1}, {66.6, 55.6}};
        double[][] c = new double[3][2];
        trans_polar2rect_deg(VLL);
        trans_polar2rect_deg(c_expected);
        im.setVLL_abc(VLL);
        im.calI(c);
        assertTrue(isDoubleMatrixEqual(c_expected, c, 0.5));
        assertEquals(0.0, c[0][0] + c[1][0] + c[2][0], 0.2);
        assertEquals(0.0, c[0][1] + c[1][1] + c[2][1], 0.2);
        assertEquals(0.0, VLL[0][0] + VLL[1][0] + VLL[2][0], 0.2);
        assertEquals(0.0, VLL[0][1] + VLL[1][1] + VLL[2][1], 0.2);

        //测试文献Induction Machine Test Case for the 34-Bus Test Feeder –Description中的例子
        double base_kva = 660, base_kv = 0.48;
        double baseZ = base_kv * base_kv / base_kva * 1000.;
        im.setUp(0.0053 * baseZ, 0.106 * baseZ, 0.007 * baseZ, 0.120 * baseZ, 4.0 * baseZ);
        im.setP(-660 * 1000.);
        //测试第一台分布式电源G1
        double[][] VLN_abc = {{281.58, 6}, {281.43, -115.3}, {282.44, 125.2}};
        double[][] VLL_abc = new double[3][2];
        trans_polar2rect_deg(VLN_abc);
        for (int j = 0; j < 3; j++) {
            VLL_abc[j][0] = VLN_abc[j][0] - VLN_abc[(j + 1) % 3][0];
            VLL_abc[j][1] = VLN_abc[j][1] - VLN_abc[(j + 1) % 3][1];
        }
        c_expected = new double[][]{{856.37, -147.3}, {891.82, 92}, {866.31, -29.7}};
        trans_polar2rect_deg(c_expected);
        im.formPara(-0.00751);
        im.setVLL_abc(VLL_abc);
        im.calI(c);
        assertTrue(isDoubleMatrixEqual(c_expected, c, 2));

        AVector z_est = im.calZ(new AVector(new double[]{-741.64, 20.75, -457.44, -5.01, -0.00751}));
        assertEquals(0.0, z_est.getValue(0), 1e-3);
        assertEquals(0.0, z_est.getValue(1), 1e-2);
        assertEquals(0.0, z_est.getValue(2), 0.4);
        assertEquals(0.0, z_est.getValue(3), 1e-3);
        assertEquals(0.0, z_est.getValue(4), 10);

        im.setSlip(0.0);
        im.calI(VLN_abc, c);
        assertEquals(-0.00751, im.getSlip(),1e-5);
        assertTrue(isDoubleMatrixEqual(c_expected, c, 2));

        //测试第二台分布式电源G2
        VLN_abc = new double[][]{{259.25, 14.8}, {257.69, -106.9}, {260.13, 133.8}};
        trans_polar2rect_deg(VLN_abc);
        //System.out.println(Arrays.deepToString(VLN_abc));
        for (int j = 0; j < 3; j++) {
            VLL_abc[j][0] = VLN_abc[j][0] - VLN_abc[(j + 1) % 3][0];
            VLL_abc[j][1] = VLN_abc[j][1] - VLN_abc[(j + 1) % 3][1];
        }
        c_expected = new double[][]{{943.55, -137.9}, {975.29, 100.8}, {941.78, -20.2}};
        trans_polar2rect_deg(c_expected);
        im.formPara(-0.00912);
        im.setVLL_abc(VLL_abc);
        im.calI(c);
        assertTrue(isDoubleMatrixEqual(c_expected, c, 2));

        im.setSlip(0.0);
        im.calI(VLN_abc, c);
        assertEquals(-0.00912, im.getSlip(),1e-4);
        assertTrue(isDoubleMatrixEqual(c_expected, c, 3));
    }

    public static boolean isDoubleMatrixEqual(double[][] a, double[][] b, double eps) {
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                if (Math.abs(a[i][j] - b[i][j]) > eps) {
                    // System.out.println("a:" + a[i][j] + "  b:" +b[i][j]);
                    return false;
                }
            }
        }
        return true;
    }

    public static void trans_polar2rect_deg(double[][] toTrans) {
        for (int i = 0; i < toTrans.length; i++) {
            double a = toTrans[i][0];
            double b = toTrans[i][1];
            toTrans[i][0] = a * Math.cos(b * Math.PI / 180.0);
            toTrans[i][1] = a * Math.sin(b * Math.PI / 180.0);
        }
    }

    public static void trans_rect2polar_deg(double[][] toTrans) {
        for (int i = 0; i < toTrans.length; i++) {
            double a = toTrans[i][0];
            double b = toTrans[i][1];
            toTrans[i][0] = Math.sqrt(a * a + b * b);
            toTrans[i][1] = Math.atan2(b, a) * 180. / Math.PI;
        }
    }
}
