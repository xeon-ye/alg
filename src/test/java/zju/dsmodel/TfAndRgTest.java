package zju.dsmodel;

import junit.framework.TestCase;

public class TfAndRgTest extends TestCase {

    public void testTransformer() {
        //书本《Distribution System Modeling and Analysis》Example 8.1
        double r_pu = 0.085 * Math.cos(85. * Math.PI / 180.);
        double x_pu = 0.085 * Math.sin(85. * Math.PI / 180.);
        Transformer tf = new Transformer(138, 12.47, 5000, r_pu, x_pu);
        tf.setConnType(Transformer.CONN_TYPE_D_GrY);
        tf.formPara();
        assertTrue(Math.abs(tf.getR()[0] - 0.2304) < 1e-3);
        assertTrue(Math.abs(tf.getX()[0] - 2.6335) < 1e-3);

        //double[][] Vtabc = {{7614.8, -56.9}, {7199.6, 180}, {7069., 64.5}};
        //FeederAndLoadTest.trans_polar2rect_deg(Vtabc);
        //double[][] Vtabc = {{2.89164, -6.20114}, {-6.51331, 1.03161}, {3.65914, 5.48665}};
        double[][] Iabc = {{484.1, -93.0}, {470.7, 151.5}, {425.4, 34.8}};
        FeederAndLoadTest.trans_polar2rect_deg(Iabc);
        double[][] Iabc_rect = {{-25.336, -483.44}, {-413.66, 224.6}, {349.32, 242.78}};
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(Iabc, Iabc_rect, 1.0));
        double[][] VLNABC = {{83224, -29.3}, {77103, -148.1}, {81843, 95.0}};
        //double[][] VLNABC_rect = {{72198.4, -41214.0}, {-66189.5, -41822.9}, {-7141.8, 81556.0}};
        FeederAndLoadTest.trans_polar2rect_deg(VLNABC);
        //assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(VLNABC, VLNABC_rect, 5.0));
        //书里面第C相电压有的地方错写成6594.9，注意
        double[][] VLGabc = {{6842.2, -65.0}, {6594.5, 171.0}, {6494.9, 56.3}};
        FeederAndLoadTest.trans_polar2rect_deg(VLGabc);

        double[][] calValue = new double[][]{{0, 0}, {0, 0}, {0, 0}};
        tf.calTailV(VLNABC, Iabc, calValue);
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(VLGabc, calValue, 10));

        tf.calHeadV(VLGabc, Iabc, calValue);
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(VLNABC, calValue, 50));

        //书本《Distribution System Modeling and Analysis》Example 8.2
        tf.setConnType(Transformer.CONN_TYPE_Y_D);
        tf.setVLL_high_rated(7.2 * Math.sqrt(3));
        tf.setVLL_low_rated(0.24);
        tf.setSn(new double[]{100.0, 50.0, 50.0});
        tf.setR_pu(new double[]{0.01, 0.015, 0.015});
        //书里面有的地方错写成0.35，注意
        tf.setX_pu(new double[]{0.04, 0.035, 0.035});
        tf.formPara();
        assertTrue(Math.abs(tf.getNt() - 30.0) < 1e-3);
        assertTrue(Math.abs(tf.getR()[0] - 0.0058) < 1e-3);
        //书里面有的地方错写成0.23，注意
        assertTrue(Math.abs(tf.getX()[0] - 0.023) < 1e-2);
        assertTrue(Math.abs(tf.getR()[1] - 0.017) < 1e-3);
        assertTrue(Math.abs(tf.getX()[1] - 0.0403) < 1e-3);
        assertTrue(Math.abs(tf.getR()[2] - 0.017) < 1e-3);
        assertTrue(Math.abs(tf.getX()[2] - 0.0403) < 1e-3);

        double[][] IABC = {{11.54, -28.04}, {8.95, -166.43}, {7.68, 101.16}};
        FeederAndLoadTest.trans_polar2rect_deg(IABC);
        //注意书上B相相角-119.06是错的
        Iabc = new double[][]{{522.9, -47.97}, {575.3, 170.01}, {360.8, 53.13}};
        FeederAndLoadTest.trans_polar2rect_deg(Iabc);
        VLNABC = new double[][]{{7367.6, 1.4}, {7532.3, -119.1}, {7406.2, 121.7}};
        FeederAndLoadTest.trans_polar2rect_deg(VLNABC);
        VLGabc = new double[][]{{138.56, -30}, {138.56, -150}, {138.56, 90}};
        FeederAndLoadTest.trans_polar2rect_deg(VLGabc);

        double[][] IDabc = {{416.7, -25.84}, {208.3, -156.87}, {208.3, 83.13}};
        FeederAndLoadTest.trans_polar2rect_deg(IDabc);
        //Iabc[0][0] = IDabc[0][0] - IDabc[2][0];
        //Iabc[0][1] = IDabc[0][1] - IDabc[2][1];
        //Iabc[1][0] = IDabc[1][0] - IDabc[0][0];
        //Iabc[1][1] = IDabc[1][1] - IDabc[0][1];
        //Iabc[2][0] = IDabc[2][0] - IDabc[1][0];
        //Iabc[2][1] = IDabc[2][1] - IDabc[1][1];

        assertTrue(Math.abs(Iabc[0][0] - IDabc[0][0] + IDabc[2][0]) < 1);
        assertTrue(Math.abs(Iabc[0][1] - IDabc[0][1] + IDabc[2][1]) < 1);
        assertTrue(Math.abs(Iabc[1][0] - IDabc[1][0] + IDabc[0][0]) < 1);
        assertTrue(Math.abs(Iabc[1][1] - IDabc[1][1] + IDabc[0][1]) < 1);
        assertTrue(Math.abs(Iabc[2][0] - IDabc[2][0] + IDabc[1][0]) < 1);
        assertTrue(Math.abs(Iabc[2][1] - IDabc[2][1] + IDabc[1][1]) < 1);


        tf.calHeadI(VLGabc, Iabc, calValue);
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(IABC, calValue, 1));

        tf.calHeadV(VLGabc, Iabc, calValue);
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(VLNABC, calValue, 5));

        tf.calTailV(VLNABC, Iabc, calValue);
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(VLGabc, calValue, 1e-1));

        //The test example is derived from the IEEE 4 Node test feeder power flow study
        tf.setConnType(Transformer.CONN_TYPE_GrY_GrY);
        tf.setVLL_high_rated(12.47);
        tf.setVLL_low_rated(4.16);
        tf.setTotalSn(1.0);
        tf.formPara();
        tf.setR(0.028842666666667);
        tf.setX(0.173056);

        VLNABC = new double[][]{
                {7139.058463169212, -37.24305083598065},
                {-3611.867993558103, -6176.417837969006},
                {-3535.149721908844, 6213.88786308571}};
        Iabc = new double[][]{
                {674.496606649087, -326.673612142448},
                {-620.156252001986, -420.794411995144},
                {-54.3408263734955, 747.468252605109}};
        double[][] Iabc2 = {
                {723.993517952942, -431.071146930717},
                {-716.248907092631, -395.532978233428},
                {24.1314618655373, 823.267046549888}};
        VLGabc = new double[][]{
                {2305.607373886339, -119.7278523082225},
                {-1259.855517512164, -1940.998359856850},
                {-1048.407014903354, 2060.802005976205}};
        IABC = new double[][]{
                {241.524702059682, -143.805611165339},
                {-238.941094908207, -131.950055288778},
                {8.05027115963392, 274.642414887533}};

        tf.calHeadI(VLGabc, Iabc2, calValue);
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(IABC, calValue, 1e-3));

        tf.calHeadV(VLGabc, Iabc, calValue);
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(VLNABC, calValue, 1e-3));

        tf.calTailV(VLNABC, Iabc, calValue);
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(VLGabc, calValue, 1e-3));

        //============================ D-D ======================================
        tf.setConnType(Transformer.CONN_TYPE_D_D);
        tf.setVLL_high_rated(12.470);
        tf.setVLL_low_rated(0.24);
        tf.setSn(new double[]{100.0, 50.0, 50.0});
        tf.setR_pu(new double[]{0.01, 0.015, 0.015});
        tf.setX_pu(new double[]{0.04, 0.035, 0.035});
        tf.formPara();
        VLGabc = new double[][]{{119.9964799, -69.28}, {-119.9964799, -69.28}, {0, 138.56}};
        Iabc = new double[][]{{350.1118972, -388.4294587}, {-566.5871759, 99.8026495}, {216.5045157, 288.671632}};
        VLNABC = new double[][]{{6479.006659, -3536.871195}, {-6342.222222, -3900.235436}, {-136.3081697, 7437.150976}};
        IABC = new double[][]{{6.735367421, -7.47252511339}, {-10.90215716, 1.920382571}, {4.170009532, 5.559992551}};

        tf.calHeadI(VLGabc, Iabc, calValue);
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(IABC, calValue, 1e-2));

        tf.calHeadV(VLGabc, Iabc, calValue);
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(VLNABC, calValue, 1));

        tf.calTailV(VLNABC, Iabc, calValue);
        assertTrue(FeederAndLoadTest.isDoubleMatrixEqual(VLGabc, calValue, 1e-2));
    }

    //------------------------- test regulator starts ----------------

    /**
     * Test Wye-Connected Regulator
     */
    public void testRG_Wye_Connected() {

        double Bandwidth = 0.002;
        //double[] Voltage_level = new double[]{120, 120, 120};
        double[] Voltage_level = new double[]{0.120, 0.120, 0.120};
        double[] R = new double[]{5.346, 5.628, 6.386};
        double[] X = new double[]{12.385, 8.723, 14.18};
        double Npt = 60;
        double CTp = 600;

        double[][] temp = new double[][]{{0, 0}, {-0, -0}, {-0, 0}};
        double[][] temp2 = new double[][]{{0, 0}, {-0, -0}, {-0, 0}};

        double[][] SubBus_voltage = new double[][]{{7.200, 0}, {-3.600, -6.23538}, {-3.600, 6.23538}};
        double[][] SubBus_current = new double[][]{{242.44, -88.24}, {-241.54, -156.86}, {22.60, 323.21}};
        double[][] V_reg = new double[][]{{7.3846, 0}, {-3.71615, -6.43656}, {-3.81457, 6.60703}};
        RG_Wye rg_y = new RG_Wye();

        rg_y = rg_y.initial(Bandwidth, Voltage_level, R, X, Npt, CTp);

        rg_y.calTailI(SubBus_voltage, SubBus_current, temp2);
        rg_y.calTailV(SubBus_voltage, SubBus_current, temp);

        rg_y.isRegulatorWorks(temp, temp2);

        rg_y = rg_y.reset();

        rg_y.calTailV(SubBus_voltage, SubBus_current, temp);

        for (int k = 0; k < 3; k++) {
            System.out.print("\n" + "=======begin=============");
            System.out.print("\n" + "Tap:" + rg_y.getTap()[k]);
            System.out.print("\n" + "A[i][i]:" + rg_y.get_A()[k][k]);
            //System.out.print("\n" + "expected_V_reg:" + norm(V_reg[k]));
            //System.out.print("\n" + "cal_rlt_V_reg:" + norm(temp[k]));
            System.out.print("\n" + "=========end===========");
        }
        //for (int i = 0; i < 3; i++)
        //    assertTrue(eq_double(norm(V_reg[i]), norm(temp[i])));
        System.out.print("\n" + "\n" + "============TESTING Wye FINISHED=============" + "\n" + "\n");
    }

    /**
     * Test OpenDelta_Connected Regulator
     */
    public void testRG_OpenDelta_Connected() {
        double Bandwidth = 0.002;
        //double[] Voltage_level = new double[]{120, 120, 120};
        double[] Voltage_level = new double[]{0.120, 0.120, 0.120};
        double[] R = new double[]{0.8, 7.2, 0};
        double[] X = new double[]{9.9, 6.7, 0};
        double Npt = 103.92;
        double CTp = 500;

/*
        double[][] SubBus_voltage = new double[][]{{12470, 0}, {-6235, -10799.34}, {-6235, 10799.34}};
        double[][] SubBus_current = new double[][]{{163.32, -261.37}, {-263.59, -17.97}, {100.12, 279.62}};
        double[][] V_reg = new double[][]{{12956, 0}, {-6395, -11076.46}, {-6437, 11149.21}};
*/

        double[][] SubBus_voltage = new double[][]{{12.470, 0}, {-6.235, -10.79934}, {-6.235, 10.79934}};
        double[][] SubBus_current = new double[][]{{163.32, -261.37}, {-263.59, -17.97}, {100.12, 279.62}};
        double[][] V_reg = new double[][]{{12.956, 0}, {-6.395, -11.07646}, {-6.437, 11.14921}};

        RG_OpenDelta rg_od = new RG_OpenDelta();

        rg_od.initial(Bandwidth, Voltage_level, R, X, Npt, CTp);

        double[][] temp = new double[][]{{0, 0}, {-0, -0}, {-0, 0}};
        double[][] temp2 = new double[][]{{0, 0}, {-0, -0}, {-0, 0}};

        rg_od.Vp2Vs(SubBus_voltage, SubBus_current, temp);
        rg_od.Ip2Is(SubBus_voltage, SubBus_current, temp2);

        rg_od.reset();

        rg_od.Vp2Vs(SubBus_voltage, SubBus_current, temp);

        for (int k = 0; k < 3; k++) {
            System.out.print("\n" + "=======begin=============");
            System.out.print("\n" + "Tap:" + rg_od.getTap()[k]);
            System.out.print("\n" + "A[i][i]:" + rg_od.get_A()[k][k]);
            //System.out.print("\n" + "expected_V_reg:" + norm(V_reg[k]));
            //System.out.print("\n" + "cal_rlt_V_reg:" + norm(temp[k]));
            System.out.print("\n" + "=========end===========");
        }

        //for (int i = 0; i < 3; i++)
        //    assertTrue(eq_double(norm(V_reg[i]), norm(temp[i])));
    }

    private double[][] mulDoubleMatrix(double[][] a, double b) {
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] * b;
            }
        }
        return c;
    }

}
