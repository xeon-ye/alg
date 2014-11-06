package zju.dsmodel;

/**
 * Created by IntelliJ IDEA.
 * User: zhouyt
 * Date: 2010-7-9
 * Time: 16:17:34
 */

/**
 * This module of regulation is base on the analysis of
 * "Distribution System Modeling and Analysis, Second Edition"
 * in page 224 to 229
 */

public class RG_OpenDelta {

    /**
     * Global constants
     * these constants are on a 120-volt base
     */
    private final double DELTA_TAP = 0.00625;
    private final double DELTA_VOLTAGE = 0.00075;

    /**
     * some basic information of voltage regulator  s
     * Such as : ID, Line_segment, Location, Connection, monitoring_phase, Bandwidth
     * Line_segment[0] is the start node of the line segment where the voltage regulator locates
     * Line_segment[1] is the end node of the line segment where the voltage regulator locates
     * Location records the node where the voltage regulator locates
     * RGstate shows the position of the voltage regulator, which "R" stands for raise position,
     * while "L" represents the lower position. Since most of the regulator are operating
     * in raise position, we initialize RGstate as "R".
     */

    //data
    private double ID;  // whether to use double or not
    private double[] Line_segment = new double[2];  // whether to use double or not
    private double Location;
    private String RG_state = "R";
    private String[] Monitor_phase = new String[]{"AB", "CB"};
    private double Bandwidth;
    private double[] Voltage_level = new double[3];


    /**
     * the basic output parameters of voltage regulator
     * Vp and Ip :the voltage and current of the primary side
     * Vs and Is :the voltage and current of the secondary side
     * all the voltages are based on line-neutral voltage
     */
    private double[][] Vp = new double[3][2];
    private double[][] Ip = new double[3][2];
    private double[][] Vs = new double[3][2];
    private double[][] Is = new double[3][2];
    /**
     * basic inner parameters of voltage regulators
     * Tap :the position of the taps
     * R&X[0-2]: R&X[ab,bc,ca]; Tap[0-2]: Tap[ab,bc,ca]
     */
    private double[] Tap = new double[3];
    private double[] R = new double[3];
    private double[] X = new double[3];
    private double CTp;
    private double Npt;

    /**
     * the so-called generalized constants
     * a : used to calculate the voltage
     * d : used to calculate the current
     * b and c : generally equals to zero since the impedance and admittance is neglected
     * a,b,c,d are used to transform Vs to Vp
     * *************************************************************
     * A : used to calculate the voltage
     * D : used to calculate the current
     * B and C : generally equals to zero since the impedance and admittance is neglected
     * A,B,C,D are used to transform Vp to Vs
     */
    private double[][] a = new double[3][3];
    private double[][] b = new double[3][3];
    private double[][] c = new double[3][3];
    private double[][] d = new double[3][3];

    private double[][] A = new double[3][3];
    private double[][] B = new double[3][3];
    private double[][] C = new double[3][3];
    private double[][] D = new double[3][3];

    ////////////////////////////////////////////////////////////

    //temp_variable
    /**
     * Since in closed delta connection, the basic voltage relationship between primary side
     * and secondary side is based on line-line voltage, we need temporary variable VLLp and VLLs
     * to record the line voltage, then use some transforming methods to obtain the line-neutral voltage
     */
    private double[] aR = new double[3];
    private double[][] eq_voltageLC = new double[3][2];

    private double[][] VLLp = new double[3][2];
    private double[][] VLLs = new double[3][2];

    public RG_OpenDelta() {

    }


    /**********************************functions*********************************************/
    /**
     * INITIAL FUNCTION**
     * this function is used to initialize a Wye-connected regulator
     * In the processing of initialization, the basic constants will be set up,
     * and the generalized constants matrices will be set up to ONES or ZEROS
     */
    public RG_OpenDelta initial(double bw, double[] vl, double[] r, double[] x, double npt, double ctp) {
        this.Bandwidth = bw;
        this.Voltage_level = vl;
        this.R = r;
        this.X = x;
        this.Npt = npt;
        this.CTp = ctp;
        this.a = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        this.b = new double[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        this.c = new double[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        this.d = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        this.A = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        this.B = new double[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        this.C = new double[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        this.D = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

        return this;
    }

    /**
     * RESET FUNCTION**
     * this function used to recalculate the generalized constants matrices,
     * after the first calculating of power flow
     */
    public RG_OpenDelta reset() {
        this.cal_Tap();
        this.cal_Tap_ceil();
        this.cal_aR();
        this.cal_a();
        this.cal_d();
        this.cal_A();
        this.cal_D();

        return this;
    }

    /**
     * tool functions**
     */
    public double getID() {
        return this.ID;
    }

    public double[] getLine_segment() {
        return this.Line_segment;
    }

    public double getLocation() {
        return this.Location;
    }

    public String getRG_state() {
        return this.RG_state;
    }

    public double[][] getVp() {
        return this.Vp;
    }

    public double[][] getVs() {
        return this.Vs;
    }

    public double[][] getIp() {
        return this.Ip;
    }

    public double[][] getIs() {
        return this.Is;
    }

    public double[] getR() {
        return this.R;
    }

    public double[] getX() {
        return this.X;
    }

    public double getCTp() {
        return this.CTp;
    }

    public double getNpt() {
        return this.Npt;
    }

    public double[] getTap() {
        return this.Tap;
    }

    public double[][] get_a() {
        return this.a;
    }

    public double[][] get_b() {
        return this.b;
    }

    public double[][] get_c() {
        return this.c;
    }

    public double[][] get_d() {
        return this.d;
    }

    public double[][] get_A() {
        return this.A;
    }

    public double[][] get_B() {
        return this.B;
    }

    public double[][] get_C() {
        return this.C;
    }

    public double[][] get_D() {
        return this.D;
    }

    public String[] getMonitor_phase() {
        return Monitor_phase;
    }

    public double getBandwidth() {
        return this.Bandwidth;
    }

    public double[] getVoltage_level() {
        return this.Voltage_level;
    }

    public void setID(double id) {
        this.ID = id;
    }

    public void setLine_segment(double[] line_segment) {
        this.Line_segment = line_segment;
    }

    public void setLocation(double location) {
        this.Location = location;
    }

    public void setRG_state(String state) {
        this.RG_state = state;
    }

    public void setVp(double[][] vpp) {
        this.Vp = vpp;
    }

    public void setVs(double[][] vss) {
        this.Vs = vss;
    }

    public void setIp(double[][] ipp) {
        this.Ip = ipp;
    }

    public void setIs(double[][] iss) {
        this.Is = iss;
    }

    public void setR(double[] rr) {
        this.R = rr;
    }

    public void setX(double[] xx) {
        this.X = xx;
    }

    public void setCTp(double ctp) {
        this.CTp = ctp;
    }

    public void setNpt(double npt) {
        this.Npt = npt;
    }

    public void setTap(double[] tap) {
        this.Tap = tap;
    }

    public void set_a(double[][] aa) {
        this.a = aa;
    }

    public void set_b(double[][] bb) {
        this.b = bb;
    }

    public void set_c(double[][] cc) {
        this.c = cc;
    }

    public void set_d(double[][] dd) {
        this.d = dd;
    }

    public void set_A(double[][] AA) {
        this.A = AA;
    }

    public void set_B(double[][] BB) {
        this.B = BB;
    }

    public void set_C(double[][] CC) {
        this.C = CC;
    }

    public void set_D(double[][] DD) {
        this.D = DD;
    }

    public void setMonitor_phase(String[] monitor_phase) {
        Monitor_phase = monitor_phase;
    }

    public void setBandwidth(double bandwidth) {
        this.Bandwidth = bandwidth;
    }

    public void setVoltage_level(double[] voltage_level) {
        this.Voltage_level = voltage_level;

    }

    /***END OF BASIC TOOL FUNCTIONS***/

    /**
     * ESSENTIAL FUNCTIONS**
     */
    private void cal_aR() {
        if (RG_state.equals("R")) {
            aR[0] = 1 - DELTA_TAP * ((int) Tap[0] + 1);
            aR[1] = 1 - DELTA_TAP * ((int) Tap[1] + 1);
            aR[2] = 1 - DELTA_TAP * ((int) Tap[2] + 1);
        } else {
            aR[0] = 1 + DELTA_TAP * ((int) Tap[0] + 1);
            aR[1] = 1 + DELTA_TAP * ((int) Tap[1] + 1);
            aR[2] = 1 + DELTA_TAP * ((int) Tap[2] + 1);
        }
    }

    public void cal_a() {   //a_ll
        //[0-2]: [ab,bc,ca]
        a[0][0] = aR[0];
        a[1][1] = aR[1];
        a[2][2] = 0;
        a[0][1] = 0;
        a[0][2] = 0;
        a[1][0] = 0;
        a[1][2] = 0;
        a[2][0] = -aR[0];
        a[2][1] = -aR[1];

    }

    public void cal_b() {
        /**Generally Zt is neglected, so b = 0
         * If we want to consider the impedance later, we can add the elements for matrix b.
         */
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                b[i][j] = 0;
            }
        }
    }

    public void cal_c() {
        /**Generally Ym is neglected, so c = 0
         * If we want to consider the admittance later, we can add the elements for matrix c.
         */
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                c[i][j] = 0;
            }
        }
    }

    public void cal_d() { //d_ll
        /**Generally Zt and Ym is neglected, so the elements in matrix d is very simple.
         * If we want to consider the impedance and admittance later,
         * we have to recalculate the elements in matrix d in some more complex way, which is not mentioned.
         */
        //[0-2]: [ab,bc,ca]
        d[0][0] = 1 / aR[0];
        d[1][1] = 0;
        d[2][2] = 1 / aR[1];
        d[0][1] = 0;
        d[1][2] = -1 / aR[1];
        d[2][0] = 0;
        d[0][2] = 0;
        d[1][0] = -1 / aR[0];
        d[2][1] = 0;

    }

    public void cal_A() {

        A[0][0] = 1 / aR[0];
        A[1][1] = 1 / aR[1];
        A[2][2] = 0;
        A[0][2] = 0;
        A[1][0] = 0;
        A[2][1] = -1 / aR[1];
        A[0][1] = 0;
        A[1][2] = 0;
        A[2][0] = -1 / aR[0];

    }

    public void cal_B() {
        /**
         * in my module, in order to use the same side parameters to get the other side parameters,
         * I redefine the matrix B as {[a]^-1 * [b] * [d]^-1} instead of {[a]^-1 * [b]} in the book
         * What's more, as it is related with matrix d, which is simple when Zt is neglected (i.e. [b]=0)
         * we set matrix B zero, or we have to recalculate matrix d in complex way.
         */
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                B[i][j] = 0;
            }
        }
    }

    public void cal_C() {
        /**
         * It goes the same as matrix B
         */
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                C[i][j] = 0;
            }
        }

    }

    public void cal_D() {

        D[0][0] = aR[0];
        D[1][1] = 0;
        D[2][2] = aR[1];
        D[0][2] = 0;
        D[1][0] = -aR[0];
        D[2][1] = 0;
        D[0][1] = 0;
        D[1][2] = -aR[1];
        D[2][0] = 0;

    }


    public void Vp2Vs(double[][] vp, double[][] ip, double[][] Vs) {
        /**
         * we only know the exact relationship of the primary voltage and the secondary voltage
         * based on line-line voltage
         * Here, I use the transforming method which is used in the transformer,
         * to transform the line-line voltage to the line-neutral voltage,
         * also the transforming of the line-neutral voltage to line-line voltage is the same.
         * But it is not sure that this method can be used in this way.
         */
        //double k = 1;
        double k = 1000;
        Vp2VLLp(vp);//the line-neutral voltage to line-line voltage

        for (int i = 0; i < 3; i++) {
            VLLs[i][0] = 0;
            VLLs[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                VLLs[i][0] += A[i][j] * VLLp[j][0] - B[i][j] * ip[j][0] / k;
                VLLs[i][1] += A[i][j] * VLLp[j][1] - B[i][j] * ip[j][1] / k;
            }
        }

        //the line-line voltage to line-neutral voltage
        //todo:to transform VLLs to Vs
        /**
         * for testing, let VLLs equals to Vs
         */
        for (int i = 0; i < 3; i++) {
            Vs[i][0] = VLLs[i][0];
            Vs[i][1] = VLLs[i][1];
        }
    }

    public double[][] Vs2Vp(double[][] vs, double[][] is) {
        /**
         * we only know the exact relationship of the primary voltage and the secondary voltage
         * based on line-line voltage
         * Here, I use the transforming method which is used in the transformer,
         * to transform the line-line voltage to the line-neutral voltage,
         * also the transforming of the line-neutral voltage to line-line voltage is the same.
         * But it is not sure that this method can be used in this way.
         */
        // double k = 1;
        double k = 1000;
        Vs2VLLs(vs);
        for (int i = 0; i < 3; i++) {
            VLLp[i][0] = 0;
            VLLp[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                VLLp[i][0] += a[i][j] * VLLs[j][0] + b[i][j] * is[j][0] / k;
                VLLp[i][1] += a[i][j] * VLLs[j][1] + b[i][j] * is[j][1] / k;
            }
        }
        VLLp2Vp(VLLp);
        return Vp;

    }

    public void Ip2Is(double[][] vp, double[][] ip, double[][] Is) {
        /**
         * the calculating of currents is exactly right
         */
        //double k = 1;
        double k = 1000;
        for (int i = 0; i < 3; i++) {
            Is[i][0] = 0;
            Is[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                Is[i][0] += -C[i][j] * vp[j][0] * k + D[i][j] * ip[j][0];
                Is[i][1] += -C[i][j] * vp[j][1] * k + D[i][j] * ip[j][1];

            }
        }
    }

    public void Is2Ip(double[][] vs, double[][] is, double[][] Ip) {
        /**
         * the calculating of currents is exactly right
         */
        //double k=1;
        double k = 1000;
        for (int i = 0; i < 3; i++) {
            Ip[i][0] = 0;
            Ip[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                Ip[i][0] += c[i][j] * vs[j][0] * k + d[i][j] * is[j][0];
                Ip[i][1] += c[i][j] * vs[j][1] * k + d[i][j] * is[j][1];

            }
        }
    }


    //todo: add judge whether the regulator need to work

    public boolean isRegulatorWorks(double[][] vs, double[][] is) {
        return false;
    }

    public void cal_Tap() {
        double[] normVLC = new double[]{0, 0};
        //double k=1;
        double k = 1000;
        //todo: calculate the equivalent voltage at the load center
        if (Monitor_phase[0].equals("AB") && Monitor_phase[1].equals("CB")) {
            eq_voltageLC[0][0] = Vs[0][0] / Npt - (R[0] * Is[0][0] - X[0] * Is[0][1]) / CTp / k;
            eq_voltageLC[0][1] = Vs[0][1] / Npt - (R[0] * Is[0][1] + X[0] * Is[0][0]) / CTp / k;
            eq_voltageLC[1][0] = Vs[1][0] / Npt + (R[1] * Is[2][0] - X[1] * Is[2][1]) / CTp / k;
            eq_voltageLC[1][1] = Vs[1][1] / Npt + (R[1] * Is[2][1] + X[1] * Is[2][0]) / CTp / k;

            normVLC[0] = Math.sqrt((eq_voltageLC[0][0] * eq_voltageLC[0][0] + eq_voltageLC[0][1] * eq_voltageLC[0][1]));
            normVLC[1] = Math.sqrt((eq_voltageLC[1][0] * eq_voltageLC[1][0] + eq_voltageLC[1][1] * eq_voltageLC[1][1]));
            Tap[0] = Math.abs(((Voltage_level[0] - 0.5 * Bandwidth) - normVLC[0]) / DELTA_VOLTAGE);
            Tap[1] = Math.abs(((Voltage_level[1] - 0.5 * Bandwidth) - normVLC[1]) / DELTA_VOLTAGE);
            Tap[2] = -1;

            System.out.print("normVLC:" + normVLC[0] + "\n");
            System.out.print("normVLC:" + normVLC[1] + "\n");


        } else if (false) {

        } else {

        }
    }

    private void cal_Tap_ceil() {
        cal_Tap();
        Tap[0] = (int) (Tap[0]) + 1;
        Tap[1] = (int) (Tap[1]) + 1;
        Tap[2] = (int) (Tap[2]) + 1;
    }

    public void VLLp2Vp(double[][] vllp) {
        //todo:to transform VLLp to Vp
        /**
         * for testing, let VLLs equals to Vs
         */
        for (int i = 0; i < 3; i++) {
            Vp[i][0] = vllp[i][0];
            Vp[i][1] = vllp[i][1];
        }
    }

    public void Vp2VLLp(double[][] vp) {
        //todo:to transform Vp to VLLp
        /**
         * for testing, let Vp equals to VLLp
         */
        for (int i = 0; i < 3; i++) {
            VLLp[i][0] = vp[i][0];
            VLLp[i][1] = vp[i][1];
        }

    }


    public void Vs2VLLs(double[][] vs) {
        //todo:to transform Vs to VLLs
        /**
         * for testing, let Vs equals to VLLs
         */
        for (int i = 0; i < 3; i++) {
            VLLs[i][0] = vs[i][0];
            VLLs[i][1] = vs[i][1];
        }
    }

    public void calTailV(double[][] headV, double[][] tailI, double[][] tailV) {
        Vp2Vs(headV, tailI, tailV);
    }

    public void calHeadI(double[][] tailV, double[][] tailI, double[][] headI) {
        Is2Ip(tailV, tailI, headI);
    }

    public void calHeadV(double[][] tailV, double[][] tailI, double[][] headV) {
        //todo:
    }

    public void _debug_forceTap(double[] tap) {
        this.setTap(tap);
        this.cal_aR();
        this.cal_a();
        this.cal_d();
        this.cal_A();
        this.cal_D();
    }

    public void formPara() {
        //To change body of implemented methods use File | Settings | File Templates.
        //todo:unfinished
    }
}
