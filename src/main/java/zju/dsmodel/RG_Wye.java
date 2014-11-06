package zju.dsmodel;

/**
 * This module of regulation is base on the analysis of
 * "Distribution System Modeling and Analysis, Second Edition"
 * in page 215 to 217
 */
public class RG_Wye {
    //0907.Mikoyan.
    //here now, we don't have any items showing whether the the regulator is of type A or B.
    //So set a parameter used for debug only.
    private final boolean isTypeB = false;

    /**
     * Global constants
     * these constants are on a 120-volt base
     */
    private final double DELTA_TAP = 0.00625;
    private final double DELTA_VOLTAGE = 0.00075;

    /**
     * implements ThreePhaseRG
     * some basic information of voltage regulators
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
    private String[] Monitor_phase = new String[]{"A", "B", "C"};
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

    /**
     * temp_variable
     * eq_voltageLC: represents the equivalent voltage of the load center,
     * which is calculated from the output current of the regulator with the help of R and X
     * aR: it records the position of the taps of each phase
     */

    private double[][] eq_voltageLC = new double[3][2];
    private double[] aR = new double[3];

    public RG_Wye() {

    }

    /**********************************functions*********************************************/
    /**
     * INITIAL FUNCTION**
     * this function is used to initialize a Wye-connected regulator
     * In the processing of initialization, the basic constants will be set up,
     * and the generalized constants matrices will be set up to ONES or ZEROS
     */
    public RG_Wye initial(double bw, double[] vl, double[] r, double[] x, double npt, double ctp) {

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
        //am_real = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        //bm_real = new double[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        //cm_real = new double[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        //dm_real = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        //A_real = new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        //B_real = new double[][]{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        return this;
    }

    /**
     * RESET FUNCTION**
     * this function used to recalculate the generalized constants matrices,
     * after the first calculating of power flow
     */
    public RG_Wye reset() {
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
    public void cal_aR() {
        if (isTypeB) {//typeB regulator.
            if (RG_state.equals("R")) {
                aR[0] = 1 - DELTA_TAP * Math.ceil(Tap[0]);
                aR[1] = 1 - DELTA_TAP * Math.ceil(Tap[1]);
                aR[2] = 1 - DELTA_TAP * Math.ceil(Tap[2]);
            } else {
                aR[0] = 1 + DELTA_TAP * Math.ceil(Tap[0]);
                aR[1] = 1 + DELTA_TAP * Math.ceil(Tap[1]);
                aR[2] = 1 + DELTA_TAP * Math.ceil(Tap[2]);
            }
        } else { //typeA regulator.
            if (!RG_state.equals("R")) {
                aR[0] = 1 - DELTA_TAP * Math.ceil(Tap[0]);
                aR[1] = 1 - DELTA_TAP * Math.ceil(Tap[1]);
                aR[2] = 1 - DELTA_TAP * Math.ceil(Tap[2]);
            } else {
                aR[0] = 1 + DELTA_TAP * Math.ceil(Tap[0]);
                aR[1] = 1 + DELTA_TAP * Math.ceil(Tap[1]);
                aR[2] = 1 + DELTA_TAP * Math.ceil(Tap[2]);
            }
        }
    }

    public void cal_a() {
        if (isTypeB) {
            a[0][0] = aR[0];
            a[1][1] = aR[1];
            a[2][2] = aR[2];
        } else {
            a[0][0] = 1. / aR[0];
            a[1][1] = 1. / aR[1];
            a[2][2] = 1. / aR[2];
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i != j)
                    a[i][j] = 0;
            }
        }

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

    public void cal_d() {
        /**Generally Zt and Ym is neglected, so the elements in matrix d is very simple.
         * If we want to consider the impedance and admittance later,
         * we have to recalculate the elements in matrix d in some more complex way, which is not mentioned.
         */
        if (isTypeB) {
            d[0][0] = 1. / aR[0];
            d[1][1] = 1. / aR[1];
            d[2][2] = 1. / aR[2];
        } else {
            d[0][0] = aR[0];
            d[1][1] = aR[1];
            d[2][2] = aR[2];
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i != j)
                    d[i][j] = 0;
            }
        }

    }

    public void cal_A() {  //matrix A is the inverting of matrix a

        if (isTypeB) {
            A[0][0] = 1. / aR[0];
            A[1][1] = 1. / aR[1];
            A[2][2] = 1. / aR[2];
        } else {
            A[0][0] = aR[0];
            A[1][1] = aR[1];
            A[2][2] = aR[2];
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i != j)
                    A[i][j] = 0;
            }
        }
    }

    public void cal_B() {
        /**
         * in my module, in order to use the same side parameters to get the other side parameters,
         * I redefine the matrix B as {[a]^-1 * [b] * [d]^-1} instead of {[a]^-1 * [b]} in the book
         * What's more, as it is related with matrix d, which is simple when Zt is neglected (i.e. [b]=0)
         * we set matrix B zero, or we have to recalculate matrix d in MORE complex way.
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

    public void cal_D() {//matrix D is the inverting of matrix d

        if (isTypeB) {
            D[0][0] = aR[0];
            D[1][1] = aR[1];
            D[2][2] = aR[2];
        } else {
            D[0][0] = 1. / aR[0];
            D[1][1] = 1. / aR[1];
            D[2][2] = 1. / aR[2];
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (i != j)
                    D[i][j] = 0;
            }
        }
    }

    public void Vs2Vp(double[][] vs, double[][] is, double[][] v) {
        //whether we consider the efficiency of the process!!!!!!!!
        /**
         for (int i = 0; i < 3; i++) {
         v[i][0] = vs[i][0] * a[i][i];
         v[i][1] = vs[i][1] * a[i][i];
         }
         *the method above is more effective then the one below
         */
        for (int i = 0; i < 3; i++) {
            double k = 1000;
            v[i][0] = 0;
            v[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                v[i][0] += a[i][j] * vs[j][0] + b[i][j] * is[j][0] / k;
                v[i][1] += a[i][j] * vs[j][1] + b[i][j] * is[j][1] / k;
            }
        }
    }

    public void Vp2Vs(double[][] vp, double[][] ip, double[][] v) {
        //whether we consider the efficiency of the process!!!!!!!!
        /**
         for (int i = 0; i < 3; i++) {
         v[i][0] = vp[i][0] * a[i][i];
         v[i][1] = vp[i][1] * a[i][i];
         }
         *the method above is more effective then the one below
         */
        for (int i = 0; i < 3; i++) {
            double k = 1000;
            v[i][0] = 0;
            v[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                v[i][0] += A[i][j] * vp[j][0] - B[i][j] * ip[j][0] / k;
                v[i][1] += A[i][j] * vp[j][1] - B[i][j] * ip[j][1] / k;
            }
        }
    }

    public void Is2Ip(double[][] vs, double[][] is, double[][] c) {
        /**
         for (int i = 0; i < 3; i++) {
         c[i][0] = is[i][0] * d[i][i];
         c[i][1] = is[i][1] * d[i][i];
         }
         *the method above is more effective then the one below
         */
        double k = 1000;
        for (int i = 0; i < 3; i++) {
            c[i][0] = 0;
            c[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                c[i][0] += this.c[i][j] * vs[j][0] * k + d[i][j] * is[j][0];
                c[i][1] += this.c[i][j] * vs[j][1] * k + d[i][j] * is[j][1];
            }
        }
    }

    public void Ip2Is(double[][] vp, double[][] ip, double[][] v) {
        /**
         for (int i = 0; i < 3; i++) {
         v[i][0] = ip[i][0] / d[i][i];
         v[i][1] = ip[i][1] / d[i][i];
         }
         *the method above is more effective then the one below
         */
        double k = 1000;
        for (int i = 0; i < 3; i++) {
            v[i][0] = 0;
            v[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                v[i][0] += -C[i][j] * vp[j][0] * k + D[i][j] * ip[j][0];
                v[i][1] += -C[i][j] * vp[j][1] * k + D[i][j] * ip[j][1];
            }
        }
    }

    //todo:No, No, No ~

    public boolean isRegulatorWorks(double[][] vs, double[][] is) {
        double para = 1000;
        double[] normVLC = new double[3];
        //first calculate the equivalent voltage of the load center
        for (int k = 0; k < 3; k++) {
            // eq_voltageLC[k][0] = (vs[k][0] * 1000.0) / Npt - (R[k] * is[k][0] - X[k] * is[k][1]) / CTp;
            //eq_voltageLC[k][1] = (vs[k][1] * 1000.0) / Npt - (R[k] * is[k][1] + X[k] * is[k][0]) / CTp;
            eq_voltageLC[k][0] = vs[k][0] / Npt - (R[k] * is[k][0] - X[k] * is[k][1]) / CTp / para;
            eq_voltageLC[k][1] = vs[k][1] / Npt - (R[k] * is[k][1] + X[k] * is[k][0]) / CTp / para;
            System.out.println("Head voltage: " + Math.sqrt(((vs[k][0]) * (vs[k][0]) + (vs[k][1]) * (vs[k][1]))) / Npt);
        }

        for (int i = 0; i < 3; i++) {
            normVLC[i] = Math.sqrt((eq_voltageLC[i][0] * eq_voltageLC[i][0] + eq_voltageLC[i][1] * eq_voltageLC[i][1]));
            System.out.println("Norm VLC " + i + " : " + normVLC[i]);
        }

        if (normVLC[0] < (Voltage_level[0] - 0.5 * Bandwidth) || normVLC[0] > (Voltage_level[0] + 0.5 * Bandwidth)) {
            return true;
        } else if (normVLC[1] < (Voltage_level[1] - 0.5 * Bandwidth) || normVLC[1] > (Voltage_level[1] + 0.5 * Bandwidth)) {
            return true;
        } else if (normVLC[2] < (Voltage_level[2] - 0.5 * Bandwidth) || normVLC[2] > (Voltage_level[2] + 0.5 * Bandwidth)) {
            return true;
        } else {
            return false;
        }
    }

    public void cal_Tap() {
        double normVLC;
        for (int i = 0; i < 3; i++) {
            normVLC = Math.sqrt((eq_voltageLC[i][0] * eq_voltageLC[i][0] + eq_voltageLC[i][1] * eq_voltageLC[i][1]));
            if (normVLC < Voltage_level[i] - 0.5 * Bandwidth) {
                Tap[i] = Math.abs(((Voltage_level[i] - 0.5 * Bandwidth) - normVLC) / DELTA_VOLTAGE);
            } else if (normVLC > Voltage_level[i] + 0.5 * Bandwidth) {
                Tap[i] = Math.abs(((Voltage_level[i] + 0.5 * Bandwidth) - normVLC) / DELTA_VOLTAGE);
            } else {
                Tap[i] = 0.0;
            }

            //testing:
            System.out.print("normVLC:" + normVLC + "\n");
            System.out.println(eq_voltageLC[i][0] + " , " + eq_voltageLC[i][1]);
        }

    }

    private void cal_Tap_ceil() {
        cal_Tap();
        Tap[0] = (int) (Tap[0]) + 1;
        Tap[1] = (int) (Tap[1]) + 1;
        Tap[2] = (int) (Tap[2]) + 1;
    }
    /**
     * THE FUNCTION IN THE INTERFACE**
     */

    /**
     * Remark by Mikoyan:
     * The function declaration in interface calTailV has input arguments
     * headV and tailI, here the version should have expired.
     */
//    public double[][] calTailV(double[][] headV, double[][] headI) {
    public void calTailV(double[][] headV, double[][] tailI, double[][] tailV) {
        /**
         * calTailV equals to Vp2Vs
         */

        Vp2Vs(headV, tailI, tailV);
        //this method is for Type-B regulators, while the reference result is for
        //Type-A regulators.

    }

    public void calTailI(double[][] headV, double[][] headI,  double[][] v) {
        /**
         * calTailV equals to Ip2Is
         */
        Ip2Is(headV, headI, v);

    }

    public void calHeadV(double[][] tailV, double[][] tailI, double[][] headV) {
        /**
         * calTailVI equals to Vs2Vp
         */
        Vs2Vp(tailV, tailI, headV);
    }

    public void calHeadI(double[][] tailV, double[][] tailI, double[][] headI) {
        /**
         * calTailVI equals to Is2Ip
         */
         Is2Ip(tailV, tailI, headI);
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
