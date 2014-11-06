package zju.dsmodel;

/**
 * Created by IntelliJ IDEA.
 * User: zhouyt
 * Date: 2010-7-9
 * Time: 9:43:21
 */

/**
 * This module of regulation is base on the analysis of
 * "Distribution System Modeling and Analysis, Second Edition"
 * in page 221 to 224
 */

/**
 * Since the current transformers are not monitoring the load side line current,
 * we cannot obtain the equivalent voltage of the load center.
 * What's more, the closed delta connection can be difficult to apply
 * Note in both the voltage and current equations that a change of the tap position in
 * one regulator will affect voltages and currents in two phase.
 * Also, in IEEE's testing of feeders, NONE of the distribution systems contains the regulator which is connected in
 * closed_delta
 */

public class RG_ClosedDelta {
    /**
     * Global constants
     * these constants are on a 120-volt base
     */
    public static final double delta_tap = 0.00625;
    public static final double delta_voltage = 0.75;
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
    private String[] Monitor_phase = new String[3];
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
    private double[][] VLLp = new double[3][2];
    private double[][] VLLs = new double[3][2];
    private double[][] eq_voltageLC = new double[3][2];

    public RG_ClosedDelta() {

    }


    //functions
    //tool functions

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

    //**********************************************************************************
    //**********************************************************************************
    //the essential functions

    public void cal_a() {
        cal_Tap();
        //[0-2]: [ab,bc,ca]
        if (RG_state.equals("R")) {
            aR[0] = 1 - delta_tap * ((int) Tap[0] + 1);
            aR[1] = 1 - delta_tap * ((int) Tap[1] + 1);
            aR[2] = 1 - delta_tap * ((int) Tap[2] + 1);

        } else {
            aR[0] = 1 + delta_tap * ((int) Tap[0] + 1);
            aR[1] = 1 + delta_tap * ((int) Tap[1] + 1);
            aR[2] = 1 + delta_tap * ((int) Tap[2] + 1);

        }
        a[0][0] = aR[0];
        a[1][1] = aR[1];
        a[2][2] = aR[2];
        a[0][1] = 1 - aR[1];
        a[1][2] = 1 - aR[2];
        a[2][0] = 1 - aR[0];
        a[0][2] = 0;
        a[1][0] = 0;
        a[2][1] = 0;

    }

    public void cal_b() {
        //cal_Tap();
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
        //cal_Tap();
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

        double divider = 0;
        cal_Tap();
        /**Generally Zt and Ym is neglected, so the elements in matrix d is very simple.
         * If we want to consider the impedance and admittance later,
         * we have to recalculate the elements in matrix d in some more complex way, which is not mentioned.
         */
        if (RG_state.equals("R")) {
            aR[0] = 1 - delta_tap * ((int) Tap[0] + 1);
            aR[1] = 1 - delta_tap * ((int) Tap[1] + 1);
            aR[2] = 1 - delta_tap * ((int) Tap[2] + 1);

        } else {
            aR[0] = 1 + delta_tap * ((int) Tap[0] + 1);
            aR[1] = 1 + delta_tap * ((int) Tap[1] + 1);
            aR[2] = 1 + delta_tap * ((int) Tap[2] + 1);

        }
        divider = aR[0] * aR[2] + aR[0] * aR[1] + aR[2] * aR[1] - aR[2] - aR[1] - aR[0] + 1;
        d[0][0] = (aR[1] * aR[2]) / divider;
        d[1][1] = (aR[0] * aR[2]) / divider;
        d[2][2] = (aR[0] * aR[1]) / divider;
        d[0][1] = ((aR[2] - 1) * (aR[1] - 1)) / divider;
        d[1][2] = ((aR[0] - 1) * (aR[2] - 1)) / divider;
        d[2][0] = ((aR[0] - 1) * (aR[1] - 1)) / divider;
        d[0][2] = (aR[1] * (aR[2] - 1)) / divider;
        d[1][0] = (aR[2] * (aR[0] - 1)) / divider;
        d[2][1] = (aR[0] * (aR[1] - 1)) / divider;

    }

    public void cal_A() {
        double divider = 0;
        cal_Tap();
        //matrix A is the inverting of matrix a
        if (RG_state.equals("R")) {
            aR[0] = 1 - delta_tap * ((int) Tap[0] + 1);
            aR[1] = 1 - delta_tap * ((int) Tap[1] + 1);
            aR[2] = 1 - delta_tap * ((int) Tap[2] + 1);

        } else {
            aR[0] = 1 + delta_tap * ((int) Tap[0] + 1);
            aR[1] = 1 + delta_tap * ((int) Tap[1] + 1);
            aR[2] = 1 + delta_tap * ((int) Tap[2] + 1);

        }

        divider = aR[0] * aR[2] + aR[0] * aR[1] + aR[2] * aR[1] - aR[2] - aR[1] - aR[0] + 1;
        A[0][0] = (aR[1] * aR[2]) / divider;
        A[1][1] = (aR[0] * aR[2]) / divider;
        A[2][2] = (aR[0] * aR[1]) / divider;
        A[0][2] = ((aR[2] - 1) * (aR[1] - 1)) / divider;
        A[1][0] = ((aR[0] - 1) * (aR[2] - 1)) / divider;
        A[2][1] = ((aR[0] - 1) * (aR[1] - 1)) / divider;
        A[0][1] = (aR[2] * (aR[1] - 1)) / divider;
        A[1][2] = (aR[0] * (aR[2] - 1)) / divider;
        A[2][0] = (aR[1] * (aR[0] - 1)) / divider;

    }

    public void cal_B() {
        //cal_Tap();
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
        //cal_Tap();
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
        cal_Tap();
        //matrix D is the inverting of matrix d
        if (RG_state.equals("R")) {
            aR[0] = 1 - delta_tap * ((int) Tap[0] + 1);
            aR[1] = 1 - delta_tap * ((int) Tap[1] + 1);
            aR[2] = 1 - delta_tap * ((int) Tap[2] + 1);

        } else {
            aR[0] = 1 + delta_tap * ((int) Tap[0] + 1);
            aR[1] = 1 + delta_tap * ((int) Tap[1] + 1);
            aR[2] = 1 + delta_tap * ((int) Tap[2] + 1);

        }
        D[0][0] = aR[0];
        D[1][1] = aR[1];
        D[2][2] = aR[2];
        D[0][2] = 1 - aR[2];
        D[1][0] = 1 - aR[0];
        D[2][1] = 1 - aR[1];
        D[0][1] = 0;
        D[1][2] = 0;
        D[2][0] = 0;

    }


    public double[][] Vp2Vs(double[][] vp, double[][] ip) {
        cal_A();
        cal_B();
        /**
         * we only know the exact relationship of the primary voltage and the secondary voltage
         * based on line-line voltage
         * Here, I use the transforming method which is used in the transformer,
         * to transform the line-line voltage to the line-neutral voltage,
         * also the transforming of the line-neutral voltage to line-line voltage is the same.
         * But it is not sure that this method can be used in this way.
         */
        Vp2VLLp(vp);
        for (int i = 0; i < 3; i++) {
            VLLs[i][0] = 0;
            VLLs[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                VLLs[i][0] += A[i][j] * VLLp[j][0] - B[i][j] * ip[j][0];
                VLLs[i][1] += A[i][j] * VLLp[j][1] - B[i][j] * ip[j][1];
            }
        }
        VLLs2Vs(VLLs);
        return Vs;
    }

    public double[][] Vs2Vp(double[][] vs, double[][] is) {
        cal_a();
        cal_b();
        /**
         * we only know the exact relationship of the primary voltage and the secondary voltage
         * based on line-line voltage
         * Here, I use the transforming method which is used in the transformer,
         * to transform the line-line voltage to the line-neutral voltage,
         * also the transforming of the line-neutral voltage to line-line voltage is the same.
         * But it is not sure that this method can be used in this way.
         */
        Vs2VLLs(vs);
        for (int i = 0; i < 3; i++) {
            VLLp[i][0] = 0;
            VLLp[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                VLLp[i][0] += a[i][j] * VLLs[j][0] + b[i][j] * is[j][0];
                VLLp[i][1] += a[i][j] * VLLs[j][1] + b[i][j] * is[j][1];
            }
        }
        VLLp2Vp(VLLp);
        return Vp;

    }

    public double[][] Ip2Is(double[][] vp, double[][] ip) {
        cal_C();
        cal_D();
        /**
         * the calculating of currents is exactly right
         */
        for (int i = 0; i < 3; i++) {
            Is[i][0] = 0;
            Is[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                Is[i][0] += -C[i][j] * vp[j][0] + D[i][j] * ip[j][0];
                Is[i][1] += -C[i][j] * vp[j][1] + D[i][j] * ip[j][1];

            }
        }
        return Is;
    }

    public double[][] Is2Ip(double[][] vs, double[][] is) {
        cal_c();
        cal_d();
        /**
         * the calculating of currents is exactly right
         */
        for (int i = 0; i < 3; i++) {
            Ip[i][0] = 0;
            Ip[i][1] = 0;
            for (int j = 0; j < 3; j++) {
                Ip[i][0] += c[i][j] * vs[j][0] + d[i][j] * is[j][0];
                Ip[i][1] += c[i][j] * vs[j][1] + d[i][j] * is[j][1];

            }
        }
        return Ip;
    }

    public void cal_Tap() {
        double normVLC = 0;
        eq_voltageLC = new double[][]{{0, 0}, {0, 0}, {0, 0}};
        //todo: calculate the equivalent voltage at the load center
        /**
         * Since the current transformers are not monitoring the load side line current,
         * we cannot obtain the equivalent voltage of the load center.
         * What's more, the closed delta connection can be difficult to apply
         * Note in both the voltage and current equations that a change of the tap position in
         * one regulator will affect voltages and currents in two phase.
         */
        for (int i = 0; i < 3; i++) {
            normVLC = Math.sqrt((eq_voltageLC[i][0] * eq_voltageLC[i][0] + eq_voltageLC[i][1] * eq_voltageLC[i][1]));
            Tap[i] = Math.abs((Voltage_level[i] - normVLC) / delta_voltage);
        }

    }

    public double[][] VLLp2Vp(double[][] vllp) {
        double[][] vp_temp = new double[][]{{0, 0}, {0, 0}, {0, 0}};
        //todo:to transform VLLp to Vp

        return vp_temp;
    }

    public double[][] Vp2VLLp(double[][] vp) {
        double[][] vllp_temp = new double[][]{{0, 0}, {0, 0}, {0, 0}};
        //todo:to transform Vp to VLLp

        return vllp_temp;
    }

    public double[][] VLLs2Vs(double[][] vlls) {
        double[][] vs_temp = new double[][]{{0, 0}, {0, 0}, {0, 0}};
        //todo:to transform VLLs to Vs

        return vs_temp;
    }

    public double[][] Vs2VLLs(double[][] vs) {
        double[][] vlls_temp = new double[][]{{0, 0}, {0, 0}, {0, 0}};
        //todo:to transform Vs to VLLs

        return vlls_temp;
    }

}
