package zju.dsmodel;

/**
 * 该类表述了基本的负荷ZIP模型
 *
 * @author Dong Shufeng
 *         Date: 2011-4-21
 */
public class BasicLoad implements DsModelCons, ThreePhaseLoad, Cloneable {

    private String mode;
    //电流单位是安培或标幺值，电流幅值和落后电压的角度
    private double[][] constantI;//constant current
    //单位是瓦特或标幺值
    private double[][] constantS;//constant power
    //单位是欧姆或标幺值
    private double[][] constantZ;

    public double[][] lineToLineCurrent;

    public double[][] lineToLineV;

    public BasicLoad(String mode) {
        this.mode = mode;
    }

    public BasicLoad() {
    }

    private void calParam_DI(double[][] s, double[][] baseKv) {
        constantI = new double[3][2];
        double rtemp;
        double itemp;
        for (int i = 0; i < 3; i++) {
            lineToLineV[i][0] = baseKv[i][0] - baseKv[(i + 1) % 3][0];
            lineToLineV[i][1] = baseKv[i][1] - baseKv[(i + 1) % 3][1];
        }
        for (int i = 0; i < 3; i++) {
            double divisor = lineToLineV[i][0] * lineToLineV[i][0] + lineToLineV[i][1] * lineToLineV[i][1];
            rtemp = (s[i][0] * lineToLineV[i][0] + s[i][1] * lineToLineV[i][1]) / divisor;
            itemp = (s[i][0] * lineToLineV[i][1] - s[i][1] * lineToLineV[i][0]) / divisor;
            constantI[i][0] = Math.sqrt(rtemp * rtemp + itemp * itemp);
            constantI[i][1] = Math.atan2(lineToLineV[i][1], lineToLineV[i][0]) - Math.atan2(itemp, rtemp);
        }
    }

    public void calParam_DZ(double[][] s, double[][] baseKv) {
        constantZ = new double[3][2];
        for (int i = 0; i < 3; i++) {
            lineToLineV[i][0] = baseKv[i][0] - baseKv[(i + 1) % 3][0];
            lineToLineV[i][1] = baseKv[i][1] - baseKv[(i + 1) % 3][1];
        }
        for (int i = 0; i < 3; i++) {
            double divisor = s[i][0] * s[i][0] + s[i][1] * s[i][1];
            if (divisor < 1e-9) {
                constantZ[i][0] = 0;
                constantZ[i][1] = 0;
            } else {
                double mul = (lineToLineV[i][0] * lineToLineV[i][0] + lineToLineV[i][1] * lineToLineV[i][1]) / divisor;
                constantZ[i][0] = s[i][0] * mul;
                constantZ[i][1] = s[i][1] * mul;
            }
        }
    }

    private void calParam_YI(double[][] s, double[][] baseKv) {
        constantI = new double[3][2];
        double rtemp;
        double itemp;
        for (int i = 0; i < 3; i++) {
            double divisor = baseKv[i][0] * baseKv[i][0] + baseKv[i][1] * baseKv[i][1];
            rtemp = (s[i][0] * baseKv[i][0] + s[i][1] * baseKv[i][1]) / divisor;
            itemp = (s[i][0] * baseKv[i][1] - s[i][1] * baseKv[i][0]) / divisor;
            constantI[i][0] = Math.sqrt(rtemp * rtemp + itemp * itemp);
            constantI[i][1] = Math.atan2(baseKv[i][1], baseKv[i][0]) - Math.atan2(itemp, rtemp);
        }
    }

    private void calParam_YZ(double[][] s, double[][] baseKv) {
        constantZ = new double[s.length][2];
        for (int i = 0; i < s.length; i++) {
            double divisor = s[i][0] * s[i][0] + s[i][1] * s[i][1];
            if (divisor < 1e-9) {
                constantZ[i][0] = 0;
                constantZ[i][1] = 0;
            } else {
                double mul = (baseKv[i][0] * baseKv[i][0] + baseKv[i][1] * baseKv[i][1]) / divisor;
                constantZ[i][0] = s[i][0] * mul;
                constantZ[i][1] = s[i][1] * mul;
            }
        }
    }

    private void calI_DI(double[][] phaseV, double[][] phaseC) {
        double angleV;
        for (int i = 0; i < 3; i++) {
            lineToLineV[i][0] = phaseV[i][0] - phaseV[(i + 1) % 3][0];
            lineToLineV[i][1] = phaseV[i][1] - phaseV[(i + 1) % 3][1];
        }
        for (int i = 0; i < 3; i++) {
            angleV = Math.atan2(lineToLineV[i][1], lineToLineV[i][0]);
            lineToLineCurrent[i][0] = constantI[i][0] * Math.cos(angleV - constantI[i][1]);
            lineToLineCurrent[i][1] = constantI[i][0] * Math.sin(angleV - constantI[i][1]);
        }
        for (int i = 0; i < 3; i++) {
            phaseC[i][0] = lineToLineCurrent[i][0] - lineToLineCurrent[(i + 2) % 3][0];
            phaseC[i][1] = lineToLineCurrent[i][1] - lineToLineCurrent[(i + 2) % 3][1];
        }
    }

    private void calI_DS(double[][] phaseV, double[][] phaseC) {
        double divisor;
        for (int i = 0; i < 3; i++) {
            lineToLineV[i][0] = phaseV[i][0] - phaseV[(i + 1) % 3][0];
            lineToLineV[i][1] = phaseV[i][1] - phaseV[(i + 1) % 3][1];
        }
        for (int i = 0; i < 3; i++) {
            divisor = lineToLineV[i][0] * lineToLineV[i][0] + lineToLineV[i][1] * lineToLineV[i][1];
            if (Math.abs(divisor) < 1e-6) {
                System.out.println("Error: 电压幅值约等于0");
                return;
            }
            lineToLineCurrent[i][0] = (constantS[i][0] * lineToLineV[i][0]
                    + constantS[i][1] * lineToLineV[i][1]) / divisor;
            lineToLineCurrent[i][1] = (constantS[i][0] * lineToLineV[i][1]
                    - constantS[i][1] * lineToLineV[i][0]) / divisor;
        }
        for (int i = 0; i < 3; i++) {
            phaseC[i][0] = lineToLineCurrent[i][0] - lineToLineCurrent[(i + 2) % 3][0];
            phaseC[i][1] = lineToLineCurrent[i][1] - lineToLineCurrent[(i + 2) % 3][1];
        }
    }

    public void calI_DZ(double phaseV[][], double[][] phaseC) {
        double[][] c = lineToLineCurrent;
        double[][] v = lineToLineV;
        double divisor;
        for (int i = 0; i < 3; i++) {
            v[i][0] = phaseV[i][0] - phaseV[(i + 1) % 3][0];
            v[i][1] = phaseV[i][1] - phaseV[(i + 1) % 3][1];
        }
        for (int i = 0; i < 3; i++) {
            divisor = constantZ[i][0] * constantZ[i][0] + constantZ[i][1] * constantZ[i][1];
            if (Math.abs(divisor) < 1e-9) {
                c[i][0] = 0;
                c[i][1] = 0;
            } else {
                c[i][0] = (constantZ[i][0] * v[i][0] + constantZ[i][1] * v[i][1]) / divisor;
                c[i][1] = (constantZ[i][0] * v[i][1] - constantZ[i][1] * v[i][0]) / divisor;
            }
        }
        for (int i = 0; i < 3; i++) {
            phaseC[i][0] = c[i][0] - c[(i + 2) % 3][0];
            phaseC[i][1] = c[i][1] - c[(i + 2) % 3][1];
        }
    }

    public void calI_YI(double phaseV[][], double[][] current) {
        for (int i = 0; i < current.length; i++) {
            double thetaV = Math.atan2(phaseV[i][1], phaseV[i][0]);
            current[i][0] = constantI[i][0] * Math.cos(thetaV - constantI[i][1]);
            current[i][1] = constantI[i][0] * Math.sin(thetaV - constantI[i][1]);
        }
    }


    public void calI_YS(double[][] v, double[][] current) {
        double divisor;
        for (int i = 0; i < current.length; i++) {
            divisor = v[i][0] * v[i][0] + v[i][1] * v[i][1];
            if (Math.abs(divisor) < 1e-6) {
                System.out.println("Error: 电压幅值约等于0");
                return;
            }
            current[i][0] = (constantS[i][0] * v[i][0] + constantS[i][1] * v[i][1]) / divisor;
            current[i][1] = (constantS[i][0] * v[i][1] - constantS[i][1] * v[i][0]) / divisor;
        }
    }

    public void calI_YZ(double v[][], double[][] current) {
        double divisor;
        for (int i = 0; i < 3; i++) {
            divisor = constantZ[i][0] * constantZ[i][0] + constantZ[i][1] * constantZ[i][1];
            if (Math.abs(divisor) < 1e-9) {
                current[i][0] = 0;
                current[i][1] = 0;
            } else {
                current[i][0] = (constantZ[i][0] * v[i][0] + constantZ[i][1] * v[i][1]) / divisor;
                current[i][1] = (constantZ[i][0] * v[i][1] - constantZ[i][1] * v[i][0]) / divisor;
            }
        }
    }

    /**
     * The function can be used in both per-unit situation and not.
     *
     * @param s         视在功率, 单位为瓦特或标幺值
     * @param baseVAmpl 单位为伏特或标幺值
     */
    public void formPara(double[][] s, double baseVAmpl) {
        if (mode.equals(LOAD_Y_PQ)) {
            setConstantS(s);
        } else if (mode.equals(LOAD_D_PQ)) {
            lineToLineV = new double[3][2];
            lineToLineCurrent = new double[3][2];
            setConstantS(s);
        } else {
            double[][] baseV = new double[][]{{baseVAmpl, 0},
                    {-0.5 * baseVAmpl, -0.5 * sqrt3 * baseVAmpl},
                    {-0.5 * baseVAmpl, 0.5 * sqrt3 * baseVAmpl}};
            if (mode.equals(LOAD_Y_I)) {
                calParam_YI(s, baseV);
            } else if (mode.equals(LOAD_Y_Z)) {
                calParam_YZ(s, baseV);
            } else if (mode.equals(LOAD_D_I)) {
                lineToLineV = new double[3][2];
                lineToLineCurrent = new double[3][2];
                calParam_DI(s, baseV);
            } else if (mode.equals(LOAD_D_Z)) {
                lineToLineV = new double[3][2];
                lineToLineCurrent = new double[3][2];
                calParam_DZ(s, baseV);
            }
        }
    }

    public void calI(double[][] v, double[][] c) {
        if (mode.equals(LOAD_D_I)) {
            calI_DI(v, c);
        } else if (mode.equals(LOAD_D_PQ)) {
            calI_DS(v, c);
        } else if (mode.equals(LOAD_D_Z)) {
            calI_DZ(v, c);
        } else if (mode.equals(LOAD_Y_I)) {
            calI_YI(v, c);
        } else if (mode.equals(LOAD_Y_PQ)) {
            calI_YS(v, c);
        } else if (mode.equals(LOAD_Y_Z))
            calI_YZ(v, c);
    }

    public String getMode() {
        return mode;
    }

    public void setMode(String mode) {
        this.mode = mode;
    }

    public double[][] getConstantI() {
        return constantI;
    }

    public void setConstantI(double[][] constantI) {
        this.constantI = constantI;
    }

    public double[][] getConstantS() {
        return constantS;
    }

    public void setConstantS(double[][] constantS) {
        this.constantS = constantS;
    }

    public double[][] getConstantZ() {
        return constantZ;
    }

    public void setConstantZ(double[][] constantZ) {
        this.constantZ = constantZ;
    }

    public double[][] getLineToLineCurrent() {
        return lineToLineCurrent;
    }

    public double[][] getLineToLineV() {
        return lineToLineV;
    }

    protected BasicLoad clone() {
        try {
            return (BasicLoad) super.clone();
        } catch (CloneNotSupportedException e) {
            e.printStackTrace();
            return null;
        }
    }
}
