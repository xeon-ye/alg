package zju.dsmodel;

/**
 * Created by IntelliJ IDEA.
 * User: LQ
 * Date: 2010-7-9
 * Time: 1:24:28
 */
public class Feeder implements GeneralBranch {

    protected double Z_real[][];

    protected double Z_imag[][];

    protected double Y_imag[][];

    private boolean isShuntYNeglected = true;

    private int[] phases;

    public Feeder() {
    }

    public void initialPhases() {
        int phaseCount = 0;
        for (int i = 0; i < 3; i++) {
            if (Math.abs(Z_real[i][i]) > DsModelCons.ZERO_LIMIT
                    || Math.abs(Z_imag[i][i]) > DsModelCons.ZERO_LIMIT)
                phaseCount++;
        }
        phases = new int[phaseCount];
        phaseCount = 0;
        for (int i = 0; i < 3; i++) {
            if (Math.abs(Z_real[i][i]) > DsModelCons.ZERO_LIMIT
                    || Math.abs(Z_imag[i][i]) > DsModelCons.ZERO_LIMIT)
                phases[phaseCount++] = i;
        }
        //phases = new int[]{0,1,2};
    }

    public double[][] getZ_real() {
        return Z_real;
    }

    public double[][] getZ_imag() {
        return Z_imag;
    }

    public double[][] getY_imag() {
        return Y_imag;
    }

    public void setZ_real(double[][] z_real) {
        this.Z_real = z_real;
    }

    public void setZ_imag(double[][] z_imag) {
        this.Z_imag = z_imag;
    }

    public void setY_imag(double[][] y_imag) {
        this.Y_imag = y_imag;
    }

    public void calTailV(double[][] headV, double[][] tailI, double[][] tailV) {
        for (int p : phases) {
            tailV[p][0] = headV[p][0];
            tailV[p][1] = headV[p][1];
            for (int j : phases) {
                if (j != p && Math.abs(Z_real[p][j]) < DsModelCons.ZERO_LIMIT
                        && Math.abs(Z_imag[p][j]) < DsModelCons.ZERO_LIMIT)
                    continue;
                if (isShuntYNeglected) {
                    tailV[p][0] -= (Z_real[p][j] * tailI[j][0] - Z_imag[p][j] * tailI[j][1]);
                    tailV[p][1] -= (Z_imag[p][j] * tailI[j][0] + Z_real[p][j] * tailI[j][1]);
                } else {
                    //todo: not finished.
                }
            }
        }
    }

    public void calHeadV(double[][] tailV, double[][] tailI, double[][] headV) {
        for (int p : phases) {
            headV[p][0] = tailV[p][0];
            headV[p][1] = tailV[p][1];
            for (int j : phases) {
                if (j != p && Math.abs(Z_real[p][j]) < DsModelCons.ZERO_LIMIT
                        && Math.abs(Z_imag[p][j]) < DsModelCons.ZERO_LIMIT)
                    continue;
                if (isShuntYNeglected) {
                    headV[p][0] += (Z_real[p][j] * tailI[j][0] - Z_imag[p][j] * tailI[j][1]);
                    headV[p][1] += (Z_imag[p][j] * tailI[j][0] + Z_real[p][j] * tailI[j][1]);
                } else {
                    //todo: not finished.
                }
            }
        }
    }

    public void calHeadI(double[][] tailV, double[][] tailI, double[][] headI) {
        if (isShuntYNeglected && headI != tailI) {
            for (int i = 0; i < tailI.length; i++)
                System.arraycopy(tailI[i], 0, headI[i], 0, headI[i].length);
        } else {
            //todo: not finished..
        }
    }

    @Override
    public int getNonZeroNumOfJac() {
        if (isShuntYNeglected) {
            int count = 0;
            for (int i : phases) {
                count += 4;
                for (int j : phases) {
                    if (j != i && Math.abs(Z_real[i][j]) < DsModelCons.ZERO_LIMIT
                            && Math.abs(Z_imag[i][j]) < DsModelCons.ZERO_LIMIT)
                        continue;
                    count += 4;
                }
            }
            return count;
        } else {
            return 0;//todo:
        }
    }

    /**
     * 该方法用于判断线路是否包含某一相
     *
     * @param phase 相位
     * @return 是否包含某相
     */
    public boolean containsPhase(int phase) {
        for (int p : phases)
            if (p == phase)
                return true;
        return false;
        //return true;
    }

    public int getPhaseIndex(int phase) {
        for (int i = 0; i < phases.length; i++)
            if (phases[i] == phase)
                return i;
        return -1;
        //return phase;
    }

    public boolean isShuntYNeglected() {
        return isShuntYNeglected;
    }

    public void setShuntYNeglected(boolean shuntYNeglected) {
        isShuntYNeglected = shuntYNeglected;
    }

    public int[] getPhases() {
        return phases;
    }
}