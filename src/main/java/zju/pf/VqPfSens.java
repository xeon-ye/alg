package zju.pf;

import jpscpu.LinearSolver;
import zju.ieeeformat.BranchData;
import zju.matrix.ASparseMatrixLink;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.util.YMatrixGetter;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-8-30
 */
public class VqPfSens {
    private LinearSolver solver = new LinearSolver();
    private ASparseMatrixLink2D Ldd;
    private ASparseMatrixLink2D Ldg;
    private ASparseMatrixLink2D Lgd;
    private ASparseMatrixLink2D Lgg;
    private ASparseMatrixLink2D L;

    private int pqSize, pvSize;
    private boolean isFirstTime;
    public double[] right;

    public VqPfSens(int pqSize, int pvSize) {
        this.pqSize = pqSize;
        this.pvSize = pvSize;
        right = new double[pvSize + pqSize];
    }

    public void formSubMarix(ASparseMatrixLink b) {
        int size1 = 0, size2 = 0, size3 = 0, size4 = 0;
        for (int i = 0; i < pqSize; i++) {
            int k = b.getIA()[i];
            while (k != -1) {
                int j = b.getJA().get(k);
                if (j >= pqSize)
                    size2++;
                else
                    size1++;
                k = b.getLINK().get(k);
            }
        }

        for (int i = pqSize; i < pvSize + pqSize; i++) {
            int k = b.getIA()[i];
            while (k != -1) {
                int j = b.getJA().get(k);
                if (j >= pqSize)
                    size4++;
                else
                    size3++;
                k = b.getLINK().get(k);
            }
        }

        Ldd = new ASparseMatrixLink2D(pqSize, pqSize, size1);
        Ldg = new ASparseMatrixLink2D(pqSize, pvSize, size2);
        Lgd = new ASparseMatrixLink2D(pvSize, pqSize, size3);
        Lgg = new ASparseMatrixLink2D(pvSize, pvSize, size4);
        L = new ASparseMatrixLink2D(pvSize + pqSize, pvSize + pqSize, size1 + size2 + size3 + size4);
        for (int i = 0; i < pqSize; i++) {
            int k = b.getIA()[i];
            while (k != -1) {
                int j = b.getJA().get(k);
                if (j == pqSize + pvSize)
                    break;
                if (j >= pqSize)
                    Ldg.setValue(i, j - pqSize, b.getVA().get(k));
                else
                    Ldd.setValue(i, j, b.getVA().get(k));
                L.setValue(i, j, b.getVA().get(k));
                k = b.getLINK().get(k);
            }
        }

        for (int i = pqSize; i < pvSize + pqSize; i++) {
            int k = b.getIA()[i];
            while (k != -1) {
                int j = b.getJA().get(k);
                if (j == pqSize + pvSize)
                    break;
                if (j >= pqSize)
                    Lgg.setValue(i - pqSize, j - pqSize, b.getVA().get(k));
                else
                    Lgd.setValue(i - pqSize, j, b.getVA().get(k));
                L.setValue(i, j, b.getVA().get(k));
                k = b.getLINK().get(k);
            }
        }
        isFirstTime = true;
    }

    public void dealBranch(BranchData branch, int sign) {
        int f = branch.getTapBusNumber() - 1;
        int t = branch.getZBusNumber() - 1;
        double r = branch.getBranchR();
        double x = branch.getBranchX();
        double g = r / (r * r + x * x);
        double b = -x / (r * r + x * x);
        //special for branchType4
        if (branch.getType() == BranchData.BRANCH_TYPE_PHASE_SHIFTER) {
            double Ktheta = branch.getTransformerAngle() * Math.PI / 180;
            if (Math.abs(Ktheta) < 1e-4) {
                return;
            }
            double Kamplitude = branch.getTransformerRatio();
            double k1 = Kamplitude * Math.cos(Ktheta);
            double k2 = Kamplitude * Math.sin(Ktheta);
            double base = Kamplitude * Kamplitude;

            if (t < pqSize)
                Ldd.increase(t, t, sign * b);
            else
                Lgg.increase(t - pqSize, t - pqSize, sign * b);
            if (f < pqSize)
                Ldd.increase(f, f, sign * b / base);
            else
                Lgg.increase(f - pqSize, f - pqSize, sign * b / base);

            double temp = -b * k1 + g * k2;
            double temp2 = -b * k1 - g * k2;
            if (t < pqSize) {
                if (f < pqSize) {
                    Ldd.increase(t, f, sign * temp / base);
                    Ldd.increase(f, t, sign * temp2 / base);
                } else {
                    Ldg.increase(t, f - pqSize, sign * temp / base);
                    Lgd.increase(f - pqSize, t, sign * temp2 / base);
                }
            } else {
                if (f < pqSize) {
                    Lgd.increase(t - pqSize, f, sign * temp / base);
                    Ldg.increase(f, t - pqSize, sign * temp2 / base);
                } else {
                    Lgg.increase(t - pqSize, f - pqSize, sign * temp / base);
                    Lgg.increase(f - pqSize, t - pqSize, sign * temp2 / base);
                }
            }
            return;
        }
        //general procedure for branchType 0,1,2,3
        double[][] ft = YMatrixGetter.getBranchAdmittance(branch);
        double[] from = ft[0];
        double[] to = ft[1];
        if (branch.getType() != BranchData.BRANCH_TYPE_ACLINE) {
            double c = 1 / branch.getTransformerRatio();
            b = c * b;
        }

        if (t < pqSize)
            Ldd.increase(t, t, sign * (to[1] + to[3]));
        else
            Lgg.increase(t - pqSize, t - pqSize, sign * (to[1] + to[3]));
        if (f < pqSize)
            Ldd.increase(f, f, sign * (from[1] + from[3]));
        else
            Lgg.increase(f - pqSize, f - pqSize, sign * (from[1] + from[3]));

        if (t < pqSize) {
            if (f < pqSize) {
                Ldd.increase(t, f, -sign * b);
                Ldd.increase(f, t, -sign * b);
            } else {
                Ldg.increase(t, f - pqSize, -sign * b);
                Lgd.increase(f - pqSize, t, -sign * b);
            }
        } else {
            if (f < pqSize) {
                Lgd.increase(t - pqSize, f, -sign * b);
                Ldg.increase(f, t - pqSize, -sign * b);
            } else {
                Lgg.increase(t - pqSize, f - pqSize, -sign * b);
                Lgg.increase(f - pqSize, t - pqSize, -sign * b);
            }
        }
    }

    public void calPvDeltaV(double[] pvBusDeltaQ, double[] pvBusDeltaV) {
        System.arraycopy(pvBusDeltaQ, 0, right, pqSize, pvSize);
        if (isFirstTime) {
            solver.solve2(L, right, true);
            isFirstTime = false;
        } else
            solver.solve2(right);
        System.arraycopy(right, pqSize, pvBusDeltaV, 0, pvSize);
    }

    public double calPvDeltaV(double pvBusDeltaQ, int busNum) {
        AVector Ldi = Ldg.getColVector(busNum - pqSize - 1);
        double[] Ldi_ori = new double[Ldi.getN()];
        System.arraycopy(Ldi.getValues(), 0, Ldi_ori, 0, Ldi.getN());
        if(isFirstTime) {
            solver.solve3(Ldd, Ldi.getValues());
            isFirstTime = false;
        } else
            solver.solve3(Ldi.getValues());
        double v = 0;
        for (int i = 0; i < Ldi.getN(); i++)
            v += Ldi_ori[i] * Ldi.getValue(i);
        return pvBusDeltaQ / (Lgg.getValue(busNum - pqSize - 1, busNum - pqSize - 1) - v);
    }

    public ASparseMatrixLink2D getLdd() {
        return Ldd;
    }

    public ASparseMatrixLink2D getLdg() {
        return Ldg;
    }

    public ASparseMatrixLink2D getLgd() {
        return Lgd;
    }

    public ASparseMatrixLink2D getLgg() {
        return Lgg;
    }

    public void dealBranch(BranchData branch) {
        dealBranch(branch, 1);
    }
}
