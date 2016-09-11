package zju.util;

import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink;
import zju.matrix.ASparseMatrixLink2D;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2007-12-5
 */
public class YMatrixGetter {
    public static final int LINE_FROM = 0;
    public static final int LINE_TO = 1;

    private IEEEDataIsland island;
    private Map<Integer, double[][]> lineAdmittance = new HashMap<>();

    private ASparseMatrixLink[] admittance = new ASparseMatrixLink[2];

    private double[] shuntG;

    private double[] shuntB;

    private int[] connectedBusCount;

    public YMatrixGetter(IEEEDataIsland island) {
        this.island = island;
    }

    public YMatrixGetter() {
    }

    public ASparseMatrixLink[] getAdmittance() {
        return admittance;
    }

    public void initialY(int size) {
        admittance[0] = new ASparseMatrixLink(size);
        admittance[1] = new ASparseMatrixLink(size);
    }

    /**
     * form admittance matrix, this method should be invoked before admittance matrix is used
     */
    public void formYMatrix() {
        lineAdmittance.clear();
        initialY(island.getBuses().size());
        shuntG = new double[island.getBuses().size()];
        shuntB = new double[island.getBuses().size()];
        //bApostrophe = new ASparseMatrixLink(island.getBuses().size());
        for (int i = 0; i < island.getBranches().size(); i++) {
            BranchData branch = island.getBranches().get(i);
            dealBranch(branch);
        }
        for (BusData bus : island.getBuses()) {
            int i = bus.getBusNumber() - 1;
            admittance[0].increase(i, i, bus.getShuntConductance());
            admittance[1].increase(i, i, bus.getShuntSusceptance());
            shuntG[i] = bus.getShuntConductance();
            shuntB[i] = bus.getShuntSusceptance();
        }
    }

    public void dealBranch(BranchData branch) {
        dealBranch(branch, 1);
    }

    public static double[][] getBranchAdmittance(BranchData branch) {
        double r = branch.getBranchR();
        double x = branch.getBranchX();
        double g = r / (r * r + x * x);
        double b = -x / (r * r + x * x);
        double[] from;
        double[] to;
        if (branch.getType() == BranchData.BRANCH_TYPE_ACLINE) {
            from = new double[]{g, b, 0.0, branch.getLineB() / 2};
            to = new double[]{g, b, 0.0, branch.getLineB() / 2};
            return new double[][]{from, to};
        } else {
            double c = 1 / branch.getTransformerRatio();
            double tmp = c * (c - 1);
            from = new double[]{c * g, c * b, tmp * g, tmp * b};
            tmp = 1 - c;
            to = new double[]{c * g, c * b, tmp * g, tmp * b};
            return new double[][]{from, to};
        }
    }

    public void dealBranch(BranchData branch, int sign, ASparseMatrixLink G, ASparseMatrixLink B) {
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
            //todo: sign is not considered here
            double Kamplitude = branch.getTransformerRatio();
            double k1 = Kamplitude * Math.cos(Ktheta);
            double k2 = Kamplitude * Math.sin(Ktheta);
            double base = Kamplitude * Kamplitude;
            G.increase(t, t, g);
            B.increase(t, t, b);
            G.increase(f, f, g / base);
            B.increase(f, f, b / base);
            double temp = -g * k1 - b * k2;
            G.increase(t, f, temp / base);
            temp = -b * k1 + g * k2;
            B.increase(t, f, temp / base);
            temp = -g * k1 + b * k2;
            G.increase(f, t, temp / base);
            temp = -b * k1 - g * k2;
            B.increase(f, t, temp / base);

            double[][] ft = getBranchAdmittance(branch);
            lineAdmittance.put(branch.getId(), ft);
            return;
        }
        //general procedure for branchType 0,1,2,3
        double[][] ft = getBranchAdmittance(branch);
        double[] from = ft[0];
        double[] to = ft[1];
        if (branch.getType() != BranchData.BRANCH_TYPE_ACLINE) {
            double c = 1 / branch.getTransformerRatio();
            g = c * g;
            b = c * b;
        }
        lineAdmittance.put(branch.getId(), ft);
        if (f >= 0) {
            G.increase(f, f, sign * (from[0] + from[2]));
            B.increase(f, f, sign * (from[1] + from[3]));
        }
        if (t >= 0) {
            G.increase(t, t, sign * (to[0] + to[2]));
            B.increase(t, t, sign * (to[1] + to[3]));
        }
        if (f >= 0 && t >= 0) {
            G.increase(f, t, -sign * g);
            G.increase(t, f, -sign * g);
            B.increase(f, t, -sign * b);
            B.increase(t, f, -sign * b);
        }
    }

    public void dealBranch(BranchData branch, int sign) {
        dealBranch(branch, sign, admittance[0], admittance[1]);
    }

    /**
     * form an array storing every bus's connected buses amount
     */
    public void formConnectedBusCount() {
        int busNumber = island.getBuses().size();
        connectedBusCount = new int[busNumber];
        for (int i = 0; i < busNumber; i++) {
            int k = getAdmittance()[0].getIA()[i];
            while (k != -1) {
                connectedBusCount[i]++;
                k = getAdmittance()[0].getLINK().get(k);
            }
            connectedBusCount[i]--;
        }
    }

    //public ASparseMatrixLink getBApostrophe() {
    //    return bApostrophe;
    //}

    /**
     * @param branchId number of line starts from 1
     * @return bus numbers of from and to
     */
    public int[] getFromTo(int branchId) {
        int i = island.getId2branch().get(branchId).getTapBusNumber();
        int j = island.getId2branch().get(branchId).getZBusNumber();
        return new int[]{i, j};
    }

    /**
     * @param branchId branchId number of line starts from 1
     * @param fromOrTo view point is head of line or tail of line
     * @return line (g, b) and (gc, bc)(connected to ground)
     */
    public double[] getLineAdmittance(int branchId, int fromOrTo) {
        switch (fromOrTo) {
            case LINE_FROM:
                return lineAdmittance.get(branchId)[0];
            case LINE_TO:
                return lineAdmittance.get(branchId)[1];
            default:
                return null;
        }
    }

    /**
     * @param branchId number of line starts from 1
     * @return line resistance and reactance
     */
    public double[] getLineRX(int branchId) {
        BranchData branch = island.getId2branch().get(branchId);
        return new double[]{branch.getBranchR(), branch.getBranchX()};
    }


    public ASparseMatrixLink2D formBApostrophe(boolean isXB) {
        int size = island.getPqBusSize() + island.getPvBusSize();
        return formBApostrophe(isXB, size);
    }

    public ASparseMatrixLink2D formBApostrophe(boolean isXB, int size) {
            ASparseMatrixLink2D bApos = new ASparseMatrixLink2D(size);
            if (isXB) {
                int f, t;
                for (BranchData branch : island.getBranches()) {
                    f = branch.getTapBusNumber() - 1;
                    t = branch.getZBusNumber() - 1;
                    if (f >= size && t < size) {
                        bApos.increase(t, t, -1.0 / branch.getBranchX());
                    } else if (f < size && t >= size) {
                        bApos.increase(f, f, -1.0 / branch.getBranchX());
                    } else if (f < size && t < size) {
                        bApos.increase(f, f, -1.0 / branch.getBranchX());
                        bApos.increase(t, t, -1.0 / branch.getBranchX());
                        bApos.increase(f, t, 1.0 / branch.getBranchX());
                        bApos.increase(t, f, 1.0 / branch.getBranchX());
                    }
                }
            } else {
                for (BranchData branch : island.getBranches()) {
                    int f = branch.getTapBusNumber() - 1;
                    int t = branch.getZBusNumber() - 1;
                    double r = branch.getBranchR();
                    double x = branch.getBranchX();
                    double b = -x / (r * r + x * x);
                    if (f >= size && t < size) {
                        bApos.increase(t, t, b);
                    } else if (f < size && t >= size) {
                        bApos.increase(f, f, b);
                    } else if (f < size && t < size) {
                        bApos.increase(f, f, b);
                        bApos.increase(t, t, b);
                        bApos.increase(f, t, -b);
                        bApos.increase(t, f, -b);
                    }
                }
            }
            //for (int i = 0; i < size; i++) {
            //    int k = Y.getAdmittance()[1].getIA()[i];
            //    while (k != -1) {
            //        int j = Y.getAdmittance()[1].getJA().get(k);
            //        if (j >= size)
            //            break;
            //        bApos.setValue(i, j, Y.getAdmittance()[1].getVA().get(k));
            //        k = Y.getAdmittance()[1].getLINK().get(k);
            //    }
            //}
            //for (BusData bus : clonedIsland.getBuses()) {
            //    int i = bus.getBusNumber() - 1;
            //    if (i >= size)
            //        continue;
            //    bApos.increase(i, i, -bus.getShuntSusceptance());
            //}
            return bApos;
        }


    public ASparseMatrixLink2D formBApostropheTwo(boolean isXB) {
        int size = island.getPqBusSize();
        ASparseMatrixLink2D bAposTwo = new ASparseMatrixLink2D(size);
        if (isXB) {
            for (int i = 0; i < island.getPqBusSize(); i++) {
                int k = getAdmittance()[1].getIA()[i];
                while (k != -1) {
                    int j = getAdmittance()[1].getJA().get(k);
                    if (j >= size)
                        break;
                    bAposTwo.setValue(i, j, getAdmittance()[1].getVA().get(k));
                    k = getAdmittance()[1].getLINK().get(k);
                }
            }
        } else {
            int f, t;
            for (BranchData branch : island.getBranches()) {
                f = branch.getTapBusNumber() - 1;
                t = branch.getZBusNumber() - 1;
                if (f >= size && t < size) {
                    bAposTwo.increase(t, t, -1.0 / branch.getBranchX());
                    bAposTwo.increase(t, t, branch.getLineB() / 2.0);
                } else if (f < size && t >= size) {
                    bAposTwo.increase(f, f, -1.0 / branch.getBranchX());
                    bAposTwo.increase(f, f, branch.getLineB() / 2.0);
                } else if (f < size && t < size) {
                    bAposTwo.increase(f, f, branch.getLineB() / 2.0);
                    bAposTwo.increase(t, t, branch.getLineB() / 2.0);
                    bAposTwo.increase(f, f, -1.0 / branch.getBranchX());
                    bAposTwo.increase(t, t, -1.0 / branch.getBranchX());
                    bAposTwo.increase(f, t, 1.0 / branch.getBranchX());
                    bAposTwo.increase(t, f, 1.0 / branch.getBranchX());
                }
            }
            for (BusData bus : island.getBuses()) {
                int i = bus.getBusNumber() - 1;
                if (i >= island.getPqBusSize())
                    continue;
                bAposTwo.increase(i, i, bus.getShuntSusceptance());
            }
        }
        return bAposTwo;
    }

    public IEEEDataIsland getIsland() {
        return island;
    }

    public void setIsland(IEEEDataIsland island) {
        this.island = island;
    }

    public double[] getShuntG() {
        return shuntG;
    }

    public double[] getShuntB() {
        return shuntB;
    }

    public int[] getConnectedBusCount() {
        return connectedBusCount;
    }
}
