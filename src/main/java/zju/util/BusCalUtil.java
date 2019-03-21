package zju.util;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.ieeeformat.BranchData;
import zju.ieeeformat.BusData;
import zju.matrix.ASparseMatrixLink;
import zju.measure.MeasTypeCons;
import zju.measure.MeasureInfo;

import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-6-21
 */
public class BusCalUtil implements MeasTypeCons {
    private static Logger log = LogManager.getLogger(BusCalUtil.class);

    public static double[] getEstValue(BusData centerBus, Map<String, Integer> busIndex, Map<String, BranchData> branches, MeasureInfo[] meases, double[][] y, double[] x) {
        int i = 0;
        double[] g = new double[meases.length];
        for (MeasureInfo info : meases) {
            int type = info.getMeasureType();
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    g[i++] = x[busIndex.get(info.getPositionId())];
                    continue;
                case TYPE_BUS_ACTIVE_POWER:
                    g[i++] = StateCalByPolar.calBusP(y, x);
                    continue;
                case TYPE_BUS_REACTIVE_POWER:
                    g[i++] = StateCalByPolar.calBusQ(y, x);
                    continue;
            }
            BranchData branch = branches.get(info.getPositionId());
            Integer headIndex = busIndex.get(String.valueOf(branch.getTapBusNumber()));
            double headV = x[headIndex];
            Integer tailIndex = busIndex.get(String.valueOf(branch.getZBusNumber()));
            double tailV = x[tailIndex];
            double deltaTheta = branch.getTapBusNumber() == centerBus.getBusNumber() ? x[y[0].length + tailIndex - 1] : -x[y[0].length + headIndex - 1];
            switch (type) {
                case TYPE_LINE_FROM_ACTIVE:
                    g[i++] = StateCalByPolar.calLinePFrom(branch, new double[]{headV, tailV, deltaTheta});
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    g[i++] = StateCalByPolar.calLinePTo(branch, new double[]{headV, tailV, deltaTheta});
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    g[i++] = StateCalByPolar.calLineQFrom(branch, new double[]{headV, tailV, deltaTheta});
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    g[i++] = StateCalByPolar.calLineQTo(branch, new double[]{headV, tailV, deltaTheta});
                    break;
                default:
                    log.warn("Not supported type:" + meases[i].getMeasureType());
                    break;
            }
        }
        return g;
    }

    public static ASparseMatrixLink getJocobian(BusData centerBus, Map<String, Integer> busIndex, Map<String, BranchData> branches, MeasureInfo[] meases, double[][] y, double[] state) {
        ASparseMatrixLink result = new ASparseMatrixLink(meases.length, 2 * busIndex.size() - 1);
        int index = 0;
        for (int k = 0; k < meases.length; k++, index++) {
            MeasureInfo info = meases[k];
            int type = info.getMeasureType();
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    result.setValue(index, busIndex.get(info.getPositionId()), 1);
                    continue;
                case TYPE_BUS_ACTIVE_POWER:
                    for (int i : busIndex.values()) {
                        if (i == 0) {
                            result.setValue(index, 0, y[0][i] * state[0] + StateCalByPolar.calBusP(y, state) / state[0]);
                        } else {
                            double thetaIJ = state[i + busIndex.size() - 1];
                            double sin = Math.sin(thetaIJ);
                            double cos = Math.cos(thetaIJ);
                            result.setValue(index, i, state[0] * (y[0][i] * cos + y[1][i] * sin));
                            result.setValue(index, i + busIndex.size() - 1, -state[0] * state[i] * (y[0][i] * sin - y[1][i] * cos));
                        }
                    }
                    continue;
                case TYPE_BUS_REACTIVE_POWER:
                    for (int i : busIndex.values()) {
                        if (i == 0) {
                            result.setValue(index, 0, -y[1][i] * state[0] + StateCalByPolar.calBusQ(y, state) / state[0]);
                        } else {
                            double thetaIJ = state[i + busIndex.size() - 1];
                            double sin = Math.sin(thetaIJ);
                            double cos = Math.cos(thetaIJ);
                            result.setValue(index, i, state[0] * (y[0][i] * sin - y[1][i] * cos));
                            result.setValue(index, i + busIndex.size() - 1, state[0] * state[i] * (y[0][i] * cos + y[1][i] * sin));
                        }
                    }
                    continue;
            }
            BranchData branch = branches.get(info.getPositionId());
            double[][] ft = YMatrixGetter.getBranchAdmittance(branch);
            double[] gbg1b1_from = ft[0];
            double[] gbg1b1_to = ft[1];
            Integer headIndex = busIndex.get(String.valueOf(branch.getTapBusNumber()));
            Integer tailIndex = busIndex.get(String.valueOf(branch.getZBusNumber()));
            boolean b = branch.getTapBusNumber() == centerBus.getBusNumber();
            int thetaIndex = b ? busIndex.size() + tailIndex - 1 : busIndex.size() + headIndex - 1;
            double thetaIJ = b ? state[thetaIndex] : -state[thetaIndex];
            double cos = Math.cos(thetaIJ);
            double sin = Math.sin(thetaIJ);
            double[] x = new double[]{state[headIndex], state[tailIndex], thetaIJ};
            switch (type) {
                case TYPE_LINE_FROM_ACTIVE:
                    result.setValue(index, headIndex, 2 * x[0] * (gbg1b1_from[0] + gbg1b1_from[2]) - x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    result.setValue(index, tailIndex, -x[0] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    double tmp = x[0] * x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos);
                    result.setValue(index, thetaIndex, b ? tmp : -tmp);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    result.setValue(index, headIndex, -x[1] * (gbg1b1_to[0] * cos - gbg1b1_to[1] * sin));
                    result.setValue(index, tailIndex, 2 * x[1] * (gbg1b1_to[0] + gbg1b1_to[2]) - x[0] * (gbg1b1_to[0] * cos - gbg1b1_to[1] * sin));
                    tmp = -x[0] * x[1] * (gbg1b1_to[0] * sin + gbg1b1_to[1] * cos);
                    result.setValue(index, thetaIndex, b ? -tmp : tmp);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    result.setValue(index, headIndex, -2 * x[0] * (gbg1b1_from[1] + gbg1b1_from[3]) - x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    result.setValue(index, tailIndex, -x[0] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    tmp = -x[0] * x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin);
                    result.setValue(index, thetaIndex, b ? tmp : -tmp);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    result.setValue(index, headIndex, x[1] * (gbg1b1_to[0] * sin + gbg1b1_to[1] * cos));
                    result.setValue(index, tailIndex, -2 * x[1] * (gbg1b1_to[1] + gbg1b1_to[3]) + x[0] * (gbg1b1_to[0] * sin + gbg1b1_to[1] * cos));
                    tmp = -x[0] * x[1] * (gbg1b1_to[0] * cos - gbg1b1_to[1] * sin);
                    result.setValue(index, thetaIndex, b ? -tmp : tmp);
                    break;
                default:
                    log.warn("Not supported type:" + type);
                    break;
            }
        }
        return result;
    }

    public static ASparseMatrixLink[] getHessian(BusData centerBus, Map<String, Integer> busIndex, Map<String, BranchData> branches, MeasureInfo[] meases, double[][] y, double[] state, ASparseMatrixLink jacobian) {
        ASparseMatrixLink[] result = new ASparseMatrixLink[meases.length];
        int n = busIndex.size() - 1;
        int i, j;
        for (int index = 0; index < meases.length; index++) {
            MeasureInfo info = meases[index];
            int type = info.getMeasureType();
            result[index] = new ASparseMatrixLink(2 * busIndex.size() - 1, 2 * busIndex.size() - 1);
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    continue;
                case TYPE_BUS_ACTIVE_POWER:
                    int k = jacobian.getIA()[index];
                    while (k != -1) {
                        j = jacobian.getJA().get(k);
                        double v = jacobian.getVA().get(k);
                        k = jacobian.getLINK().get(k);
                        if (j == 0) {
                            result[index].setValue(0, 0, 2 * y[0][0]);
                        } else if (j < busIndex.size()) {
                            double v1 = v / state[0];
                            result[index].setValue(j, 0, v1);
                        } else {
                            double v1 = v / state[j - n];
                            result[index].setValue(j, j - n, v1);
                            result[index].setValue(j, 0, v / state[0]);
                            double gij = y[0][j - n];
                            double bij = y[1][j - n];
                            double cos = Math.cos(state[j]);
                            double sin = Math.sin(state[j]);
                            double mij = -state[0] * state[j - n] * (gij * cos + bij * sin);
                            result[index].setValue(j, j, mij);
                        }
                    }
                    continue;
                case TYPE_BUS_REACTIVE_POWER:
                    k = jacobian.getIA()[index];
                    while (k != -1) {
                        j = jacobian.getJA().get(k);
                        double v = jacobian.getVA().get(k);
                        k = jacobian.getLINK().get(k);
                        if (j == 0) {
                            result[index].setValue(0, 0, -2 * y[1][0]);
                        } else if (j < busIndex.size()) {
                            double v1 = v / state[0];
                            result[index].setValue(j, 0, v1);
                        } else {
                            double v1 = v / state[j - n];
                            result[index].setValue(j, j - n, v1);
                            result[index].setValue(j, 0, v / state[0]);
                            double gij = y[0][j - n];
                            double bij = y[1][j - n];
                            double cos = Math.cos(state[j]);
                            double sin = Math.sin(state[j]);
                            double hij = state[0] * state[j - n] * (gij * sin - bij * cos);
                            result[index].setValue(j, j, -hij);
                        }
                    }
                    continue;
            }

            BranchData branch = branches.get(info.getPositionId());
            double[][] ft = YMatrixGetter.getBranchAdmittance(branch);
            double[] gbg1b1_from = ft[0];
            double[] gbg1b1_to = ft[1];
            i = busIndex.get(String.valueOf(branch.getTapBusNumber()));
            j = busIndex.get(String.valueOf(branch.getZBusNumber()));
            boolean b = branch.getTapBusNumber() == centerBus.getBusNumber();
            int thetaIndex = b ? busIndex.size() + j - 1 : busIndex.size() + i - 1;
            double thetaIJ = b ? state[thetaIndex] : -state[thetaIndex];
            double cos = Math.cos(thetaIJ);
            double sin = Math.sin(thetaIJ);
            double[] x = new double[]{state[i], state[j], thetaIJ};
            switch (type) {
                case TYPE_LINE_FROM_ACTIVE:
                    result[index].setValue(i, i, 2 * (gbg1b1_from[0] + gbg1b1_from[2]));
                    result[index].setValue(i, j, -gbg1b1_from[0] * cos - gbg1b1_from[1] * sin);
                    if (!b)
                        result[index].setValue(i, i + n, -x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    else
                        result[index].setValue(i, j + n, x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));

                    result[index].setValue(j, i, -gbg1b1_from[0] * cos - gbg1b1_from[1] * sin);
                    if (!b)
                        result[index].setValue(j, i + n, -x[0] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    else
                        result[index].setValue(j, j + n, x[0] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));

                    if (!b) {
                        result[index].setValue(i + n, i, -x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                        result[index].setValue(i + n, j, -x[0] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                        result[index].setValue(i + n, i + n, x[0] * x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    } else {
                        result[index].setValue(j + n, i, x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                        result[index].setValue(j + n, j, x[0] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                        result[index].setValue(j + n, j + n, x[0] * x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    }
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    result[index].setValue(i, i, -2 * (gbg1b1_from[1] + gbg1b1_from[3]));
                    result[index].setValue(i, j, -gbg1b1_from[0] * sin + gbg1b1_from[1] * cos);
                    if (!b)
                        result[index].setValue(i, i + n, x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    else
                        result[index].setValue(i, j + n, -x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));

                    result[index].setValue(j, i, -gbg1b1_from[0] * sin + gbg1b1_from[1] * cos);
                    if (!b)
                        result[index].setValue(j, i + n, x[0] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    else
                        result[index].setValue(j, j + n, -x[0] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));

                    if (!b) {
                        result[index].setValue(i + n, i, x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                        result[index].setValue(i + n, j, x[0] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                        result[index].setValue(i + n, i + n, x[0] * x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    } else {
                        result[index].setValue(j + n, i, -x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                        result[index].setValue(j + n, j, -x[0] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                        result[index].setValue(j + n, i + n, -x[0] * x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    }
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    sin = -sin;
                    result[index].setValue(j, j, 2 * (gbg1b1_to[0] + gbg1b1_to[2]));
                    result[index].setValue(j, i, -gbg1b1_to[0] * cos - gbg1b1_to[1] * sin);
                    if (b)
                        result[index].setValue(j, j + n, -x[0] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    else
                        result[index].setValue(j, i + n, x[0] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));

                    result[index].setValue(i, j, -gbg1b1_to[0] * cos - gbg1b1_to[1] * sin);
                    if (b)
                        result[index].setValue(i, j + n, -x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    else
                        result[index].setValue(i, i + n, x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));

                    if (b) {
                        result[index].setValue(j + n, j, -x[0] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                        result[index].setValue(j + n, i, -x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                        result[index].setValue(j + n, j + n, x[0] * x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    } else {
                        result[index].setValue(i + n, j, x[0] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                        result[index].setValue(i + n, i, x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                        result[index].setValue(i + n, i + n, x[0] * x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    }
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    sin = -sin;
                    result[index].setValue(j, j, -2 * (gbg1b1_to[1] + gbg1b1_to[3]));
                    result[index].setValue(j, i, -gbg1b1_to[0] * sin + gbg1b1_to[1] * cos);
                    if (b)
                        result[index].setValue(j, j + n, x[0] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    else
                        result[index].setValue(j, i + n, -x[0] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));

                    result[index].setValue(i, j, -gbg1b1_to[0] * sin + gbg1b1_to[1] * cos);
                    if (b)
                        result[index].setValue(i, j + n, x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    else
                        result[index].setValue(i, i + n, -x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));

                    if (b) {
                        result[index].setValue(j + n, j, x[0] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                        result[index].setValue(j + n, i, x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                        result[index].setValue(j + n, j + n, x[0] * x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    } else {
                        result[index].setValue(i + n, j, -x[0] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                        result[index].setValue(i + n, i, -x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                        result[index].setValue(i + n, i + n, x[0] * x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    }
                    break;
                default:
                    log.warn("unsupported measure type: " + type);
                    break;
            }
        }
        return result;
    }
}
