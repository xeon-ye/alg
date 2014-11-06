package zju.util;

import org.apache.log4j.Logger;
import zju.ieeeformat.BranchData;
import zju.matrix.ASparseMatrixLink;
import zju.matrix.ASparseMatrixLink2D;
import zju.measure.MeasTypeCons;
import zju.measure.MeasureInfo;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2008-6-18
 */
public class BranchCalUtil implements MeasTypeCons {
    private static Logger log = Logger.getLogger(BranchCalUtil.class);

    public static ASparseMatrixLink getJacobian(BranchData branch, MeasureInfo[] infos, double[] x) {
        ASparseMatrixLink result = new ASparseMatrixLink(infos.length, 3);
        int index = 0;
        double[][] ft = YMatrixGetter.getBranchAdmittance(branch);
        double[] gbg1b1_from = ft[0];
        double[] gbg1b1_to = ft[1];
        double cos = Math.cos(x[2]);
        double sin = Math.sin(x[2]);
        for (MeasureInfo info : infos) {
            int type = info.getMeasureType();
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    if (info.getPositionId().equals(String.valueOf(branch.getTapBusNumber()))) {
                        result.setValue(index, 0, 1.0);
                    } else {
                        result.setValue(index, 1, 1.0);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    result.setValue(index, 0, 2 * x[0] * (gbg1b1_from[0] + gbg1b1_from[2]) - x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    result.setValue(index, 1, -x[0] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    double tmp = x[0] * x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos);
                    result.setValue(index, 2, tmp);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    result.setValue(index, 0, -x[1] * (gbg1b1_to[0] * cos - gbg1b1_to[1] * sin));
                    result.setValue(index, 1, 2 * x[1] * (gbg1b1_to[0] + gbg1b1_to[2]) - x[0] * (gbg1b1_to[0] * cos - gbg1b1_to[1] * sin));
                    tmp = -x[0] * x[1] * (gbg1b1_to[0] * sin + gbg1b1_to[1] * cos);
                    result.setValue(index, 2, -tmp);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    result.setValue(index, 0, -2 * x[0] * (gbg1b1_from[1] + gbg1b1_from[3]) - x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    result.setValue(index, 1, -x[0] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    tmp = -x[0] * x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin);
                    result.setValue(index, 2, tmp);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    result.setValue(index, 0, x[1] * (gbg1b1_to[0] * sin + gbg1b1_to[1] * cos));
                    result.setValue(index, 1, -2 * x[1] * (gbg1b1_to[1] + gbg1b1_to[3]) + x[0] * (gbg1b1_to[0] * sin + gbg1b1_to[1] * cos));
                    tmp = -x[0] * x[1] * (gbg1b1_to[0] * cos - gbg1b1_to[1] * sin);
                    result.setValue(index, 2, -tmp);
                    break;
                default:
                    log.warn("Not supported type:" + info.getMeasureType());
                    break;
            }
            index++;
        }
        return result;
    }

    public static ASparseMatrixLink getJacobianOfGB(MeasureInfo[] infos, double[] x) {
        ASparseMatrixLink result = new ASparseMatrixLink(infos.length, 5);
        int index = 0;
        double cos = Math.cos(x[2]);
        double sin = Math.sin(x[2]);
        for (MeasureInfo info : infos) {
            int type = info.getMeasureType();
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    result.setValue(index, 0, x[0] * x[0] - x[0] * x[1] * cos);
                    result.setValue(index, 1, -x[0] * x[1] * sin);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    result.setValue(index, 0, x[1] * x[1] - x[0] * x[1] * cos);
                    result.setValue(index, 1, x[0] * x[1] * sin);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    result.setValue(index, 0, -x[0] * x[1] * sin);
                    result.setValue(index, 1, -x[0] * x[0] + x[0] * x[1] * cos);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    result.setValue(index, 0, x[0] * x[1] * sin);
                    result.setValue(index, 1, -x[1] * x[1] + x[0] * x[1] * cos);
                    break;
                default:
                    log.warn("Not supported type:" + info.getMeasureType());
                    break;
            }
            index++;
        }
        return result;
    }


    public static ASparseMatrixLink[] getHessian(BranchData branch, MeasureInfo[] meases, double[] x) {
        ASparseMatrixLink[] result = new ASparseMatrixLink[meases.length];
        int index = 0;
        int n = 2;
        int i = 0, j = 1;
        double[][] ft = YMatrixGetter.getBranchAdmittance(branch);
        double[] gbg1b1_from = ft[0];
        double[] gbg1b1_to = ft[1];
        double thetaIJ = x[2];
        double cos = Math.cos(thetaIJ);
        double sin = Math.sin(thetaIJ);
        for (MeasureInfo info : meases) {
            int type = info.getMeasureType();
            result[index] = new ASparseMatrixLink2D(4, 4);
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    result[index].setValue(i, i, 2 * (gbg1b1_from[0] + gbg1b1_from[2]));
                    result[index].setValue(i, j, -gbg1b1_from[0] * cos - gbg1b1_from[1] * sin);
                    result[index].setValue(i, i + n, x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    //result[index].setValue(i, j + n, -x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));

                    result[index].setValue(j, i, -gbg1b1_from[0] * cos - gbg1b1_from[1] * sin);
                    result[index].setValue(j, i + n, x[0] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    result[index].setValue(j, j + n, -x[0] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));

                    result[index].setValue(i + n, i, x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    result[index].setValue(i + n, j, x[0] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    result[index].setValue(i + n, i + n, x[0] * x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    //result[index].setValue(i + n, j + n, -x[0] * x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));

                    //result[index].setValue(j + n, i, -x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    //result[index].setValue(j + n, j, -x[0] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    //result[index].setValue(j + n, i + n, -x[0] * x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    //result[index].setValue(j + n, j + n, x[0] * x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    result[index].setValue(i, i, -2 * (gbg1b1_from[1] + gbg1b1_from[3]));
                    result[index].setValue(i, j, -gbg1b1_from[0] * sin + gbg1b1_from[1] * cos);
                    result[index].setValue(i, i + n, -x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    //result[index].setValue(i, j + n, x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));

                    result[index].setValue(j, i, -gbg1b1_from[0] * sin + gbg1b1_from[1] * cos);
                    result[index].setValue(j, i + n, -x[0] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    //result[index].setValue(j, j + n, x[0] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));

                    result[index].setValue(i + n, i, -x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    result[index].setValue(i + n, j, -x[0] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    result[index].setValue(i + n, i + n, x[0] * x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    result[index].setValue(i + n, j + n, -x[0] * x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));

                    //result[index].setValue(j + n, i, x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    //result[index].setValue(j + n, j, x[0] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    //result[index].setValue(j + n, i + n, -x[0] * x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    //result[index].setValue(j + n, j + n, x[0] * x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    sin = -sin;
                    result[index].setValue(j, j, 2 * (gbg1b1_to[0] + gbg1b1_to[2]));
                    result[index].setValue(j, i, -gbg1b1_to[0] * cos - gbg1b1_to[1] * sin);
                    //result[index].setValue(j, j + n, x[0] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    result[index].setValue(j, i + n, -x[0] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));

                    result[index].setValue(i, j, -gbg1b1_to[0] * cos - gbg1b1_to[1] * sin);
                    //result[index].setValue(i, j + n, x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    result[index].setValue(i, i + n, -x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));

                    //result[index].setValue(j + n, j, x[0] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    //result[index].setValue(j + n, i, x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    //result[index].setValue(j + n, j + n, x[0] * x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    //result[index].setValue(j + n, i + n, -x[0] * x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));

                    result[index].setValue(i + n, j, -x[0] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    result[index].setValue(i + n, i, -x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    result[index].setValue(i + n, j + n, -x[0] * x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    result[index].setValue(i + n, i + n, x[0] * x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    sin = -sin;
                    result[index].setValue(j, j, -2 * (gbg1b1_to[1] + gbg1b1_to[3]));
                    result[index].setValue(j, i, -gbg1b1_to[0] * sin + gbg1b1_to[1] * cos);
                    //result[index].setValue(j, j + n, -x[0] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    result[index].setValue(j, i + n, x[0] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));

                    result[index].setValue(i, j, -gbg1b1_to[0] * sin + gbg1b1_to[1] * cos);
                    //result[index].setValue(i, j + n, -x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    result[index].setValue(i, i + n, x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));

                    //result[index].setValue(j + n, j, -x[0] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    //result[index].setValue(j + n, i, -x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    //result[index].setValue(j + n, j + n, x[0] * x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    //result[index].setValue(j + n, i + n, -x[0] * x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));

                    result[index].setValue(i + n, j, x[0] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    result[index].setValue(i + n, i, x[1] * (gbg1b1_to[0] * cos + gbg1b1_to[1] * sin));
                    result[index].setValue(i + n, j + n, -x[0] * x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    result[index].setValue(i + n, i + n, x[0] * x[1] * (gbg1b1_to[0] * sin - gbg1b1_to[1] * cos));
                    break;
                default:
                    log.warn("unsupported measure type: " + type);
                    break;
            }
            index++;
        }
        return result;
    }

    public static ASparseMatrixLink[] getHessianOfGBYc(MeasureInfo[] meases, double[] x) {
        ASparseMatrixLink[] result = new ASparseMatrixLink[meases.length];
        int index = 0;
        double thetaIJ = x[2];
        double cos = Math.cos(thetaIJ);
        double sin = Math.sin(thetaIJ);
        for (MeasureInfo info : meases) {
            int type = info.getMeasureType();
            result[index] = new ASparseMatrixLink2D(3, 3);
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    result[index].setValue(0, 0, 2 * x[0] - x[1] * cos);
                    result[index].setValue(0, 1, -x[1] * sin);

                    result[index].setValue(1, 0, -x[0] * cos);
                    result[index].setValue(1, 1, -x[0] * sin);

                    result[index].setValue(2, 0, x[1] * x[0] * sin);
                    result[index].setValue(2, 1, -x[0] * x[1] * cos);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    result[index].setValue(0, 0, -x[1] * sin);
                    result[index].setValue(0, 1, -2 * x[0] + x[1] * cos);
                    result[index].setValue(0, 2, -2 * x[0]);

                    result[index].setValue(1, 0, -x[0] * sin);
                    result[index].setValue(1, 1, x[0] * cos);

                    result[index].setValue(2, 0, -x[1] * x[0] * cos);
                    result[index].setValue(2, 1, -x[0] * x[1] * sin);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    result[index].setValue(0, 0, -x[1] * cos);
                    result[index].setValue(0, 1, x[1] * sin);

                    result[index].setValue(1, 0, 2 * x[1] - x[0] * cos);
                    result[index].setValue(1, 1, x[0] * sin);

                    result[index].setValue(2, 0, x[1] * x[0] * sin);
                    result[index].setValue(2, 1, x[0] * x[1] * cos);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    result[index].setValue(0, 0, x[1] * sin);
                    result[index].setValue(0, 1, x[1] * cos);

                    result[index].setValue(1, 0, x[0] * sin);
                    result[index].setValue(1, 1, -2 * x[1] + x[0] * cos);
                    result[index].setValue(1, 2, -2 * x[1]);

                    result[index].setValue(2, 0, x[1] * x[0] * cos);
                    result[index].setValue(2, 1, -x[0] * x[1] * sin);
                    break;
                default:
                    log.warn("unsupported measure type: " + type);
                    break;
            }
            index++;
        }
        return result;
    }

    public static ASparseMatrixLink[] getHessianOfGB(MeasureInfo[] meases, double[] x) {
        ASparseMatrixLink[] result = new ASparseMatrixLink[meases.length];
        int index = 0;
        double thetaIJ = x[2];
        double cos = Math.cos(thetaIJ);
        double sin = Math.sin(thetaIJ);
        for (MeasureInfo info : meases) {
            int type = info.getMeasureType();
            result[index] = new ASparseMatrixLink2D(3, 3);
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    result[index].setValue(0, 0, 2 * x[0] - x[1] * cos);
                    result[index].setValue(0, 1, -x[1] * sin);

                    result[index].setValue(1, 0, -x[0] * cos);
                    result[index].setValue(1, 1, -x[0] * sin);

                    result[index].setValue(2, 0, x[1] * x[0] * sin);
                    result[index].setValue(2, 1, -x[0] * x[1] * cos);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    result[index].setValue(0, 0, -x[1] * sin);
                    result[index].setValue(0, 1, -2 * x[0] + x[1] * cos);

                    result[index].setValue(1, 0, -x[0] * sin);
                    result[index].setValue(1, 1, x[0] * cos);

                    result[index].setValue(2, 0, -x[1] * x[0] * cos);
                    result[index].setValue(2, 1, -x[0] * x[1] * sin);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    result[index].setValue(0, 0, -x[1] * cos);
                    result[index].setValue(0, 1, x[1] * sin);

                    result[index].setValue(1, 0, 2 * x[1] - x[0] * cos);
                    result[index].setValue(1, 1, x[0] * sin);

                    result[index].setValue(2, 0, x[1] * x[0] * sin);
                    result[index].setValue(2, 1, x[0] * x[1] * cos);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    result[index].setValue(0, 0, x[1] * sin);
                    result[index].setValue(0, 1, x[1] * cos);

                    result[index].setValue(1, 0, x[0] * sin);
                    result[index].setValue(1, 1, -2 * x[1] + x[0] * cos);

                    result[index].setValue(2, 0, x[1] * x[0] * cos);
                    result[index].setValue(2, 1, -x[0] * x[1] * sin);
                    break;
                default:
                    log.warn("unsupported measure type: " + type);
                    break;
            }
            index++;
        }
        return result;
    }

    public static double[] getEstValue(BranchData branch, MeasureInfo[] meases, double[] x) {
        int i = 0;
        double[] g = new double[meases.length];
        for (MeasureInfo info : meases) {
            int type = info.getMeasureType();
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    if (info.getPositionId().equals(String.valueOf(branch.getTapBusNumber()))) {
                        g[i++] = x[0];
                    } else {
                        g[i++] = x[1];
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    g[i++] = StateCalByPolar.calLinePFrom(branch, x);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    g[i++] = StateCalByPolar.calLinePTo(branch, x);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    g[i++] = StateCalByPolar.calLineQFrom(branch, x);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    g[i++] = StateCalByPolar.calLineQTo(branch, x);
                    break;
                default:
                    log.warn("Not supported type:" + meases[i].getMeasureType());
                    break;
            }
        }
        return g;
    }

    public static double[] getGBYc(BranchData branch) {
        double r = branch.getBranchR();
        double x = branch.getBranchX();
        return getGBYc(r, x, branch.getLineB());
    }

    public static double[] getGBYc(double r, double x, double lineB) {
        double g = r / (r * r + x * x);
        double b = -x / (r * r + x * x);
        double yc = lineB / 2.0;
        return new double[]{g, b, yc};
    }
}
