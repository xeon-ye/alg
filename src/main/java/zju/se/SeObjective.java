package zju.se;

import cern.colt.matrix.DoubleMatrix2D;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.measure.MeasVector;
import zju.measure.MeasureInfo;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-10-20
 */
public class SeObjective {
    public static final int OBJ_TYPE_WLS = 1;
    public static final int OBJ_TYPE_QL = 2;
    public static final int OBJ_TYPE_QC = 3;
    public static final int OBJ_TYPE_SIGMOID = 4;
    public static final int OBJ_TYPE_PARTION = 5;
    public static final int OBJ_TYPE_MNMR = 6;
    public static final int OBJ_TYPE_MSE = 7;

    protected MeasVector meas;
    protected MeasureInfo[] aMeas = new MeasureInfo[0];
    protected int[] aMeasPos = new int[0];
    protected MeasureInfo[] vMeas = new MeasureInfo[0];
    protected int[] vMeasPos = new int[0];
    protected MeasureInfo[] pMeas = new MeasureInfo[0];
    protected int[] pMeasPos = new int[0];
    protected MeasureInfo[] qMeas = new MeasureInfo[0];
    protected int[] qMeasPos = new int[0];

    protected int[] measInObjFunc;
    double thresholds[];

    double a[], b[];
    double shortenRate = 1.0;
    // Parzen 窗宽
    double mesSigma;
    int objType = OBJ_TYPE_WLS;

    public boolean eval_f(int n, double[] x, boolean new_x, double[] obj_value, int offset) {
        switch (objType) {
            case OBJ_TYPE_WLS:
                return eval_f_wls(x, obj_value, offset);
            case OBJ_TYPE_MSE:
                return eval_f_mse(x, obj_value, offset);
            case OBJ_TYPE_QL:
                return eval_f_ql(x, obj_value, offset);
            case OBJ_TYPE_QC:
                return eval_f_qc(x, obj_value, offset);
            case OBJ_TYPE_SIGMOID:
                return eval_f_sigmoid(x, obj_value, offset);
            //case OBJ_TYPE_PARTION:
            //       return eval_f_(n, x, new_x, obj_value, offset);
        }
        return false;
    }

    // to estimate jacobian matrix of the objective function

    public boolean eval_grad_f(int n, double[] x, boolean new_x, double[] grad_f, int offset) {
        switch (objType) {
            case OBJ_TYPE_WLS:
                return eval_grad_f_wls(x, grad_f, offset);
            case OBJ_TYPE_MSE:
                return eval_grad_f_mse(x, grad_f, offset);
            case OBJ_TYPE_QL:
                return eval_grad_f_ql(x, grad_f, offset);
            case OBJ_TYPE_QC:
                return eval_grad_f_qc(x, grad_f, offset);
            case OBJ_TYPE_SIGMOID:
                return eval_grad_f_sigmoid(x, grad_f, offset);
            //case OBJ_TYPE_PARTION:
            //       return eval_f_(n, x, new_x, obj_value, offset);
        }
        return false;
    }

    public void getHessStruc(DoubleMatrix2D hessian) {
        switch (objType) {
            case OBJ_TYPE_WLS:
            case OBJ_TYPE_MSE:
                for (int pos : aMeasPos)
                    hessian.setQuick(pos, pos, 1.0);
                for (int pos : vMeasPos)
                    hessian.setQuick(pos, pos, 1.0);
                for (int pos : pMeasPos)
                    hessian.setQuick(pos, pos, 1.0);
                for (int pos : qMeasPos)
                    hessian.setQuick(pos, pos, 1.0);
                break;
            default:
                break;
        }
    }

    public void getHessStruc(int[] rows, int[] cols, DoubleMatrix2D hessian, int offset) {
        int start = hessian.cardinality();
        switch (objType) {
            case OBJ_TYPE_WLS:
            case OBJ_TYPE_MSE:
                for (int i = offset, j = 0; j < measInObjFunc.length; i++, j++) {
                    //hessian.setQuick(i, i, 1.0);
                    rows[start] = i;
                    cols[start++] = i;
                }
                break;
            case OBJ_TYPE_QL:
            case OBJ_TYPE_QC:
            case OBJ_TYPE_SIGMOID:
                for (int i = offset, j = 0; j < measInObjFunc.length; i++, j++) {
                    //hessian.setQuick(i, i, 1.0);
                    rows[start] = i;
                    cols[start++] = i;
                }
                break;
            default:
                break;
        }
    }

    public void fillHessian(double[] x, double[] values, MySparseDoubleMatrix2D hessian, double obj_factor, int offset) {
        int start = hessian.cardinality();
        switch (objType) {
            case OBJ_TYPE_WLS:
                for (int i = 0; i < aMeasPos.length; i++)
                    hessian.addQuick(aMeasPos[i], aMeasPos[i], obj_factor * aMeas[i].getWeight());
                for (int i = 0; i < vMeasPos.length; i++)
                    hessian.addQuick(vMeasPos[i], vMeasPos[i], obj_factor * vMeas[i].getWeight());
                for (int i = 0; i < pMeasPos.length; i++)
                    hessian.addQuick(pMeasPos[i], pMeasPos[i], obj_factor * pMeas[i].getWeight());
                for (int i = 0; i < qMeasPos.length; i++)
                    hessian.addQuick(qMeasPos[i], qMeasPos[i], obj_factor * qMeas[i].getWeight());
                for (int pos : measInObjFunc)
                    //hessian.addQuick(i, i, obj_factor * meas.getWeight().getValue(pos));
                    values[start++] = obj_factor * meas.getWeight().getValue(pos);
                break;
            case OBJ_TYPE_MSE:
                double squareMesSigma = mesSigma * mesSigma;
                for (int i = 0; i < aMeasPos.length; i++) {
                    double v = x[aMeasPos[i]] - aMeas[i].getValue();
                    hessian.addQuick(aMeasPos[i], aMeasPos[i], -obj_factor * aMeas[i].getWeight() * Math.exp(-v * v / (2 * squareMesSigma)) * (v * v / (squareMesSigma * squareMesSigma) - 1 / (squareMesSigma)));
                }
                for (int i = 0; i < vMeasPos.length; i++) {
                    double v = x[vMeasPos[i]] - vMeas[i].getValue();
                    hessian.addQuick(vMeasPos[i], vMeasPos[i], -obj_factor * vMeas[i].getWeight() * Math.exp(-v * v / (2 * squareMesSigma)) * (v * v / (squareMesSigma * squareMesSigma) - 1 / (squareMesSigma)));
                }
                for (int i = 0; i < pMeasPos.length; i++) {
                    double v = x[pMeasPos[i]] - pMeas[i].getValue();
                    hessian.addQuick(pMeasPos[i], pMeasPos[i], -obj_factor * pMeas[i].getWeight() * Math.exp(-v * v / (2 * squareMesSigma)) * (v * v / (squareMesSigma * squareMesSigma) - 1 / (squareMesSigma)));
                }
                for (int i = 0; i < qMeasPos.length; i++) {
                    double v = x[qMeasPos[i]] - qMeas[i].getValue();
                    hessian.addQuick(qMeasPos[i], qMeasPos[i], -obj_factor * qMeas[i].getWeight() * Math.exp(-v * v / (2 * squareMesSigma)) * (v * v / (squareMesSigma * squareMesSigma) - 1 / (squareMesSigma)));
                }
                for (int pos : measInObjFunc) {
                    double v = x[pos + offset];
                    values[start++] = -obj_factor * meas.getWeight().getValue(pos) * Math.exp(-v * v / (2 * squareMesSigma)) * (v * v / (squareMesSigma * squareMesSigma) - 1 / (squareMesSigma));
                }
                break;
            case OBJ_TYPE_QL:
                for (int j = 0; j < measInObjFunc.length; j++) {
                    double v = x[j + offset];
                    double delta = thresholds[j];
                    if (Math.abs(v) <= delta)
                        //hessian.addQuick(i, i, obj_factor * meas.getWeight().getValue(measInObjFunc[j]));
                        values[start++] = obj_factor * meas.getWeight().getValue(measInObjFunc[j]);
                }
                break;
            case OBJ_TYPE_QC:
                for (int i = 0; i < measInObjFunc.length; i++) {
                    double v = x[i + offset];
                    double delta = thresholds[i];
                    if (Math.abs(v) > delta / 2.0 && Math.abs(v) < delta)
                        //hessian.addQuick(i, i, obj_factor * 2.0);
                        values[start++] = obj_factor * meas.getWeight().getValue(measInObjFunc[i]);
                }
                break;
            case OBJ_TYPE_SIGMOID:
                for (int i = 0; i < measInObjFunc.length; i++) {
                    double v = x[i + offset];
                    double y1 = 1.0 / (1.0 + Math.exp(-b[i] * (v - a[i])));
                    double y2 = 1.0 / (1.0 + Math.exp(b[i] * (v + a[i])));
                    double y = shortenRate * b[i] * b[i] * (y1 * (1.0 - y1) * (1.0 - 2.0 * y1)
                            + y2 * (1.0 - y2) * (1.0 - 2.0 * y2));
                    //hessian.addQuick(i, i, obj_factor * y);
                    values[start++] = obj_factor * y;
                }
                break;
            default:
                break;
        }
    }

    // to estimate the objective function
    public boolean eval_f_sigmoid(double[] x, double[] obj_value, int offset) {
        obj_value[0] = 0;
        for (int i : measInObjFunc) {
            double v = x[i + offset];
            double tmp1 = 1.0 / (1.0 + Math.exp(-b[i] * (v - a[i])));
            double tmp2 = 1.0 / (1.0 + Math.exp(b[i] * (v + a[i])));
            obj_value[0] += shortenRate * (tmp1 + tmp2);
        }
        return true;
    }

    public boolean eval_grad_f_sigmoid(double[] x, double[] grad_f, int offset) {
        for (int i = 0; i < x.length; i++)
            grad_f[i] = 0;
        for (int i : measInObjFunc) {
            double v = x[i + offset];
            double tmp1 = 1.0 / (1.0 + Math.exp(-b[i] * (v - a[i])));
            double tmp2 = 1.0 / (1.0 + Math.exp(b[i] * (v + a[i])));
            grad_f[i + offset] = shortenRate * b[i] * (tmp1 * (1 - tmp1) - tmp2 * (1 - tmp2));
        }
        return true;
    }

    public boolean eval_f_ql(double[] x, double[] obj_value, int offset) {
        obj_value[0] = 0;
        for (int i : measInObjFunc) {
            double v = x[i + offset];
            double delta = thresholds[i];
            if (v <= delta)
                obj_value[0] += meas.getWeight().getValue(i) * v * v * 0.5;
            else
                obj_value[0] += meas.getWeight().getValue(i) * delta * delta * 0.5;
        }
        return true;
    }

    public boolean eval_grad_f_ql(double[] x, double[] grad_f, int offset) {
        for (int i = 0; i < x.length; i++)
            grad_f[i] = 0;
        for (int i : measInObjFunc) {
            double v = x[i + offset];
            double delta = thresholds[i];
            if (Math.abs(v) <= delta)
                grad_f[i + offset] = meas.getWeight().getValue(i) * v;
            else
                grad_f[i + offset] = 0;
        }
        return true;
    }

    public boolean eval_f_qc(double[] x, double[] obj_value, int offset) {
        obj_value[0] = 0;
        for (int i : measInObjFunc) {
            double v = x[i + offset];
            double delta = thresholds[i];
            if (v <= delta) //todo:
                obj_value[0] += meas.getWeight().getValue(i) * v * v * 0.5;
            else
                obj_value[0] += meas.getWeight().getValue(i) * delta * delta * 0.5;
        }
        return true;
    }

    public boolean eval_grad_f_qc(double[] x, double[] grad_f, int offset) {
        for (int i = 0; i < x.length; i++)
            grad_f[i] = 0;
        for (int i : measInObjFunc) {
            double v = x[i + offset];
            double delta = thresholds[i];
            if (Math.abs(v) <= delta / 2.0)
                grad_f[i + offset] = 0;
            else if (Math.abs(v) > delta / 2.0 && Math.abs(v) < delta)
                grad_f[i + offset] = 2 * (v - delta / 2);
            else
                grad_f[i + offset] = 0;
        }
        return true;
    }

    public boolean eval_f_wls(double[] x, double[] obj_value, int offset) {
        obj_value[0] = 0;
        for (int i = 0; i < aMeasPos.length; i++) {
            double v = x[aMeasPos[i]] - aMeas[i].getValue();
            obj_value[0] += shortenRate * aMeas[i].getWeight() * v * v * 0.5;
        }
        for (int i = 0; i < vMeasPos.length; i++) {
            double v = x[vMeasPos[i]] - vMeas[i].getValue();
            obj_value[0] += shortenRate * vMeas[i].getWeight() * v * v * 0.5;
        }
        for (int i = 0; i < pMeasPos.length; i++) {
            double v = x[pMeasPos[i]] - pMeas[i].getValue();
            obj_value[0] += shortenRate * pMeas[i].getWeight() * v * v * 0.5;
        }
        for (int i = 0; i < qMeasPos.length; i++) {
            double v = x[qMeasPos[i]] - qMeas[i].getValue();
            obj_value[0] += shortenRate * qMeas[i].getWeight() * v * v * 0.5;
        }
        for (int i : measInObjFunc) {
            double v = x[i + offset];
            obj_value[0] += shortenRate * meas.getWeight().getValue(i) * v * v * 0.5;
        }
        return true;
    }

    public boolean eval_grad_f_wls(double[] x, double[] grad_f, int offset) {
        for (int i = 0; i < x.length; i++)
            grad_f[i] = 0;
        for (int i = 0; i < aMeasPos.length; i++) {
            double v = x[aMeasPos[i]] - aMeas[i].getValue();
            grad_f[aMeasPos[i]] = shortenRate * aMeas[i].getWeight() * v;
        }
        for (int i = 0; i < vMeasPos.length; i++) {
            double v = x[vMeasPos[i]] - vMeas[i].getValue();
            grad_f[vMeasPos[i]] = shortenRate * vMeas[i].getWeight() * v;
        }
        for (int i = 0; i < pMeasPos.length; i++) {
            double v = x[pMeasPos[i]] - pMeas[i].getValue();
            grad_f[pMeasPos[i]] = shortenRate * pMeas[i].getWeight() * v;
        }
        for (int i = 0; i < qMeasPos.length; i++) {
            double v = x[qMeasPos[i]] - qMeas[i].getValue();
            grad_f[qMeasPos[i]] = shortenRate * qMeas[i].getWeight() * v;
        }
        for (int i : measInObjFunc) {
            double v = x[i + offset];
            grad_f[i + offset] = shortenRate * v * meas.getWeight().getValue(i);
        }
        return true;
    }

    public boolean eval_f_mse(double[] x, double[] obj_value, int offset) {
        double doubleSquareMesSigma = 2 * mesSigma * mesSigma;
        obj_value[0] = 0;
        for (int i = 0; i < aMeasPos.length; i++) {
            double v = x[aMeasPos[i]] - aMeas[i].getValue();
            obj_value[0] += shortenRate * aMeas[i].getWeight() * Math.exp(- v * v / doubleSquareMesSigma);
        }
        for (int i = 0; i < vMeasPos.length; i++) {
            double v = x[vMeasPos[i]] - vMeas[i].getValue();
            obj_value[0] += shortenRate * vMeas[i].getWeight() * Math.exp(- v * v / doubleSquareMesSigma);
        }
        for (int i = 0; i < pMeasPos.length; i++) {
            double v = x[pMeasPos[i]] - pMeas[i].getValue();
            obj_value[0] += shortenRate * pMeas[i].getWeight() * Math.exp(- v * v / doubleSquareMesSigma);
        }
        for (int i = 0; i < qMeasPos.length; i++) {
            double v = x[qMeasPos[i]] - qMeas[i].getValue();
            obj_value[0] += shortenRate * qMeas[i].getWeight() * Math.exp(- v * v / doubleSquareMesSigma);
        }
        for (int i : measInObjFunc) {
            double v = x[i + offset];
            obj_value[0] += shortenRate * meas.getWeight().getValue(i) * Math.exp(- v * v / doubleSquareMesSigma);
        }
        obj_value[0] = -obj_value[0];
        return true;
    }

    public boolean eval_grad_f_mse(double[] x, double[] grad_f, int offset) {
        double squareMesSigma = mesSigma * mesSigma;
        for (int i = 0; i < x.length; i++)
            grad_f[i] = 0;
        for (int i = 0; i < aMeasPos.length; i++) {
            double v = x[aMeasPos[i]] - aMeas[i].getValue();
            grad_f[aMeasPos[i]] = -shortenRate * aMeas[i].getWeight() * (-v / squareMesSigma * Math.exp(-v * v / (2 * squareMesSigma)));
        }
        for (int i = 0; i < vMeasPos.length; i++) {
            double v = x[vMeasPos[i]] - vMeas[i].getValue();
            grad_f[vMeasPos[i]] = -shortenRate * vMeas[i].getWeight() * (-v / squareMesSigma * Math.exp(-v * v / (2 * squareMesSigma)));
        }
        for (int i = 0; i < pMeasPos.length; i++) {
            double v = x[pMeasPos[i]] - pMeas[i].getValue();
            grad_f[pMeasPos[i]] = -shortenRate * pMeas[i].getWeight() * (-v / squareMesSigma * Math.exp(-v * v / (2 * squareMesSigma)));
        }
        for (int i = 0; i < qMeasPos.length; i++) {
            double v = x[qMeasPos[i]] - qMeas[i].getValue();
            grad_f[qMeasPos[i]] = -shortenRate * qMeas[i].getWeight() * (-v / squareMesSigma * Math.exp(-v * v / (2 * squareMesSigma)));
        }
        for (int i : measInObjFunc) {
            double v = x[i + offset];
            grad_f[i + offset] = -shortenRate * (-v / squareMesSigma * Math.exp(-v * v / (2 * squareMesSigma))) * meas.getWeight().getValue(i);
        }
        return true;
    }

    public int[] getMeasInObjFunc() {
        return measInObjFunc;
    }

    public void setMeasInObjFunc(int[] measInObjFunc) {
        this.measInObjFunc = measInObjFunc;
    }

    public double getShortenRate() {
        return shortenRate;
    }

    public void setShortenRate(double shortenRate) {
        this.shortenRate = shortenRate;
    }

    public double getMesSigma() {
        return mesSigma;
    }

    public void setMesSigma(double mesSigma) {
        this.mesSigma = mesSigma;
    }

    public void setInitialMesSigma() {
        this.mesSigma = 10 / Math.sqrt(2);
    }

    public int getObjType() {
        return objType;
    }

    public void setObjType(int objType) {
        this.objType = objType;
    }

    public double[] getThresholds() {
        return thresholds;
    }

    public void setThresholds(double[] thresholds) {
        this.thresholds = thresholds;
    }

    public MeasureInfo[] getVMeas() {
        return vMeas;
    }

    public void setVMeas(MeasureInfo[] vMeas) {
        this.vMeas = vMeas;
    }

    public int[] getVMeasPos() {
        return vMeasPos;
    }

    public void setVMeasPos(int[] vMeasPos) {
        this.vMeasPos = vMeasPos;
    }

    public MeasureInfo[] getPMeas() {
        return pMeas;
    }

    public void setPMeas(MeasureInfo[] pMeas) {
        this.pMeas = pMeas;
    }

    public int[] getPMeasPos() {
        return pMeasPos;
    }

    public void setPMeasPos(int[] pMeasPos) {
        this.pMeasPos = pMeasPos;
    }

    public MeasureInfo[] getQMeas() {
        return qMeas;
    }

    public void setQMeas(MeasureInfo[] qMeas) {
        this.qMeas = qMeas;
    }

    public int[] getQMeasPos() {
        return qMeasPos;
    }

    public void setQMeasPos(int[] qMeasPos) {
        this.qMeasPos = qMeasPos;
    }

    public MeasureInfo[] getAMeas() {
        return aMeas;
    }

    public void setAMeas(MeasureInfo[] aMeas) {
        this.aMeas = aMeas;
    }

    public int[] getAMeasPos() {
        return aMeasPos;
    }

    public void setAMeasPos(int[] aMeasPos) {
        this.aMeasPos = aMeasPos;
    }

    public MeasVector getMeas() {
        return meas;
    }

    public void setMeas(MeasVector meas) {
        this.meas = meas;
    }

    public double[] getA() {
        return a;
    }

    public void setA(double[] a) {
        this.a = a;
    }

    public double[] getB() {
        return b;
    }

    public void setB(double[] b) {
        this.b = b;
    }
}
