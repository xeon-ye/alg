package zju.common;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2009-3-14
 */
public interface IpoptModel {
    public void setStartingPoint(double[] x);

    boolean isPrintPath();

    boolean eval_f(int n, double[] x, boolean new_x, double[] obj_value);

    boolean eval_grad_f(int n, double[] x, boolean new_x, double[] grad_f);

    boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g);

    boolean eval_jac_g(int n, double[] x, boolean new_x, int m, int nele_jac, int[] iRow, int[] jCol, double[] values);

    boolean eval_h(int n, double[] x, boolean new_x, double obj_factor, int m, double[] lambda, boolean new_lambda, int nele_hess, int[] iRow, int[] jCol, double[] values);

    void setXLimit(double[] x_L, double[] x_U);

    void setGLimit(double[] g_L, double[] g_U);

    int getNele_jac();

    int getNele_hess();

    int getM();

    int getN();

    void fillState(double[] x);
}