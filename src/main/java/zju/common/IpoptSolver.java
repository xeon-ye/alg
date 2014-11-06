package zju.common;

import org.coinor.Ipopt;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 *         Date: 2009-3-14
 */
public class IpoptSolver extends Ipopt {
    private IpoptModel model;
    private boolean isConverged;
    private double objective;
    private Map<String, String> strOption = new HashMap<String, String>();
    private Map<String, Integer> intOption = new HashMap<String, Integer>();
    private Map<String, Double> doubleOption = new HashMap<String, Double>();

    public IpoptSolver(IpoptModel model) {
        this.model = model;
    }

    public void solve() {
        int maxIters = intOption.containsKey(KEY_MAX_ITER) ? intOption.get(KEY_MAX_ITER) : 500;
        double tol = 1e-4;
        //double tol = doubleOption.containsKey(KEY_TOL) ? doubleOption.get(KEY_TOL) : 1e-4;
        solve(maxIters, tol, true);
    }

    public void solve(int maxIters, double tol, boolean isDispose) {
        solve(maxIters, tol, isDispose, false);
    }

    /**
     * @param maxIters               最大迭代次数
     * @param tol                    总体收敛精度
     * @param isDispose              是否回收内存
     * @param isHessianApproximation 是否进行Hessian矩阵的自动模拟
     */
    public void solve(int maxIters, double tol, boolean isDispose, boolean isHessianApproximation) {
        this.create(model.getN(), model.getM(), model.getNele_jac(), model.getNele_hess(), C_STYLE);
        if (!model.isPrintPath())
            this.setIntegerOption(KEY_PRINT_LEVEL, 1);

        //open it, can use hessian approximation
        if (isHessianApproximation)
            setStringOption(Ipopt.KEY_HESSIAN_APPROXIMATION, "limited-memory");

        setNumericOption(KEY_TOL, tol);
        //setNumericOption(Ipopt.KEY_COMPL_INF_TOL, tol);
        //setNumericOption(Ipopt.KEY_CONSTR_VIOL_TOL, tol);
        //setNumericOption(Ipopt.KEY_DUAL_INF_TOL, tol);
        //
        //setNumericOption(Ipopt.KEY_ACCEPTABLE_TOL,0.01);
        //setStringOption(KEY_DERIVATIVE_TEST, "first-order");
        //setNumericOption(KEY_DERIVATIVE_TEST_TOL, 1e-2);
        for (String key : strOption.keySet())
            setStringOption(key, strOption.get(key));
        for (String key : intOption.keySet())
            setIntegerOption(key, intOption.get(key));
        for (String key : doubleOption.keySet())
            setNumericOption(key, doubleOption.get(key));
        setIntegerOption(KEY_MAX_ITER, maxIters);

        // solve the problem
        int result = OptimizeNLP();
        isConverged = (result == Ipopt.SOLVE_SUCCEEDED || result == Ipopt.ACCEPTABLE_LEVEL);
        objective = getObjVal();
        //fill state in model
        if (isConverged)
            model.fillState(getState());
        if (isDispose)
            dispose();
    }

    public boolean isConverged() {
        return isConverged;
    }

    public double getObjective() {
        return objective;
    }

    public void setOption(String key, String value) {
        strOption.put(key, value);
    }

    public void setOption(String key, Integer value) {
        intOption.put(key, value);
    }

    public void setOption(String key, Double value) {
        doubleOption.put(key, value);
    }

    @Override
    protected boolean get_bounds_info(int n, double[] x_l, double[] x_u, int m, double[] g_l, double[] g_u) {
        model.setXLimit(x_l, x_u);
        model.setGLimit(g_l, g_u);
        return true;
    }

    @Override
    protected boolean get_starting_point(int n, boolean init_x, double[] x, boolean init_z, double[] z_L, double[] z_U, int m, boolean init_lambda, double[] lambda) {
        model.setStartingPoint(x);
        return true;
    }

    protected boolean eval_f(int n, double[] x, boolean new_x, double[] obj_value) {
        return model.eval_f(n, x, new_x, obj_value);
    }

    protected boolean eval_grad_f(int n, double[] x, boolean new_x, double[] grad_f) {
        return model.eval_grad_f(n, x, new_x, grad_f);
    }

    protected boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g) {
        return model.eval_g(n, x, new_x, m, g);
    }

    protected boolean eval_jac_g(int n, double[] x, boolean new_x, int m, int nele_jac, int[] iRow, int[] jCol, double[] values) {
        return model.eval_jac_g(n, x, new_x, m, nele_jac, iRow, jCol, values);
    }

    protected boolean eval_h(int n, double[] x, boolean new_x, double obj_factor, int m, double[] lambda, boolean new_lambda, int nele_hess, int[] iRow, int[] jCol, double[] values) {
        return model.eval_h(n, x, new_x, obj_factor, m, lambda, new_lambda, nele_hess, iRow, jCol, values);
    }
}
