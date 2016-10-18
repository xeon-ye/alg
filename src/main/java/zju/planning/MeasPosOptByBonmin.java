package zju.planning;

import org.coinor.Bonmin;
import zju.dsmodel.DsTopoIsland;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.se.MeasPosOpt;

/**
 * Created by arno on 16-10-17.
 * @author Dong Shufeng
 */
public class MeasPosOptByBonmin extends MeasPosOpt {

    public MeasPosOptByBonmin(IEEEDataIsland island) {
        super(island);
    }

    public MeasPosOptByBonmin(DsTopoIsland dsIsland, boolean isP2pNeglected) {
        super(dsIsland, isP2pNeglected);
    }

    class Solver extends Bonmin {
        protected MySparseDoubleMatrix2D jacobian, hessian;

        public Solver() {
            int n = binaryNum + (size * size + size) / 2;
            int m = (size * size + size) / 2 + 1;
            int nele_jac = 0;//todo
            int nele_hess = 0;
            createBonmin(n, m, nele_jac, nele_hess, C_STYLE);
        }

        @Override
        protected boolean get_variables_types(int n, int[] var_types) {
            int i = 0;
            for(; i < binaryNum; i++)
                var_types[i] = BONMIN_BINARY;
            for(; i < n; i++)
                var_types[i] = BONMIN_CONTINUOUS;
            return true;
        }

        @Override
        protected boolean get_variables_linearity(int n, int[] var_types) {
            for(int i = 0; i < n; i++)
                var_types[i] = IPOPT_TNLP_LINEAR;
            return true;
        }

        @Override
        protected boolean get_constraints_linearity(int m, int[] const_types) {
            for(int i = 0; i < m; i++)
                const_types[i] = IPOPT_TNLP_NON_LINEAR;
            return true;
        }

        @Override
        protected boolean get_bounds_info(int n, double[] x_l, double[] x_u, int m, double[] g_l, double[] g_u) {
            int i = 0;
            for(; i < binaryNum; i++) {
                x_l[i] = 0;
                x_u[i] = 1;
            }
            for(; i < n; i++) {
                x_l[i] = -2e14;
                x_u[i] = 2e14;
            }
            int count = 0;
            for (int row = 0; row < size; row++) {
                for (int j = row; j < size; j++, count++) {
                    if (j == row) {
                        g_u[count] = 1 + 1e-6;
                        g_l[count] = 1 - 1e-6;
                    } else {
                        g_u[count] = 1e-6;
                        g_l[count] = -1e-6;
                    }
                }
            }
            g_l[m - 1] = 0;
            g_u[m - 1] = maxDevNum;
            return true;
        }

        @Override
        protected boolean get_starting_point(int n, boolean init_x, double[] x, boolean init_z, double[] z_L, double[] z_U, int m, boolean init_lambda, double[] lambda) {
            int i = 0;
            for(; i < binaryNum; i++)
                x[i] = 0;
            for(; i < n; i++)
                x[i] = 0; //todo
            return true;
        }

        @Override
        protected boolean eval_f(int n, double[] x, boolean new_x, double[] obj_value) {
            for(int row = 0; row < size; row++)
                obj_value[0] += x[binaryNum + row * size - row * (row + 1)/2 + row];
            return true;
        }

        @Override
        protected boolean eval_grad_f(int n, double[] x, boolean new_x, double[] grad_f) {
            for(int i = 0; i < n; i++)
                grad_f[i] = 0.;
            for(int row = 0; row < size; row++)
                grad_f[binaryNum + row * size - row * (row + 1)/2 + row] = 1.0;
            return true;
        }

        @Override
        protected boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g) {
            for(int i = 0; i < m; i++)
                g[i] = 0.;
            int k, col;
            for (int row = 0; row < size; row++) {
                for(int i = 0; i < Ds.length; i++) {
                    ASparseMatrixLink2D d = Ds[i];
                    k = d.getIA()[row];
                    while (k != -1) {
                        col = d.getJA().get(k);
                        for (int j = row; j < size; j++) {
                            if (col > j)
                                if(i > 0)
                                    g[row * size - row * (row + 1)/2 + j] += x[i - 1] * x[binaryNum +  j * size - j * (j + 1) / 2 + col] * d.getVA().get(k);
                                else
                                    g[row * size - row * (row + 1)/2 + j] += x[binaryNum +  j * size - j * (j + 1) / 2 + col] * d.getVA().get(k);
                            else {
                                if(i > 0)
                                    g[row * size - row * (row + 1)/2 + j] += x[i - 1] * x[binaryNum + col * size - col * (col + 1) / 2 + j] * d.getVA().get(k);
                                else
                                    g[row * size - row * (row + 1)/2 + j] += x[binaryNum + col * size - col * (col + 1) / 2 + j] * d.getVA().get(k);
                            }
                        }
                        k = d.getLINK().get(k);
                    }
                }
            }
            //最后一个是设备个数的约束
            for(int i = 0; i < binaryNum; i++)
                g[m - 1] += x[i];
            return true;
        }

        @Override
        protected boolean eval_jac_g(int n, double[] x, boolean new_x, int m, int nele_jac, int[] iRow, int[] jCol, double[] values) {
            updateJacobian(x, new_x);
            final int[] count = {0};
            if (values == null) {
                jacobian.forEachNonZero((row, col, v) -> {
                    iRow[count[0]] = row;
                    jCol[count[0]] = col;
                    count[0]++;
                    return v;
                });
            } else {
                jacobian.forEachNonZero((row, col, v) -> {
                    values[count[0]] = v;
                    count[0]++;
                    return v;
                });
            }
            return true;
        }

        @Override
        protected boolean eval_h(int n, double[] x, boolean new_x, double obj_factor, int m, double[] lambda, boolean new_lambda, int nele_hess, int[] iRow, int[] jCol, double[] values) {
            return false;
        }

        protected void updateJacobian(double[] x, boolean isNewX) {
            int k, col;
            for (int row = 0; row < size; row++) {
                for(int i = 0; i < Ds.length; i++) {
                    ASparseMatrixLink2D d = Ds[i];
                    k = d.getIA()[row];
                    while (k != -1) {
                        col = d.getJA().get(k);
                        for (int j = row; j < size; j++) {
                            int rowInJac = row * size - row * (row + 1) / 2 + j;
                            int varIndex;
                            if (col > j) {
                                varIndex = binaryNum + j * size - j * (j + 1) / 2 + col;
                            } else {
                                varIndex = binaryNum + col * size - col * (col + 1) / 2 + j;
                            }
                            if(i > 0)
                                jacobian.setQuick(rowInJac, i - 1, x[varIndex] * d.getVA().get(k));
                            else
                                jacobian.addQuick(rowInJac, varIndex, d.getVA().get(k));

                        }
                        k = d.getLINK().get(k);
                    }
                }
            }
        }
    }

    @Override
    public void doOpt(boolean isThreePhase) {
        prepare(isThreePhase);

        Solver solver = new Solver();
        //solver.setStringOption("bonmin.algorithm","B-BB");
        solver.setStringOption("bonmin.algorithm","B-OA");
        //solver.setStringOption("bonmin.algorithm","B-QG");
        //solver.setStringOption("bonmin.algorithm","B-Hyb");
        //solver.setStringOption("bonmin.algorithm","B-Ecp");
        //solver.setStringOption("bonmin.algorithm","B-iFP");

        solver.OptimizeMINLP();
        //Below see results
        double x[] = solver.getState();

        solver.dispose();
    }
}
