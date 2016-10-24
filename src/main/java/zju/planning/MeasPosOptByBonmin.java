package zju.planning;

import org.apache.log4j.Logger;
import org.coinor.Bonmin;
import zju.dsmodel.DsTopoIsland;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.MySparseDoubleMatrix2D;

/**
 * Created by arno on 16-10-17.
 * @author Dong Shufeng
 */
public class MeasPosOptByBonmin extends MeasPosOpt {

    private static Logger log = Logger.getLogger(MeasPosOptByBonmin.class);

    public MeasPosOptByBonmin(IEEEDataIsland island) {
        super(island);
    }

    public MeasPosOptByBonmin(DsTopoIsland dsIsland, boolean isP2pNeglected) {
        super(dsIsland, isP2pNeglected);
    }

    class Solver extends Bonmin {

        int n, m;
        protected MySparseDoubleMatrix2D jacobian, hessian;

        public Solver() {
            n = binaryNum + (size * size + size) / 2;
            m = (size * size + size) / 2 + 1;
            //填充jacobian矩阵
            jacobian = new MySparseDoubleMatrix2D(m, n);
            double[] ini_x = new double[n];
            get_starting_point(n, true, ini_x, false, null, null, m, false, null);
            updateJacobian(ini_x, true);
            for(int i = 0; i < binaryNum; i++)
                jacobian.setQuick(m - 1, i, 1.0);
            //为Hessian开辟内存
            hessian = new MySparseDoubleMatrix2D(n, n);
            updateHessian(null);

            int nele_jac = jacobian.cardinality();
            int nele_hess = hessian.cardinality();

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
            obj_value[0] = 0.0;
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
            int k, col, index, indexInG;
            for (int row = 0; row < size; row++) {
                for(int i = 0; i < Ds.length; i++) {
                    ASparseMatrixLink2D d = Ds[i];
                    k = d.getIA()[row];
                    while (k != -1) {
                        col = d.getJA().get(k);
                        for (int j = row; j < size; j++) {
                            indexInG = row * size - row * (row + 1) / 2 + j;
                            if (col > j) {
                                index = binaryNum + j * size - j * (j + 1) / 2 + col;
                            } else
                                index = binaryNum + col * size - col * (col + 1) / 2 + j;
                            if(i > 0)
                                g[indexInG] += x[i - 1] * x[index] * d.getVA().get(k);
                            else
                                g[indexInG] += x[index] * d.getVA().get(k);
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
            final int[] count = {0};
            if (values == null) {
                jacobian.forEachNonZero((row, col, v) -> {
                    iRow[count[0]] = row;
                    jCol[count[0]] = col;
                    count[0]++;
                    return v;
                });
            } else {
                updateJacobian(x, new_x);
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
            final int[] idx = new int[]{0};
            if (values == null) {
                hessian.forEachNonZero((i, j, v) -> {
                    iRow[idx[0]] = i;
                    jCol[idx[0]] = j;
                    idx[0]++;
                    return v;
                });
            } else {
                updateHessian(lambda);
                hessian.forEachNonZero((i, j, v) -> {
                    values[idx[0]] = v;
                    idx[0]++;
                    return v;
                });
            }
            return true;
        }

        private void updateHessian(double[] lambda) {
            hessian.forEachNonZero((i, j, v) -> 0.0);
            int k, col;
            for (int row = 0; row < size; row++) {
                for(int i = 1; i < Ds.length; i++) {
                    ASparseMatrixLink2D d = Ds[i];
                    k = d.getIA()[row];
                    while (k != -1) {
                        col = d.getJA().get(k);
                        for (int j = row; j < size; j++) {
                            int rowInJac = row * size - row * (row + 1) / 2 + j;
                            int varIndex;
                            if (col > j) {
                                varIndex = binaryNum + j * size - j * (j + 1) / 2 + col;
                            } else
                                varIndex = binaryNum + col * size - col * (col + 1) / 2 + j;
                            if(lambda != null) {
                                hessian.addQuick(i - 1, varIndex, d.getVA().get(k) * lambda[rowInJac]);
                                //hessian.addQuick(varIndex, i - 1, d.getVA().get(k) * lambda[rowInJac]);
                            } else {
                                hessian.addQuick(i - 1, varIndex, d.getVA().get(k));
                                //hessian.addQuick(varIndex, i - 1, d.getVA().get(k));
                            }
                            if(i > 0) {
                                if(lambda != null) {
                                    hessian.addQuick(i - 1, varIndex, d.getVA().get(k) * lambda[rowInJac]);
                                    //hessian.addQuick(varIndex, i - 1, d.getVA().get(k) * lambda[rowInJac]);
                                } else {
                                    hessian.addQuick(i - 1, varIndex, d.getVA().get(k));
                                    //hessian.addQuick(varIndex, i - 1, d.getVA().get(k));
                                }
                            }
                        }
                        k = d.getLINK().get(k);
                    }
                }
            }
            //System.out.println("============= hessian ====================");
            //hessian.printOnScree();//todo:
        }

        protected void updateJacobian(double[] x, boolean isNewX) {
            if(!isNewX)
                return;
            jacobian.forEachNonZero((i, j, v) -> {
                if(i < m - 1)
                    return 0.0;
                else
                    return v;
            });
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
                            if(i > 0) {
                                jacobian.addQuick(rowInJac, i - 1, x[varIndex] * d.getVA().get(k));
                                jacobian.addQuick(rowInJac, varIndex, x[i - 1] * d.getVA().get(k));
                            } else
                                jacobian.addQuick(rowInJac, varIndex, d.getVA().get(k));
                        }
                        k = d.getLINK().get(k);
                    }
                }
            }
            //System.out.println("================= jacobian ===========");
            //jacobian.printOnScree();//todo:
        }
    }

    @Override
    public void doOpt(boolean isThreePhase) {
        prepare(isThreePhase);

        Solver solver = new Solver();
        //solver.setStringOption("bonmin.algorithm","B-BB");
        //solver.setStringOption("bonmin.algorithm","B-OA");
        solver.setStringOption("bonmin.algorithm","B-QG");
        //solver.setStringOption("bonmin.algorithm","B-Hyb");
        //solver.setStringOption("bonmin.algorithm","B-Ecp");
        //solver.setStringOption("bonmin.algorithm","B-iFP");

        int status = solver.OptimizeMINLP();
        if (status < 0) {
            log.warn("计算不收敛.");
        } else { //状态位显示计算收敛
            log.info("计算结果.");
            //Below see results
            double x[] = solver.getState();
            for (int i = 0; i < binaryNum; i++)
                System.out.print(x[i] + "\t");
            System.out.println();
        }
        solver.dispose();
    }
}
