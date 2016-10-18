package zju.planning;

import org.coinor.Bonmin;
import zju.dsmodel.DsTopoIsland;
import zju.ieeeformat.IEEEDataIsland;
import zju.matrix.ASparseMatrixLink2D;
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
            //对约束的参数赋值
            int k, col, index, count, rowInA = 1;
            int nonZeroOfRow = 0, nonZeroOfCol[] = new int[size], nonZeroOfCurrent;
            for (int row = 0; row < size; row++) {
                for (int j = row; j < size; j++, rowInA++) {
                    //记录当前行的前j列共有多少个非零元
                    nonZeroOfCol[j] = 0;
                    for (ASparseMatrixLink2D d : Ds)
                        nonZeroOfCol[j] += d.getNA()[row] * (j - row + 1);
                }

                nonZeroOfCurrent = 0;//记录当前
                count = 0; //记录当前矩阵的位置
                for (ASparseMatrixLink2D d : Ds) {
                    k = d.getIA()[row];
                    while (k != -1) {
                        col = d.getJA().get(k);
                        for (int j = row; j < size; j++) {
                            index = nonZeroOfRow + nonZeroOfCurrent;
                            if (j > row)
                                index += nonZeroOfCol[j - 1];
                            values[index] = d.getVA().get(k);
                            if (col > j) {
                                iRow[index] = row;
                                jCol[index] = binaryNum + count * (size - 1) * (size + 2) / 2 + j * size - j * (j + 1) / 2 + col;
                            } else {
                                jCol[index] = binaryNum + count * (size - 1) * (size + 2) / 2 + col * size - col * (col + 1) / 2 + j;
                            }
                            nonZeroOfCurrent++;
                            k = d.getLINK().get(k);
                        }
                        count++;
                    }
                }
                //记录前row行一共多少个非零元
                for (ASparseMatrixLink2D d : Ds)
                    nonZeroOfRow += d.getNA()[row] * (size - row);
            }
            return true;
        }

        @Override
        protected boolean eval_h(int n, double[] x, boolean new_x, double obj_factor, int m, double[] lambda, boolean new_lambda, int nele_hess, int[] iRow, int[] jCol, double[] values) {
            return false;
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
