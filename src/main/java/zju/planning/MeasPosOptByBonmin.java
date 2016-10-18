package zju.planning;

import org.coinor.Bonmin;
import zju.dsmodel.DsTopoIsland;
import zju.ieeeformat.IEEEDataIsland;
import zju.se.MeasPosOpt;

/**
 * Created by arno on 16-10-17.
 */
public class MeasPosOptByBonmin extends MeasPosOpt {

    public MeasPosOptByBonmin(IEEEDataIsland island) {
        super(island);
    }

    public MeasPosOptByBonmin(DsTopoIsland dsIsland, boolean isP2pNeglected) {
        super(dsIsland, isP2pNeglected);
    }

    class Solver extends Bonmin {
        @Override
        protected boolean get_variables_types(int n, int[] var_types) {
            return true;
        }

        @Override
        protected boolean get_variables_linearity(int n, int[] var_types) {
            return false;
        }

        @Override
        protected boolean get_constraints_linearity(int m, int[] const_types) {
            return false;
        }

        @Override
        protected boolean get_bounds_info(int n, double[] x_l, double[] x_u, int m, double[] g_l, double[] g_u) {
            return false;
        }

        @Override
        protected boolean get_starting_point(int n, boolean init_x, double[] x, boolean init_z, double[] z_L, double[] z_U, int m, boolean init_lambda, double[] lambda) {
            return false;
        }

        @Override
        protected boolean eval_f(int n, double[] x, boolean new_x, double[] obj_value) {
            return false;
        }

        @Override
        protected boolean eval_grad_f(int n, double[] x, boolean new_x, double[] grad_f) {
            return false;
        }

        @Override
        protected boolean eval_g(int n, double[] x, boolean new_x, int m, double[] g) {
            return false;
        }

        @Override
        protected boolean eval_jac_g(int n, double[] x, boolean new_x, int m, int nele_jac, int[] iRow, int[] jCol, double[] values) {
            return false;
        }

        @Override
        protected boolean eval_h(int n, double[] x, boolean new_x, double obj_factor, int m, double[] lambda, boolean new_lambda, int nele_hess, int[] iRow, int[] jCol, double[] values) {
            return false;
        }
    }
}
