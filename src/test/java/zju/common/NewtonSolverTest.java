package zju.common;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import junit.framework.TestCase;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;


/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/5/7
 */
public class NewtonSolverTest extends TestCase {

    public AVector state = new AVector(new double[]{0.0, 0.0});

    public DoubleMatrix2D jacobian = new SparseDoubleMatrix2D(4, 2);

    public AVector z_est = new AVector(4);

    /*
    某次实验得到了四个数据点 (x, y)：(1, 6)、(2, 5)、(3, 7)、(4, 10)。我们希望找出一条和这四个点最匹配的直线 y=\beta_1+\beta_2 x，
    即找出在某种“最佳情况”下能够大致符合如下超定线性方程组的 \beta_1 和 \beta_2：

    \beta_1  +  1\beta_2  = 6
    \beta_1  +  2\beta_2  = 5
    \beta_1  +  3\beta_2  = 7
    \beta_1  +  4\beta_2  = 10

    最小二乘法采用的手段是尽量使得等号两边的方差最小，也就是找出这个函数的最小值：
    最小值可以通过对 S(\beta_1, \beta_2) 分别求 \beta_1 和 \beta_2 的偏导数，然后使它们等于零得到。
    如此就得到了一个只有两个未知数的方程组，很容易就可以解出：
    也就是说直线 y=3.5+1.4x 是最佳的。
    */
    public void testWlsFromWiki() {
        NewtonSolver solver = new NewtonSolver();
        jacobian.setQuick(0, 0, 1.0);
        jacobian.setQuick(0, 1, 1.0);
        jacobian.setQuick(1, 0, 1.0);
        jacobian.setQuick(1, 1, 2.0);
        jacobian.setQuick(2, 0, 1.0);
        jacobian.setQuick(2, 1, 3.0);
        jacobian.setQuick(3, 0, 1.0);
        jacobian.setQuick(3, 1, 4.0);

        solver.setModel(new NewtonWlsModel() {
            @Override
            public AVector getWeight() {
                return new AVector(new double[]{1.0, 1.0, 1.0, 1.0});
            }

            @Override
            public boolean isJacLinear() {
                return true;
            }

            @Override
            public int getMaxIter() {
                return 50;
            }

            @Override
            public double getTolerance() {
                return 1e-4;
            }

            @Override
            public boolean isTolSatisfied(double[] delta) {
                return false;
            }

            @Override
            public AVector getInitial() {
                return state;
            }

            @Override
            public DoubleMatrix2D getJocobian(AVector state) {
                return jacobian;
            }

            @Override
            public ASparseMatrixLink2D getJacobianStruc() {
                return null;
            }

            @Override
            public AVector getZ() {
                return new AVector(new double[]{6, 5, 7, 10});
            }

            @Override
            public double[] getDeltaArray() {
                return null;
            }

            @Override
            public AVector calZ(AVector state) {
                z_est.setValue(0, state.getValue(0) + state.getValue(1));
                z_est.setValue(1, state.getValue(0) + 2. * state.getValue(1));
                z_est.setValue(2, state.getValue(0) + 3. * state.getValue(1));
                z_est.setValue(3, state.getValue(0) + 4. * state.getValue(1));
                return z_est;
            }

            @Override
            public boolean isJacStrucChange() {
                return false;
            }
        });
        assertTrue(solver.solveWls(false));
        assertEquals(3.5, state.getValue(0), 1e-4);
        assertEquals(1.4, state.getValue(1), 1e-4);
    }
}
