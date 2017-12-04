package jpscpu;

import cern.colt.matrix.DoubleMatrix2D;
import zju.matrix.ASparseMatrixLink2D;
import zju.util.ColtMatrixUtil;

/**
 * Created by IntelliJ IDEA.
 * Author: Fang Rui
 * Date: 17-11-30
 * Time: 下午4:43
 */
public class LinearSolverMT {

    public static final int SUPERLU_DRIVE_0 = 0;
    public static final int SUPERLU_DRIVE_1 = 1;
    public static final int MLP_DRIVE_CBC = 2;
    public static final int MLP_DRIVE_SYM = 3;

    private int drive = SUPERLU_DRIVE_0;

    private int[] perm_c, perm_r, etree;

    private int m, n, nnz;

    private double[] a;

    private int[] asub, xa;

    private SCformat L;

    private NCformat U;

    private double[] R, C;

    private int[] equed_int;

    private int nprocs = 1;

    static {
        SoFileLoader.loadSoFiles();
    }

    public LinearSolverMT(int nprocs) {
        this.nprocs = nprocs;
    }

    /**
     * 计算  Ax = b,使用pdgssv引擎
     *
     * @param m    A的行数
     * @param n    A的列数
     * @param nnz  A的非零元个数
     * @param a    A的非零元的数值
     * @param asub A的非零元行号
     * @param xa   xa的第i个元素表示前i行共有多少个非零元
     * @param b    向量b
     * @return 计算结果x
     */
    private native double[] solve0_mt(int m, int n, int nnz, double[] a, int[] asub, int[] xa, double[] b, int nprocs);

    /**
     * 计算  Ax = b,使用pdgssvx引擎
     *
     * @param m    A的行数
     * @param n    A的列数
     * @param nnz  A的非零元个数
     * @param a    A的非零元的数值
     * @param asub A的非零元行号
     * @param xa   xa的第i个元素表示前i行共有多少个非零元
     * @param b    向量b
     * @return 计算结果x
     */
    private native double[] solve1_mt(int m, int n, int nnz, double[] a, int[] asub, int[] xa, double[] b, int nprocs);


    /**
     * 求解Ax = b, 求解结果存储在right中，第一次求解可以调用该方法
     *
     * @param jacStruc jacobian矩阵的结构
     * @param left     A矩阵
     * @param right    b向量，求解结果存储在该向量中
     * @return right
     */
    public double[] solve(ASparseMatrixLink2D jacStruc, DoubleMatrix2D left, double[] right) {
        m = jacStruc.getM();
        n = jacStruc.getN();
        nnz = jacStruc.getVA().size();
        a = new double[jacStruc.getVA().size()];
        asub = new int[jacStruc.getVA().size()];
        xa = new int[jacStruc.getN() + 1];
        jacStruc.getSluStrucNC(asub, xa);
        ColtMatrixUtil.getSluMatrixNC(left, a, asub, xa);
        return solve(m, n, nnz, a, asub, xa, right);
    }

    public double[] solve(int m, int n, int nnz, double[] a, int[] asub, int[] xa, double[] b) {
        switch (drive) {
            case SUPERLU_DRIVE_0:
                return solve0_mt(m, n, nnz, a, asub, xa, b, nprocs);
            case SUPERLU_DRIVE_1:
                return solve1_mt(m, n, nnz, a, asub, xa, b, nprocs);
            default:
                return null;
        }
    }


}
