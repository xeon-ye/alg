package jpscpu;

import cern.colt.matrix.DoubleMatrix1D;
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

    private int drive = SUPERLU_DRIVE_0;

    private int[] perm_c, etree, colcnt_h, part_super_h;

    private int m, n, nnz;

    private double[] a;

    private int[] asub, xa;

    private SCPformat L;

    private NCPformat U;

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
     * <br>计算  Ax = b,使用pdgssvx引擎, solver2和solve3配合适合多次计算，而A的结构不变，只是值变化的情况</br>
     * <br>solve2是第一次调用时使用，perm_c、etree、colcnt_h、part_super_h用于存储第一次对A进行LU分解的结构信息，在cpp中进行赋值</br>
     *
     * @param m      A的行数
     * @param n      A的列数
     * @param nnz    A的非零元个数
     * @param a      A的非零元的数值
     * @param asub   A的非零元行号
     * @param xa     xa的第i个元素表示前i行共有多少个非零元
     * @param b      向量b
     * @param perm_c 用于存储A分解后的信息
     * @param etree  用于存储A分解后的信息
     * @return 计算结果x
     */
    private native double[] solve2_mt(int m, int n, int nnz, double[] a, int[] asub, int[] xa, double[] b,
                                      int nprocs, int[] perm_c, int[] etree, int[] colcnt_h, int[] part_super_h,
                                      SCPformat L, NCPformat U);

    /**
     * <br>计算  Ax = b,使用pdgssvx引擎, solver2和solve3配合适合多次计算，而A的结构不变，只是值变化的情况</br>
     * <br>solve3是非第一次调用时使用，perm_c、etree、colcnt_h、part_super_h已经存储了第一次对A进行LU分解的结构信息</br>
     *
     * @param m      A的行数
     * @param n      A的列数
     * @param nnz    A的非零元个数
     * @param a      A的非零元的数值
     * @param asub   A的非零元行号
     * @param xa     xa的第i个元素表示前i行共有多少个非零元
     * @param b      向量b
     * @param perm_c 用于存储A分解后的信息
     * @param etree  用于存储A分解后的信息
     * @return 计算结果x
     */
    private native double[] solve3_mt(int m, int n, int nnz, double[] a, int[] asub, int[] xa, double[] b,
                                      int nprocs, int[] perm_c, int[] etree, int[] colcnt_h, int[] part_super_h,
                                      int nnz_L, int nsuper_L, double[] nzval_L, int[] nzval_colbeg_L, int[] nzval_colend_L,
                                      int[] rowind_L, int[] rowind_colbeg_L, int[] rowind_colend_L, int[] col_to_sup_L,
                                      int[] sup_to_colbeg_L, int[] sup_to_colend_L,
                                      int nnz_U, double[] nzval_U, int[] rowind_U, int[] colbeg_U, int[] colend_U);


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

    /**
     * 求解Ax = b, 求解结果存储在right中，非第一次求解可以调用该方法
     *
     * @param left  A矩阵
     * @param right b向量，求解结果存储在该向量中
     * @return right
     */
    public double[] solve(DoubleMatrix2D left, double[] right) {
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

    /**
     * 对于Ax=b, A的结构不变且需要多次求解的问题，第一次求解可以使用此方法
     *
     * @param jacStruc A矩阵结构
     * @param left     矩阵A
     * @param b        向量b
     * @return x
     */
    public double[] solve2(ASparseMatrixLink2D jacStruc, DoubleMatrix2D left, double[] b) {
        m = jacStruc.getM();
        n = jacStruc.getN();
        nnz = jacStruc.getVA().size();
        a = new double[jacStruc.getVA().size()];
        asub = new int[jacStruc.getVA().size()];
        xa = new int[jacStruc.getN() + 1];
        jacStruc.getSluStrucNC(asub, xa);
        ColtMatrixUtil.getSluMatrixNC(left, a, asub, xa);
        perm_c = new int[n];
        etree = new int[n];
        colcnt_h = new int[n];
        part_super_h = new int[n];
        L = new SCPformat();
        U = new NCPformat();
//        System.out.println(colcnt_h[0]);
        return solve2_mt(m, n, nnz, a, asub, xa, b, nprocs, perm_c, etree, colcnt_h, part_super_h, L, U);
    }

    /**
     * 对于Ax=b, A的结构不变且需要多次求解的问题
     *
     * @param left        矩阵A
     * @param b           向量b
     * @param isFirstTime 是否是第一次计算
     * @return x
     */
    public double[] solve2(ASparseMatrixLink2D left, double[] b, boolean isFirstTime) {
        if (isFirstTime) {
            m = left.getM();
            n = left.getN();
            nnz = left.getVA().size();
            a = new double[left.getVA().size()];
            asub = new int[left.getVA().size()];
            xa = new int[left.getN() + 1];
            left.getSluStrucNC(a, asub, xa);
            perm_c = new int[n];
            etree = new int[n];
            colcnt_h = new int[n];
            part_super_h = new int[n];
            L = new SCPformat();
            U = new NCPformat();
            return solve2_mt(m, n, nnz, a, asub, xa, b, nprocs, perm_c, etree, colcnt_h, part_super_h, L, U);
        } else {
            left.getSluStrucNC(a);
            return solve3_mt(m, n, nnz, a, asub, xa, b, nprocs, perm_c, etree, colcnt_h, part_super_h,
                    L.getNnz(), L.getNsuper(), L.getNzval(), L.getNzval_colbeg(), L.getNzval_colend(), L.getRowind(),
                    L.getRowind_colbeg(), L.getRowind_colend(), L.getCol_to_sup(), L.getSup_to_colbeg(), L.getSup_to_colend(),
                    U.getNnz(), U.getNzval(), U.getRowind(), U.getColbeg(), U.getColend());
        }
    }

    public double[] solve2(DoubleMatrix1D right) {
        return solve2(right.toArray());
    }

    /**
     * 对于Ax=b, A的结构和值均不变且需要多次求解的问题， 非第一次求解可以使用方法
     *
     * @param right 向量b
     * @return x
     */
    public double[] solve2(double[] right) {
        return solve3_mt(m, n, nnz, a, asub, xa, right, nprocs, perm_c, etree, colcnt_h, part_super_h,
                L.getNnz(), L.getNsuper(), L.getNzval(), L.getNzval_colbeg(), L.getNzval_colend(), L.getRowind(),
                L.getRowind_colbeg(), L.getRowind_colend(), L.getCol_to_sup(), L.getSup_to_colbeg(), L.getSup_to_colend(),
                U.getNnz(), U.getNzval(), U.getRowind(), U.getColbeg(), U.getColend());

    }

    /**
     * 对于Ax=b, A的结构不变,只有值发生变化，且需要多次求解的问题，非第一次求解可以使用方法
     *
     * @param left 矩阵A
     * @param b    向量b
     * @return x
     */
    public double[] solve2(DoubleMatrix2D left, double[] b) {
        ColtMatrixUtil.getSluMatrixNC(left, a, asub, xa);
//        System.out.println("第二次求解");
        return solve3_mt(m, n, nnz, a, asub, xa, b, nprocs, perm_c, etree, colcnt_h, part_super_h,
                L.getNnz(), L.getNsuper(), L.getNzval(), L.getNzval_colbeg(), L.getNzval_colend(), L.getRowind(),
                L.getRowind_colbeg(), L.getRowind_colend(), L.getCol_to_sup(), L.getSup_to_colbeg(), L.getSup_to_colend(),
                U.getNnz(), U.getNzval(), U.getRowind(), U.getColbeg(), U.getColend());
    }

    public int getDrive() {
        return drive;
    }

    public void setDrive(int drive) {
        this.drive = drive;
    }
}
