package zju.common;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.linalg.LUDecompositionQuick;
import jpscpu.LinearSolver;
import jpscpu.LinearSolverMT;
import org.apache.log4j.Logger;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.util.ColtMatrixUtil;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 * Date: 2008-8-9
 */
public class NewtonSolver {

    private static Logger log = Logger.getLogger(NewtonSolver.class);
    //colt包求解器
    public static final int LINEAR_SOLVER_COLT = 1;
    //SuperLU求解器
    public static final int LINEAR_SOLVER_SUPERLU = 2;
    //SuperLU_MT求解器
    public static final int LINEAR_SOLVER_SUPERLU_MT = 3;
    //默认为SuperLU求解器
    private int linearSolver = LINEAR_SOLVER_SUPERLU;
    //具体问题的模型
    private NewtonModel model;
    //状态变量，有具体问题模型提供
    private AVector state;
    //Jacobian矩阵的结构
    private ASparseMatrixLink2D jacStruc;
    //JNI封装的SuperLU线性求解器
    private LinearSolver sluSolver = new LinearSolver();
    //JNI封装的SuperLU_MT线性求解器
    private LinearSolverMT sluMTSolver = new LinearSolverMT(4);
    //jacobian矩阵的结构是否可以重用
    private boolean isJacStrucReuse = false;
    //迭代次数
    public int iterNum;

    public NewtonSolver() {
    }

    public NewtonSolver(NewtonModel model) {
        this.model = model;
    }

    /**
     * 求解方程
     *
     * @return 是否收敛
     */
    public boolean solve() {
        state = model.getInitial(); // 初始化状态变量
        AVector z = model.getZ(); // 量测值
        AVector z_est; // 量测估计值

        iterNum = 0;
        double[] result = model.getDeltaArray();
        DoubleMatrix2D left;
        while (iterNum < model.getMaxIter()) {
            iterNum++;
            log.debug("At iteration " + iterNum);
            //compute estimated measurement
            long start = System.nanoTime();
            z_est = model.calZ(state);
            log.debug("Time used for forming right hand b : " + (System.nanoTime() - start) / 1000 + " us");

            if (z == null) {
                for (int i = 0; i < z_est.getN(); i++)
                    result[i] = -z_est.getValue(i);
            } else {
                for (int i = 0; i < z_est.getN(); i++)
                    result[i] = z.getValue(i) - z_est.getValue(i);
            }
            if (model.isTolSatisfied(result))
                return true;
            start = System.nanoTime();
            //-----  evaluate jacobian  -----
            left = model.getJocobian(state);
            log.debug("Time used for forming jocobian matrix J : " + (System.nanoTime() - start) / 1000 + "us");
            if (linearSolver == LINEAR_SOLVER_COLT) {
                //LUDecompositionQuick luSolver = new LUDecompositionQuick(1e-6);
                LUDecompositionQuick solver = new LUDecompositionQuick();
                solver.decompose(left);
                solver.solve(new DenseDoubleMatrix1D(result));
            } else if (linearSolver == LINEAR_SOLVER_SUPERLU) {
                if (!model.isJacStrucChange() && jacStruc != null && isJacStrucReuse) {
                    sluSolver.solve2(left, result);
                } else if (!model.isJacStrucChange() && iterNum == 1) {
                    jacStruc = model.getJacobianStruc();
                    sluSolver.solve2(jacStruc, left, result);
                } else if (!model.isJacStrucChange()) {
                    sluSolver.solve2(left, result);
                } else {
                    jacStruc = model.getJacobianStruc();
                    sluSolver.solve(jacStruc, left, result);
                }
            } else if (linearSolver == LINEAR_SOLVER_SUPERLU_MT) {
                if (!model.isJacStrucChange() && jacStruc != null && isJacStrucReuse) {
                    sluMTSolver.solve2(left, result);
                } else if (!model.isJacStrucChange() && iterNum == 1) {
                    jacStruc = model.getJacobianStruc();
                    sluMTSolver.solve2(jacStruc, left, result);
                } else if (!model.isJacStrucChange()) {
                    sluMTSolver.solve2(left, result);
                } else {
                    jacStruc = model.getJacobianStruc();
                    sluMTSolver.solve(jacStruc, left, result);
                }
            }
            log.debug("计算Jx=b用时: " + (System.nanoTime() - start) / 1000 + " us");
            //check for convergence
            double max = 0;
            int columns = left.columns();
            for (int i = 0; i < columns; i++)
                if (Math.abs(result[i]) > max)
                    max = Math.abs(result[i]);
            if (max < model.getTolerance())
                return true;
            //update state
            for (int i = 0; i < state.getN(); i++)
                state.setValue(i, state.getValue(i) + result[i]);
        }
        return false;
    }

    /**
     * 求解WLS问题
     *
     * @return 是否收敛
     */
    public boolean solveWls() {
        iterNum = 0;
        state = model.getInitial();
        //量测值
        AVector z = model.getZ();
        //估计值
        AVector z_est;
        //权重
        final AVector weight = ((NewtonWlsModel) model).getWeight();
        //Jacobian矩阵的结果
        final ASparseMatrixLink2D jacStruc = new ASparseMatrixLink2D(weight.getN(), state.getN());
        //Jacobian和H'WH
        DoubleMatrix2D H = null, left = new MySparseDoubleMatrix2D(state.getN(), state.getN());
        //状态量的Delta值
        DoubleMatrix1D result = new DenseDoubleMatrix1D(state.getN());
        //存储量测值和估计值之间的偏差
        DoubleMatrix1D delta = new DenseDoubleMatrix1D(weight.getN());

        int k1, k2, i1, i2;
        double s;
        boolean isExist;

        while (iterNum <= model.getMaxIter()) {
            iterNum++;
            log.debug("At iteration " + iterNum);
            //compute estimated measurement
            long start = System.currentTimeMillis();
            //计算估计值
            z_est = model.calZ(state);
            log.debug("计算估计值用时: " + (System.currentTimeMillis() - start) + " ms");
            if (z == null) {
                for (int i = 0; i < z_est.getN(); i++)
                    delta.setQuick(i, -z_est.getValue(i));
            } else {
                for (int i = 0; i < z_est.getN(); i++)
                    delta.setQuick(i, z.getValue(i) - z_est.getValue(i));
            }
            //-----  evaluate jacobian  -----
            if (!((NewtonWlsModel) model).isJacLinear() || iterNum == 1) {
                H = model.getJocobian(state);
                //记录开始时间
                start = System.currentTimeMillis();
                if (iterNum == 1)
                    ColtMatrixUtil.toMyMatrix(H, jacStruc);
                //计算H'*W*H
                for (int row = 0; row < H.columns(); row++) {
                    k1 = jacStruc.getJA2()[row];
                    s = 0;
                    while (k1 != -1) {
                        i1 = jacStruc.getIA2().get(k1);
                        s += weight.getValue(i1) * H.getQuick(i1, row) * H.getQuick(i1, row);
                        k1 = jacStruc.getLINK2().get(k1);
                    }
                    left.setQuick(row, row, s);
                    for (int col = row + 1; col < H.columns(); col++) {
                        k1 = jacStruc.getJA2()[row];
                        k2 = jacStruc.getJA2()[col];
                        s = 0;
                        isExist = false;
                        while (true) {
                            i1 = jacStruc.getIA2().get(k1);
                            i2 = jacStruc.getIA2().get(k2);
                            if (i1 == i2) {
                                s += weight.getValue(i1) * H.getQuick(i1, row) * H.getQuick(i2, col);
                                k1 = jacStruc.getLINK2().get(k1);
                                k2 = jacStruc.getLINK2().get(k2);
                                isExist = true;
                            } else if (i1 < i2) {
                                k1 = jacStruc.getLINK2().get(k1);
                            } else {
                                k2 = jacStruc.getLINK2().get(k2);
                            }
                            if (k1 == -1 || k2 == -1)
                                break;
                        }
                        if (!isExist)
                            continue;
                        left.setQuick(row, col, s);
                        left.setQuick(col, row, s);
                    }
                }
                log.debug("Form H'*W*H matrix time used: " + (System.currentTimeMillis() - start) + " ms");
            }
            start = System.currentTimeMillis();
            //计算H'*W*(z - z_est)
            //HW.zMult(delta, result);
            for (int row = 0; row < H.columns(); row++) {
                s = 0;
                k1 = jacStruc.getJA2()[row];
                while (k1 != -1) {
                    i1 = jacStruc.getIA2().get(k1);
                    s += weight.getValue(i1) * H.getQuick(i1, row) * delta.getQuick(i1);
                    k1 = jacStruc.getLINK2().get(k1);
                }
                result.setQuick(row, s);
            }
            //所用时间
            log.debug("Form H'*W*(z - z_est) matrix time used:" + (System.currentTimeMillis() - start) + " ms");
            start = System.currentTimeMillis();
            if (iterNum == 1) {//第一次迭代
                ASparseMatrixLink2D gainStruc = new ASparseMatrixLink2D(left.rows(), left.columns(), left.cardinality());
                ColtMatrixUtil.toMyMatrix(left, gainStruc);
                //第一次求解
                double[] r = sluSolver.solve2(gainStruc, result.toArray(), true);
                result.assign(r);
            } else {
                double[] r = sluSolver.solve2(left, result.toArray());
                result.assign(r);
            }
            log.debug("Solve Ax = b time used: " + (System.currentTimeMillis() - start) + " ms");

            //check for convergence
            double max = 0;
            for (int i = 0; i < H.columns(); i++)
                if (Math.abs(result.getQuick(i)) > max)
                    max = Math.abs(result.getQuick(i));
            if (max < model.getTolerance())
                return true;
            //update state
            for (int i = 0; i < state.getN(); i++)
                state.setValue(i, state.getValue(i) + result.getQuick(i));
        }
        return false;
    }

    public int getLinearSolver() {
        return linearSolver;
    }

    public void setLinearSolver(int linearSolver) {
        this.linearSolver = linearSolver;
    }

    public void setModel(NewtonModel model) {
        this.model = model;
    }

    public boolean isJacStrucReuse() {
        return isJacStrucReuse;
    }

    public void setJacStrucReuse(boolean jacStrucReuse) {
        isJacStrucReuse = jacStrucReuse;
    }

    public int getIterNum() {
        return iterNum;
    }

}
