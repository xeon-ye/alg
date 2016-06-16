package zju.se.bdfilter;

import cern.colt.matrix.DoubleMatrix2D;
import weka.clusterers.SimpleKMeans;
import weka.core.*;
import zju.common.NewtonModel;
import zju.common.NewtonSolver;
import zju.ieeeformat.BranchData;
import zju.matrix.ASparseMatrixLink2D;
import zju.matrix.AVector;
import zju.matrix.MySparseDoubleMatrix2D;
import zju.measure.MeasTypeCons;
import zju.measure.MeasureInfo;
import zju.util.StateCalByPolar;
import zju.util.YMatrixGetter;

import java.text.DecimalFormat;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 13-11-24
 */
public class BranchBasedFilter implements NewtonModel, MeasTypeCons {

    public static final DecimalFormat numFormat = new DecimalFormat("#.####");

    //量测顺序为：Pij,Qij,Pji,Qji,Ui,Uj
    private static int[][] all_cases = {
            {0, 1, 3},
            {0, 1, 4},
            {0, 1, 5},
            {0, 3, 4},
            {0, 3, 5},
            {1, 2, 3},
            {1, 2, 4},
            {1, 2, 5},
            {2, 3, 4},
            {2, 3, 5},
            {0, 4, 5},
            {2, 4, 5},
    };
    private static int[] has_bad_p = {0, 1, 2, 3, 4, 10};
    private static int[] has_bad_q = {0, 5};
    private static int[] has_bad_v_i = {1, 3, 6, 8, 10, 11};
    private static int[] has_bad_v_j = {2, 4, 7, 9, 10, 11};

    //需要辨识的支路
    BranchData branch;
    //支路上的量测
    MeasureInfo[] meases;
    //Vi,Vj,Theta
    AVector state;

    public MySparseDoubleMatrix2D jac;
    //
    private AVector g;
    //用于计算v,theta
    private AVector z_meas;
    //用于计算v,theta
    private double[] delta;
    //当前选用的量测组合
    private int[] selectedMeasures;
    //聚类方法
    public SimpleKMeans clusterer;
    //聚类的参数
    public FastVector attributes;
    //用于聚类的数据
    public Instances dataset;

    private boolean isViBad = false;
    private boolean isVjBad = false;
    private boolean hasBadQ = false;
    private boolean hasBadP = false;
    public NewtonSolver solver;

    public BranchBasedFilter() throws Exception {
        g = new AVector(3);
        state = new AVector(3);
        z_meas = new AVector(3);
        delta = new double[3];
        //selectedMeasures = new int[3];
        jac = new MySparseDoubleMatrix2D(3, 3, 9, 0.2, 0.9);
        //jacStruc = new ASparseMatrixLink2D(3, 3);

        //初始化样本的属性
        attributes = new FastVector();
        attributes.addElement(new Attribute("vi"));
        attributes.addElement(new Attribute("vj"));
        attributes.addElement(new Attribute("theta"));
        //初始化数据集
        dataset = new Instances("Test-dataset", attributes, all_cases.length);
        for (int[] ignored : all_cases)
            dataset.add(new DenseInstance(3));
        //初始化聚类对象及其参数
        clusterer = new SimpleKMeans();
        clusterer.setOptions(weka.core.Utils.splitOptions(
                "-N 2 -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -S 10 -O"));
        solver = new NewtonSolver();
        solver.setLinearSolver(NewtonSolver.LINEAR_SOLVER_SUPERLU);
        solver.setModel(this);
    }

    public void doFilter() throws Exception {
        hasBadQ = false;
        isViBad = false;
        hasBadP = false;

        for (int index = 0; index < all_cases.length; index++) {
            selectedMeasures = all_cases[index];
            for (int i = 0; i < selectedMeasures.length; i++)
                z_meas.setValue(i, meases[selectedMeasures[i]].getValue());
            if (!solver.solve()) {
                System.out.println("Not converged.");
                return;
            } else {
                for (int j = 0; j < state.getN(); j++)
                    dataset.instance(index).setValue(j, state.getValue(j));
                //dataset.instance(index).setValue(0, state.getValue(0) - state.getValue(1));
                //dataset.instance(index).setValue(1, state.getValue(2));
                //System.out.println(Arrays.toString(selectedMeasures));
                //state.printOnScreen();
                //state.printOnScreen(numFormat);
            }
        }
        //进行聚类
        clusterer.buildClusterer(dataset);
        //打印聚类结果的情况
        //System.out.println(Arrays.toString(clusterer.getAssignments()));
        //ClusterEvaluation eval = new ClusterEvaluation();
        //eval.setClusterer(clusterer);
        //eval.evaluateClusterer(dataset);
        //System.out.println(eval.clusterResultsToString());
        //分析聚类结果
        if (clusterer.getNumClusters() > 1) {
            int[] clusterResult = clusterer.getAssignments();
            //检查判据
            hasBadQ = isHasBad(clusterResult, has_bad_q);
            hasBadP = false;
            isViBad = false;
            isVjBad = false;
            if (hasBadQ)
                return;
            hasBadP = isHasBad(clusterResult, has_bad_p);
            if (hasBadP)
                return;
            isViBad = isHasBad(clusterResult, has_bad_v_i);
            if (isViBad)
                return;
            isVjBad = isHasBad(clusterResult, has_bad_v_j);
        }
    }

    private boolean isHasBad(int[] clusterResult, int[] shouldBeTogether) {
        int tmp = clusterResult[shouldBeTogether[0]];
        for (int i = 0, j = 0; i < clusterResult.length; i++) {
            if (i == shouldBeTogether[j]) {
                if (clusterResult[i] != tmp)
                    return false;
                if (j < shouldBeTogether.length - 1)
                    j++;
            } else {
                if (clusterResult[i] == tmp)
                    return false;
            }
        }
        return true;
    }

    @Override
    public int getMaxIter() {
        return 50;
    }

    @Override
    public double getTolerance() {
        return 1e-5;
    }

    @Override
    public boolean isTolSatisfied(double[] delta) {
        return false;
    }

    @Override
    public AVector getInitial() {
        state.setValue(0, 1.0);
        state.setValue(1, 1.0);
        state.setValue(2, 0.0);
        return state;
    }

    @Override
    public DoubleMatrix2D getJocobian(AVector state) {
        fillJacobian(state.getValues());
        return jac;
    }

    @Override
    public ASparseMatrixLink2D getJacobianStruc() {//todo: not efficient
        int index = 0;
        ASparseMatrixLink2D jacStruc = new ASparseMatrixLink2D(3, 3);

        for (int i : selectedMeasures) {
            MeasureInfo info = meases[i];
            int type = info.getMeasureType();
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    if (info.getPositionId().equals(String.valueOf(branch.getTapBusNumber()))) {
                        jacStruc.setValue(index, 0, 1.0);
                    } else {
                        jacStruc.setValue(index, 1, 1.0);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                case TYPE_LINE_TO_ACTIVE:
                case TYPE_LINE_FROM_REACTIVE:
                case TYPE_LINE_TO_REACTIVE:
                    jacStruc.setValue(index, 0, 1.0);
                    jacStruc.setValue(index, 1, 1.0);
                    jacStruc.setValue(index, 2, 1.0);
                    break;
                default:
                    break;
            }
            index++;
        }
        return jacStruc;
    }

    @Override
    public AVector getZ() {
        return z_meas;
    }

    @Override
    public double[] getDeltaArray() {
        return delta;
    }

    @Override
    public AVector calZ(AVector state) {
        getEstValue(state.getValues(), g.getValues());
        return g;
    }

    @Override
    public boolean isJacStrucChange() {
        return false;
    }

    public void fillJacobian(double[] x) {
        int index = 0;
        double[][] ft = YMatrixGetter.getBranchAdmittance(branch);
        double[] gbg1b1_from = ft[0];
        double[] gbg1b1_to = ft[1];
        double cos = Math.cos(x[2]);
        double sin = Math.sin(x[2]);
        for (int i : selectedMeasures) {
            MeasureInfo info = meases[i];
            int type = info.getMeasureType();
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    if (info.getPositionId().equals(String.valueOf(branch.getTapBusNumber()))) {
                        jac.setQuick(index, 0, 1.0);
                    } else {
                        jac.setQuick(index, 1, 1.0);
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    jac.setQuick(index, 0, 2 * x[0] * (gbg1b1_from[0] + gbg1b1_from[2]) - x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    jac.setQuick(index, 1, -x[0] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin));
                    double tmp = x[0] * x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos);
                    jac.setQuick(index, 2, tmp);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    jac.setQuick(index, 0, -x[1] * (gbg1b1_to[0] * cos - gbg1b1_to[1] * sin));
                    jac.setQuick(index, 1, 2 * x[1] * (gbg1b1_to[0] + gbg1b1_to[2]) - x[0] * (gbg1b1_to[0] * cos - gbg1b1_to[1] * sin));
                    tmp = -x[0] * x[1] * (gbg1b1_to[0] * sin + gbg1b1_to[1] * cos);
                    jac.setQuick(index, 2, -tmp);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    jac.setQuick(index, 0, -2 * x[0] * (gbg1b1_from[1] + gbg1b1_from[3]) - x[1] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    jac.setQuick(index, 1, -x[0] * (gbg1b1_from[0] * sin - gbg1b1_from[1] * cos));
                    tmp = -x[0] * x[1] * (gbg1b1_from[0] * cos + gbg1b1_from[1] * sin);
                    jac.setQuick(index, 2, tmp);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    jac.setQuick(index, 0, x[1] * (gbg1b1_to[0] * sin + gbg1b1_to[1] * cos));
                    jac.setQuick(index, 1, -2 * x[1] * (gbg1b1_to[1] + gbg1b1_to[3]) + x[0] * (gbg1b1_to[0] * sin + gbg1b1_to[1] * cos));
                    tmp = -x[0] * x[1] * (gbg1b1_to[0] * cos - gbg1b1_to[1] * sin);
                    jac.setQuick(index, 2, -tmp);
                    break;
                default:
                    break;
            }
            index++;
        }
    }

    public void getEstValue(double[] x, double[] g) {
        int i = 0;
        for (int index : selectedMeasures) {
            MeasureInfo info = meases[index];
            int type = info.getMeasureType();
            switch (type) {
                case TYPE_BUS_VOLOTAGE:
                    if (info.getPositionId().equals(String.valueOf(branch.getTapBusNumber()))) {
                        g[i++] = x[0];
                    } else {
                        g[i++] = x[1];
                    }
                    break;
                case TYPE_LINE_FROM_ACTIVE:
                    g[i++] = StateCalByPolar.calLinePFrom(branch, x);
                    break;
                case TYPE_LINE_TO_ACTIVE:
                    g[i++] = StateCalByPolar.calLinePTo(branch, x);
                    break;
                case TYPE_LINE_FROM_REACTIVE:
                    g[i++] = StateCalByPolar.calLineQFrom(branch, x);
                    break;
                case TYPE_LINE_TO_REACTIVE:
                    g[i++] = StateCalByPolar.calLineQTo(branch, x);
                    break;
                default:
                    break;
            }
        }
    }

    public boolean isHasBadQ() {
        return hasBadQ;
    }

    public boolean isViBad() {
        return isViBad;
    }

    public boolean isVjBad() {
        return isVjBad;
    }

    public boolean isHasBadP() {
        return hasBadP;
    }

    public void setBranch(BranchData branch) {
        this.branch = branch;
    }

    public void setMeases(MeasureInfo[] meases) {
        this.meases = meases;
    }
}
