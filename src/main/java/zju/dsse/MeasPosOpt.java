package zju.dsse;

import jpscpu.LinearSolver;
import org.apache.log4j.Logger;
import zju.dsmodel.DistriSys;
import zju.measure.MeasureInfo;

/**
 * 量测位置优化
 * @author Dong Shufeng
 * Date: 2016/7/12
 */
public class MeasPosOpt {

    private static Logger log = Logger.getLogger(MeasPosOpt.class);

    private DistriSys distriSys;

    private MeasureInfo[] existMeas;

    //可以布置的位置
    private int[] cand_pos;
    //每个位置可以布置的量测集合
    private int[][] meas_types;

    public MeasPosOpt(DistriSys distriSys) {
        this.distriSys = distriSys;
    }

    public void doOpt() {
        int row_count = 0;
        int element_count = 0;
        //开辟内存
        double objValue[] = new double[0];
        //objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数
        //变量下限//
        //column里元素的个数等于矩阵C里系数的个数,详见starts解释
        double columnLower[] = new double[objValue.length];
        //变量上限
        double columnUpper[] = new double[objValue.length];
        //指明那些是整数
        int whichInt[] = new int[objValue.length];

        //约束下限
        double rowLower[] = new double[row_count];
        //约束上限
        double rowUpper[] = new double[row_count];
        //约束中非零元系数
        double element[] = new double[element_count];
        //上面系数对应的列
        int column[] = new int[element_count];
        //每一行起始位置
        int starts[] = new int[rowLower.length + 1];
        starts[0] = 0;

        int obj_count = 0, whichint_count = 0;

        int numberRows = rowLower.length;
        int numberColumns = columnLower.length;
        double result[] = new double[numberColumns];
        //进行求解
        LinearSolver solver = new LinearSolver();
        solver.setDrive(LinearSolver.MLP_DRIVE_CBC);
        int status = solver.solveMlp(numberColumns, numberRows, objValue,
                columnLower, columnUpper, rowLower, rowUpper, element, column, starts, whichInt, result);
        if (status < 0) {
            log.warn("计算不收敛.");
        } else { //状态位显示计算收敛
            log.info("计算结束.");
        }
    }
}
