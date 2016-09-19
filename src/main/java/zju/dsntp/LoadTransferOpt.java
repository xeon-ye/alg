package zju.dsntp;

import jpscpu.LinearSolver;
import org.apache.log4j.Logger;
import zju.devmodel.MapObject;
import zju.dsmodel.DistriSys;
import zju.dsmodel.DsDevices;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 转供路径搜索
 * @author Dong Shufeng
 * @date 2016/9/19
 */
public class LoadTransferOpt extends PathBasedModel {

    private static Logger log = Logger.getLogger(LoadTransferOpt.class);

    int[] errorFeeder;

    String[] errorSupply;
    //电源容量
    Map<String, Double> supplyCap;
    //馈线容量
    Map<String, Double> feederCap;

    public LoadTransferOpt(DistriSys sys) {
        super(sys);
    }

    public void doOpt() {
        DsDevices devices = sys.getDevices();
        //从系统中删除故障的馈线
        List<MapObject> toRemove = new ArrayList<>(errorFeeder.length);
        for(int i : errorFeeder)
            toRemove.add(devices.getFeeders().get(i));
        devices.getFeeders().removeAll(toRemove);
        //将故障的电源从supplyCn中删除
        String[] normalSupply = new String[sys.getSupplyCns().length - errorSupply.length];
        Double[] supplyBaseKv = new Double[normalSupply.length];
        int index = 0;
        boolean isNoraml;
        for(int i = 0; i < sys.getSupplyCns().length; i++) {
            String cnId = sys.getSupplyCns()[i];
            isNoraml = true;
            for (String errorS : errorSupply) {
                if (cnId.equals(errorS)) {
                    isNoraml = false;
                    break;
                }
            }
            if(isNoraml) {
                supplyBaseKv[index] = sys.getSupplyCnBaseKv()[i];
                normalSupply[index++] = cnId;
            }
        }
        sys.setSupplyCns(normalSupply);
        sys.setSupplyCnBaseKv(supplyBaseKv);
        //重新形成拓扑图
        sys.buildOrigTopo(devices);

        //开始构造线性规划模型

        //objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数
        double objValue[] = new double[0];
        //状态变量下限, column里元素的个数等于矩阵C里系数的个数
        double columnLower[] = new double[objValue.length];
        //状态变量上限
        double columnUpper[] = new double[objValue.length];
        //指明那些是整数
        int whichInt[] = new int[0];

        //约束下限
        double rowLower[] = new double[0];
        //约束上限
        double rowUpper[] = new double[rowLower.length];
        //约束中非零元系数
        double element[] = new double[0];
        //上面系数对应的列
        int column[] = new int[element.length];
        //每一行起始位置
        int starts[] = new int[rowLower.length + 1];
        starts[0] = 0;

        //todo:对上面这些参数赋值

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
            log.info("计算结果.");
        }
    }

    public void setErrorFeeder(int[] errorFeeder) {
        this.errorFeeder = errorFeeder;
    }

    public void setErrorSupply(String[] errorSupply) {
        this.errorSupply = errorSupply;
    }

    public void setSupplyCap(Map<String, Double> supplyCap) {
        this.supplyCap = supplyCap;
    }

    public void setFeederCap(Map<String, Double> feederCap) {
        this.feederCap = feederCap;
    }
}
