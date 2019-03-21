package zju.hems;


import jpscpu.LinearSolver;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Created by cz274 on 16/08/2016.
 *
 */

public class P2pMlpModel {
    private static Logger log = LogManager.getLogger(P2pMlpModel.class);

    //储能能量初值,储能最大变化量
    private double iniEnergy, PG, R, maxBatteryChange, minBatteryChange;
    private double[]  DCOut;
    //放电效率和充电效率，如果这两者不一样则目前程序还没有处理
    private double dischargeEff = 0.7, chargeEff = 1.3;
    //储能能量上限,储能能量下限,负荷需求,电价,优化结果
    private double[] energy_L, energy_U, pNeed, pricePerKwh;
    //优化结果
    private double result[];
    /**
     * 求解优化模型
     *
     * @return 是否收敛
     */


    public boolean doEsOpt_p2p() {
        //约束的个数
        //element_count 约束参数的个数

        int
                z7_count=0, z9_count=0, z11_count=0,
                p_count=0,x_count = 0, row_count = 0, element_count = 0;


        for (int k = 0; k < pNeed.length; k++) {

            p_count+=6;
            z7_count+=2;
            z9_count+=2;
            z11_count+=2;

            row_count += 11;
            element_count += 26;

        }

        //x_L =< iniEnerge + x_1 <= x_U, x_L =< iniEnerge + x_1 + x_2 < x_L,...,
        //x_1 + x_2 + ... + x_n = 0
        //累计这两个条件非零元的个数. 假设第一个时刻三段线,last_count=3（四个元素去掉0的情况）,element_count=c+3,
        // 第二时刻放电两段线,last_count=3+2=5,element_count=c+3+5,
        // 如此累计,直到最后一个条件，累积到第48个, last_count=4+3+....+3=Y, 将这些非零元个数加在element_count里
        //因为最后一项加了x1+x2+....+xn(x48),长度跟最后一个约束一样，所以没有再计算个数
        //element_count第二部分列数是xL<=inienergy+x1<=xU这几个条件非零元素

        row_count += pNeed.length;
        int lastCount = 0;

        //count the element in battery constraints
        //每一行多两个参数，battery 这一时刻的charge/discharge--lastCount

        for (int k = 0; k < pNeed.length; k++){

            lastCount += 2;
            element_count += lastCount;
        }


        //开辟内存
        double objValue[] = new double[p_count +z7_count + z9_count + z11_count];
        //objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数， y value of breakpoints
        //变量下限//
        //column里元素的个数等于矩阵C里系数的个数,详见starts解释
        double columnLower[] = new double[objValue.length];
        //变量上限
        double columnUpper[] = new double[objValue.length];
        //指明那些是整数
        int whichInt[] = new int[z7_count + z9_count + z11_count];

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
        //starts 解释见笔记本...
        int obj_count = 0, whichint_count = 0;
        element_count = 0;
        row_count = 0;

        for (int k = 0; k < pNeed.length; k++) {

            objValue[obj_count]= pricePerKwh[k];
            columnLower[obj_count] = 0.0;
            columnUpper[obj_count++] = PG;

            objValue[obj_count]= 0.0;
            columnLower[obj_count] = 0.0;
            columnUpper[obj_count++] = PG;

            objValue[obj_count]= 0.0;
            columnLower[obj_count] = 0.0;
            columnUpper[obj_count++] = R;

            objValue[obj_count]= 0.0;
            columnLower[obj_count] = 0.0;
            columnUpper[obj_count++] = R;

            objValue[obj_count]= 0.0;
            columnLower[obj_count] = 0.0;
            columnUpper[obj_count++] = maxBatteryChange;

            objValue[obj_count]= 0.0;
            columnLower[obj_count] = 0.0;
            columnUpper[obj_count++] = maxBatteryChange;


            for (int i = 0; i < 6; i++) {
                objValue[obj_count] = 0.0;
                columnLower[obj_count] = 0.0;
                columnUpper[obj_count] = 1.0;
                whichInt[whichint_count++] = obj_count;
                obj_count++;
            }
            formElements(rowLower, rowUpper, element, column,
                    starts, row_count, element_count, obj_count, k);

            //后面一个method，确定系数C的矩阵，用starts方法
            //objValue 是c(x)=[w1*c(b1),w2*c(b2),w3*c(b3),w4*c(b4),0,0,0,w1'*c(b1'),w2'*c(b2')....]其中的w*c(b)是y值， 0 是赋予整数的位置为0；
            //obj_count是[w1,w2,w3,w4,z1,z2,z3,w1',w2',w3',z1',z2'.......]个数. element_count =0; row_count = 0; rowLower[0]=?, element[0]=?, column[0]=? 后面会自动加吗?rowLower[1]?
            //columnLower and columnUpper是 w 和 z 的范围, 所以是变量下限
            row_count += 11;
            element_count += 26;

        }


        // object_count 加7 if 三段，加5 if 两段，加1 if 不分段; y值的个数+整数的个数
        // row_count, element_count:约束的个数以及约束中变量的个数
        // object_count max=7*48?=336, row_count max=288, element_count max=17*48=816 如果不考虑上下限
        //columnLower[objValue.length]=object_count, rowLower[row_count<=288], element[element_count<=816], column[element_count<=816]

        //输入battery constraints的参数
        x_count = 0;
        lastCount = 0;

        double[] bat= new double [2];

        for (int k = 0; k < pNeed.length; k++) {
            bat[0]= 1.0;
            bat[1]= 1.0;
            lastCount = formElements2(rowLower, rowUpper, element, column, starts,
                    element_count, x_count, bat, k, lastCount);
            x_count += 12;


            element_count += lastCount;
            //element_count 在加last_count之前是只考虑分段函数变量约束时，约束中变量的个数.
            // lastCount 怎么算的？
            // x_count：x值（分段函数中转变成w,z值）的个数
            // starts 用来做什么？
        }

        int numberRows = rowLower.length;
        int numberColumns = columnLower.length;
        result = new double[numberColumns];

        //进行求解
        LinearSolver solver = new LinearSolver();

        solver.setDrive(LinearSolver.MLP_DRIVE_CBC);
        int status = solver.solveMlp(numberColumns, numberRows, objValue,
                columnLower, columnUpper, rowLower, rowUpper, element, column, starts, whichInt, result);

        System.out.println("KB4: " + (double) (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024);
        if (status < 0) {
            log.warn("计算不收敛.");
        } else { //状态位显示计算收敛
            log.info("计算结束.");

            return true;
            //x_count统计什么？
            //return true 是否收敛，怎样就收敛？
            //AVector.printOnScreen(optCharge);
        }
        return false;
    }

    private void formElements(double[] rowLower, double[] rowUpper,
                              double[] element, int[] column, int[] starts,
                              int row_count, int element_count, int obj_count, int n){

        //AC power balance
        rowLower[row_count] = pNeed[n];
        rowUpper[row_count++] = pNeed[n];
        //此等式约束有4个变量
        starts[row_count] = starts[row_count - 1] + 4;

        //element 是每个w,z变量前的系数，w1-z1<=0,第一二个变量系数为1，-1，w2-z1-z2<=0,系数为1，-1，-1
        //column 是每个变量所在的列数

        //obj_count 变量个数
        element[element_count] = 1.0;
        column[element_count++] = obj_count - 12;
        element[element_count] = -1.0;
        column[element_count++] = obj_count - 11;
        element[element_count] = dischargeEff;
        column[element_count++] = obj_count - 10;
        element[element_count] = -1.0;
        column[element_count++] = obj_count - 9;


        //DC power balance
        rowLower[row_count] = DCOut[n];
        rowUpper[row_count++] = DCOut[n];
        //此等式约束有4个变量
        starts[row_count] = starts[row_count - 1] + 4;

        element[element_count] = -1.0;
        column[element_count++] = obj_count - 10;
        element[element_count] = chargeEff;
        column[element_count++] = obj_count - 9;
        element[element_count] = 1.0;
        column[element_count++] = obj_count - 8;
        element[element_count] = -1.0;
        column[element_count++] = obj_count - 7;


        //home-grid constraints
        for (int i = 0; i < 2; i++) {
            rowLower[row_count] = -PG;
            rowUpper[row_count++] = 0.0;
            //-PG，0由来：x1-PG*x7<=0, x1最小值0，x7最大值1，结果最小-PG，最大是0
            starts[row_count] = starts[row_count - 1] + 2;

            element[element_count] = 1.0;
            column[element_count++] = obj_count - 12+ i;
            element[element_count] = -PG;
            column[element_count++] = obj_count - 6 + i;
        }


        //Home-grid in&out only one
        rowLower[row_count] = 1;
        rowUpper[row_count++] = 1;
        starts[row_count] = starts[row_count - 1] + 2;

        //x7+x8=1
        element[element_count] = 1.0;
        column[element_count++] = obj_count - 6;
        element[element_count] = 1.0;
        column[element_count++] = obj_count - 5;

        //AC/DC constraints
        for (int i = 0; i < 2; i++) {
            rowLower[row_count] = -R;
            rowUpper[row_count++] = 0.0;
            starts[row_count] = starts[row_count - 1] + 2;

            element[element_count] = 1.0;
            column[element_count++] = obj_count - 10+ i;
            element[element_count] = -R;
            column[element_count++] = obj_count - 4 + i;
        }


        //AC/DC in & out only one
        rowLower[row_count] = 1;
        rowUpper[row_count++] = 1;
        starts[row_count] = starts[row_count - 1] + 2;

        //x9+x10=1
        element[element_count] = 1.0;
        column[element_count++] = obj_count - 4;
        element[element_count] = 1.0;
        column[element_count++] = obj_count - 3;


        //battery constraints
        for (int i = 0; i < 2; i++) {
            rowLower[row_count] = -maxBatteryChange;
            rowUpper[row_count++] = 0.0;
            starts[row_count] = starts[row_count - 1] + 2;

            element[element_count] = 1.0;
            column[element_count++] = obj_count - 8 + i;
            element[element_count] = -maxBatteryChange;
            column[element_count++] = obj_count - 2 + i;
        }

        //battery in & out only one
        rowLower[row_count] = 1;
        rowUpper[row_count++] = 1;
        starts[row_count] = starts[row_count - 1] + 2;

        //x11+x12=1
        element[element_count] = 1.0;
        column[element_count++] = obj_count - 2;
        element[element_count] = 1.0;
        column[element_count++] = obj_count - 1;
    }

    private int formElements2(double[] rowLower, double[] rowUpper,
                              double[] element, int[] column, int[] starts,
                              int element_count, int obj_count,
                              double[] b, int n, int lastCount) {
        //x_L =< iniEnerge + x_1 <= x_U, x_L =< iniEnerge + x_1 + x_2 < x_L,...,
        //x_1 + x_2 + ... + x_n = finalEnergyChanged
        //n=0...47一共48个数。
        int row = rowLower.length - pNeed.length + n;

        if (n == pNeed.length - 1) {
            //todo: MAY NOT RIGHT
            rowLower[row] = 0.0;
            rowUpper[row] = 0.0;
            //最后一个约束本该对应 x_1+....x_48 = 0
        } else {
            rowLower[row] = energy_L[n] - iniEnergy;
            rowUpper[row] = energy_U[n] - iniEnergy;
        }
        if (n > 0) {
            System.arraycopy(element, element_count - lastCount, element, element_count, lastCount);
            System.arraycopy(column, element_count - lastCount, column, element_count, lastCount);
            element_count += lastCount;
        }
        //System提供了一个静态方法arraycopy(),我们可以使用它来实现数组之间的复制.
        //其函数原型是：
        //public static void arraycopy(Object src, int srcPos, Object dest, int destPos, int length)
        //src:源数组; srcPos:源数组要复制的起始位置; dest:目的数组; destPos:目的数组放置的起始位置; length:复制的长度.

        starts[row + 1] = starts[row] + lastCount;

        //复制 battery 约束的参数给element 和列数给column
        for (int i = 0; i < b.length; i++) {
            if (Math.abs(b[i]) < 1e-17)
                continue;
            element[element_count] = b[i];
            //这里+4因为battery 变量是x5 and x6
            column[element_count++] = obj_count + 4 + i;
            //column[element_count++] = obj_count + i;
            starts[row + 1]++;
            lastCount++;
            //lastCount的个数等于b中非零的个数，（b的个数等于2）
        }
        return lastCount;
    }
    public double getIniEnergy() {
        return iniEnergy;
    }

    public void setIniEnergy(double iniEnergy) {
        this.iniEnergy = iniEnergy;
    }

    public double getMaxBatteryChange() {
        return maxBatteryChange;
    }

    public double getMinBatteryChange() {
        return minBatteryChange;
    }

    public void setMinBatteryChange(double minBatteryChange) {
        this.minBatteryChange = minBatteryChange;
    }

    public void setMaxBatteryChange(double maxBatteryChange) {
        this.maxBatteryChange = maxBatteryChange;
    }

    public double getDischargeEff() {
        return dischargeEff;
    }


    public void setDischargeEff(double dischargeEff) {
        this.dischargeEff = dischargeEff;
    }


    public double getChargeEff() {
        return chargeEff;
    }


    public void setChargeEff(double chargeEff) {
        this.chargeEff = chargeEff;
    }


    public double[] getEnergy_L() {
        return energy_L;
    }

    public void setEnergy_L(double[] energy_L) {
        this.energy_L = energy_L;
    }

    public double[] getEnergy_U() {
        return energy_U;
    }

    public void setEnergy_U(double[] energy_U) {
        this.energy_U = energy_U;
    }

    public double[] getpNeed() {
        return pNeed;
    }

    public void setpNeed(double[] pNeed) {
        this.pNeed = pNeed;
    }

    public double[] getPricePerKwh() {
        return pricePerKwh;
    }

    public void setPricePerKwh(double[] pricePerKwh) {
        this.pricePerKwh = pricePerKwh;
    }

    public double[] getDCOut() {
        return DCOut;
    }

    public void setDCOut(double[] DCOut) {
        this.DCOut = DCOut;
    }

    public double[] getResult() {
        return result;
    }

    public void setResult(double[] result) {
        this.result = result;
    }



    public double getPG() {
        return PG;
    }

    public void setPG(double PG) {
        this.PG = PG;
    }

    public double getRating() {
        return R;
    }

    public void setRating(double R) {
        this.R = R;
    }
}
