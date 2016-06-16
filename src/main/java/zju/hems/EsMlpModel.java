package zju.hems;

import jpscpu.LinearSolver;
import org.apache.log4j.Logger;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/6/11
 */
public class EsMlpModel {
    private static Logger log = Logger.getLogger(EsMlpModel.class);

    //储能能量初值,储能最大变化量
    private double iniEnergy, finalEnergyChanged = 0;
    private double[] maxEnergyChange, minEnergeChage;
    //放电效率和充电效率，如果这两者不一样则目前程序还没有处理
    private double dischargeEff = 0.7, chargeEff = 1.3;
    //储能能量上限,储能能量下限,负荷需求,电价,优化结果
    double[] energy_L, energy_U, pNeed, pricePerKwh;
    //优化结果
    double optCharge[];

    /**
     * 求解优化模型
     *
     * @return 是否收敛
     */

    public boolean doEsOpt() {
        //分别是w,z,x的数组长度,约束的个数以及约束中变量的个数, 分段函数中用w,z来表示x
        //element_count 最前面的元素是w1<=z1,w2<=z1+z2……z1+z2=1,w1+w2+w3=1这几个条件中的非零元素
        int w_count = 0, z_count = 0, x_count = 0, row_count = 0, element_count = 0;
        double b2; //放电量
        for (int k = 0; k < pNeed.length; k++) {
            double p = pNeed[k];
            if (p > 0) {
                b2 = -p / dischargeEff;
                if (b2 > minEnergeChage[k]) {//第一种情况，三段线
                    if (maxEnergyChange[k] > 0) {
                        w_count += 4;
                        z_count += 3;
                        row_count += 6;
                        element_count += 17;
                    } else if (maxEnergyChange[k] > b2) { //第二种情况，两段线
                        w_count += 3;
                        z_count += 2;
                        row_count += 5;
                        element_count += 12;
                    } else {//此时已经不是分段函数
                        x_count += 1;
                    }
                } else if (minEnergeChage[k] < 0 && maxEnergyChange[k] > 0) { //第二种情况，两段线
                    w_count += 3;
                    z_count += 2;
                    row_count += 5;
                    element_count += 12;
                } else { //此时已经不是分段函数
                    x_count += 1;
                }
                //w_1 <= z_1, w_2 <= z_1 + z_2, ..., w_n <= z_n-1 + z_n, w_n+1 <= zn
                //z_1 + z_2 + ... + z_n = 1;
                //w_1 + w_2 + ... + w_n+1 = 1;
            } else {
                b2 = -p / chargeEff;
                if (b2 < maxEnergyChange[k] && minEnergeChage[k] < b2) {//两段线
                    w_count += 3;
                    z_count += 2;
                    row_count += 5;
                    element_count += 12;
                } else //此时已经不是分段函数
                    x_count += 1;
            }
        }
        //x_L =< iniEnerge + x_1 <= x_U, x_L =< iniEnerge + x_1 + x_2 < x_L,...,
        //x_1 + x_2 + ... + x_n = 0
        //累计这两个条件非零元的个数. 假设第一个时刻三段线,last_count=3（四个元素去掉0的情况）,element_count=c+3,
        // 第二时刻放电两段线,last_count=3+2=5,element_count=c+3+5,
        // 如此累计,直到最后一个条件，累积到第48个, last_count=4+3+....+3=Y, 将这些非零元个数加在element_count里
        //因为最后一项加了x1+x2+....+xn(x48),长度跟最后一个约束一样，所以没有再计算个数
        //element_count第二部分列数是xL<=inienergy+x1<=xU这几个条件非零元素
        row_count += pNeed.length;
        double[] forThreeLines = new double[4];
        double[] forTwoLines = new double[3];
        double[] forOneLine = {1.0};
        int lastCount = 0;
        double[] b;
        for (int k = 0; k < pNeed.length; k++) {
            double p = pNeed[k];
            if (p > 0) {
                b2 = -p / dischargeEff;
                if (b2 > minEnergeChage[k]) {//第一种情况，三段线
                    if (maxEnergyChange[k] > 0) {
                        forThreeLines[0] = minEnergeChage[k];
                        forThreeLines[1] = b2;
                        forThreeLines[2] = 0;
                        forThreeLines[3] = maxEnergyChange[k];
                        b = forThreeLines;
                    } else if (maxEnergyChange[k] > b2) { //第二种情况，两段线
                        forTwoLines[0] = minEnergeChage[k];
                        forTwoLines[1] = b2;
                        forTwoLines[2] = maxEnergyChange[k];
                        b = forTwoLines;
                    } else {//此时已经不是分段函数
                        b = forOneLine;
                    }
                } else if (minEnergeChage[k] < 0 && maxEnergyChange[k] > 0) { //第二种情况，两段线
                    forTwoLines[0] = minEnergeChage[k];
                    forTwoLines[1] = 0;
                    forTwoLines[2] = maxEnergyChange[k];
                    b = forTwoLines;
                } else  //此时已经不是分段函数
                    b = forOneLine;
            } else {
                b2 = -p / chargeEff;
                if (b2 < maxEnergyChange[k] && minEnergeChage[k] < b2) {//两段线
                    forTwoLines[0] = minEnergeChage[k];
                    forTwoLines[1] = b2;
                    forTwoLines[2] = maxEnergyChange[k];
                    b = forTwoLines;
                } else //此时已经不是分段函数
                    b = forOneLine;
            }
            for (double aB : b) {
                if (Math.abs(aB) < 1e-17)
                    continue;
                lastCount++;
            }
            element_count += lastCount;
            //b=bs1={b1,b2,0,maxEnergyChange}循环检测b中的每个元素
        }

        //开辟内存
        double objValue[] = new double[w_count + z_count + x_count];
        //objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数
        //变量下限//
        //column里元素的个数等于矩阵C里系数的个数,详见starts解释
        double columnLower[] = new double[objValue.length];
        //变量上限
        double columnUpper[] = new double[objValue.length];
        //指明那些是整数
        int whichInt[] = new int[z_count];

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
            double p = pNeed[k];
            if (p > 0) {
                b2 = -p / dischargeEff;
                if (b2 > minEnergeChage[k]) {//第一种情况，三段线
                    if (maxEnergyChange[k] > 0) {
                        objValue[obj_count] = 0.0;
                        columnLower[obj_count] = 0.0;
                        columnUpper[obj_count++] = 1.0;

                        objValue[obj_count] = 0.0;
                        columnLower[obj_count] = 0.0;
                        columnUpper[obj_count++] = 1.0;

                        objValue[obj_count] = p * pricePerKwh[k];
                        columnLower[obj_count] = 0.0;
                        columnUpper[obj_count++] = 1.0;

                        objValue[obj_count] = (p + chargeEff * maxEnergyChange[k]) * pricePerKwh[k];
                        columnLower[obj_count] = 0.0;
                        columnUpper[obj_count++] = 1.0;
                        for (int i = 0; i < 3; i++) {
                            objValue[obj_count] = 0.0;
                            columnLower[obj_count] = 0.0;
                            columnUpper[obj_count] = 1.0;
                            whichInt[whichint_count++] = obj_count;
                            obj_count++;
                        }
                        formElements(rowLower, rowUpper, element, column,
                                starts, row_count, element_count, obj_count, 3);
                        //后面一个method
                        //objValue 是c(x)=[w1*c(b1),w2*c(b2),w3*c(b3),w4*c(b4),0,0,0,w1'*c(b1'),w2'*c(b2')....]其中的w*c(b)是y值， 0 是赋予整数的位置为0；
                        //obj_count是[w1,w2,w3,w4,z1,z2,z3,w1',w2',w3',z1',z2'.......]个数. element_count =0; row_count = 0; rowLower[0]=?, element[0]=?, column[0]=? 后面会自动加吗?rowLower[1]?
                        //columnLower and columnUpper是 w 和 z 的范围, 所以是变量下限
                        row_count += 6;
                        element_count += 17;
                        //每有一个放电三段线情况，element_count加17 ，row_count加6， element 跟约束的参数有关， row 跟有几个整数优化的约束条件有关
                    } else if (maxEnergyChange[k] > b2) { //第二种情况，两段线
                        objValue[obj_count] = 0.0;
                        columnLower[obj_count] = 0.0;
                        columnUpper[obj_count++] = 1.0;

                        objValue[obj_count] = 0.0;
                        columnLower[obj_count] = 0.0;
                        columnUpper[obj_count++] = 1.0;

                        objValue[obj_count] = (maxEnergyChange[k] - b2) * dischargeEff * pricePerKwh[k];
                        columnLower[obj_count] = 0.0;
                        columnUpper[obj_count++] = 1.0;

                        for (int i = 0; i < 2; i++) {
                            objValue[obj_count] = 0.0;
                            columnLower[obj_count] = 0.0;
                            columnUpper[obj_count] = 1.0;
                            whichInt[whichint_count++] = obj_count;
                            obj_count++;
                        }
                        formElements(rowLower, rowUpper, element, column,
                                starts, row_count, element_count, obj_count, 2);
                        row_count += 5;
                        element_count += 12;
                    } else {//此时已经不是分段函数
                        objValue[obj_count] = 0.0;
                        columnLower[obj_count] = minEnergeChage[k];
                        columnUpper[obj_count++] = maxEnergyChange[k];
                    }
                } else if (minEnergeChage[k] < 0 && maxEnergyChange[k] > 0) { //第二种情况，两段线
                    objValue[obj_count] = (p + dischargeEff * minEnergeChage[k]) * pricePerKwh[k];
                    columnLower[obj_count] = 0.0;
                    columnUpper[obj_count++] = 1.0;

                    objValue[obj_count] = p * pricePerKwh[k];
                    columnLower[obj_count] = 0.0;
                    columnUpper[obj_count++] = 1.0;

                    objValue[obj_count] = (p + chargeEff * maxEnergyChange[k]) * pricePerKwh[k];
                    columnLower[obj_count] = 0.0;
                    columnUpper[obj_count++] = 1.0;

                    for (int i = 0; i < 2; i++) {
                        objValue[obj_count] = 0.0;
                        columnLower[obj_count] = 0.0;
                        columnUpper[obj_count] = 1.0;
                        whichInt[whichint_count++] = obj_count;
                        obj_count++;
                    }
                    formElements(rowLower, rowUpper, element, column,
                            starts, row_count, element_count, obj_count, 2);
                    row_count += 5;
                    element_count += 12;
                } else { //此时已经不是分段函数
                    if (maxEnergyChange[k] < 0) {
                        objValue[obj_count] = dischargeEff * pricePerKwh[k];
                    } else
                        objValue[obj_count] = chargeEff * pricePerKwh[k];
                    columnLower[obj_count] = minEnergeChage[k];
                    columnUpper[obj_count++] = maxEnergyChange[k];
                }
                //w_1 <= z_1, w_2 <= z_1 + z_2, ..., w_n <= z_n-1 + z_n, w_n+1 <= zn
                //z_1 + z_2 + ... + z_n = 1;
                //w_1 + w_2 + ... + w_n+1 = 1;
            } else {
                b2 = -p / chargeEff;
                if (b2 < maxEnergyChange[k] && minEnergeChage[k] < b2) {//两段线
                    objValue[obj_count] = 0.0;
                    columnLower[obj_count] = 0.0;
                    columnUpper[obj_count++] = 1.0;

                    objValue[obj_count] = 0.0;
                    columnLower[obj_count] = 0.0;
                    columnUpper[obj_count++] = 1.0;

                    objValue[obj_count] = (maxEnergyChange[k] - b2) * chargeEff * pricePerKwh[k];
                    columnLower[obj_count] = 0.0;
                    columnUpper[obj_count++] = 1.0;

                    for (int i = 0; i < 2; i++) {
                        objValue[obj_count] = 0.0;
                        columnLower[obj_count] = 0.0;
                        columnUpper[obj_count] = 1.0;
                        whichInt[whichint_count++] = obj_count;
                        obj_count++;
                    }
                    formElements(rowLower, rowUpper, element, column,
                            starts, row_count, element_count, obj_count, 2);
                    row_count += 5;
                    element_count += 12;
                } else {//此时已经不是分段函数
                    if (maxEnergyChange[k] < b2)
                        objValue[obj_count] = 0.0;
                    else
                        objValue[obj_count] = chargeEff * pricePerKwh[k];
                    columnLower[obj_count] = minEnergeChage[k];
                    columnUpper[obj_count++] = maxEnergyChange[k];
                }
            }
        }
        // object_count 加7 if 三段，加5 if 两段，加1 if 不分段; y值的个数+整数的个数
        // row_count, element_count:约束的个数以及约束中变量的个数
        // object_count max=7*48?=336, row_count max=288, element_count max=17*48=816 如果不考虑上下限
        //columnLower[objValue.length]=object_count, rowLower[row_count<=288], element[element_count<=816], column[element_count<=816]

        x_count = 0;
        lastCount = 0;
        for (int k = 0; k < pNeed.length; k++) {
            double p = pNeed[k];
            if (p > 0) {
                b2 = -p / dischargeEff;
                if (b2 > minEnergeChage[k]) {//第一种情况，三段线
                    if (maxEnergyChange[k] > 0) {
                        forThreeLines[0] = minEnergeChage[k];
                        forThreeLines[1] = b2;
                        forThreeLines[2] = 0;
                        forThreeLines[3] = maxEnergyChange[k];
                        lastCount = formElements2(rowLower, rowUpper, element, column, starts,
                                element_count, x_count, forThreeLines, k, lastCount);
                        x_count += 7;
                    } else if (maxEnergyChange[k] > b2) { //第二种情况，两段线
                        forTwoLines[0] = minEnergeChage[k];
                        forTwoLines[1] = b2;
                        forTwoLines[2] = maxEnergyChange[k];
                        lastCount = formElements2(rowLower, rowUpper, element, column, starts,
                                element_count, x_count, forTwoLines, k, lastCount);
                        x_count += 5;
                    } else {//此时已经不是分段函数
                        lastCount = formElements2(rowLower, rowUpper, element, column, starts,
                                element_count, x_count, forOneLine, k, lastCount);
                        x_count += 1;
                    }
                } else if (minEnergeChage[k] < 0 && maxEnergyChange[k] > 0) { //第二种情况，两段线
                    forTwoLines[0] = minEnergeChage[k];
                    forTwoLines[1] = 0;
                    forTwoLines[2] = maxEnergyChange[k];
                    lastCount = formElements2(rowLower, rowUpper, element, column, starts,
                            element_count, x_count, forTwoLines, k, lastCount);
                    x_count += 5;
                } else {//此时已经不是分段函数
                    lastCount = formElements2(rowLower, rowUpper, element, column, starts,
                            element_count, x_count, forOneLine, k, lastCount);
                    x_count += 1;
                }
            } else {
                b2 = -p / chargeEff;
                if (b2 < maxEnergyChange[k] && minEnergeChage[k] < b2) {//两段线
                    forTwoLines[0] = minEnergeChage[k];
                    forTwoLines[1] = b2;
                    forTwoLines[2] = maxEnergyChange[k];
                    lastCount = formElements2(rowLower, rowUpper, element, column, starts,
                            element_count, x_count, forTwoLines, k, lastCount);
                    x_count += 5;
                } else {//此时已经不是分段函数
                    lastCount = formElements2(rowLower, rowUpper, element, column, starts,
                            element_count, x_count, forOneLine, k, lastCount);
                    x_count += 1;
                }
            }
            element_count += lastCount;
            //element_count 在加last_count之前是只考虑分段函数变量约束时，约束中变量的个数.
            // lastCount 怎么算的？
            // x_count：x值（分段函数中转变成w,z值）的个数
            // starts 用来做什么？
        }

        int numberRows = rowLower.length;
        int numberColumns = columnLower.length;
        double result[] = new double[numberColumns];
        //进行求解
        LinearSolver solver = new LinearSolver();
        //ASparseMatrixLink2D mat = new ASparseMatrixLink2D(numberRows, numberColumns);
        //for(int i = 1; i < starts.length; i++) {
        //    for(int j = starts[i - 1]; j < starts[i]; j++)
        //        mat.setValue(i - 1, column[j], element[j]);
        //}
        //double[] a = new double[mat.getVA().size()];
        //int[] asub = new int[mat.getVA().size()];
        //int[] xa = new int[mat.getN() + 1];
        //mat.getSluStrucNC(a, asub, xa);
        //solver.setDrive(LinearSolver.MLP_DRIVE_SYM);
        //int status = solver.solveMlp(numberColumns, numberRows, objValue,
        //        columnLower, columnUpper, rowLower, rowUpper, a, asub, xa, whichInt, result);

        solver.setDrive(LinearSolver.MLP_DRIVE_CBC);
        int status = solver.solveMlp(numberColumns, numberRows, objValue,
                columnLower, columnUpper, rowLower, rowUpper, element, column, starts, whichInt, result);
        if (status < 0) {
            log.warn("计算不收敛.");
        } else { //状态位显示计算收敛
            log.info("计算结束.");
            optCharge = new double[pNeed.length];

            //恢复x值，从w,z恢复成x
            x_count = 0;
            for (int k = 0; k < pNeed.length; k++) {

                double p = pNeed[k];
                if (p > 0) {
                    b2 = -p / dischargeEff;
                    if (b2 > minEnergeChage[k]) {//第一种情况，三段线
                        if (maxEnergyChange[k] > 0) {
                            forThreeLines[0] = minEnergeChage[k];
                            forThreeLines[1] = b2;
                            forThreeLines[2] = 0;
                            forThreeLines[3] = maxEnergyChange[k];
                            for (int i = 0; i < 4; i++)
                                optCharge[k] += forThreeLines[i] * result[x_count + i];
                            x_count += 7;
                        } else if (maxEnergyChange[k] > b2) { //第二种情况，两段线
                            forTwoLines[0] = minEnergeChage[k];
                            forTwoLines[1] = b2;
                            forTwoLines[2] = maxEnergyChange[k];
                            for (int i = 0; i < 3; i++)
                                optCharge[k] += forTwoLines[i] * result[x_count + i];
                            x_count += 5;
                        } else {//此时已经不是分段函数
                            optCharge[k] = result[x_count];
                            x_count += 1;
                        }
                    } else if (minEnergeChage[k] < 0 && maxEnergyChange[k] > 0) { //第二种情况，两段线
                        forTwoLines[0] = minEnergeChage[k];
                        forTwoLines[1] = 0;
                        forTwoLines[2] = maxEnergyChange[k];
                        for (int i = 0; i < 3; i++)
                            optCharge[k] += forTwoLines[i] * result[x_count + i];
                        x_count += 5;
                    } else {//此时已经不是分段函数
                        optCharge[k] = result[x_count];
                        x_count += 1;
                    }
                } else {
                    b2 = -p / chargeEff;
                    if (b2 < maxEnergyChange[k] && minEnergeChage[k] < b2) {//两段线
                        forTwoLines[0] = minEnergeChage[k];
                        forTwoLines[1] = b2;
                        forTwoLines[2] = maxEnergyChange[k];
                        for (int i = 0; i < 3; i++)
                            optCharge[k] += forTwoLines[i] * result[x_count + i];
                        x_count += 5;
                    } else {//此时已经不是分段函数
                        optCharge[k] = result[x_count];
                        x_count += 1;
                    }
                }
            }
            return true;
            //x_count统计什么？
            //return true 是否收敛，怎样就收敛？
            //AVector.printOnScreen(optCharge);
        }
        return false;
    }

    private void formElements(double[] rowLower, double[] rowUpper,
                              double[] element, int[] column, int[] starts,
                              int row_count, int element_count, int obj_count, int n) {
        //w_1 <= z_1, w_2 <= z_1 + z_2, ..., w_n <= z_n-1 + z_n, w_n+1 <= zn
        //三段线返回n=3, 两段线返回n=2, 因为三段线有4=n+1个此类约束：w1<=z1,w2<=z1+z2,w3<=z2+z3,w4<=z3, 两段线有3个此类约束
        for (int i = 0; i < n + 1; i++) {
            rowLower[row_count] = -2.0;
            rowUpper[row_count++] = 0.0;
            //-2，0由来：w1<=z1, w1-z1<=0, w1最小值0，z1最大值1，结果最小-1，设置成-2，最大是0
            if (i == 0 || i == n)
                starts[row_count] = starts[row_count - 1] + 2;
                //前面声明starts[0]=0,row_count在上句中已经变成1，此时给starts[1]赋值
                //此处2代表2个变量，当i=0时， 为w1,z1， 当i=n时， 为w4, z3
            else
                starts[row_count] = starts[row_count - 1] + 3;
            //此处3 代表三个变量，即w2<=z1+z2中的3个变量，和w3<=z2+z3中的3个变量
        }
        element[element_count] = 1.0;
        //element 是每个w,z变量前的系数，w1-z1<=0,第一二个变量系数为1，-1，w2-z1-z2<=0,系数为1，-1，-1
        column[element_count++] = obj_count - 2 * n - 1;
        //确定上一行决定的系数是哪一个变量，(系数对应的列），如果是三段线，obj_count=7, 分别是[w1,w2,w3,w4,z1,z2,z3],此时1是w1的系数，w1在位置0=obj_count-(2*n+1)
        element[element_count] = -1.0;
        column[element_count++] = obj_count - n;
        //z1 begin，在上面数列中，z1在第4个位置 4=obj_count(7)-n
        for (int i = 1; i < n; i++) {
            element[element_count] = 1.0;
            column[element_count++] = obj_count - 2 * n + i - 1;
            element[element_count] = -1.0;
            column[element_count++] = obj_count - n + i - 1;
            element[element_count] = -1.0;
            column[element_count++] = obj_count - n + i;
        }
        element[element_count] = 1.0;
        column[element_count++] = obj_count - n - 1;
        element[element_count] = -1.0;
        column[element_count++] = obj_count - 1;

        //z_1 + z_2 + ... + z_n = 1;
        rowLower[row_count] = 1.0;
        rowUpper[row_count++] = 1.0;
        //因为是等式约束=1, 所以上下限都是1
        starts[row_count] = starts[row_count - 1] + n;
        //n为第二个约束中，z1+z2+z3=1或z1+z2=1中z的个数

        for (int i = 0; i < n; i++) {
            element[element_count] = 1.0;
            column[element_count++] = obj_count - n + i;
        }
        //w_1 + w_2 + ... + w_n+1 = 1;
        rowLower[row_count] = 1.0;
        rowUpper[row_count++] = 1.0;
        starts[row_count] = starts[row_count - 1] + n + 1;
        for (int i = 0; i < n + 1; i++) {
            element[element_count] = 1.0;
            column[element_count++] = obj_count - 2 * n - 1 + i;
        }
    }

    private int formElements2(double[] rowLower, double[] rowUpper,
                              double[] element, int[] column, int[] starts,
                              int element_count, int obj_count,
                              double[] b, int n, int lastCount) {
        //x_L =< iniEnerge + x_1 <= x_U, x_L =< iniEnerge + x_1 + x_2 < x_L,...,
        //x_1 + x_2 + ... + x_n = finalEnergyChanged
        //n=0...47一共48个数。
        int row = rowLower.length - pNeed.length + n;
        //第一个=rowLower-48+0, point to 第n个条件
        if (n == pNeed.length - 1) {
            //todo: MAY NOT RIGHT
            rowLower[row] = finalEnergyChanged;
            rowUpper[row] = finalEnergyChanged;
            //最后一个约束本该对应 x_1+....x_48 = finalEnergyChanged
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
        //这行是初始, initial, 赋值参见下面循环,starts[row+1]++

        //if 三段线,b.length=length{b1,b2,0,maxEnergyChange}=4
        for (int i = 0; i < b.length; i++) {
            if (Math.abs(b[i]) < 1e-17)
                continue;
            element[element_count] = b[i];
            column[element_count++] = obj_count + i;
            //第一时刻第一个输入column[element_count]=0,第一时刻column[element_count]=[0,1,2...]第二时刻从7开始. 为什么从零开始?
            starts[row + 1]++;
            lastCount++;
        }
        return lastCount;
    }

    public double getIniEnergy() {
        return iniEnergy;
    }

    public void setIniEnergy(double iniEnergy) {
        this.iniEnergy = iniEnergy;
    }

    public double[] getMaxEnergyChange() {
        return maxEnergyChange;
    }

    public double[] getMinEnergeChage() {
        return minEnergeChage;
    }

    public void setMinEnergeChage(double[] minEnergeChage) {
        this.minEnergeChage = minEnergeChage;
    }

    public void setMaxEnergyChange(double[] maxEnergyChange) {
        this.maxEnergyChange = maxEnergyChange;
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

    public double[] getOptCharge() {
        return optCharge;
    }

    public void setOptCharge(double[] optCharge) {
        this.optCharge = optCharge;
    }

    public double getFinalEnergyChanged() {
        return finalEnergyChanged;
    }

    public void setFinalEnergyChanged(double finalEnergyChanged) {
        this.finalEnergyChanged = finalEnergyChanged;
    }
}