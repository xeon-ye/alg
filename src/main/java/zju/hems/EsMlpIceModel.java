package zju.hems;

import jpscpu.LinearSolver;
import org.apache.log4j.Logger;

/**
 * Created by Administrator on 2016/11/1.
 */
public class EsMlpIceModel {
    private static Logger log = Logger.getLogger(EsMlpModel.class);

    //储能能量初值,储能最大变化量,每段时间/h,蓄冰槽初始容量
    private double initEnergy, finalEnergyChange = 0, initIceCapacity, finalIceChange;
    private double[] maxEnergyChange, minEnergyChange;
    //放电效率和充电效率，如果这两者不一样则目前程序还没有处理,制冷机能效比,蓄冰槽供冷效率
    private double dischargeEff, chargeEff, EER = 4, coolEff = 0.85;
    //储能能量上限,储能能量下限,负荷需求,电价,系统需冷量
    double[] energy_L, energy_U, pNeed, pricePerKwh, sysQNeed;
    //优化结果
    double optCharge[], optQi[], optQk[];
    //maxQi为蓄冰槽最大蓄冷量, minQi为蓄冰槽最小蓄冷量（负数）, maxQk为制冷机组最大供冷量,iceCapacity为蓄冰槽容量
    double minBatteryChange, maxBatteryChange, maxQiChange, minQiChange, maxQk, iceCapacity;
    //时段数
    int periodNum;
    double power[];

    /**
     * 求解优化模型
     *
     * @return 是否收敛
     */

    public boolean doISAndESOpt() {
        double period = 24.0/periodNum;
        int obj_count = 0, whichInt_count = 0,row_count = 0, element_count = 0, lastCount;

        //开辟内存
        double objValue[] = new double[11*periodNum];
        //objValue 是优化问题中的变量的系数,　如min Cx 中 C矩阵里的系数
        //变量下限//
        //column里元素的个数等于矩阵C里系数的个数,详见starts解释
        double columnLower[] = new double[objValue.length];
        //变量上限
        double columnUpper[] = new double[objValue.length];
        //指明那些是整数
        int whichInt[] = new int[4*periodNum];

        //约束下限
        double rowLower[] = new double[(5+5+1+1+1+1+1+1)*periodNum];
        //约束上限
        double rowUpper[] = new double[rowLower.length];
        //约束中非零元系数
        double element[] = new double[(12+12+3+2+5)*periodNum+3*periodNum*(periodNum+1)/2+periodNum*(periodNum+1)+2*periodNum];
        //上面系数对应的列
        int column[] = new int[element.length];
        //每一行起始位置
        int starts[] = new int[rowLower.length + 1];
        starts[0] = 0;

        for(int i = 0; i < periodNum; i++) {
            //交直流转换分段函数变量系数
            objValue[obj_count] = period * pricePerKwh[i] * dischargeEff * minEnergyChange[i];   //minEnergyChange已经考虑了光伏和直流负荷
            columnLower[obj_count] = 0;
            columnUpper[obj_count++] = 1;

            objValue[obj_count] = 0;
            columnLower[obj_count] = 0;
            columnUpper[obj_count++] = 1;

            objValue[obj_count] = period * maxEnergyChange[i] * chargeEff * pricePerKwh[i];
            columnLower[obj_count] = 0;
            columnUpper[obj_count++] = 1;

            for (int k = 0; k < 2; k++) {
                objValue[obj_count] = 0;
                columnLower[obj_count] = 0;
                columnUpper[obj_count] = 1;
                whichInt[whichInt_count++] = obj_count;
                obj_count++;
            }
            //分两段函数的约束条件
            formElements(rowLower, rowUpper, element, column,
                    starts, row_count, element_count, obj_count);
            row_count += 5;
            element_count += 12;

            //蓄冰槽消耗功率分段函数变量系数
            objValue[obj_count] = period * pricePerKwh[i] * coolEff * minQiChange / EER;   //minEnergeChange已经考虑了光伏和直流负荷
            columnLower[obj_count] = 0;
            columnUpper[obj_count++] = 1;

            objValue[obj_count] = 0;
            columnLower[obj_count] = 0;
            columnUpper[obj_count++] = 1;

            objValue[obj_count] = period * pricePerKwh[i] * maxQiChange / EER;
            columnLower[obj_count] = 0;
            columnUpper[obj_count++] = 1;

            for (int k = 0; k < 2; k++) {
                objValue[obj_count] = 0;
                columnLower[obj_count] = 0;
                columnUpper[obj_count] = 1;
                whichInt[whichInt_count++] = obj_count;
                obj_count++;
            }
            //分两段函数的约束条件
            formElements(rowLower, rowUpper, element, column,
                    starts, row_count, element_count, obj_count);
            row_count += 5;
            element_count += 12;

            //制冷机变量系数
            objValue[obj_count] = 0.025 * period * pricePerKwh[i] * (1+1.0/EER);
            columnLower[obj_count] = 0;
            columnUpper[obj_count++] = maxQk;

            //电池单位时间充放电限制，minBatteryChange <= Pb(t) <= maxBatteryChange
            starts[row_count] = element_count;
            rowLower[row_count] = minBatteryChange;
            rowUpper[row_count++] = maxBatteryChange;

            element[element_count] =  minBatteryChange;
            column[element_count++] = obj_count - 11;
            element[element_count] =  minBatteryChange-minEnergyChange[i];
            column[element_count++] = obj_count - 10;
            element[element_count] =  maxBatteryChange;
            column[element_count++] = obj_count - 9;

            //蓄冰槽单位时间蓄冷、供冷量限制，minQi <= Qi(t) <= maxQi
            starts[row_count] = element_count;
            rowLower[row_count] = minQiChange;
            rowUpper[row_count++] = maxQiChange;

            element[element_count] =  minQiChange;
            column[element_count++] = obj_count - 6;
            element[element_count] =  maxQiChange;
            column[element_count++] = obj_count - 4;

            //不向系统倒送电，p(t) >= 0
            starts[row_count] = element_count;
            rowLower[row_count] = -pNeed[i];
            rowUpper[row_count++] = chargeEff * maxEnergyChange[i] + maxQiChange/EER + 0.025 * (1+1.0/EER) * maxQk;
//            rowUpper[row_count++] = Double.MAX_VALUE;

            element[element_count] = dischargeEff * minEnergyChange[i];
            column[element_count++] = obj_count - 11;
            element[element_count] = maxEnergyChange[i] * chargeEff;
            column[element_count++] = obj_count - 9;
            element[element_count] = coolEff * minQiChange / EER;
            column[element_count++] = obj_count - 6;
            element[element_count] = maxQiChange / EER;
            column[element_count++] = obj_count - 4;
            element[element_count] =  0.025 * (1+1.0/EER);
            column[element_count++] = obj_count - 1;
        }

        //电池内蓄电量的限制，Emin <= E(t) <= Emax
        lastCount = 0;
        for(int i = 0; i < periodNum; i++) {
            starts[row_count] = element_count;
            rowLower[row_count] = (energy_L[i] - initEnergy)/period;
            rowUpper[row_count++] = (energy_U[i] - initEnergy)/period;

            System.arraycopy(element, element_count - lastCount, element, element_count, lastCount);
            System.arraycopy(column, element_count - lastCount, column, element_count, lastCount);
            element_count += lastCount;

            element[element_count] =  minBatteryChange;
            column[element_count++] = 11*i+0;
            element[element_count] =  minBatteryChange-minEnergyChange[i];
            column[element_count++] = 11*i+1;
            element[element_count] =  maxBatteryChange;
            column[element_count++] = 11*i+2;
            lastCount += 3;
        }
        rowLower[row_count-1] = finalEnergyChange;
        rowUpper[row_count-1] = finalEnergyChange;

        //蓄冰槽容量限制
        lastCount = 0;
        for(int i = 0; i < periodNum; i++) {
            starts[row_count] = element_count;
            rowLower[row_count] = -initIceCapacity/period;
            rowUpper[row_count++] = (iceCapacity - initIceCapacity)/period;

            System.arraycopy(element, element_count - lastCount, element, element_count, lastCount);
            System.arraycopy(column, element_count - lastCount, column, element_count, lastCount);
            element_count += lastCount;

            element[element_count] =  minQiChange;
            column[element_count++] = 11*i+5;
            element[element_count] =  maxQiChange;
            column[element_count++] = 11*i+7;
            lastCount += 2;
        }
        rowLower[row_count-1] = finalIceChange;
        rowUpper[row_count-1] = finalIceChange;

        //系统供冷量约束,Qk(t)-Qi(t)=Qsys,其中Qi(t)这一项在 Qi(t)<0 时存在
        for(int i = 0; i < periodNum; i++) {
            starts[row_count] = element_count;
            rowLower[row_count] = sysQNeed[i];
            rowUpper[row_count++] = sysQNeed[i];

            element[element_count] =  1;
            column[element_count++] = 11*i+10;
            element[element_count] =  -coolEff*minQiChange;
            column[element_count++] = 11*i+5;
        }

        //约束条件的总条数
        starts[row_count] = element_count;

//        for(int i = 0; i < columnLower.length; i++){
//            System.out.printf("%.3f %.1f %.1f\n", objValue[i], columnLower[i], columnUpper[i]);
//        }
//        for(int i = 0; i < 4*periodNum; i++) {
//            System.out.printf("%d\n", whichInt[i]);
//        }
//        for(int i = 0; i < rowLower.length; i++)
//            System.out.printf("%.1f %.1f %d\n", rowLower[i], rowUpper[i], starts[i]);
//        for(int i = 0; i < element.length; i++)
//            System.out.printf("%4.1f\n", element[i]);
//        System.out.printf("\n");
//        for(int i = 0; i < column.length; i++)
//            System.out.printf("%4d ", column[i]);

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
            optCharge = new double[pNeed.length];
            optQi = new double[periodNum];
            optQk = new double[periodNum];
            power = new double[periodNum];

            for (int i = 0; i < periodNum; i++) {
                power[i] = pNeed[i]+dischargeEff * minEnergyChange[i]*result[11*i]+maxEnergyChange[i] * chargeEff*result[11*i+2]+
                        coolEff * minQiChange / EER*result[11*i+5]+maxQiChange / EER*result[11*i+7]+0.025 * (1+1.0/EER)*result[11*i+10];
                optCharge[i] = minBatteryChange*result[11*i]+(minBatteryChange-minEnergyChange[i])*result[11*i+1]+maxBatteryChange*result[11*i+2];
                optQi[i] = -(minQiChange*result[11*i+5]+maxQiChange*result[11*i+7]);
                optQk[i] = result[11*i+10];
            }
            return true;
        }
        return false;
    }

    //两分段函数的约束条件
    private void formElements(double[] rowLower, double[] rowUpper,
                              double[] element, int[] column, int[] starts,
                              int row_count, int element_count, int obj_count) {
        //w1 <= z1
        starts[row_count] = element_count;
        rowLower[row_count] = -1;
        rowUpper[row_count++] = 0;

        element[element_count] = 1; //element 是每个w,z变量前的系数，w1-z1<=0,第一二个变量系数为1，-1
        column[element_count++] = obj_count - 5;    //确定上一行决定的系数是哪一个变量，(系数对应的列）
        element[element_count] = -1.0;
        column[element_count++] = obj_count - 2;    //z1 begin，在上面数列中，z1在第4个位置

        //w2 <= z1+z2
        starts[row_count] = element_count;
        rowLower[row_count] = -2;
        rowUpper[row_count++] = 0;

        element[element_count] = 1.0;
        column[element_count++] = obj_count - 4;
        element[element_count] = -1.0;
        column[element_count++] = obj_count - 2;
        element[element_count] = -1.0;
        column[element_count++] = obj_count - 1;

        //w3 <= z2
        starts[row_count] = element_count;
        rowLower[row_count] = -1;
        rowUpper[row_count++] = 0;

        element[element_count] = 1.0;
        column[element_count++] = obj_count - 3;
        element[element_count] = -1.0;
        column[element_count++] = obj_count - 1;

        //z1 + z2 = 1;
        starts[row_count] = element_count;
        rowLower[row_count] = 1;
        rowUpper[row_count++] = 1;  //等式约束, 上下限相同

        element[element_count] = 1;
        column[element_count++] = obj_count - 2;
        element[element_count] = 1;
        column[element_count++] = obj_count - 1;

        //w1 + w2 + w3 = 1;
        starts[row_count] = element_count;
        rowLower[row_count] = 1;
        rowUpper[row_count++] = 1;

        for(int i = 0; i < 3; i++) {
            element[element_count] = 1;
            column[element_count++] = obj_count - 5 + i;
        }
    }

    public double getIniEnergy() {
        return initEnergy;
    }

    public void setIniEnergy(double iniEnergy) {
        this.initEnergy = iniEnergy;
    }

    public double[] getMaxEnergyChange() {
        return maxEnergyChange;
    }

    public double[] getMinEnergyChange() {
        return minEnergyChange;
    }

    public void setMinEnergeChange(double[] minEnergeChange) {
        this.minEnergyChange = minEnergeChange;
    }

    public void setMaxEnergyChange(double[] maxEnergyChange) {
        this.maxEnergyChange = maxEnergyChange;
    }

    public void setPeriodNum(int periodNum) {
        this.periodNum = periodNum;
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

    public void setSysQNeed(double[] sysQNeed) {
        this.sysQNeed = sysQNeed;
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

    public double[] getOptQi() {
        return optQi;
    }

    public double[] getOptQk() {
        return optQk;
    }

    public void setOptCharge(double[] optCharge) {
        this.optCharge = optCharge;
    }

    public double getFinalEnergyChanged() {
        return finalEnergyChange;
    }

    public void setFinalEnergyChanged(double finalEnergyChanged) {
        this.finalEnergyChange = finalEnergyChanged;
    }

    public void setMaxQiChange(double maxQiChange) {
        this.maxQiChange = maxQiChange;
    }

    public void setMinQiChange(double minQiChange) {
        this.minQiChange = minQiChange;
    }

    public void  setIceCapacity(double iceCapacity) {
        this.iceCapacity = iceCapacity;
    }

    public void setInitIceCapacity(double initIceCapacity) {
        this.initIceCapacity = initIceCapacity;
    }

    public void setMaxQk(double maxQk) {
        this.maxQk = maxQk;
    }

    public void setMinBatteryChange(double minBatteryChange) {
        this.minBatteryChange = minBatteryChange;
    }

    public void setMaxBatteryChange(double maxBatteryChange) {
        this.maxBatteryChange = maxBatteryChange;
    }

    public void setFinalIceChange(double finalIceChange) {
        this.finalIceChange = finalIceChange;
    }
}
