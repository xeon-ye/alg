package zju.hems;

import org.junit.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import static org.junit.Assert.assertTrue;

/**
 * Created by xcs on 2016/12/11.
 */
public class EsMlpIceModelTest {
    private double[] dcEnergy_l; //下限
    private double[] dcEnergy_u; //上限
    private double[] pNeeded, acLoad, dcLoad, pvOut; //负荷有功
    private double[] sysQNeed; //系统需要供冷量
    private double[] pricePerKwh;//价格
    private double[] minEnergyChange;
    private double[] maxEnergyChange;
    private double finalEnergyChange = 0, finalIceChange = 0;
    private int periodNum;   //时段数

    //traditional method
    boolean isTraditional = false;
    private double[] DCOut;//DC power
    private double minBatteryChange;
    private double maxBatteryChange;

    @Test
    public void doISAndESOpt() throws Exception {

    }

    private void readFile(String filePath) throws IOException {
        //开始读入每半小时所需的能量和电价
        BufferedReader r = new BufferedReader(new InputStreamReader(this.getClass().getResourceAsStream(filePath)));
        int count = 0;
        while (r.readLine() != null)
            count++;
        r.close();

        periodNum = count;
        //设定上下限
        dcEnergy_l = new double[count];
        dcEnergy_u = new double[count];
        minEnergyChange = new double[count];
        maxEnergyChange = new double[count];
        acLoad = new double[count];
        dcLoad = new double[count];
        pvOut = new double[count];
        //traditional method
        minBatteryChange = -2.15;
        maxBatteryChange =2.15;
        for (int i = 0; i < dcEnergy_l.length; i++) {
            dcEnergy_l[i] = 5.76;
            dcEnergy_u[i] = 17.28;
            minEnergyChange[i] = -2.15;
            maxEnergyChange[i] = 2.15;
        }

        pNeeded = new double[count];
        sysQNeed = new double[count];
        pricePerKwh = new double[count];
        DCOut = new double[count];
        r = new BufferedReader(new InputStreamReader(this.getClass().getResourceAsStream(filePath)));
        count = 0;
        String str;
        String[] customerList;
        while ((str = r.readLine()) != null) {
            customerList = str.split(",");
            //AC负荷功率
            acLoad[count] = Double.parseDouble(customerList[0]);
            //DC负荷功率
            dcLoad[count] = Double.parseDouble(customerList[1]);
            //PV输出功率
            pvOut[count] = Double.parseDouble(customerList[2]);
            //dc输出
            DCOut[count] = (dcLoad[count] - pvOut[count]);
            //电价
            pricePerKwh[count] = Double.parseDouble(customerList[3]) / 100;

//            sysQNeed[count] = Double.parseDouble(customerList[4]);
            sysQNeed[count] = 0;

            pNeeded[count] = acLoad[count];
            minEnergyChange[count] = minEnergyChange[count]+dcLoad[count]-pvOut[count];
            maxEnergyChange[count] = maxEnergyChange[count]+dcLoad[count]-pvOut[count];
            count++;
        }
        r.close();
        //读入结束
    }

    @Test
    public void testCase1() throws IOException {
        String filePath = "/other/send_Ashton_winter_weekday.csv";
//        String filePath = "/other/send_Ashton_winter_weekday_2.csv";
//        String filePath = "/other/send_Ashley_winter_holi_weekday_3.csv";
//        String filePath = "/other/send_Ashley_summer_holi_weekend_3.csv";
//        String filePath = "/other/testcase.txt";
        readFile(filePath);

        //开始设置优化的条件
        EsMlpIceModel esOpt = new EsMlpIceModel();
        esOpt.setEnergy_L(dcEnergy_l);
        esOpt.setEnergy_U(dcEnergy_u);
        esOpt.setChargeEff(1);//充电效率
        esOpt.setDischargeEff(1);//放电效率
        esOpt.setFinalEnergyChanged(finalEnergyChange);//todo:
        esOpt.setMinEnergeChange(minEnergyChange);//存储最大变化量
        esOpt.setMaxEnergyChange(maxEnergyChange);//存储最大变化量
        esOpt.setPricePerKwh(pricePerKwh);//每kwh能量的价格
        esOpt.setpNeed(pNeeded);//所需的能量
        esOpt.setIniEnergy(6);//储能能量的初值
        esOpt.setPeriodNum(periodNum);
        esOpt.setSysQNeed(sysQNeed);
        esOpt.setMaxQiChange(0);
        esOpt.setMinQiChange(0);
        esOpt.setIceCapacity(0);
        esOpt.setInitIceCapacity(0);
        esOpt.setMaxQk(0);
        esOpt.setMinBatteryChange(minBatteryChange);
        esOpt.setMaxBatteryChange(maxBatteryChange);
        esOpt.setFinalIceChange(finalIceChange);

        //开始优化
        boolean r = esOpt.doISAndESOpt();
        //判断是否收敛
        assertTrue(r);
        //如果不收敛怎么显示？//
        //打印优化结果
        int index = 1;
        double add = 0;
        for (double d : esOpt.getOptCharge()) {
            add += d;
            System.out.printf("%d   %4.2f   %f    %f\n", index, d, add, esOpt.power[index-1]);
//            System.out.println((index) + "\t" + d + "\t" + pricePerKwh[index - 1]);
            index++;
        }
//        index = 1;
//        for (double d : esOpt.getOptQi()) {
//            System.out.println((index) + "\t" + d + "\t" + pricePerKwh[index - 1]);
//            index++;
//        }
//        index = 1;
//        for (double d : esOpt.getOptQk()) {
//            System.out.println((index) + "\t" + d + "\t" + pricePerKwh[index - 1]);
//            index++;
//        }
    }

}