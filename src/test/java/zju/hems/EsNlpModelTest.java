package zju.hems;

import junit.framework.TestCase;
import org.supercsv.io.CsvListReader;
import org.supercsv.io.ICsvListReader;
import org.supercsv.prefs.CsvPreference;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 2014/5/23
 */
public class EsNlpModelTest extends TestCase {

    public double[] dcEnergy_l; //下限
    public double[] dcEnergy_u; //上限
    public double[] pNeeded; //负荷有功
    public double[] pricePerKwh;//价格
    public double[] minEnergeChage;
    public double[] maxEnergyChange;
    private double finalEnergyChange;

    public EsNlpModelTest(){
    }

    /**
     *
     * @param file 读入负荷数据
     * @throws IOException 如果文件不存在将抛出异常
     */
    private void readFile(String file) throws IOException {
        //开始读入每半小时所需的能量和电价
        ICsvListReader listReader;
        Reader r = new InputStreamReader(this.getClass().getResourceAsStream(file));
        listReader = new CsvListReader(r, CsvPreference.STANDARD_PREFERENCE);
        int count = 0;
        while (listReader.read() != null) {
            count++;
        }
        listReader.close();

        //设定上下限
        dcEnergy_l = new double[count];
        dcEnergy_u = new double[count];
        minEnergeChage = new double[count];
        maxEnergyChange = new double[count];
        for (int i = 0; i < dcEnergy_l.length; i++) {
            dcEnergy_l[i] = 5.76;
            dcEnergy_u[i] = 17.28;
            minEnergeChage[i] = -2.15;
            maxEnergyChange[i] = 2.15;
        }
        pNeeded = new double[count];
        pricePerKwh = new double[count];
        r = new InputStreamReader(this.getClass().getResourceAsStream(file));
        listReader = new CsvListReader(r, CsvPreference.STANDARD_PREFERENCE);
        count = 0;
        List<String> customerList;
        finalEnergyChange = 0.0;
        while ((customerList = listReader.read()) != null) {
            //AC负荷功率
            double acLoadP = Double.parseDouble(customerList.get(0));
            //DC负荷功率
            double dcLoadP = Double.parseDouble(customerList.get(1));
            //PV输出功率
            double pvOutputP = Double.parseDouble(customerList.get(2));
            //将功率转化为能量
            //pNeeded[count] = (acLoadP + dcLoadP - pvOutputP) * 0.5;
            //电价
            pricePerKwh[count] = Double.parseDouble(customerList.get(3)) / 100;

            pNeeded[count] = acLoadP * 0.5;
            minEnergeChage[count] += dcLoadP * 0.5;
            minEnergeChage[count] -= pvOutputP * 0.5;
            maxEnergyChange[count] += dcLoadP * 0.5;
            maxEnergyChange[count] -= pvOutputP * 0.5;
            finalEnergyChange += (dcLoadP - pvOutputP) * 0.5;
            //对于直流系统，设存储能量的上下限为x_l[count],x_u[count]，直流系统的充电量为x[count]
            //则第1个时间段的充电
            //则第2个时间段的充电 x_l[1] < x[0] + x[1] - dcLoad[0] - dcLoad[1] + pvOutput[0]  + pvOutput[1] < x_u[1]
            //依次类推
            //对于上面的约束可以写成
            //  x_l[0] + dcLoad[0] - pvOutput[0] < x[0] < x_u[0] + dcLoad[0] - pvOutput[0]
            //  x_l[0] + dcLoad[0] + dcLoad[1] - pvOutput[0] - pvOutput[1] < x[1] < x_u[0] + dcLoad[0] + dcLoad[1] - pvOutput[0] - pvOutput[1]
            //依次类推，将约束两边的上下限用dcEnergy_l[count],dcEnergy_u[count]表示，写成程序就是下面两句
            dcEnergy_l[count] += finalEnergyChange;
            dcEnergy_u[count] += finalEnergyChange;
            count++;
        }
        listReader.close();
        //读入结束
    }

    public void testAshley() throws IOException {
        String[] files = {
                "/other/send_Ashton_winter_weekday.csv",
                "/other/send_Ashton_winter_weekday_2.csv",
                "/other/send_Ashley_winter_holi_weekday_3.csv",
                "/other/send_Ashley_summer_holi_weekend_3.csv",
        };
        for (String file : files) {
            System.out.println("Test Hems opt from : " + file);
            readFile(file);
            doEsOpt2();
        }
    }

    /**
     * 测试平滑后非线性模型的例子
     *
     * @throws IOException
     */
    public void doEsOpt1() throws IOException {
        //开始设置优化的条件
        EsNlpModel esOpt = new EsNlpModel();
        esOpt.setX_L(dcEnergy_l);
        esOpt.setX_U(dcEnergy_u);
        esOpt.setEsChargeEff(1.0);//充电效率
        esOpt.setMaxEnergyChange(2.15);//储能最大变化量
        esOpt.setPricePerKwh(pricePerKwh);//每kwh能量的价格
        esOpt.setpNeed(pNeeded);//所需的能量
        esOpt.setIniEnergy(6);//储能能量的初值
        esOpt.setBuffer(0.001);//设置平滑圆心距离Y轴的距离

        //开始优化
        esOpt.doEsOpt();
        //判断是否收敛
        assertTrue(esOpt.isConverged());
        //打印优化结果
        for (double d : esOpt.getResult())
            System.out.println(d);
    }

    public void doEsOpt2() throws IOException {
        //开始设置优化的条件
        EsMlpModel esOpt = new EsMlpModel();
        esOpt.setEnergy_L(dcEnergy_l);
        esOpt.setEnergy_U(dcEnergy_u);
        esOpt.setChargeEff(1);//充电效率
        esOpt.setDischargeEff(1);//放电效率
        esOpt.setFinalEnergyChanged(finalEnergyChange);//todo:
        esOpt.setMinEnergeChage(minEnergeChage);//存储最大变化量
        esOpt.setMaxEnergyChange(maxEnergyChange);//存储最大变化量
        esOpt.setPricePerKwh(pricePerKwh);//每kwh能量的价格
        esOpt.setpNeed(pNeeded);//所需的能量
        esOpt.setIniEnergy(6);//储能能量的初值

        //开始优化
        boolean r = esOpt.doEsOpt();
        //判断是否收敛
        assertTrue(r);
        //如果不收敛怎么显示？//
        //打印优化结果
        int index = 1;
        for (double d : esOpt.getOptCharge()) {
            System.out.println((index) + "\t" + d + "\t" + pricePerKwh[index - 1]);
            index++;
        }
    }
}
