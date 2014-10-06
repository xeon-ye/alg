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

    public final double[] x_l; //下限
    public final double[] x_u; //上限
    public final double[] pNeeded; //负荷有功
    public final double[] pricePerKwh;//价格
    public final double[] minEnergeChage;
    public final double[] maxEnergyChange;
    private double finalEnergyChange;

    public EsNlpModelTest() throws IOException {

        //开始读入每半小时所需的能量和电价
        ICsvListReader listReader;
        //String file = "/other/send_Ashton_winter_weekday.csv";
        //String file = "/other/send_Ashton_winter_weekday_2.csv";
        String file = "/other/send_Brentry_winter_weekday_3.csv";
        Reader r = new InputStreamReader(this.getClass().getResourceAsStream(file));
        listReader = new CsvListReader(r, CsvPreference.STANDARD_PREFERENCE);
        int count = 0;
        while (listReader.read() != null) {
            count++;
        }
        listReader.close();

        //设定上下限
        x_l = new double[count];
        x_u = new double[count];
        minEnergeChage = new double[count];
        maxEnergyChange = new double[count];
        for (int i = 0; i < x_l.length; i++) {
            x_l[i] = 5.76;
            x_u[i] = 17.28;
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
            x_l[count] += dcLoadP*0.5;
            x_l[count] -= pvOutputP*0.5;
            x_u[count] += dcLoadP*0.5;
            x_u[count] -= pvOutputP*0.5;
            minEnergeChage[count] += dcLoadP*0.5;
            minEnergeChage[count] -= pvOutputP*0.5;
            maxEnergyChange[count] += dcLoadP*0.5;
            maxEnergyChange[count] -= pvOutputP*0.5;
            finalEnergyChange += (dcLoadP - pvOutputP)*0.5;
            count++;
        }
        listReader.close();
        //读入结束
    }

    /**
     * 测试平滑后非线性模型的例子
     *
     * @throws IOException
     */
    public void testEsOpt1() throws IOException {
        //开始设置优化的条件
        EsNlpModel esOpt = new EsNlpModel();
        esOpt.setX_L(x_l);
        esOpt.setX_U(x_u);
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

    public void testEsOpt2() throws IOException {
        //开始设置优化的条件
        EsMlpModel esOpt = new EsMlpModel();
        esOpt.setEnergy_L(x_l);
        esOpt.setEnergy_U(x_u);
        esOpt.setChargeEff(1.15);//存储效率
        esOpt.setDischargeEff(0.87);//存储效率
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
