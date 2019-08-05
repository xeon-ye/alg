package zju.hems;

import junit.framework.TestCase;

import java.io.*;
import java.util.*;

public class SelfOptModelTest  extends TestCase {

    int periodNum = 96; // 一天的时段数
    double[] acLoad = new double[periodNum];    // 交流负荷
    double[] dcLoad = new double[periodNum];    // 直流负荷
    double[] coolingLoad = new double[periodNum];   // 冷负荷
    double[] elecPrices = new double[periodNum];    // 电价
    double[] gasPrices = new double[periodNum];    // 天然气价格
    double[] steamPrices = new double[periodNum];    // 园区CHP蒸汽价格
    double[] pvPowers = new double[periodNum];   // 光伏出力
    double[] Lsteams = new double[periodNum];   // 蒸汽负荷

    public void testCase1() throws IOException {
        InputStream inputStream = this.getClass().getResourceAsStream("/iesfiles/selfopt/input.csv");
        readData(inputStream);
        List<AbsorptionChiller> absorptionChillers = new LinkedList<>();
        AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 1000, 0.8);
        absorptionChillers.add(absorptionChiller);
        List<AirCon> airCons = new LinkedList<>();
        AirCon airCon = new AirCon(0.0097, 1, 1.05, 0, 500, 4.3);
        airCons.add(airCon);
        List<Converter> converters = new LinkedList<>();
        Converter converter = new Converter(0.95, 0.95);
        converters.add(converter);
        List<GasBoiler> gasBoilers = new LinkedList<>();
        GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500);
        gasBoilers.add(gasBoiler);
        List<GasTurbine> gasTurbines = new LinkedList<>();
        GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 200, 50, 1000, - 500, 500, 0);
        gasTurbines.add(gasTurbine);
        List<IceStorageAc> iceStorageAcs = new LinkedList<>();
        IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                0.002, 500, 3000, 0.1, 0.95, 0.1, 1.05, 500, 500);
        iceStorageAcs.add(iceStorageAc);
        List<Photovoltaic> photovoltaics = new LinkedList<>();
        List<SteamLoad> steamLoads = new LinkedList<>();
        List<Storage> storages = new LinkedList<>();
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, photovoltaics, steamLoads, storages);
        Map<String, User> users = new HashMap<>();
        users.put(user.getUserId(), user);
        Map<String, double[]> gatePower = new HashMap<>();   // 用户关口功率
        Microgrid microgrid = new Microgrid(users);
        SelfOptModel selfOptModel = new SelfOptModel(microgrid, periodNum, acLoad, dcLoad, coolingLoad, elecPrices, gasPrices, steamPrices, gatePower);
        selfOptModel.doSelfOpt();
    }

    public void readData(InputStream inputStream) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
        String data;
        int t = 0;
        while ((data = br.readLine()) != null) {
            String[] newdata = data.split(";", 8);
            acLoad[t] = Double.parseDouble(newdata[0]);
            dcLoad[t] = Double.parseDouble(newdata[1]);
            pvPowers[t] = Double.parseDouble(newdata[2]);
            elecPrices[t] = Double.parseDouble(newdata[3]);
            coolingLoad[t] = Double.parseDouble(newdata[4]);
            Lsteams[t] = Double.parseDouble(newdata[6]);
            gasPrices[t] = 0.349;
            steamPrices[t] = 0.465;
            t += 1;
        }
    }
}