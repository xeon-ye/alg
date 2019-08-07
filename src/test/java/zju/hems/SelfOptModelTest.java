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
        List<AbsorptionChiller> absorptionChillers = new ArrayList<>(1);
        AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 1000, 0.8);
        absorptionChillers.add(absorptionChiller);
        List<AirCon> airCons = new ArrayList<>(1);
        AirCon airCon = new AirCon(0.0097, 1, 1.05, 0, 500, 4.3);
        airCons.add(airCon);
        List<Converter> converters = new ArrayList<>(1);
        Converter converter = new Converter(0.95, 0.95);
        converters.add(converter);
        List<GasBoiler> gasBoilers = new ArrayList<>(1);
        GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500);
        gasBoilers.add(gasBoiler);
        List<GasTurbine> gasTurbines = new ArrayList<>(1);
        GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 200, 50, 1000, - 500, 500, 0);
        gasTurbines.add(gasTurbine);
        List<IceStorageAc> iceStorageAcs = new ArrayList<>(1);
        IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                0.002, 500, 3000, 0.1, 0.95, 0.1, 1.05, 500, 500);
        iceStorageAcs.add(iceStorageAc);
        List<Photovoltaic> photovoltaics = new ArrayList<>(1);
        Photovoltaic photovoltaic = new Photovoltaic(0.0005, pvPowers);
        photovoltaics.add(photovoltaic);
        List<SteamLoad> steamLoads = new ArrayList<>(1);
        SteamLoad steamLoad = new SteamLoad(Lsteams, 0.8);
        steamLoads.add(steamLoad);
        List<Storage> storages = new ArrayList<>(3);
        Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
        for (int i = 0; i < 3; i++) {
            storages.add(storage);
        }
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, photovoltaics, steamLoads, storages);
        Map<String, User> users = new HashMap<>();
        users.put(user.getUserId(), user);
        Microgrid microgrid = new Microgrid(users);
        Map<String, double[]> gatePowers = new HashMap<>();   // 用户关口功率
        double[] gatePower = new double[96];
        for (int i = 0; i < gatePower.length; i++) {
            gatePower[i] = 8000;
        }
        gatePowers.put(user.getUserId(), gatePower);
        SelfOptModel selfOptModel = new SelfOptModel(microgrid, periodNum, acLoad, dcLoad, coolingLoad, elecPrices, gasPrices, steamPrices, gatePowers);
        selfOptModel.doSelfOpt();
        Map<String, UserResult> microgridResult = selfOptModel.getMicrogridResult();
        for (UserResult userResult : microgridResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result.csv", userResult);
            }
        }
    }

    public void readData(InputStream inputStream) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
        String data;
        int t = 0;
        br.readLine();
        while ((data = br.readLine()) != null) {
            String[] newdata = data.split(",", 8);
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

    public void writeResult(String filePath, UserResult userResult) throws IOException {
        FileOutputStream out;
        OutputStreamWriter osw;
        BufferedWriter bw;

        out = new FileOutputStream(new File(filePath));
        osw = new OutputStreamWriter(out);
        bw = new BufferedWriter(osw);
        for (int j = 0; j < periodNum; j++) {
            for (int i = 0; i < userResult.getFrigesP().size(); i++) {
                bw.write(userResult.getFrigesP().get(i)[j] + ",");
            }
            for (int i = 0; i < userResult.getIceTanksP().size(); i++) {
                bw.write(userResult.getIceTanksP().get(i)[j] + ",");
            }
            for (int i = 0; i < userResult.getIceTanksQ().size(); i++) {
                bw.write(userResult.getIceTanksQ().get(i)[j] + ",");
            }
            for (int i = 0; i < userResult.getGasTurbinesState().size(); i++) {
                bw.write(userResult.getGasTurbinesState().get(i)[j] + ",");
            }
            for (int i = 0; i < userResult.getGasTurbinesP().size(); i++) {
                bw.write(userResult.getGasTurbinesP().get(i)[j] + ",");
            }
            for (int i = 0; i < userResult.getStoragesP().size(); i++) {
                bw.write(userResult.getStoragesP().get(i)[j] + ",");
            }
            for (int i = 0; i < userResult.getConvertersP().size(); i++) {
                bw.write(userResult.getConvertersP().get(i)[j] + ",");
            }
            bw.write(userResult.getPurP()[j] + ",");
            for (int i = 0; i < userResult.getAirConsP().size(); i++) {
                bw.write(userResult.getAirConsP().get(i)[j] + ",");
            }
            for (int i = 0; i < userResult.getGasBoilersState().size(); i++) {
                bw.write(userResult.getGasBoilersState().get(i)[j] + ",");
            }
            for (int i = 0; i < userResult.getGasBoilersH().size(); i++) {
                bw.write(userResult.getGasBoilersH().get(i)[j] + ",");
            }
            for (int i = 0; i < userResult.getAbsorptionChillersH().size(); i++) {
                bw.write(userResult.getAbsorptionChillersH().get(i)[j] + ",");
            }
            bw.write(String.valueOf(userResult.getPurH()[j]));
            bw.newLine();
        }
        bw.close();
        osw.close();
        out.close();
    }
}