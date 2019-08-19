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

    public SelfOptModel selfOptTestModel() throws IOException {
        InputStream inputStream = this.getClass().getResourceAsStream("/iesfiles/selfopt/input_with_cold.csv");
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
        GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500, 0);
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
        for (int i = 0; i < 2; i++) {
            Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
            storages.add(storage);
        }
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, photovoltaics, steamLoads, storages);
        Map<String, User> users = new HashMap<>();
        users.put(user.getUserId(), user);
        Microgrid microgrid = new Microgrid(users);
        Map<String, double[]> gatePowers = new HashMap<>();   // 用户关口功率
        double[] gatePower = new double[periodNum];
        for (int i = 0; i < periodNum; i++) {
            gatePower[i] = 8000;
        }
        gatePowers.put(user.getUserId(), gatePower);
        return new SelfOptModel(microgrid, periodNum, acLoad, dcLoad, coolingLoad, elecPrices, gasPrices, steamPrices, gatePowers);
    }

    public DemandRespModel demandRespTestModel() throws IOException {
        InputStream inputStream = this.getClass().getResourceAsStream("/iesfiles/selfopt/input_with_cold.csv");
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
        GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500, 0);
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
        for (int i = 0; i < 2; i++) {
            Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
            storages.add(storage);
        }
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, photovoltaics, steamLoads, storages);
        Map<String, User> users = new HashMap<>();
        users.put(user.getUserId(), user);
        Microgrid microgrid = new Microgrid(users);
        Map<String, double[]> gatePowers = new HashMap<>();   // 用户关口功率
        double[] gatePower = new double[periodNum];
        for (int i = 0; i < periodNum; i++) {
            gatePower[i] = 8000;
        }
        gatePowers.put(user.getUserId(), gatePower);
        return new DemandRespModel(microgrid, periodNum, acLoad, dcLoad, coolingLoad, elecPrices, gasPrices, steamPrices, gatePowers);
    }

    public void testSelfOpt() throws IOException {
        SelfOptModel selfOptModel = selfOptTestModel();
        selfOptModel.mgSelfOpt();
        Map<String, UserResult> microgridResult = selfOptModel.getMicrogridResult();
        for (UserResult userResult : microgridResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result_with_cold.csv", userResult);
            }
        }
    }

    public void testDemandResp() throws IOException {
        DemandRespModel demandRespModel = demandRespTestModel();
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        System.out.println(selfOptResult.get("1").getMinCost());
        System.out.println("---------自趋优计算结束---------");

        Map<String, User> users = demandRespModel.getMicrogrid().getUsers();
        // 原始关口功率
        Map<String, double[]> origGatePowers = new HashMap<>();
        for (String userId : users.keySet()) {
            double[] ogGatePower = new double[periodNum];
            for (int i = 0; i < periodNum; i++) {
                ogGatePower[i] = demandRespModel.getGatePowers().get(userId)[i];
            }
            origGatePowers.put(userId, ogGatePower);
        }
        // 关口功率指令
        int[] peakShaveTime = new int[periodNum];
        for (int i = 72; i < 76; i++) {
            peakShaveTime[i] = 1;
        }
        demandRespModel.setPeakShaveTime(peakShaveTime);
        Map<String, double[]> insGatePowers = new HashMap<>();
        for (String userId : users.keySet()) {
            double[] insGatePower = new double[periodNum];
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    insGatePower[i] = 3000;
                } else {
                    insGatePower[i] = origGatePowers.get(userId)[i];
                }
            }
            insGatePowers.put(userId, insGatePower);
        }
        demandRespModel.setInsGatePowers(insGatePowers);
        // 采样点数
        int sampleNum = 20;
        // 采样范围
        double sampleStart = 1.65;
        double sampleEnd = 1.66;
        Map<String, double[]> increCosts = new HashMap<>(users.size());
        // 应削峰量
        Map<String, double[]> peakShavePowers = new HashMap<>(users.size());
        for (String userId : users.keySet()) {
            increCosts.put(userId, new double[sampleNum]);
            peakShavePowers.put(userId, new double[periodNum]);
            double[] purP = selfOptResult.get(userId).getPurP();
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    peakShavePowers.get(userId)[i] = purP[i] - demandRespModel.getInsGatePowers().get(userId)[i];
                }
            }
        }
        for (int i = 0; i < sampleNum; i++) {
            Map<String, double[]> newGatePowers = new HashMap<>();
            for (String userId : selfOptResult.keySet()) {
                double[] purP = selfOptResult.get(userId).getPurP();
                double[] newGatePower = new double[periodNum];
                for (int j = 0; j < periodNum; j++) {
                    if (peakShaveTime[j] == 1) {
                        newGatePower[j] = purP[j] - peakShavePowers.get(userId)[j] * (sampleStart + (sampleEnd - sampleStart) * (i + 1) / sampleNum);
                    } else {
                        newGatePower[j] = origGatePowers.get(userId)[j];
                    }
                }
                newGatePowers.put(userId, newGatePower);
            }
            demandRespModel.setGatePowers(newGatePowers);
            demandRespModel.mgDemandResp();
            Map<String, UserResult> microgridResult = demandRespModel.getMicrogridResult();
            for (String userId : microgridResult.keySet()) {
                UserResult userResult = microgridResult.get(userId);
                System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
                if (userResult.getStatus().equals("Optimal")) {
                    System.out.println(userResult.getMinCost());
                    writeResult("D:\\user" + userResult.getUserId() + "Result_DR.csv", userResult);
                }
                increCosts.get(userId)[i] = userResult.getMinCost() - selfOptResult.get(userId).getMinCost();
            }
        }
        for (String userId : users.keySet()) {
            double[] increCost = increCosts.get(userId);
            for (int i = 0; i < sampleNum; i++) {
                System.out.println((sampleStart + (sampleEnd - sampleStart) * (i + 1) / sampleNum) + "," + increCost[i]);
            }
        }
    }

    public void testDistDemandResp() throws IOException {
        DemandRespModel demandRespModel = demandRespTestModel();
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        System.out.println(selfOptResult.get("1").getMinCost());
        System.out.println("---------自趋优计算结束---------");
        demandRespModel.setSelfOptResult(selfOptResult);
        Map<String, User> users = demandRespModel.getMicrogrid().getUsers();
        // 原始关口功率
        Map<String, double[]> origGatePowers = new HashMap<>();
        for (String userId : users.keySet()) {
            double[] ogGatePower = new double[periodNum];
            for (int i = 0; i < periodNum; i++) {
                ogGatePower[i] = demandRespModel.getGatePowers().get(userId)[i];
            }
            origGatePowers.put(userId, ogGatePower);
        }
        // 关口功率指令
        int[] peakShaveTime = new int[periodNum];
        for (int i = 72; i < 76; i++) {
            peakShaveTime[i] = 1;
        }
        demandRespModel.setPeakShaveTime(peakShaveTime);
        Map<String, double[]> insGatePowers = new HashMap<>();
        for (String userId : users.keySet()) {
            double[] insGatePower = new double[periodNum];
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    insGatePower[i] = 3000;
                } else {
                    insGatePower[i] = origGatePowers.get(userId)[i];
                }
            }
            insGatePowers.put(userId, insGatePower);
        }
        demandRespModel.setInsGatePowers(insGatePowers);
        demandRespModel.calPeakShavePowers();   // 应削峰量
        demandRespModel.setClearingPrice(3);
        demandRespModel.mgDistDemandResp();
        Map<String, UserResult> microgridResult = demandRespModel.getMicrogridResult();
        for (String userId : microgridResult.keySet()) {
            UserResult userResult = microgridResult.get(userId);
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result_DR.csv", userResult);
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