package zju.hems;

import junit.framework.TestCase;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.graph.SimpleGraph;

import java.io.*;
import java.util.*;

import static java.lang.Math.abs;
import static java.lang.Math.pow;

public class SelfOptTest  extends TestCase {

    int periodNum = 96; // 一天的时段数
    double t = 0.25;    // 每个时段15分钟
    double[] elecPrices = new double[periodNum];    // 电价
    double[] gasPrices = new double[periodNum];    // 天然气价格
    double[] steamPrices = new double[periodNum];    // 园区CHP蒸汽价格
    double[] chargePrices = new double[periodNum];    // 充电桩充电价格

    public Microgrid microgridModel() throws IOException {
        Map<String, User> users = new HashMap<>();
        InputStream inputStream;
        // 用户1
        List<AbsorptionChiller> absorptionChillers = new ArrayList<>(2);
        for (int i = 0; i < 0; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
            absorptionChillers.add(absorptionChiller);
        }
        List<AirCon> airCons = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons.add(airCon);
        }
        List<Converter> converters = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(1, 1);
            converters.add(converter);
        }
        List<GasBoiler> gasBoilers = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500, 0);
            gasBoilers.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 0.3, 200, 50, 1000, -500, 500, 0);
            gasTurbines.add(gasTurbine);
        }
        List<IceStorageAc> iceStorageAcs = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
            iceStorageAcs.add(iceStorageAc);
        }
        List<Storage> storages = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
//            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 50000, 0.05, 0.95, 0.05, 0.05, 0.5, 0.5, 0.0025, 0.95, 0.95);
//            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 50000, 0.05, 0.95, 0.4975489443718, 0.05, 0.5, 0.5, 0.0025, 0.95, 0.95);    // 日内4点
            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 50000, 0.05, 0.95, 0.835, 0.05, 0.5, 0.5, 0.0025, 0.95, 0.95);    // 日内7点
            storages.add(storage);
        }
        List<ChargingPile> chargingPiles = new ArrayList<>(10);
        ChargingPile chargingPile1 = new ChargingPile(24, 120, 1, 85, 0, 0, 0);
        ChargingPile chargingPile2 = new ChargingPile(12, 60, 2, 0, 8, 0, 0);
        ChargingPile chargingPile3 = new ChargingPile(18, 90, 3, 0, 0, 85, 0);
        ChargingPile chargingPile4 = new ChargingPile(6, 30, 4, 0, 0, 0, 20);
        ChargingPile chargingPile5 = new ChargingPile(0, 21, 1, 30, 0, 0, 0);
        ChargingPile chargingPile6 = new ChargingPile(0, 7, 2, 0, 24, 0, 0);
        ChargingPile chargingPile7 = new ChargingPile(0, 14, 3, 0, 0, 40, 0);
        ChargingPile chargingPile8 = new ChargingPile(12, 60, 4, 0, 0, 0, 33.8);
        ChargingPile chargingPile9 = new ChargingPile(18, 90, 2, 0, 6, 0, 0);
        ChargingPile chargingPile10 = new ChargingPile(24, 120, 3, 0, 0, 75, 0);
        chargingPiles.add(chargingPile1);
        chargingPiles.add(chargingPile2);
        chargingPiles.add(chargingPile3);
        chargingPiles.add(chargingPile4);
        chargingPiles.add(chargingPile5);
        chargingPiles.add(chargingPile6);
        chargingPiles.add(chargingPile7);
        chargingPiles.add(chargingPile8);
        chargingPiles.add(chargingPile9);
        chargingPiles.add(chargingPile10);
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, storages, 4500, chargingPiles);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzData/input_rncp_user1.csv");
        readUserData(inputStream, user);
        users.put(user.getUserId(), user);

        // 用户2
//        List<AbsorptionChiller> absorptionChillers2 = new ArrayList<>(4);
//        for (int i = 0; i < 3; i++) {
//            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
//            absorptionChillers2.add(absorptionChiller);
//        }
//        List<AirCon> airCons2 = new ArrayList<>(2);
//        for (int i = 0; i < 1; i++) {
//            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
//            airCons2.add(airCon);
//        }
//        List<Converter> converters2 = new ArrayList<>(1);
//        for (int i = 0; i < 1; i++) {
//            Converter converter = new Converter(0.95, 0.95);
//            converters2.add(converter);
//        }
//        List<GasBoiler> gasBoilers2 = new ArrayList<>(1);
//        for (int i = 0; i < 1; i++) {
//            GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500, 0);
//            gasBoilers2.add(gasBoiler);
//        }
//        List<GasTurbine> gasTurbines2 = new ArrayList<>(1);
//        for (int i = 0; i < 1; i++) {
//            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 0.3, 200, 50, 1000, -500, 500, 0);
//            gasTurbines2.add(gasTurbine);
//        }
//        List<IceStorageAc> iceStorageAcs2 = new ArrayList<>(2);
//        for (int i = 0; i < 2; i++) {
//            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
//                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
//            iceStorageAcs2.add(iceStorageAc);
//        }
//        List<Storage> storages2 = new ArrayList<>(1);
//        for (int i = 0; i < 1; i++) {
//            Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
//            storages2.add(storage);
//        }
//        User user2 = new User("2", absorptionChillers2, airCons2, converters2, gasBoilers2, gasTurbines2, iceStorageAcs2, storages2, 2100);
//        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzData/input_user2.csv");
//        readUserData(inputStream, user2);
//        users.put(user2.getUserId(), user2);
//
//        // 用户3
//        List<AbsorptionChiller> absorptionChillers3 = new ArrayList<>(1);
//        for (int i = 0; i < 1; i++) {
//            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
//            absorptionChillers3.add(absorptionChiller);
//        }
//        List<AirCon> airCons3 = new ArrayList<>(1);
//        for (int i = 0; i < 1; i++) {
//            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
//            airCons3.add(airCon);
//        }
//        List<Converter> converters3 = new ArrayList<>(1);
//        for (int i = 0; i < 1; i++) {
//            Converter converter = new Converter(0.95, 0.95);
//            converters3.add(converter);
//        }
//        List<GasBoiler> gasBoilers3 = new ArrayList<>(1);
//        for (int i = 0; i < 1; i++) {
//            GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500, 0);
//            gasBoilers3.add(gasBoiler);
//        }
//        List<GasTurbine> gasTurbines3 = new ArrayList<>(1);
//        for (int i = 0; i < 1; i++) {
//            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 0.3, 200, 50, 1000, -500, 500, 0);
//            gasTurbines3.add(gasTurbine);
//        }
//        List<IceStorageAc> iceStorageAcs3 = new ArrayList<>(1);
//        List<Storage> storages3 = new ArrayList<>(1);
//        User user3 = new User("3", absorptionChillers3, airCons3, converters3, gasBoilers3, gasTurbines3, iceStorageAcs3, storages3, 1600);
//        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzData/input_user3.csv");
//        readUserData(inputStream, user3);
//        users.put(user3.getUserId(), user3);

        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzData/energy_rncp_price.csv");
        readEnergyPrice(inputStream);

        return new Microgrid(users);
    }

    public Microgrid wanliModel() throws IOException {
        Map<String, User> users = new HashMap<>();
        InputStream inputStream;
        // 用户1
        List<AbsorptionChiller> absorptionChillers = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 3877, 0.9);
            absorptionChillers.add(absorptionChiller);
        }
        List<AirCon> airCons = new ArrayList<>(3);
        AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 684, 3.4);
        airCons.add(airCon);
        airCon = new AirCon(0.0097, 1, 1.00, 0, 300, 2.8);
        airCons.add(airCon);
        airCon = new AirCon(0.009701, 1, 1.00, 0, 1000, 5);
        airCons.add(airCon);
        List<Converter> converters = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(1, 1);
            converters.add(converter);
        }
        List<GasBoiler> gasBoilers = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.9, 100, 10000, 5000, 0);
            gasBoilers.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.7, 0, 200, 200, 15000, -7500, 7500, 0);
            gasTurbines.add(gasTurbine);
        }
        List<IceStorageAc> iceStorageAcs = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
            iceStorageAcs.add(iceStorageAc);
        }
        List<Storage> storages = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 36000, 0.1, 0.9, 0.1, 0.1, 1, 1, 0, 1, 1);
//            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 36000, 0.1, 0.9, 0.5, 0.1, 1, 1, 0, 1, 1); // 日内4点
//            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 50000, 0.05, 0.95, 0.4975489443718, 0.05, 0.5, 0.5, 0.0025, 0.95, 0.95);    // 日内4点
//            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 50000, 0.05, 0.95, 0.835, 0.05, 0.5, 0.5, 0.0025, 0.95, 0.95);    // 日内7点
            storages.add(storage);
        }
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, storages, 4500);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzTestData/input_user1_rn_new.csv");
        readUserData(inputStream, user);
        users.put(user.getUserId(), user);

        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzTestData/energy_price.csv");
        readEnergyPrice(inputStream);

        return new Microgrid(users);
    }

    public Microgrid wanbaoModel() throws IOException {
        Map<String, User> users = new HashMap<>();
        InputStream inputStream;
        // 用户1
        List<AbsorptionChiller> absorptionChillers = new ArrayList<>(1);
        List<AirCon> airCons = new ArrayList<>(3);
        List<Converter> converters = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(1, 1);
            converters.add(converter);
        }
        List<GasBoiler> gasBoilers = new ArrayList<>(1);
        List<GasTurbine> gasTurbines = new ArrayList<>(1);
        List<IceStorageAc> iceStorageAcs = new ArrayList<>(1);
        List<Storage> storages = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Storage storage = new Storage(0.005, 0.00075, 2000, 2000, 4000, 0.05, 0.95, 0.05, 0.05, 1, 1, 0, 1, 1);
//            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 36000, 0.1, 0.9, 0.66, 0.1, 1, 1, 0.0025, 0.95, 0.95); // 日内4点
//            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 50000, 0.05, 0.95, 0.4975489443718, 0.05, 0.5, 0.5, 0.0025, 0.95, 0.95);    // 日内4点
//            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 50000, 0.05, 0.95, 0.835, 0.05, 0.5, 0.5, 0.0025, 0.95, 0.95);    // 日内7点
            storages.add(storage);
        }
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, storages, 4500);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzTestData/input_user2.csv");
        readUserData(inputStream, user);
        users.put(user.getUserId(), user);

        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzTestData/energy_price.csv");
        readEnergyPrice(inputStream);

        return new Microgrid(users);
    }

    public Microgrid wanliSSModel() throws IOException {
        Map<String, User> users = new HashMap<>();
        InputStream inputStream;
        // 用户1
        List<AbsorptionChiller> absorptionChillers = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 10750, 0.8);
            absorptionChillers.add(absorptionChiller);
        }
        List<AirCon> airCons = new ArrayList<>(3);
        AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 1341, 4.3);
        airCons.add(airCon);
        airCon = new AirCon(0.0097, 1, 1.00, 0, 1341, 4.3);
        airCons.add(airCon);
        airCon = new AirCon(0.009701, 1, 1.00, 0, 1341, 4.3);
        airCons.add(airCon);
        List<Converter> converters = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(1, 1);
            converters.add(converter);
        }
        List<GasBoiler> gasBoilers = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.9, 100, 10000, 5000, 0);
            gasBoilers.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.7, 0, 200, 200, 15000, -7500, 7500, 0);
            gasTurbines.add(gasTurbine);
        }
        List<IceStorageAc> iceStorageAcs = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
            iceStorageAcs.add(iceStorageAc);
        }
        List<Storage> storages = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
//            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 36000, 0.1, 0.9, 0.66, 0.1, 1, 1, 0.0025, 0.95, 0.95); // 日内4点
            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 36000, 0.1, 0.9, 0.685, 0.1, 1, 1, 0.0025, 0.95, 0.95); // 实时4点
//            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 50000, 0.05, 0.95, 0.4975489443718, 0.05, 0.5, 0.5, 0.0025, 0.95, 0.95);    // 日内4点
//            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 50000, 0.05, 0.95, 0.835, 0.05, 0.5, 0.5, 0.0025, 0.95, 0.95);    // 日内7点
            storages.add(storage);
        }
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, storages, 4500);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzTestData/input_ss_user1.csv");
        readUserData(inputStream, user);
        users.put(user.getUserId(), user);

        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzTestData/energy_ss_price.csv");
        readEnergyPrice(inputStream);

        return new Microgrid(users);
    }

    public Microgrid distIDRModel() throws IOException {
        Map<String, User> users = new HashMap<>();
        InputStream inputStream;
        // 用户1
        List<AbsorptionChiller> absorptionChillers = new ArrayList<>(2);
        for (int i = 0; i < 0; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
            absorptionChillers.add(absorptionChiller);
        }
        List<AirCon> airCons = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons.add(airCon);
        }
        List<Converter> converters = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(1, 1);
            converters.add(converter);
        }
        List<GasBoiler> gasBoilers = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500, 0);
            gasBoilers.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 0.3, 200, 50, 1000, -500, 500, 0);
            gasTurbines.add(gasTurbine);
        }
        List<IceStorageAc> iceStorageAcs = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
            iceStorageAcs.add(iceStorageAc);
        }
        List<Storage> storages = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Storage storage = new Storage(0.005, 0.00075, 6000, 6000, 50000, 0.05, 0.95, 0.05, 0.05, 0.5, 0.5, 0.0025, 0.95, 0.95);
            storages.add(storage);
        }
        InterruptibleLoad interruptibleLoad = new InterruptibleLoad(6.1 * 1e-5, 1.6639, 800);
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, storages, 5500, interruptibleLoad);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzData/input_user1.csv");
        readUserData(inputStream, user);
        users.put(user.getUserId(), user);

        // 用户2
        List<AbsorptionChiller> absorptionChillers2 = new ArrayList<>(4);
        for (int i = 0; i < 0; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
            absorptionChillers2.add(absorptionChiller);
        }
        List<AirCon> airCons2 = new ArrayList<>(2);
        for (int i = 0; i < 0; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons2.add(airCon);
        }
        List<Converter> converters2 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters2.add(converter);
        }
        List<GasBoiler> gasBoilers2 = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 0, 0.85, 0, 1000, 500, 0);
            gasBoilers2.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines2 = new ArrayList<>(1);
        for (int i = 0; i < 0; i++) {
            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 0.3, 0, 50, 1000, -500, 500, 0);
            gasTurbines2.add(gasTurbine);
        }
        List<IceStorageAc> iceStorageAcs2 = new ArrayList<>(2);
        for (int i = 0; i < 0; i++) {
            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
            iceStorageAcs2.add(iceStorageAc);
        }
        List<Storage> storages2 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Storage storage = new Storage(0.005, 0.00075, 2000, 2000, 4000, 0, 1, 0, 0, 1, 1, 0.0025, 1, 1);
            storages2.add(storage);
        }
        InterruptibleLoad interruptibleLoad2 = new InterruptibleLoad(6.1 * 1e-5, 1.6639, 800);
        User user2 = new User("2", absorptionChillers2, airCons2, converters2, gasBoilers2, gasTurbines2, iceStorageAcs2, storages2, 1200, interruptibleLoad2);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzData/input_user2.csv");
        readUserData2(inputStream, user2);
        users.put(user2.getUserId(), user2);

        inputStream = this.getClass().getResourceAsStream("/iesfiles/gzData/energy_price.csv");
        readEnergyPrice(inputStream);

        return new Microgrid(users);
    }

    public void testSelfOpt() throws IOException {
        Microgrid microgrid = wanliModel();
        // 日内自趋优参数设置
        double[] gatePowers = microgrid.getUsers().get("1").getGatePowers();
        /** 报告中的参数
        // 日前需求响应（时间段70-75）
//        gatePowers[70] = 9666.229 - 3000 * 1;
//        gatePowers[71] = 10036.99 - 3000 * 1;
//        gatePowers[72] = 10373.09 - 3000 * 1;
//        gatePowers[73] = 7490.521 - 3000 * 1;
//        gatePowers[74] = 7110.637 - 3000 * 1;
//        gatePowers[75] = 8431.826 - 3000 * 1;
        // 日内自趋优
        gatePowers[55] = 9666.229 - 3000 * 1;
        gatePowers[56] = 10036.99 - 3000 * 1;
        gatePowers[57] = 10373.09 - 3000 * 1;
        gatePowers[58] = 7490.521 - 3000 * 1;
        gatePowers[59] = 7110.637 - 3000 * 1;
        gatePowers[60] = 8431.826 - 3000 * 1;
        // 日内需求响应
        gatePowers[30] = 3754.767 - 2000 * 0.2;
        gatePowers[31] = 4435.746 - 2000 * 0.2;
        gatePowers[32] = 4002.511 - 2000 * 0.2;
        gatePowers[33] = 4057.738 - 2000 * 0.2;
         **/
        // 日内需求响应
        gatePowers[44] = 3754.767 - 2075.83;
        gatePowers[45] = 4435.746 - 2631.31;
        gatePowers[46] = 4002.511 - 2585.75;
        gatePowers[47] = 4057.738 - 2301.92;
        SelfOptModel selfOptModel = new SelfOptModel(microgrid, periodNum, t, elecPrices, gasPrices, steamPrices);
        selfOptModel.mgSelfOpt();
        Map<String, UserResult> microgridResult = selfOptModel.getMicrogridResult();
        for (UserResult userResult : microgridResult.values()) {
            String userId = userResult.getUserId();
            System.out.println(userId + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result.csv", userResult, microgrid.getUsers().get(userId).getAcLoad(), microgrid.getUsers().get(userId).getPhotovoltaic().getPower());
            }
        }
    }

    public void testDR() throws IOException {
        Microgrid microgrid = wanliModel();
        SelfOptModel selfOptModel = new SelfOptModel(microgrid, periodNum, t, elecPrices, gasPrices, steamPrices, chargePrices);
        selfOptModel.mgSelfOpt();
        Map<String, UserResult> microgridResult = selfOptModel.getMicrogridResult();
        double[] purP = microgridResult.get("1").getPurP();

        // 实时需求响应
//        periodNum--;
//        microgrid = wanliSSModel();
//        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, t, elecPrices, gasPrices, steamPrices, chargePrices);
//        // 关口功率指令
//        Map<String, User> users = microgrid.getUsers();
//        int[] peakShaveTime = new int[periodNum];
//        for (int i = 0; i < periodNum; i++) {
//            peakShaveTime[i] = 1;
//        }
//        demandRespModel.setPeakShaveTime(peakShaveTime);
//        for (String userId : users.keySet()) {
//            User user = users.get(userId);
//            for (int i = 0; i < periodNum; i++) {
//                user.getGatePowers()[i] = purP[i + 1];
//            }
//        }
        // 日内削峰需求响应
        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, t, elecPrices, gasPrices, steamPrices, chargePrices);
        // 关口功率指令
        Map<String, User> users = microgrid.getUsers();
        int[] peakShaveTime = new int[periodNum];
        for (int i = 12; i < 16; i++) {
            peakShaveTime[i] = 1;
        }
        demandRespModel.setPeakShaveTime(peakShaveTime);
        for (String userId : users.keySet()) {
            User user = users.get(userId);
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    user.getGatePowers()[i] = purP[i] - 6000;
                }
            }
        }
        demandRespModel.mgOrigDemandResp();
        microgridResult = demandRespModel.getMicrogridResult();
        for (UserResult userResult : microgridResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result_DR.csv", userResult, microgrid.getUsers().get("1").getAcLoad(), microgrid.getUsers().get("1").getPhotovoltaic().getPower());
            }
        }
    }

    public void testDispatchOpt() throws IOException {
        Microgrid microgrid = microgridModel();
        double dispatchTime = 24;
        int periodNum = 96;
        for (User user : microgrid.getUsers().values()) {
            user.setWindPower(new WindPower(0.0005, new double[periodNum]));
        }
//        DispatchOptModel dispatchOptModel = new DispatchOptModel(microgrid, dispatchTime, periodNum, elecPrices, gasPrices, steamPrices);
//        dispatchOptModel.mgDispatchOpt();
//        Map<String, UserResult> microgridResult = dispatchOptModel.getMicrogridResult();
        SelfOptDispatch selfOptDispatch = new SelfOptDispatch();
        Map<String, UserResult> microgridResult = selfOptDispatch.doDispatchOpt(microgrid, dispatchTime, periodNum, elecPrices, gasPrices, steamPrices);
        for (UserResult userResult : microgridResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
//                writeResult("D:\\user" + userResult.getUserId() + "DispatchResult.csv", userResult);
            }
        }
    }

    public void testDemandResp() throws IOException {
        Microgrid microgrid = microgridModel();
        double[] gatePowers = microgrid.getUsers().get("1").getGatePowers();
        gatePowers[54] = 3072.699547;
        gatePowers[55] = 6127.099991;
        gatePowers[56] = 3331.616531;
        gatePowers[57] = 2958.342735;
        gatePowers[58] = 4289.747597;
        gatePowers[59] = 2907.299995;
        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, t, elecPrices, gasPrices, steamPrices);
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        for (UserResult userResult : selfOptResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result.csv", userResult, microgrid.getUsers().get("1").getAcLoad(), microgrid.getUsers().get("1").getPhotovoltaic().getPower());
            }
        }
        System.out.println("---------自趋优计算结束---------");

        Map<String, User> users = microgrid.getUsers();
        // 原始关口功率
        Map<String, double[]> origGatePowers = new HashMap<>();
        for (String userId : users.keySet()) {
            double[] ogGatePower = new double[periodNum];
            for (int i = 0; i < periodNum; i++) {
                ogGatePower[i] = users.get(userId).getGatePowers()[i];
            }
            origGatePowers.put(userId, ogGatePower);
        }
        // 关口功率指令
        int[] peakShaveTime = new int[periodNum];
        for (int i = 28; i < 32; i++) {
            peakShaveTime[i] = 1;
        }
        demandRespModel.setPeakShaveTime(peakShaveTime);
        Map<String, double[]> insGatePowers = new HashMap<>();
        for (String userId : users.keySet()) {
            double[] insGatePower = new double[periodNum];
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    insGatePower[i] = selfOptResult.get(userId).getPurP()[i] - 2000;
                } else {
                    insGatePower[i] = origGatePowers.get(userId)[i];
                }
            }
            insGatePowers.put(userId, insGatePower);
        }
        // 采样点数
        int sampleNum = 1;
        // 采样范围
        double sampleStart = 0;
        double sampleEnd = 0;
        Map<String, double[]> increCosts = new HashMap<>(users.size());
        // 应削峰量
        Map<String, double[]> peakShavePowers = new HashMap<>(users.size());
        for (String userId : users.keySet()) {
            increCosts.put(userId, new double[sampleNum]);
            peakShavePowers.put(userId, new double[periodNum]);
            double[] purP = selfOptResult.get(userId).getPurP();
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    peakShavePowers.get(userId)[i] = purP[i] - insGatePowers.get(userId)[i];
                }
            }
        }
        for (int i = 0; i < sampleNum; i++) {
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
                microgrid.getUsers().get(userId).setGatePowers(newGatePower);
            }
            demandRespModel.mgDemandResp();
            Map<String, UserResult> microgridResult = demandRespModel.getMicrogridResult();
            for (String userId : microgridResult.keySet()) {
                UserResult userResult = microgridResult.get(userId);
                System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
                if (userResult.getStatus().equals("Optimal")) {
                    System.out.println(userResult.getMinCost());
                    writeResult("D:\\user" + userResult.getUserId() + "Result_DR.csv", userResult, microgrid.getUsers().get("1").getAcLoad(), microgrid.getUsers().get("1").getPhotovoltaic().getPower());
                }
                increCosts.get(userId)[i] = userResult.getMinCost() - selfOptResult.get(userId).getMinCost();
            }
        }
        for (String userId : users.keySet()) {
            double[] increCost = increCosts.get(userId);
            for (int i = 0; i < sampleNum; i++) {
                System.out.println(userId + "," + (sampleStart + (sampleEnd - sampleStart) * (i + 1) / sampleNum) + "," + increCost[i]);
            }
        }
    }

    public void testCenDistIDR() throws IOException {
        Microgrid microgrid = microgridModel();
        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, t, elecPrices, gasPrices, steamPrices);
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        for (UserResult userResult : selfOptResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result.csv", userResult, microgrid.getUsers().get("1").getAcLoad(), microgrid.getUsers().get("1").getPhotovoltaic().getPower());
            }
        }
        demandRespModel.setSelfOptResult(selfOptResult);
        System.out.println("---------自趋优计算结束---------");
        Map<String, User> users = microgrid.getUsers();
        // 原始关口功率
        double[] parkGatePower = new double[periodNum]; // 园区关口功率
        Map<String, double[]> origGatePowers = new HashMap<>();
        for (String userId : users.keySet()) {
            double[] ogGatePower = new double[periodNum];
            double[] purP = selfOptResult.get(userId).getPurP();
            for (int i = 0; i < periodNum; i++) {
                ogGatePower[i] = microgrid.getUsers().get(userId).getGatePowers()[i];
                parkGatePower[i] += purP[i];
            }
            origGatePowers.put(userId, ogGatePower);
        }
        // 关口功率指令
        int[] peakShaveTime = new int[periodNum];
        for (int i = 45; i < 49; i++) {
            peakShaveTime[i] = 1;
        }
        demandRespModel.setPeakShaveTime(peakShaveTime);
        double[] parkPeakShavePower = new double[periodNum];
        for (int i = 0; i < periodNum; i++) {
            if (peakShaveTime[i] == 1) {
                parkPeakShavePower[i] = parkGatePower[i] - 11000;
            }
        }
        demandRespModel.calPeakShavePowers(parkGatePower, parkPeakShavePower);   // 应削峰量
        System.out.println("---------各用户应削峰量---------");
        Map<String, double[]> peakShavePowers = demandRespModel.getPeakShavePowers();
        for (String userId : users.keySet()) {
            System.out.print(userId + "\t");
            double[] peakShavePower = peakShavePowers.get(userId);
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    System.out.print(peakShavePower[i] + "\t");
                }
            }
            System.out.println();
        }
        System.out.println("---------100%需求响应计算开始---------");
        Map<String, double[]> shaveGatePowers = demandRespModel.getShaveGatePowers();
        for (String userId : users.keySet()) {
            users.get(userId).setGatePowers(shaveGatePowers.get(userId));
        }
        demandRespModel.mgDemandResp();
        demandRespModel.setDemandRespResult(demandRespModel.getMicrogridResult());
        for (String userId : users.keySet()) {
            users.get(userId).setGatePowers(origGatePowers.get(userId));
        }
        for (String userId : demandRespModel.getMicrogridResult().keySet()) {
            UserResult userResult = demandRespModel.getMicrogridResult().get(userId);
            System.out.println(userId + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userId + "Result_DR.csv", userResult, microgrid.getUsers().get("1").getAcLoad(), microgrid.getUsers().get("1").getPhotovoltaic().getPower());
            }
        }
        System.out.println("---------100%需求响应计算结束---------");
        List<Map<String, Double>> timeShaveCapRatios = demandRespModel.getTimeShaveCapRatios();
        double clearingPrice = 0.54;
        double lastClearingPrice = clearingPrice;
        demandRespModel.setClearingPrice(clearingPrice);
        demandRespModel.mgCenDistDemandResp();
        Map<String, UserResult> microgridResult = demandRespModel.getMicrogridResult();
        List<Offer> offers = demandRespModel.getOffers();
        List<Offer> lastOffers = offers;
        List<Map<String, Double>> bidRatiosList = new ArrayList<>();
        List<Map<String, Double>> lastBidRatiosList = new ArrayList<>();
        double maxRatio = 0;   // 最大削峰比例
        for (int i = 0; i < periodNum; i++) {
            if (peakShaveTime[i] == 1) {
                Map<String, Double> bidRatios = new HashMap<>();
                Map<String, Double> lastBidRatios = new HashMap<>();
                for (Offer offer : offers) {
                    maxRatio += offer.getMaxPeakShaveRatio() * offer.getPeakShaveCapRatio();
                    bidRatios.put(offer.getUserId(), offer.getMaxPeakShaveRatio());
                    lastBidRatios.put(offer.getUserId(), offer.getMaxPeakShaveRatio());
                }
                bidRatiosList.add(bidRatios);
                lastBidRatiosList.add(lastBidRatios);
            }
        }
        int iterNum = 1;
        List<Double> clearingPrices = new ArrayList<>();
        List<Double> maxRatios = new ArrayList<>();
        clearingPrices.add(clearingPrice);
        maxRatios.add(maxRatio);
        while (maxRatio > 1) {
            lastClearingPrice = clearingPrice;
            lastOffers = offers;
            lastBidRatiosList = bidRatiosList;
            microgridResult = demandRespModel.getMicrogridResult();
            bidRatiosList.clear();
            for (int i = 0; i < timeShaveCapRatios.size(); i++) {
                Map<String, Double> timeShaveCapRatio = timeShaveCapRatios.get(i);
                ClearingModel clearingModel = new ClearingModel(offers, 0.54, timeShaveCapRatio);
                clearingModel.clearing();
                if (clearingPrice > clearingModel.getClearingPrice()) {
                    clearingPrice = clearingModel.getClearingPrice();
                }
                bidRatiosList.add(clearingModel.getBidRatios());
            }
            // 出清价格变化上限
            if (lastClearingPrice - clearingPrice > 0.1) {
                clearingPrice = lastClearingPrice - 0.1;
            }
            demandRespModel.setClearingPrice(clearingPrice);
            demandRespModel.mgCenDistDemandResp();
            offers = demandRespModel.getOffers();
            maxRatio = 0;   // 最大削峰比例
            for (Offer offer : offers) {
                maxRatio += offer.getMaxPeakShaveRatio() * offer.getPeakShaveCapRatio();
            }
            clearingPrices.add(clearingPrice);
            maxRatios.add(maxRatio);
            iterNum++;
        }
        System.out.println("---------市场出清计算结束---------");
        System.out.println("---------最终报价情况---------");
        for (Offer offer : lastOffers) {
            System.out.println(offer.getUserId() + "\t" + offer.getPrice() + "\t" + offer.getMaxPeakShaveRatio());
        }
        System.out.println("---------出清价格和中标比例、容量---------");
        System.out.println(lastClearingPrice);
        for (Map<String, Double> lastBidRatios : lastBidRatiosList) {
            for (String key : lastBidRatios.keySet()) {
                System.out.print(key + ":\t" + lastBidRatios.get(key) + "\t");
            }
            System.out.println();
        }
        System.out.println();
        int count = 45;
        for (Map<String, Double> lastBidRatios : lastBidRatiosList) {
            for (String key : lastBidRatios.keySet()) {
                System.out.print(key + ":\t" + lastBidRatios.get(key) * peakShavePowers.get(key)[count] + "\t");
            }
            count++;
            System.out.println();
        }
        System.out.println("---------出清价格和申报容量变化---------");
        for (int i = 0; i < iterNum; i++) {
            System.out.println(clearingPrices.get(i) + "\t" + maxRatios.get(i));
        }
        System.out.println("---------用户自趋优结果---------");
        Map<String, Double> peakShaveRatios = demandRespModel.getPeakShaveRatios();
        for (String userId : microgridResult.keySet()) {
            UserResult userResult = microgridResult.get(userId);
            System.out.println(userId + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                System.out.println(peakShaveRatios.get(userId));
                writeResult("D:\\user" + userId + "Result_DR.csv", userResult, microgrid.getUsers().get("1").getAcLoad(), microgrid.getUsers().get("1").getPhotovoltaic().getPower());
            }
        }
        System.out.println("---------分布式需求响应计算结束---------");
    }

    public void testCenIDR() throws IOException {
        Microgrid microgrid = distIDRModel();
        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, t, elecPrices, gasPrices, steamPrices);
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        for (UserResult userResult : selfOptResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result.csv", userResult, microgrid.getUsers().get("1").getAcLoad(), microgrid.getUsers().get("1").getPhotovoltaic().getPower());
            }
        }
        System.out.println("---------自趋优计算结束---------");

        Map<String, User> users = microgrid.getUsers();
        // 原始关口功率
        Map<String, double[]> origGatePowers = new HashMap<>();
        for (String userId : users.keySet()) {
            double[] ogGatePower = new double[periodNum];
            for (int i = 0; i < periodNum; i++) {
                ogGatePower[i] = users.get(userId).getGatePowers()[i];
            }
            origGatePowers.put(userId, ogGatePower);
        }
        // 关口功率指令
        int[] peakShaveTime = new int[periodNum];
        double[] gatePowerSum = new double[periodNum];
        for (int i = 45; i < 49; i++) {
            peakShaveTime[i] = 1;
            gatePowerSum[i] = 10000;
        }
        gatePowerSum[48] = 9000;
//        gatePowerSum[45] = 10496.01;
//        gatePowerSum[46] = 9790.01;
//        gatePowerSum[47] = 9859.644;
//        gatePowerSum[48] = 8960.268;
        demandRespModel.setPeakShaveTime(peakShaveTime);
        demandRespModel.setGatePowerSum(gatePowerSum);
        demandRespModel.cenIDRIL();
        for (String userId : demandRespModel.getMicrogridResult().keySet()) {
            UserResult userResult = demandRespModel.getMicrogridResult().get(userId);
            System.out.println(userId + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userId + "Result_DR.csv", userResult, microgrid.getUsers().get("1").getAcLoad(), microgrid.getUsers().get("1").getPhotovoltaic().getPower());
            }
        }
    }

    // 要能向园区售电
    public void testDistIDR() throws IOException {
        Microgrid microgrid = distIDRModel();
        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, t, elecPrices, gasPrices, steamPrices);
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        for (UserResult userResult : selfOptResult.values()) {
            String userId = userResult.getUserId();
            System.out.println(userId + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result.csv", userResult, microgrid.getUsers().get(userId).getAcLoad(), microgrid.getUsers().get(userId).getPhotovoltaic().getPower());
            }
        }
        demandRespModel.setSelfOptResult(selfOptResult);
        System.out.println("---------自趋优计算结束---------");
        Map<String, User> users = microgrid.getUsers();
        // 原始关口功率
        double[] parkGatePower = new double[periodNum]; // 园区关口功率
        Map<String, double[]> origGatePowers = new HashMap<>();
        for (String userId : users.keySet()) {
            double[] ogGatePower = new double[periodNum];
            double[] purP = selfOptResult.get(userId).getPurP();
            for (int i = 0; i < periodNum; i++) {
                ogGatePower[i] = microgrid.getUsers().get(userId).getGatePowers()[i];
                parkGatePower[i] += purP[i];
            }
            origGatePowers.put(userId, ogGatePower);
        }
        // 关口功率指令
        int[] peakShaveTime = new int[periodNum];
        for (int i = 64; i < 68; i++) {
            peakShaveTime[i] = 1;
        }
        demandRespModel.setPeakShaveTime(peakShaveTime);
        double[] parkPeakShavePower = new double[periodNum];
        for (int i = 0; i < periodNum; i++) {
            if (peakShaveTime[i] == 1) {
                parkPeakShavePower[i] = 1000;
            }
        }
        demandRespModel.calPeakShavePowers(parkGatePower, parkPeakShavePower);   // 应削峰量
        System.out.println("---------各用户应削峰量---------");
        Map<String, double[]> peakShavePowers = demandRespModel.getPeakShavePowers();
        for (String userId : users.keySet()) {
            System.out.print(userId + "\t");
            double[] peakShavePower = peakShavePowers.get(userId);
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    System.out.print(peakShavePower[i] + "\t");
                }
            }
            System.out.println();
        }
        // 通信网络
        UndirectedGraph<String, String> g = new SimpleGraph<>(String.class);
        for (String userId : users.keySet()) {
            g.addVertex(userId);
        }
        g.addEdge("1", "2", "1_2");
        // 初始化边际成本
        Map<String, double[]> mcs = new HashMap<>(users.size());
        for (String userId : users.keySet()) {
            double[] mc = new double[periodNum];
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    mc[i] = 0.6;
                }
            }
            mcs.put(userId, mc);
        }
        demandRespModel.setMcs(mcs);
        demandRespModel.mgDistIDR();
        // 开始迭代
        Map<String, double[]> lastMcs = new HashMap<>();
        Map<String, double[]> lastPeakShaveCaps = new HashMap<>();
        Map<String, double[]> peakShaveCaps;
        double e1 = 0.001;
        double e2 = 1;
        Map<String, Double> error1s = new HashMap<>(users.size());
        Map<String, Double> error2s = new HashMap<>(users.size());
        double maxError1 = Double.MAX_VALUE;
        double maxError2 = Double.MAX_VALUE;
        Map<String, List<Double>> error1Map = new HashMap<>(users.size());
        Map<String, List<Double>> error2Map = new HashMap<>(users.size());
        for (String userId : users.keySet()) {
            error1Map.put(userId, new LinkedList<>());
            error2Map.put(userId, new LinkedList<>());
        }
        // 用户1边际成本变化
        List<List<Double>> mc1 = new ArrayList<>(4);
        List<List<Double>> peakShaveCap1 = new ArrayList<>(4);
        for (int i = 0; i < 4; i++) {
            mc1.add(new LinkedList<>());
            peakShaveCap1.add(new LinkedList<>());
        }
        // 用户2边际成本变化
        List<List<Double>> mc2 = new ArrayList<>(4);
        List<List<Double>> peakShaveCap2 = new ArrayList<>(4);
        for (int i = 0; i < 4; i++) {
            mc2.add(new LinkedList<>());
            peakShaveCap2.add(new LinkedList<>());
        }
        int iterNum = 1;
        double[][] p = {{250, 250, 250, 250},{750, 750, 750, 750}};
        while (maxError1 > e1 || maxError2 > e2) {
            double w1 = 0.056 / pow(iterNum, 0.001);
            double w2 = 0.75 * 1e-4 / pow(iterNum, 0.85);
            // 更新边际成本
            for (String userId : users.keySet()) {
                double[] lastMc = new double[periodNum];
                double[] lastPeakShaveCap = new double[periodNum];
                double[] mc = mcs.get(userId);
                double[] peakShaveCap = demandRespModel.getPeakShaveCaps().get(userId);
                int count = 0;
                for (int i = 0; i < periodNum; i++) {
                    if (peakShaveTime[i] == 1) {
                        lastMc[i] = mc[i];
                        lastPeakShaveCap[i] = peakShaveCap[i];

                        switch (userId) {
                            case "1":
                                mc1.get(count).add(mc[i]);
                                peakShaveCap1.get(count).add(peakShaveCap[i]);
                                count++;
                                break;
                            case "2":
                                mc2.get(count).add(mc[i]);
                                peakShaveCap2.get(count).add(peakShaveCap[i]);
                                count++;
                                break;
                        }
                    }
                }
                lastMcs.put(userId, lastMc);
                lastPeakShaveCaps.put(userId, lastPeakShaveCap);
            }
            int count = 0;
            for (String userId : users.keySet()) {
                double[] lastMc = lastMcs.get(userId);
                double[] lastPeakShaveCap = lastPeakShaveCaps.get(userId);
                double[] mc = new double[periodNum];
                for (int i = 0; i < periodNum; i++) {
                    if (peakShaveTime[i] == 1) {
                        mc[i] = lastMc[i];
                        for (String e : g.edgesOf(userId)) {
                            String adjNode = g.getEdgeTarget(e);
                            if (adjNode.equals(userId)) {
                                adjNode = g.getEdgeSource(e);
                            }
                            double[] adjMc = lastMcs.get(adjNode);
                            mc[i] -= w1 * (lastMc[i] - adjMc[i]);
                        }
//                        mc[i] -= w2 * (lastPeakShaveCap[i] - parkPeakShavePower[i] / users.size());
                        mc[i] -= w2 * (lastPeakShaveCap[i] - p[count][i - 64]);
                    }
                }
                mcs.put(userId, mc);
                count++;
            }
            demandRespModel.setMcs(mcs);
            // 更新IDR容量
            demandRespModel.mgDistIDR();

            peakShaveCaps = demandRespModel.getPeakShaveCaps();
            for (String userId : users.keySet()) {
                double[] lastMc = lastMcs.get(userId);
                double[] lastPeakShaveCap = lastPeakShaveCaps.get(userId);
                double[] mc = mcs.get(userId);
                double[] peakShaveCap = peakShaveCaps.get(userId);
                double error1 = 0;
                double error2 = 0;
                for (int i = 0; i < periodNum; i++) {
                    if (peakShaveTime[i] == 1) {
                        error1 += abs(mc[i] - lastMc[i]);
                        error2 += abs(peakShaveCap[i] - lastPeakShaveCap[i]);
                    }
                }
                error1s.put(userId, error1);
                error2s.put(userId, error2);
                error1Map.get(userId).add(error1);
                error2Map.get(userId).add(error2);
            }
            maxError1 = 0;
            maxError2 = 0;
            for (String userId : users.keySet()) {
                if (maxError1 < error1s.get(userId)) {
                    maxError1 = error1s.get(userId);
                }
                if (maxError2 < error2s.get(userId)) {
                    maxError2 = error2s.get(userId);
                }
            }
            if (iterNum > 51) {
                break;
            }
            iterNum++;
        }
        iterNum--;
        System.out.println("---------迭代计算结束\t" + "迭代" + iterNum + "次---------");
        for (String userId : users.keySet()) {
            List<Double> error1 = error1Map.get(userId);
            List<Double> error2 = error2Map.get(userId);
            System.out.println("边际成本迭代总变化：");
            for (Double error : error1) {
                System.out.printf("%f\t", error);
            }
            System.out.println();
            System.out.println("IDR容量迭代总变化：");
            for (Double error : error2) {
                System.out.printf("%f\t", error);
            }
            System.out.println();
        }
        System.out.println("---------用户1边际成本变化---------");
        for (int i = 0; i < 4; i++) {
            List<Double> mc = mc1.get(i);
            for (double c : mc) {
                System.out.printf("%f,", c);
            }
            System.out.println();
        }
        System.out.println("---------用户1的IDR容量变化---------");
        for (int i = 0; i < 4; i++) {
            List<Double> peakShaveCap = peakShaveCap1.get(i);
            for (double cap : peakShaveCap) {
                System.out.printf("%f,", cap);
            }
            System.out.println();
        }
        System.out.println("---------用户2边际成本变化---------");
        for (int i = 0; i < 4; i++) {
            List<Double> mc = mc2.get(i);
            for (double c : mc) {
                System.out.printf("%f,", c);
            }
            System.out.println();
        }
        System.out.println("---------用户2的IDR容量变化---------");
        for (int i = 0; i < 4; i++) {
            List<Double> peakShaveCap = peakShaveCap2.get(i);
            for (double cap : peakShaveCap) {
                System.out.printf("%f,", cap);
            }
            System.out.println();
        }
        System.out.println("---------用户IDR结果---------");
        System.out.println("---------IDR边际成本和容量---------");
        double[] caps = new double[4];
        for (String userId : users.keySet()) {
            double[] mc = mcs.get(userId);
            double[] peakShaveCap = demandRespModel.getPeakShaveCaps().get(userId);
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    System.out.printf("%f\t", mc[i]);
                }
            }
            System.out.printf("IDR容量：\t");
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    System.out.printf("%f\t", peakShaveCap[i]);
                    caps[i - 64] += peakShaveCap[i];
                }
            }
            System.out.println();
        }
        System.out.printf("IDR总容量：\t");
        for (int i = 0; i < 4; i++) {
            System.out.printf("%f\t", caps[i]);
        }
        System.out.println();
        Map<String, UserResult> microgridResult = demandRespModel.getMicrogridResult();
        for (String userId : users.keySet()) {
            UserResult userResult = microgridResult.get(userId);
            System.out.println(userId + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userId + "Result_DR.csv", userResult, microgrid.getUsers().get("1").getAcLoad(), microgrid.getUsers().get("1").getPhotovoltaic().getPower());
            }
        }
        System.out.println("---------分布式IDR计算结束---------");
    }

    public void readUserData(InputStream inputStream, User user) throws IOException {
        double[] acLoad = new double[periodNum];
        double[] dcLoad = new double[periodNum];
        double[] pvPowers = new double[periodNum];
        double[] coolingLoad1 = new double[periodNum];
        double[] coolingLoad2 = new double[periodNum];
        double[] heatLoad = new double[periodNum];
        double[] Lsteams = new double[periodNum];
        double[] gatePowers = new double[periodNum];
        BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
        String data;
        int t = 0;
        br.readLine();
        while ((data = br.readLine()) != null) {
            String[] newdata = data.split(",", 8);
            acLoad[t] = Double.parseDouble(newdata[0]);
            dcLoad[t] = Double.parseDouble(newdata[1]);
            pvPowers[t] = Double.parseDouble(newdata[2]);
            coolingLoad1[t] = Double.parseDouble(newdata[3]);
            coolingLoad2[t] = Double.parseDouble(newdata[4]);
            heatLoad[t] = Double.parseDouble(newdata[5]);
            Lsteams[t] = Double.parseDouble(newdata[6]);
            gatePowers[t] = Double.parseDouble(newdata[7]);
            t += 1;
        }
        user.setSteamLoad(new SteamLoad(Lsteams, 0.2));
        user.setPhotovoltaic(new Photovoltaic(0.0005, pvPowers));
        user.setAcLoad(acLoad);
        user.setDcLoad(dcLoad);
        user.setHeatLoad(heatLoad);
        user.setCoolingLoad1(coolingLoad1);
        user.setCoolingLoad2(coolingLoad2);
        user.setGatePowers(gatePowers);
    }

    public void readUserData2(InputStream inputStream, User user) throws IOException {
        double[] acLoad = new double[periodNum];
        double[] dcLoad = new double[periodNum];
        double[] origPvPowers = new double[3 * periodNum];
        double[] pvPowers = new double[periodNum];
        double[] coolingLoad = new double[periodNum];
        double[] heatLoad = new double[periodNum];
        double[] Lsteams = new double[periodNum];
        double[] gatePowers = new double[periodNum];
        BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
        String data;
        int t = 0;
        br.readLine();
        while ((data = br.readLine()) != null) {
            String[] newdata = data.split(",", 7);
            acLoad[t] = Double.parseDouble(newdata[0]);
            dcLoad[t] = Double.parseDouble(newdata[1]);
            origPvPowers[t] = Double.parseDouble(newdata[2]);
            coolingLoad[t] = Double.parseDouble(newdata[3]);
            heatLoad[t] = Double.parseDouble(newdata[4]);
            Lsteams[t] = Double.parseDouble(newdata[5]);
            gatePowers[t] = Double.parseDouble(newdata[6]);
            t += 1;
            if (t == periodNum) {
                break;
            }
        }
        while ((data = br.readLine()) != null) {
            String[] newdata = data.split(",", 7);
            origPvPowers[t] = Double.parseDouble(newdata[2]);
            t += 1;
        }
        for (int i = 0; i < periodNum; i++) {
            pvPowers[i] = origPvPowers[3 * i];
            acLoad[i] += pvPowers[i];
        }
        user.setSteamLoad(new SteamLoad(Lsteams, 0.2));
        user.setPhotovoltaic(new Photovoltaic(0, pvPowers));
        user.setAcLoad(acLoad);
        user.setDcLoad(dcLoad);
        user.setHeatLoad(heatLoad);
        user.setCoolingLoad(coolingLoad);
        user.setGatePowers(gatePowers);
    }

    public void readEnergyPrice(InputStream inputStream) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
        String data;
        int t = 0;
        br.readLine();
        while ((data = br.readLine()) != null) {
            String[] newdata = data.split(",", 3);
            elecPrices[t] = Double.parseDouble(newdata[0]);
            gasPrices[t] = Double.parseDouble(newdata[1]);
            steamPrices[t] = Double.parseDouble(newdata[2]);
            chargePrices[t] = elecPrices[t] + 0.8;
            t += 1;
        }
    }

    public void writeResult(String filePath, UserResult userResult, double[] acLoads, double[] pvPowers) throws IOException {
        FileOutputStream out;
        OutputStreamWriter osw;
        BufferedWriter bw;

        out = new FileOutputStream(new File(filePath));
        osw = new OutputStreamWriter(out);
        bw = new BufferedWriter(osw);

        for (int i = 0; i < userResult.getFrigesP().size(); i++) {
            bw.write("制冷机" + (i + 1) + "耗电功率" + ",");
        }
        for (int i = 0; i < userResult.getIceTanksP().size(); i++) {
            bw.write("蓄冰槽" + (i + 1) + "耗电功率" + ",");
        }
        for (int i = 0; i < userResult.getIceTanksQ().size(); i++) {
            bw.write("蓄冰槽" + (i + 1) + "制冷功率" + ",");
        }
        for (int i = 0; i < userResult.getGasTurbinesState().size(); i++) {
            bw.write("燃气轮机" + (i + 1) + "启停状态" + ",");
        }
        for (int i = 0; i < userResult.getGasTurbinesP().size(); i++) {
            bw.write("燃气轮机" + (i + 1) + "产电功率" + ",");
        }
        for (int i = 0; i < userResult.getStoragesP().size(); i++) {
            bw.write("储能" + (i + 1) + "充电功率(外部)" + ",");
        }
        for (int i = 0; i < userResult.getConvertersP().size(); i++) {
            bw.write("变流器" + (i + 1) + "AC-DC交流侧功率" + ",");
        }
        bw.write("向电网购电功率" + ",");
        for (int i = 0; i < userResult.getAirConsP().size(); i++) {
            bw.write("中央空调" + (i + 1) + "耗电功率" + ",");
        }
        for (int i = 0; i < userResult.getGasBoilersState().size(); i++) {
            bw.write("燃气锅炉" + (i + 1) + "启停状态" + ",");
        }
        for (int i = 0; i < userResult.getGasBoilersH().size(); i++) {
            bw.write("燃气锅炉" + (i + 1) + "产热状态" + ",");
        }
        for (int i = 0; i < userResult.getAbsorptionChillersH().size(); i++) {
            bw.write("吸收式制冷机" + (i + 1) + "耗热功率" + ",");
        }
        bw.write("向园区购热功率,");
        if (userResult.getChargingPilesState() != null) {
            for (int i = 0; i < userResult.getChargingPilesState().size(); i++) {
                bw.write("充电桩" + (i + 1) + "启停状态" + ",");
            }
            for (int i = 0; i < userResult.getChargingPilesP().size(); i++) {
                bw.write("充电桩" + (i + 1) + "功率" + ",");
            }
        }
        bw.write("负荷,");
        bw.write("光伏出力");
        bw.newLine();

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
            bw.write(userResult.getPurH()[j] + ",");
            if (userResult.getChargingPilesState() != null) {
                for (int i = 0; i < userResult.getChargingPilesState().size(); i++) {
                    bw.write(userResult.getChargingPilesState().get(i)[j] + ",");
                }
                for (int i = 0; i < userResult.getChargingPilesP().size(); i++) {
                    bw.write(userResult.getChargingPilesP().get(i)[j] + ",");
                }
            }
            bw.write(acLoads[j] + "," + pvPowers[j] + ",");
            bw.newLine();
        }
        bw.close();
        osw.close();
        out.close();
    }
}