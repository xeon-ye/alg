package zju.hems;

import junit.framework.TestCase;
import org.jgrapht.UndirectedGraph;
import org.jgrapht.graph.SimpleGraph;

import java.io.*;
import java.util.*;

import static java.lang.Math.abs;
import static java.lang.Math.pow;

public class SelfOptModelTest  extends TestCase {

    int periodNum = 96; // 一天的时段数
    double[] elecPrices = new double[periodNum];    // 电价
    double[] gasPrices = new double[periodNum];    // 天然气价格
    double[] steamPrices = new double[periodNum];    // 园区CHP蒸汽价格

    public Microgrid microgridModel() throws IOException {
        Map<String, User> users = new HashMap<>();
        InputStream inputStream;
        // 用户1
        List<AbsorptionChiller> absorptionChillers = new ArrayList<>(2);
        for (int i = 0; i < 2; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
            absorptionChillers.add(absorptionChiller);
        }
        List<AirCon> airCons = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons.add(airCon);
        }
        List<Converter> converters = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters.add(converter);
        }
        List<GasBoiler> gasBoilers = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500, 0);
            gasBoilers.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 0.3, 200, 50, 1000, -500, 500, 0);
            gasTurbines.add(gasTurbine);
        }
        List<IceStorageAc> iceStorageAcs = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
            iceStorageAcs.add(iceStorageAc);
        }
        List<Storage> storages = new ArrayList<>(3);
        for (int i = 0; i < 3; i++) {
            Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
            storages.add(storage);
        }
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, storages, 4500);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/selfopt/input_user1.csv");
        readUserData(inputStream, user);
        users.put(user.getUserId(), user);

        // 用户2
        List<AbsorptionChiller> absorptionChillers2 = new ArrayList<>(4);
        for (int i = 0; i < 3; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
            absorptionChillers2.add(absorptionChiller);
        }
        List<AirCon> airCons2 = new ArrayList<>(2);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons2.add(airCon);
        }
        List<Converter> converters2 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters2.add(converter);
        }
        List<GasBoiler> gasBoilers2 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500, 0);
            gasBoilers2.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines2 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 0.3, 200, 50, 1000, -500, 500, 0);
            gasTurbines2.add(gasTurbine);
        }
        List<IceStorageAc> iceStorageAcs2 = new ArrayList<>(2);
        for (int i = 0; i < 2; i++) {
            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
            iceStorageAcs2.add(iceStorageAc);
        }
        List<Storage> storages2 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
            storages2.add(storage);
        }
        User user2 = new User("2", absorptionChillers2, airCons2, converters2, gasBoilers2, gasTurbines2, iceStorageAcs2, storages2, 2100);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/selfopt/input_user2.csv");
        readUserData(inputStream, user2);
        users.put(user2.getUserId(), user2);

        // 用户3
        List<AbsorptionChiller> absorptionChillers3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
            absorptionChillers3.add(absorptionChiller);
        }
        List<AirCon> airCons3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons3.add(airCon);
        }
        List<Converter> converters3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters3.add(converter);
        }
        List<GasBoiler> gasBoilers3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500, 0);
            gasBoilers3.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 0.3, 200, 50, 1000, -500, 500, 0);
            gasTurbines3.add(gasTurbine);
        }
        List<IceStorageAc> iceStorageAcs3 = new ArrayList<>(1);
        List<Storage> storages3 = new ArrayList<>(1);
        User user3 = new User("3", absorptionChillers3, airCons3, converters3, gasBoilers3, gasTurbines3, iceStorageAcs3, storages3, 1600);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/selfopt/input_user3.csv");
        readUserData(inputStream, user3);
        users.put(user3.getUserId(), user3);

        // 用户4
        List<AbsorptionChiller> absorptionChillers4 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
            absorptionChillers4.add(absorptionChiller);
        }
        List<AirCon> airCons4 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons4.add(airCon);
        }
        List<Converter> converters4 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters4.add(converter);
        }
        List<GasBoiler> gasBoilers4 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500, 0);
            gasBoilers4.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines4 = new ArrayList<>(1);
        List<IceStorageAc> iceStorageAcs4 = new ArrayList<>(1);
        List<Storage> storages4 = new ArrayList<>(3);
        User user4 = new User("4", absorptionChillers4, airCons4, converters4, gasBoilers4, gasTurbines4, iceStorageAcs4, storages4, 1800);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/selfopt/input_user4.csv");
        readUserData(inputStream, user4);
        users.put(user4.getUserId(), user4);

        // 用户5
        List<AbsorptionChiller> absorptionChillers5 = new ArrayList<>(1);
        List<AirCon> airCons5 = new ArrayList<>(1);
        List<Converter> converters5 = new ArrayList<>(1);
        List<GasBoiler> gasBoilers5 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 100, 0.85, 0, 1000, 500, 0);
            gasBoilers5.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines5 = new ArrayList<>(1);
        List<IceStorageAc> iceStorageAcs5 = new ArrayList<>(1);
        List<Storage> storages5 = new ArrayList<>(1);
        User user5 = new User("5", absorptionChillers5, airCons5, converters5, gasBoilers5, gasTurbines5, iceStorageAcs5, storages5, 3800);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/selfopt/input_user5.csv");
        readUserData(inputStream, user5);
        users.put(user5.getUserId(), user5);

        inputStream = this.getClass().getResourceAsStream("/iesfiles/selfopt/energy_price.csv");
        readEnergyPrice(inputStream);

        return new Microgrid(users);
    }

    public Microgrid mgModelPes() throws IOException {
        Map<String, User> users = new HashMap<>();
        InputStream inputStream;
        // 用户1
        List<AbsorptionChiller> absorptionChillers = new ArrayList<>(0);
        List<AirCon> airCons = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons.add(airCon);
        }
        List<Converter> converters = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(1, 1);
            converters.add(converter);
        }
        List<GasBoiler> gasBoilers = new ArrayList<>(0);
        List<GasTurbine> gasTurbines = new ArrayList<>(0);
        List<IceStorageAc> iceStorageAcs = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
            iceStorageAcs.add(iceStorageAc);
        }
        List<Storage> storages = new ArrayList<>(3);
        for (int i = 0; i < 3; i++) {
            Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
            storages.add(storage);
        }
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, storages, 4500);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/IDR/input_user1.csv");
        readUserData(inputStream, user);
        users.put(user.getUserId(), user);

        // 用户2
        List<AbsorptionChiller> absorptionChillers2 = new ArrayList<>(0);
        List<AirCon> airCons2 = new ArrayList<>(2);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons2.add(airCon);
        }
        List<Converter> converters2 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters2.add(converter);
        }
        List<GasBoiler> gasBoilers2 = new ArrayList<>(0);
        List<GasTurbine> gasTurbines2 = new ArrayList<>(0);
        List<IceStorageAc> iceStorageAcs2 = new ArrayList<>(2);
        for (int i = 0; i < 2; i++) {
            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
            iceStorageAcs2.add(iceStorageAc);
        }
        List<Storage> storages2 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
            storages2.add(storage);
        }
        User user2 = new User("2", absorptionChillers2, airCons2, converters2, gasBoilers2, gasTurbines2, iceStorageAcs2, storages2, 2100);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/IDR/input_user2.csv");
        readUserData(inputStream, user2);
        users.put(user2.getUserId(), user2);

        // 用户3
        List<AbsorptionChiller> absorptionChillers3 = new ArrayList<>(0);
        List<AirCon> airCons3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons3.add(airCon);
        }
        List<Converter> converters3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters3.add(converter);
        }
        List<GasBoiler> gasBoilers3 = new ArrayList<>(0);
        List<GasTurbine> gasTurbines3 = new ArrayList<>(0);
        List<IceStorageAc> iceStorageAcs3 = new ArrayList<>(0);
        List<Storage> storages3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
            storages3.add(storage);
        }
        User user3 = new User("3", absorptionChillers3, airCons3, converters3, gasBoilers3, gasTurbines3, iceStorageAcs3, storages3, 1600);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/IDR/input_user3.csv");
        readUserData(inputStream, user3);
        users.put(user3.getUserId(), user3);

        // 用户4
        List<AbsorptionChiller> absorptionChillers4 = new ArrayList<>(0);
        List<AirCon> airCons4 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons4.add(airCon);
        }
        List<Converter> converters4 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters4.add(converter);
        }
        List<GasBoiler> gasBoilers4 = new ArrayList<>(0);
        List<GasTurbine> gasTurbines4 = new ArrayList<>(0);
        List<IceStorageAc> iceStorageAcs4 = new ArrayList<>(0);
        List<Storage> storages4 = new ArrayList<>(0);
        User user4 = new User("4", absorptionChillers4, airCons4, converters4, gasBoilers4, gasTurbines4, iceStorageAcs4, storages4, 1800);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/IDR/input_user4.csv");
        readUserData(inputStream, user4);
        users.put(user4.getUserId(), user4);

        inputStream = this.getClass().getResourceAsStream("/iesfiles/IDR/energy_price.csv");
        readEnergyPrice(inputStream);

        return new Microgrid(users);
    }

    public Microgrid distIDRModel() throws IOException {
        Map<String, User> users = new HashMap<>();
        InputStream inputStream;
        // 用户1
        List<AbsorptionChiller> absorptionChillers = new ArrayList<>(2);
        for (int i = 0; i < 2; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
            absorptionChillers.add(absorptionChiller);
        }
        List<AirCon> airCons = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons.add(airCon);
        }
        List<Converter> converters = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters.add(converter);
        }
        List<GasBoiler> gasBoilers = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 0, 0.85, 0, 1000, 500, 0);
            gasBoilers.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 0.3, 0, 50, 1000, -500, 500, 0);
            gasTurbines.add(gasTurbine);
        }
        List<IceStorageAc> iceStorageAcs = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
            iceStorageAcs.add(iceStorageAc);
        }
        List<Storage> storages = new ArrayList<>(3);
        for (int i = 0; i < 3; i++) {
            Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
            storages.add(storage);
        }
        User user = new User("1", absorptionChillers, airCons, converters, gasBoilers, gasTurbines, iceStorageAcs, storages, 4500);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/CIIDR/input_user1.csv");
        readUserData(inputStream, user);
        users.put(user.getUserId(), user);

        // 用户2
        List<AbsorptionChiller> absorptionChillers2 = new ArrayList<>(4);
        for (int i = 0; i < 3; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
            absorptionChillers2.add(absorptionChiller);
        }
        List<AirCon> airCons2 = new ArrayList<>(2);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons2.add(airCon);
        }
        List<Converter> converters2 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters2.add(converter);
        }
        List<GasBoiler> gasBoilers2 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 0, 0.85, 0, 1000, 500, 0);
            gasBoilers2.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines2 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 0.3, 0, 50, 1000, -500, 500, 0);
            gasTurbines2.add(gasTurbine);
        }
        List<IceStorageAc> iceStorageAcs2 = new ArrayList<>(2);
        for (int i = 0; i < 2; i++) {
            IceStorageAc iceStorageAc = new IceStorageAc(0.01, 1, 3, 3, 0.9, 1,
                    0.002, 500, 3000, 0.1, 0.95, 0.1, 1.00, 500, 500);
            iceStorageAcs2.add(iceStorageAc);
        }
        List<Storage> storages2 = new ArrayList<>(2);
        for (int i = 0; i < 2; i++) {
            Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
            storages2.add(storage);
        }
        User user2 = new User("2", absorptionChillers2, airCons2, converters2, gasBoilers2, gasTurbines2, iceStorageAcs2, storages2, 2100);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/CIIDR/input_user2.csv");
        readUserData(inputStream, user2);
        users.put(user2.getUserId(), user2);

        // 用户3
        List<AbsorptionChiller> absorptionChillers3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
            absorptionChillers3.add(absorptionChiller);
        }
        List<AirCon> airCons3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons3.add(airCon);
        }
        List<Converter> converters3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters3.add(converter);
        }
        List<GasBoiler> gasBoilers3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 0, 0.85, 0, 1000, 500, 0);
            gasBoilers3.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines3 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasTurbine gasTurbine = new GasTurbine(0.063, 0.33, 0.6, 0.3, 0, 50, 1000, -500, 500, 0);
            gasTurbines3.add(gasTurbine);
        }
        List<IceStorageAc> iceStorageAcs3 = new ArrayList<>(1);
        List<Storage> storages3 = new ArrayList<>(2);
        for (int i = 0; i < 2; i++) {
            Storage storage = new Storage(0.005, 0.00075, 1250, 1250, 13000, 0.1, 0.9, 0.1, 0.5, 0.5, 0.0025, 0.95, 0.95);
            storages3.add(storage);
        }
        User user3 = new User("3", absorptionChillers3, airCons3, converters3, gasBoilers3, gasTurbines3, iceStorageAcs3, storages3, 1600);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/CIIDR/input_user3.csv");
        readUserData(inputStream, user3);
        users.put(user3.getUserId(), user3);

        // 用户4
        List<AbsorptionChiller> absorptionChillers4 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AbsorptionChiller absorptionChiller = new AbsorptionChiller(0.00008, 0, 500, 0.8);
            absorptionChillers4.add(absorptionChiller);
        }
        List<AirCon> airCons4 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            AirCon airCon = new AirCon(0.0097, 1, 1.00, 0, 500, 4.3);
            airCons4.add(airCon);
        }
        List<Converter> converters4 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            Converter converter = new Converter(0.95, 0.95);
            converters4.add(converter);
        }
        List<GasBoiler> gasBoilers4 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 0, 0.85, 0, 1000, 500, 0);
            gasBoilers4.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines4 = new ArrayList<>(1);
        List<IceStorageAc> iceStorageAcs4 = new ArrayList<>(1);
        List<Storage> storages4 = new ArrayList<>(3);
        User user4 = new User("4", absorptionChillers4, airCons4, converters4, gasBoilers4, gasTurbines4, iceStorageAcs4, storages4, 1800);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/CIIDR/input_user4.csv");
        readUserData(inputStream, user4);
        users.put(user4.getUserId(), user4);

        // 用户5
        List<AbsorptionChiller> absorptionChillers5 = new ArrayList<>(1);
        List<AirCon> airCons5 = new ArrayList<>(1);
        List<Converter> converters5 = new ArrayList<>(1);
        List<GasBoiler> gasBoilers5 = new ArrayList<>(1);
        for (int i = 0; i < 1; i++) {
            GasBoiler gasBoiler = new GasBoiler(0.04, 0, 0.85, 0, 1000, 500, 0);
            gasBoilers5.add(gasBoiler);
        }
        List<GasTurbine> gasTurbines5 = new ArrayList<>(1);
        List<IceStorageAc> iceStorageAcs5 = new ArrayList<>(1);
        List<Storage> storages5 = new ArrayList<>(1);
        User user5 = new User("5", absorptionChillers5, airCons5, converters5, gasBoilers5, gasTurbines5, iceStorageAcs5, storages5, 3800);
        inputStream = this.getClass().getResourceAsStream("/iesfiles/CIIDR/input_user5.csv");
        readUserData(inputStream, user5);
        users.put(user5.getUserId(), user5);

        inputStream = this.getClass().getResourceAsStream("/iesfiles/CIIDR/energy_price.csv");
        readEnergyPrice(inputStream);

        return new Microgrid(users);
    }

    public void testSelfOpt() throws IOException {
        Microgrid microgrid = microgridModel();
        SelfOptModel selfOptModel = new SelfOptModel(microgrid, periodNum, elecPrices, gasPrices, steamPrices);
        selfOptModel.mgSelfOpt();
        Map<String, UserResult> microgridResult = selfOptModel.getMicrogridResult();
        for (UserResult userResult : microgridResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result.csv", userResult);
            }
        }
    }

    public void testDemandResp() throws IOException {
        Microgrid microgrid = microgridModel();
        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, elecPrices, gasPrices, steamPrices);
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        for (UserResult userResult : selfOptResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result.csv", userResult);
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
        for (int i = 45; i < 49; i++) {
            peakShaveTime[i] = 1;
        }
        demandRespModel.setPeakShaveTime(peakShaveTime);
        Map<String, double[]> insGatePowers = new HashMap<>();
        for (String userId : users.keySet()) {
            double[] insGatePower = new double[periodNum];
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    insGatePower[i] = 1673.9;
                } else {
                    insGatePower[i] = origGatePowers.get(userId)[i];
                }
            }
            insGatePowers.put(userId, insGatePower);
        }
        // 采样点数
        int sampleNum = 10;
        // 采样范围
        double sampleStart = 0.5;
        double sampleEnd = 1;
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
                    writeResult("D:\\user" + userResult.getUserId() + "Result_DR.csv", userResult);
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
        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, elecPrices, gasPrices, steamPrices);
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        for (UserResult userResult : selfOptResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result.csv", userResult);
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
                writeResult("D:\\user" + userId + "Result_DR.csv", userResult);
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
                writeResult("D:\\user" + userId + "Result_DR.csv", userResult);
            }
        }
        System.out.println("---------分布式需求响应计算结束---------");
    }

    public void testDRPes() throws IOException {
        Microgrid microgrid = mgModelPes();
        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, elecPrices, gasPrices, steamPrices);
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        for (UserResult userResult : selfOptResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\pesUser" + userResult.getUserId() + "Result.csv", userResult);
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
        for (int i = 45; i < 49; i++) {
            peakShaveTime[i] = 1;
        }
        demandRespModel.setPeakShaveTime(peakShaveTime);
        Map<String, double[]> insGatePowers = new HashMap<>();
        for (String userId : users.keySet()) {
            double[] insGatePower = new double[periodNum];
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    insGatePower[i] = 1760;
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
                    writeResult("D:\\user" + userResult.getUserId() + "Result_DR.csv", userResult);
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

    public void testCenDistIDRPes() throws IOException {
        Microgrid microgrid = mgModelPes();
        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, elecPrices, gasPrices, steamPrices);
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        for (UserResult userResult : selfOptResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\pesUser" + userResult.getUserId() + "Result.csv", userResult);
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
                writeResult("D:\\pesUser" + userId + "Result_DR.csv", userResult);
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
                writeResult("D:\\pesUser" + userId + "Result_DR.csv", userResult);
            }
        }
        System.out.println("---------分布式需求响应计算结束---------");
    }

    public void testCenIDR() throws IOException {
        Microgrid microgrid = distIDRModel();
        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, elecPrices, gasPrices, steamPrices);
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        for (UserResult userResult : selfOptResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result.csv", userResult);
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
            gatePowerSum[i] = 8000;
        }
        demandRespModel.setPeakShaveTime(peakShaveTime);
        demandRespModel.setGatePowerSum(gatePowerSum);
        demandRespModel.cenIDR();
        for (String userId : demandRespModel.getMicrogridResult().keySet()) {
            UserResult userResult = demandRespModel.getMicrogridResult().get(userId);
            System.out.println(userId + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userId + "Result_DR.csv", userResult);
            }
        }
    }

    public void testDistIDR() throws IOException {
        Microgrid microgrid = distIDRModel();
        DemandRespModel demandRespModel = new DemandRespModel(microgrid, periodNum, elecPrices, gasPrices, steamPrices);
        demandRespModel.mgSelfOpt();
        Map<String, UserResult> selfOptResult = demandRespModel.getMicrogridResult();
        for (UserResult userResult : selfOptResult.values()) {
            System.out.println(userResult.getUserId() + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userResult.getUserId() + "Result.csv", userResult);
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
                parkPeakShavePower[i] = parkGatePower[i] - 8000;
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
                writeResult("D:\\user" + userId + "Result_DR.csv", userResult);
            }
        }
        System.out.println("---------100%需求响应计算结束---------");
        // 通信网络
        UndirectedGraph<String, String> g = new SimpleGraph<>(String.class);
        for (String userId : users.keySet()) {
            g.addVertex(userId);
        }
        g.addEdge("1", "2", "1_2");
        g.addEdge("1", "3", "1_3");
        g.addEdge("2", "3", "2_3");
        g.addEdge("3", "4", "3_4");
        g.addEdge("3", "5", "3_5");
        // 初始化边际成本
        Map<String, double[]> mcs = new HashMap<>(users.size());
        for (String userId : users.keySet()) {
            double[] mc = new double[periodNum];
            for (int i = 0; i < periodNum; i++) {
                if (peakShaveTime[i] == 1) {
                    mc[i] = 0.54;
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
        double e1 = 0.0005;
        double e2 = 0.01;
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
        int iterNum = 1;
        while (maxError1 > e1 || maxError2 > e2) {
            //todo
            double w1 = 0.3 / pow(iterNum, 0.01);
            double w2 = 1 * 1e-4 / pow(iterNum, 0.95);
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

                        if (userId.equals("1")) {
                            mc1.get(count).add(mc[i]);
                            peakShaveCap1.get(count).add(peakShaveCap[i]);
                            count++;
                        }
                    }
                }
                lastMcs.put(userId, lastMc);
                lastPeakShaveCaps.put(userId, lastPeakShaveCap);
            }
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
                        //todo 减总削峰量还是应削峰量？
                        mc[i] -= w2 * (lastPeakShaveCap[i] - parkPeakShavePower[i] / users.size());
                    }
                }
                mcs.put(userId, mc);
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
            if (iterNum > 100) {
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
                System.out.printf("%f\t", c);
            }
            System.out.println();
        }
        System.out.println("---------用户1的IDR容量变化---------");
        for (int i = 0; i < 4; i++) {
            List<Double> peakShaveCap = peakShaveCap1.get(i);
            for (double cap : peakShaveCap) {
                System.out.printf("%f\t", cap);
            }
            System.out.println();
        }
        System.out.println("---------用户IDR结果---------");
        System.out.println("---------IDR边际成本和容量---------");
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
                }
            }
            System.out.println();
        }
        Map<String, UserResult> microgridResult = demandRespModel.getMicrogridResult();
        for (String userId : users.keySet()) {
            UserResult userResult = microgridResult.get(userId);
            System.out.println(userId + "\t" + userResult.getStatus());
            if (userResult.getStatus().equals("Optimal")) {
                System.out.println(userResult.getMinCost());
                writeResult("D:\\user" + userId + "Result_DR.csv", userResult);
            }
        }
        System.out.println("---------分布式IDR计算结束---------");
    }

    public void readUserData(InputStream inputStream, User user) throws IOException {
        double[] acLoad = new double[periodNum];
        double[] dcLoad = new double[periodNum];
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
            String[] newdata = data.split(",", 8);
            acLoad[t] = Double.parseDouble(newdata[0]);
            dcLoad[t] = Double.parseDouble(newdata[1]);
            pvPowers[t] = Double.parseDouble(newdata[2]);
            coolingLoad[t] = Double.parseDouble(newdata[3]);
            heatLoad[t] = Double.parseDouble(newdata[4]);
            Lsteams[t] = Double.parseDouble(newdata[5]);
            gatePowers[t] = Double.parseDouble(newdata[6]);
            t += 1;
        }
        user.setSteamLoad(new SteamLoad(Lsteams, 0.2));
        user.setPhotovoltaic(new Photovoltaic(0.0005, pvPowers));
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
        bw.write("向园区购热功率");
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
            bw.write(String.valueOf(userResult.getPurH()[j]));
            bw.newLine();
        }
        bw.close();
        osw.close();
        out.close();
    }
}