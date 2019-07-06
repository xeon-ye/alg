package zju.bpamodel;

import zju.bpamodel.pf.*;

import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import static java.lang.Math.abs;

public class BpaPfModelRw {

    public static void CreateTables(String dbFile) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        String TABLE_DATA_NAME = "PowerExchange";
        String initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " type     varchar(3) NOT NULL," +
                " chgCode     varchar(1) NULL," +
                " areaName     varchar(10) NULL," +
                " areaBusName     varchar(8) NULL," +
                " areaBusBaseKv              decimal(4,0) NULL, " +
                " exchangePower              decimal(8,0) NULL, " +
                " zoneName     varchar(60) NULL," +
                " area1Name     varchar(10) NULL," +
                " area2Name     varchar(10) NULL" +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "Bus";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " type     varchar(3) NOT NULL," +
                " chgCode     varchar(1) NULL," +
                " owner     varchar(3) NULL," +
                " name     varchar(8) NOT NULL," +
                " baseKv              decimal(4,1) NOT NULL, " +
                " zoneName     varchar(2) NULL," +
                " loadMw              decimal(5,4) NULL, " +
                " loadMvar           decimal(5,4)     NULL, " +
                " shuntMw           decimal(4,3)     NULL, " +
                " shuntMvar              decimal(4,3) NULL, " +
                " genMwMax              decimal(4,3) NULL, " +
                " genMw              decimal(5,4) NULL, " +
                " genMvarShed              decimal(5,4) NULL, " +
                " genMvarMax              decimal(5,4) NULL, " +
                " genMvarMin              decimal(5,4) NULL, " +
                " vAmplMax              decimal(4,3) NULL, " +
                " vAmplDesired              decimal(4,3) NULL, " +
                " vAmplMin              decimal(4,3) NULL, " +
                " slackBusVAngle              decimal(4,3) NULL, " +
                " remoteCtrlBusName     varchar(8) NOT NULL," +
                " remoteCtrlBusBaseKv              decimal(4,3) NULL, " +
                " genMvarPercent              decimal(3,2) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "AcLine";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " type     varchar(3) NOT NULL," +
                " chgCode     varchar(1) NULL," +
                " owner     varchar(3) NULL," +
                " linkMeterCode              INTEGER NULL, " +
                " busName1     varchar(8) NOT NULL," +
                " busName2     varchar(8) NOT NULL," +
                " baseKv1              decimal(4,3) NULL, " +
                " baseKv2              decimal(4,3) NULL, " +
                " circuit     varchar(1) DEFAULT NULL," +
                " baseI              decimal(4,3) NULL, " +
                " shuntLineNum              INTEGER NULL, " +
                " r           decimal(6,5)     NULL, " +
                " x           decimal(6,5)     NULL, " +
                " halfG              decimal(6,5) NULL, " +
                " halfB              decimal(6,5) NULL, " +
                " length              decimal(4,1) NULL, " +
                " desc     varchar(8) NULL," +
                " onlineDate     varchar(3) NULL," +
                " offlineDate     varchar(3) NULL" +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "Transformer";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " type     varchar(3) NOT NULL," +
                " chgCode     varchar(1) NULL," +
                " owner     varchar(3) NULL," +
                " busName1     varchar(8) NOT NULL," +
                " busName2     varchar(8) NOT NULL," +
                " baseKv1              decimal(4,3) NULL, " +
                " baseKv2              decimal(4,3) NULL, " +
                " circuit     varchar(1) DEFAULT NULL," +
                " baseMva              decimal(4,3) NULL, " +
                " linkMeterCode              INTEGER NULL, " +
                " shuntTransformerNum              INTEGER NULL, " +
                " r           decimal(6,5)     NULL, " +
                " x           decimal(6,5)     NULL, " +
                " g              decimal(6,5) NULL, " +
                " b              decimal(6,5) NULL, " +
                " tapKv1              decimal(5,2) NULL, " +
                " tapKv2              decimal(5,2) NULL, " +
                " phaseAngle              decimal(5,2) NULL, " +
                " onlineDate     varchar(3) NULL," +
                " offlineDate     varchar(3) NULL," +
                " desc     varchar(8) NULL" +
                ")";
        sqliteDb.initDb(initSql);
    }

    public static void parseAndSave(String filePath, String dbFile) {
        parseAndSave(new File(filePath), dbFile);
    }

    public static void parseAndSave(File file, String dbFile) {
        try {
            parseAndSave(new FileInputStream(file), dbFile);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static void parseAndSave(InputStream in, String dbFile) {
        ElectricIsland model = BpaPfModelParser.parse(in, "GBK");
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<String> sqls = new LinkedList<>();
        String TABLE_DATA_NAME = "PowerExchange";
        for (PowerExchange powerExchange : model.getPowerExchanges()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + powerExchange.getType() + powerExchange.getSubType() + "'," +
                    "'" + powerExchange.getChgCode() + "','" + powerExchange.getAreaName() + "'," +
                    "'" + powerExchange.getAreaBusName() + "'," + powerExchange.getAreaBusBaseKv() + "," +
                    powerExchange.getExchangePower() + ",'" + powerExchange.getZoneName() + "'," +
                    "'" + powerExchange.getArea1Name() + "','" + powerExchange.getArea2Name() + "'" +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "Bus";
        for (Bus bus : model.getBuses()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'B" + bus.getSubType() + "'," +
                    "'" + bus.getChgCode() + "','" + bus.getOwner() + "'," +
                    "'" + bus.getName() + "'," + bus.getBaseKv() + "," +
                    "'" + bus.getZoneName() + "'," + bus.getLoadMw() + "," +
                    bus.getLoadMvar() + "," + bus.getShuntMw() + "," +
                    bus.getShuntMvar() + "," + bus.getGenMwMax() + "," +
                    bus.getGenMw() + "," + bus.getGenMvarShed() + "," +
                    bus.getGenMvarMax() + "," + bus.getGenMvarMin() + "," +
                    bus.getvAmplMax() + "," + bus.getvAmplDesired() + "," +
                    bus.getvAmplMin() + "," + bus.getSlackBusVAngle() + "," +
                    "'" + bus.getRemoteCtrlBusName() + "'," + bus.getRemoteCtrlBusBaseKv() + "," +
                    bus.getGenMvarPercent() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "AcLine";
        for (AcLine acLine : model.getAclines()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'L','" + acLine.getChgCode() + "','" + acLine.getOwner() + "'," +
                    acLine.getLinkMeterCode() + ",'" + acLine.getBusName1() + "'," +
                    "'" + acLine.getBusName2() + "'," + acLine.getBaseKv1() + "," +
                    acLine.getBaseKv2() + ",'" + acLine.getCircuit() + "'," +
                    acLine.getBaseI() + "," + acLine.getShuntLineNum() + "," +
                    acLine.getR() + "," + acLine.getX() + "," +
                    acLine.getHalfG() + "," + acLine.getHalfB() + "," +
                    acLine.getLength() + ",'" + acLine.getDesc() + "'," +
                    "'" + acLine.getOnlineDate() + "','" + acLine.getOfflineDate() + "'" +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "Transformer";
        for (Transformer transformer : model.getTransformers()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'T" + transformer.getSubType() + "'," +
                    "'" + transformer.getChgCode() + "','" + transformer.getOwner() + "'," +
                    "'" + transformer.getBusName1() + "','" + transformer.getBusName2() + "'," +
                    transformer.getBaseKv1() + "," + transformer.getBaseKv2() + "," +
                    "'" + transformer.getCircuit() + "'," + transformer.getBaseMva() + "," +
                    transformer.getLinkMeterCode() + "," + transformer.getShuntTransformerNum() + "," +
                    transformer.getR() + "," + transformer.getX() + "," +
                    transformer.getG() + "," + transformer.getB() + "," +
                    transformer.getTapKv1() + "," + transformer.getTapKv2() + "," +
                    transformer.getPhaseAngle() + ",'" + transformer.getOnlineDate() + "'," +
                    "'" + transformer.getOfflineDate() + "','" + transformer.getDesc() + "'" +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);
    }

    public static void write(String dbFile, String caseID, String inputPath, String outputPath) {
       write(dbFile, caseID, new File(inputPath), new File(outputPath));
    }

    public static void write(String dbFile, String[] busNames, double[] baseKvs, double[] genMws, String caseID, String inputPath, String outputPath) {
        write(dbFile, busNames, baseKvs, genMws, caseID, new File(inputPath), new File(outputPath));
    }

    public static void write(String dbFile, String caseID, File inputFile, File outputFile) {
        try {
            write(dbFile, caseID, new FileInputStream(inputFile), new FileOutputStream(outputFile));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static void write(String dbFile, String[] busNames, double[] baseKvs, double[] genMws, String caseID, File inputFile, File outputFile) {
        try {
            write(dbFile, busNames, baseKvs, genMws, caseID, new FileInputStream(inputFile), new FileOutputStream(outputFile));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static void write(String dbFile, String caseID,InputStream in, OutputStream out) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        ElectricIsland modifiedModel = new ElectricIsland();
        List<Object> objects = sqliteDb.queryData("PowerExchange");
        List<PowerExchange> powerExchanges = new ArrayList<>(objects.size());
        for (Object object : objects) {
            powerExchanges.add((PowerExchange) object);
        }
        modifiedModel.setPowerExchanges(powerExchanges);
        objects.clear();
        objects = sqliteDb.queryData("Bus");
        List<Bus> buses = new ArrayList<>(objects.size());
        for (Object object : objects) {
            buses.add((Bus) object);
        }
        modifiedModel.setBuses(buses);
        objects.clear();
        objects = sqliteDb.queryData("AcLine");
        List<AcLine> acLines = new ArrayList<>(objects.size());
        for (Object object : objects) {
            acLines.add((AcLine) object);
        }
        modifiedModel.setAclines(acLines);
        objects.clear();
        objects = sqliteDb.queryData("Transformer");
        List<Transformer> transformers = new ArrayList<>(objects.size());
        for (Object object : objects) {
            transformers.add((Transformer) object);
        }
        modifiedModel.setTransformers(transformers);
        objects.clear();

        BpaPfModelWriter.readAndWrite(in, "GBK", out, "GBK", caseID, modifiedModel);
    }

    public static void write(String dbFile, String[] busNames, double[] baseKvs, double[] genMws, String caseID,InputStream in, OutputStream out) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        ElectricIsland modifiedModel = new ElectricIsland();
        List<Object> objects = sqliteDb.queryData("PowerExchange");
        List<PowerExchange> powerExchanges = new ArrayList<>(objects.size());
        for (Object object : objects) {
            powerExchanges.add((PowerExchange) object);
        }
        modifiedModel.setPowerExchanges(powerExchanges);
        objects.clear();
        objects = sqliteDb.queryData("Bus");
        List<Bus> buses = new ArrayList<>(objects.size());
        for (Object object : objects) {
            Bus bus = (Bus) object;
            for (int i = 0; i < busNames.length; i++) {
                if (bus.getName().equals(busNames[i]) && abs(bus.getBaseKv() - baseKvs[i]) < 1e-5) {
                    bus.setGenMw(genMws[i]);
                    break;
                }
            }
            buses.add(bus);
        }
        modifiedModel.setBuses(buses);
        objects.clear();
        objects = sqliteDb.queryData("AcLine");
        List<AcLine> acLines = new ArrayList<>(objects.size());
        for (Object object : objects) {
            acLines.add((AcLine) object);
        }
        modifiedModel.setAclines(acLines);
        objects.clear();
        objects = sqliteDb.queryData("Transformer");
        List<Transformer> transformers = new ArrayList<>(objects.size());
        for (Object object : objects) {
            transformers.add((Transformer) object);
        }
        modifiedModel.setTransformers(transformers);
        objects.clear();

        BpaPfModelWriter.readAndWrite(in, "GBK", out, "GBK", caseID, modifiedModel);
    }
}
