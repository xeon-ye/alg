package zju.bpamodel;

import zju.bpamodel.pf.Bus;
import zju.bpamodel.pf.ElectricIsland;
import zju.bpamodel.pf.PowerExchange;
import zju.bpamodel.swi.*;

import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

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
                " bus1Name     varchar(8) NOT NULL," +
                " bus2Name     varchar(8) NOT NULL," +
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
                " bus1Name     varchar(8) NOT NULL," +
                " bus2Name     varchar(8) NOT NULL," +
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
                    powerExchange.getExchangePower() + ",'" + powerExchange.getZoneName() + "'" +
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

        //todo:
        sqls.clear();
        TABLE_DATA_NAME = "AcLine";
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
    }

    public static void write(String dbFile, String inputPath, String outputPath) {
       write(dbFile, new File(inputPath), new File(outputPath));
    }

    public static void write(String dbFile, File inputFile, File outputFile) {
        try {
            write(dbFile, new FileInputStream(inputFile), new FileOutputStream(outputFile));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static void write(String dbFile, InputStream in, OutputStream out) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        BpaSwiModel modifiedModel = new BpaSwiModel();
        List<Object> objects = sqliteDb.queryData("Generator");
        modifiedModel.generators = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.generators.add((Generator) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("GeneratorDW");
        modifiedModel.generatorDws = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.generatorDws.add((GeneratorDW) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("Exciter");
        modifiedModel.exciters = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.exciters.add((Exciter) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("ExciterExtraInfo");
        modifiedModel.exciterExtraInfos = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.exciterExtraInfos.add((ExciterExtraInfo) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("PSS");
        modifiedModel.pssList = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.pssList.add((PSS) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("PSSExtraInfo");
        modifiedModel.pssExtraInfos = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.pssExtraInfos.add((PSSExtraInfo) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("PrimeMover");
        modifiedModel.primeMovers = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.primeMovers.add((PrimeMover) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("Governor");
        modifiedModel.governors = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.governors.add((Governor) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("GovernorExtraInfo");
        modifiedModel.governorExtraInfos = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.governorExtraInfos.add((GovernorExtraInfo) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("PV");
        modifiedModel.pvs = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.pvs.add((PV) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("BC");
        modifiedModel.bcs = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.bcs.add((BC) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("BCExtraInfo");
        modifiedModel.bcExtraInfos = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.bcExtraInfos.add((BCExtraInfo) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("Servo");
        modifiedModel.servos = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.servos.add((Servo) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("Load");
        modifiedModel.loads = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.loads.add((Load) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("ShortCircuitFault");
        modifiedModel.shortCircuitFaults = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.shortCircuitFaults.add((ShortCircuitFault) object);
        }
        objects.clear();
        objects = sqliteDb.queryData("FLTCard");
        modifiedModel.fltCards = new ArrayList<>(objects.size());
        for (Object object : objects) {
            modifiedModel.fltCards.add((FLTCard) object);
        }
        objects.clear();
        modifiedModel.ff = (FFCard) sqliteDb.queryData("FFCard").get(0);

        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
    }
}
