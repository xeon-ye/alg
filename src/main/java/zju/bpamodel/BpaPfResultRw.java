package zju.bpamodel;

import zju.bpamodel.pf.*;
import zju.bpamodel.pfr.BranchPfResult;
import zju.bpamodel.pfr.BusPfResult;
import zju.bpamodel.pfr.PfResult;
import zju.bpamodel.pfr.TransformerPfResult;

import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class BpaPfResultRw {

    public static void CreateTables(String dbFile) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        String TABLE_DATA_NAME = "BusPfResult";
        String initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " name     varchar(8) NOT NULL," +
                " vInKv              decimal(7,4) NULL, " +
                " angleInDegree           decimal(6,3)     NULL, " +
                " genP           decimal(7,4)     NULL, " +
                " genQ              decimal(7,4) NULL, " +
                " loadP              decimal(7,4) NULL, " +
                " loadQ              decimal(7,4) NULL, " +
                " isVoltageLimit              INTEGER NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "BranchPfResult";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " busName1     varchar(8) NOT NULL," +
                " busName2     varchar(8) NULL," +
                " branchP              decimal(7,4) NULL, " +
                " branchQ          decimal(7,4)     NULL, " +
                " branchPLoss           decimal(7,4)     NULL, " +
                " branchQLoss              decimal(7,4) NULL, " +
                " isOverLoad     INTEGER NULL" +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "TransformerPfResult";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " busName1     varchar(8) NOT NULL," +
                " busName2     varchar(8) NULL," +
                " baseKv1              decimal(4,3) NULL, " +
                " baseKv2              decimal(4,3) NULL, " +
                " transformerP              decimal(7,4) NULL, " +
                " transformerQ          decimal(7,4)     NULL, " +
                " transformerPLoss           decimal(7,4)     NULL, " +
                " transformerQLoss              decimal(7,4) NULL, " +
                " isOverLoad     INTEGER NULL" +
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
        PfResult r = BpaPfResultParser.parse(in, "GBK");
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<String> sqls = new LinkedList<>();
        String TABLE_DATA_NAME = "BusPfResult";
        for (BusPfResult bus : r.getBusData().values()) {
            int isVoltageLimit = 0;
            if (bus.isVoltageLimit()) {
                isVoltageLimit = 1;
            }
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + bus.getName() + "'," + bus.getvInKv() + "," +
                    bus.getAngleInDegree() + "," + bus.getGenP() + "," +
                    bus.getGenQ() + "," + bus.getLoadP() + "," +
                    bus.getLoadQ() + "," + isVoltageLimit +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "BranchPfResult";
        for (BranchPfResult acLine : r.getBranchData().values()) {
            int isOverLoad = 0;
            if (acLine.isOverLoad()) {
                isOverLoad = 1;
            }
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + acLine.getBusName1() + "','" + acLine.getBusName2() + "'," +
                    acLine.getBranchP() + "," + acLine.getBranchQ() + "," +
                    acLine.getBranchPLoss() + "," + acLine.getBranchQLoss() + "," +
                    isOverLoad +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "TransformerPfResult";
        for (TransformerPfResult transformer : r.getTransformerData().values()) {
            int isOverLoad = 0;
            if (transformer.isOverLoad()) {
                isOverLoad = 1;
            }
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + transformer.getBusName1() + "','" + transformer.getBusName2() + "'," +
                    transformer.getBaseKv1() + "," + transformer.getBaseKv2() + "," +
                    transformer.getTransformerP() + "," + transformer.getTransformerQ() + "," +
                    transformer.getTransformerPLoss() + "," + transformer.getTransformerQLoss() + "," +
                    isOverLoad +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);
    }
}
