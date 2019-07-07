package zju.bpamodel;

import zju.bpamodel.swir.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;

public class BpaSwiSwxResultRw {

    public static void CreateTables(String dbFile) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        String TABLE_DATA_NAME = "GeneratorData";
        String initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " calendar     varchar(8) NOT NULL," +
                " busName1     varchar(8) NOT NULL," +
                " baseKv1          decimal(6,3)     NULL, " +
                " busName2     varchar(8) NOT NULL," +
                " baseKv2          decimal(6,3)     NULL, " +
                " time              decimal(7,3) NULL, " +
                " relativeAngle              decimal(8,4) NULL, " +
                " freqDeviation              decimal(8,4) NULL, " +
                " fieldVoltage              decimal(7,4) NULL, " +
                " mechPower              decimal(9,4) NULL, " +
                " elecPower              decimal(9,4) NULL, " +
                " regulatorOutput              decimal(7,4) NULL, " +
                " reactivePower              decimal(9,4) NULL, " +
                " fieldCurrent          decimal(7,4)     NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "BusData";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " calendar     varchar(8) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv          decimal(6,3)     NULL, " +
                " time              decimal(7,3) NULL, " +
                " posSeqVol          decimal(7,4)     NULL, " +
                " freqDeviation          decimal(8,4)     NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "LineData";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " calendar     varchar(8) NOT NULL," +
                " busName1     varchar(8) NOT NULL," +
                " baseKv1          decimal(6,3)     NULL, " +
                " busName2     varchar(8) NOT NULL," +
                " baseKv2          decimal(6,3)     NULL, " +
                " time              decimal(7,3) NULL, " +
                " activePower          decimal(9,4)     NULL, " +
                " reactivePower          decimal(9,4)     NULL " +
                ")";
        sqliteDb.initDb(initSql);
    }

    public static void parseAndSave(String filePath, String dbFile, String calendar) {
        parseAndSave(new File(filePath), dbFile, calendar);
    }

    public static void parseAndSave(File file, String dbFile, String calendar) {
        try {
            parseAndSave(new FileInputStream(file), dbFile, calendar);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static void parseAndSave(InputStream in, String dbFile, String calendar) {
        SwiSwxResult r = BpaSwiSwxResultParser.parse(in, "GBK");
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<String> sqls = new LinkedList<>();
        String TABLE_DATA_NAME = "GeneratorData";
        String sql = "delete from " + TABLE_DATA_NAME + " where calendar='" + calendar + "'";
        sqliteDb.executeSql(sql);
        for (GeneratorData generatorData : r.getGeneratorDataList()) {
            for (GenOneStepData genOneStepData : generatorData.getGenOneStepDataList()) {
                String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                        "'" + calendar + "'," +
                        "'" + generatorData.getBusName1() + "'," + generatorData.getBaseKv1() + "," +
                        "'" + generatorData.getBusName2() + "'," + generatorData.getBaseKv2() + "," +
                        genOneStepData.getTime() + "," + genOneStepData.getRelativeAngle() + "," +
                        genOneStepData.getFreqDeviation() + "," + genOneStepData.getFieldVoltage() + "," +
                        genOneStepData.getMechPower() + "," + genOneStepData.getElecPower() + "," +
                        genOneStepData.getRegulatorOutput() + "," + genOneStepData.getReactivePower() + "," +
                        genOneStepData.getFieldCurrent() +
                        ")";
                sqls.add(insertSql);
            }
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "BusData";
        sql = "delete from " + TABLE_DATA_NAME + " where calendar='" + calendar + "'";
        sqliteDb.executeSql(sql);
        for (BusData busData : r.getBusDataList()) {
            for (BusOneStepData busOneStepData : busData.getBusOneStepDataList()) {
                String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                        "'" + calendar + "'," +
                        "'" + busData.getBusName() + "'," + busData.getBaseKv() + "," +
                        busOneStepData.getTime() + "," + busOneStepData.getPosSeqVol() + "," +
                        busOneStepData.getFreqDeviation() +
                        ")";
                sqls.add(insertSql);
            }
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "LineData";
        sql = "delete from " + TABLE_DATA_NAME + " where calendar='" + calendar + "'";
        sqliteDb.executeSql(sql);
        for (LineData lineData : r.getLineDataList()) {
            for (LineOneStepData lineOneStepData : lineData.getLineOneStepDataList()) {
                String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                        "'" + calendar + "'," +
                        "'" + lineData.getBusName1() + "'," + lineData.getBaseKv1() + "," +
                        "'" + lineData.getBusName2() + "'," + lineData.getBaseKv2() + "," +
                        lineOneStepData.getTime() + "," + lineOneStepData.getActivePower() + "," +
                        lineOneStepData.getReactivePower() +
                        ")";
                sqls.add(insertSql);
            }
        }
        sqliteDb.executeSqls(sqls);
    }
}
