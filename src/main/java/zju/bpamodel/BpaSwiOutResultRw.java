package zju.bpamodel;

import zju.bpamodel.swir.Damping;
import zju.bpamodel.swir.MonitorData;
import zju.bpamodel.swir.SwiOutResult;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;

public class BpaSwiOutResultRw {

    public static void CreateTables(String dbFile) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        String TABLE_DATA_NAME = "MonitorData";
        String initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " calendar     varchar(8) NOT NULL," +
                " time              decimal(7,3) NULL, " +
                " busName1     varchar(8) NOT NULL," +
                " baseKv1              decimal(4,3) NULL, " +
                " busName2     varchar(8) NOT NULL," +
                " baseKv2              decimal(4,3) NULL, " +
                " relativeAngle              decimal(7,3) NULL, " +
                " minVBusName     varchar(8) NOT NULL," +
                " minVBusBaseKv              decimal(4,3) NULL, " +
                " minVInPu          decimal(6,4)     NULL, " +
                " maxVBusName     varchar(8) NOT NULL," +
                " maxVBusBaseKv              decimal(4,3) NULL, " +
                " maxVInPu          decimal(6,4)     NULL, " +
                " minFreqBusName     varchar(8) NOT NULL," +
                " minFreqBusBaseKv              decimal(4,3) NULL, " +
                " minFreq          decimal(6,3)     NULL, " +
                " maxFreqBusName     varchar(8) NOT NULL," +
                " maxFreqBusBaseKv              decimal(4,3) NULL, " +
                " maxFreq          decimal(6,3)     NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "Damping";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " calendar     varchar(8) NOT NULL," +
                " busName1     varchar(8) NOT NULL," +
                " baseKv1          decimal(6,3)     NULL, " +
                " busName2     varchar(8) NOT NULL," +
                " baseKv2          decimal(6,3)     NULL, " +
                " variableName     varchar(10) NOT NULL," +
                " oscillationAmp1          decimal(8,5)     NULL, " +
                " oscillationFreq1          decimal(7,5)     NULL, " +
                " attenuationCoef1          decimal(8,5)     NULL, " +
                " dampingRatio1          decimal(7,5)     NULL, " +
                " oscillationAmp2          decimal(8,5)     NULL, " +
                " oscillationFreq2          decimal(7,5)     NULL, " +
                " attenuationCoef2          decimal(8,5)     NULL, " +
                " dampingRatio2          decimal(7,5)     NULL " +
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
        SwiOutResult r = BpaSwiOutResultParser.parse(in, "GBK");
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<String> sqls = new LinkedList<>();
        String TABLE_DATA_NAME = "MonitorData";
        String sql = "delete from " + TABLE_DATA_NAME + " where calendar='" + calendar + "'";
        sqliteDb.executeSql(sql);
        for (MonitorData monitorData : r.getMonitorDataList()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + calendar + "'," +
                    monitorData.getTime() + "," +
                    "'" + monitorData.getRelativeAngle().getBusName1() + "'," +
                    monitorData.getRelativeAngle().getBaseKv1() + "," +
                    "'" + monitorData.getRelativeAngle().getBusName2() + "'," +
                    monitorData.getRelativeAngle().getBaseKv2() + "," +
                    monitorData.getRelativeAngle().getRelativeAngle() + "," +
                    "'" + monitorData.getMinBusVoltage().getName() + "'," +
                    monitorData.getMinBusVoltage().getBaseKv() + "," +
                    monitorData.getMinBusVoltage().getvInPu() + "," +
                    "'" + monitorData.getMaxBusVoltage().getName() + "'," +
                    monitorData.getMaxBusVoltage().getBaseKv() + "," +
                    monitorData.getMaxBusVoltage().getvInPu() + "," +
                    "'" + monitorData.getMinBusFreq().getName() + "'," +
                    monitorData.getMinBusFreq().getBaseKv() + "," +
                    monitorData.getMinBusFreq().getFreq() + "," +
                    "'" + monitorData.getMaxBusFreq().getName() + "'," +
                    monitorData.getMaxBusFreq().getBaseKv() + "," +
                    monitorData.getMaxBusFreq().getFreq() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        TABLE_DATA_NAME = "Damping";
        sql = "delete from " + TABLE_DATA_NAME + " where calendar='" + calendar + "'";
        sqliteDb.executeSql(sql);
        for (Damping damping : r.getDampings()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + calendar + "'," +
                    "'" + damping.getBusName1() + "'," + damping.getBaseKv1() + "," +
                    "'" + damping.getBusName2() + "'," + damping.getBaseKv2() + "," +
                    "'" + damping.getVariableName() + "'," + damping.getOscillationAmp1() + "," +
                    damping.getOscillationFreq1() + "," + damping.getAttenuationCoef1() + "," +
                    damping.getDampingRatio1() + "," + damping.getOscillationAmp2() + "," +
                    damping.getOscillationFreq2() + "," + damping.getAttenuationCoef2() + "," +
                    damping.getDampingRatio2() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);
    }
}
