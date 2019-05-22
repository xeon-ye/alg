package zju.bpamodel;

import zju.bpamodel.swir.MonitorData;
import zju.bpamodel.swir.SwiResult;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;

public class BpaSwiResultRw {

    public static void CreateTables(String dbFile) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        String TABLE_DATA_NAME = "MonitorData";
        String initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " time              decimal(7,3) NULL, " +
                " busName1     varchar(8) NOT NULL," +
                " busName2     varchar(8) NOT NULL," +
                " relativeAngle              decimal(7,3) NULL, " +
                " minVBusName     varchar(8) NOT NULL," +
                " minVInPu          decimal(6,4)     NULL, " +
                " maxVBusName     varchar(8) NOT NULL," +
                " maxVInPu          decimal(6,4)     NULL, " +
                " minFreqBusName     varchar(8) NOT NULL," +
                " minFreq          decimal(6,3)     NULL, " +
                " maxFreqBusName     varchar(8) NOT NULL," +
                " maxFreq          decimal(6,3)     NULL " +
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
        SwiResult r = BpaSwiResultParser.parse(in, "GBK");
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<String> sqls = new LinkedList<>();
        String TABLE_DATA_NAME = "MonitorData";
        for (MonitorData monitorData : r.getMonitorDataList()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    monitorData.getTime() + "," +
                    "'" + monitorData.getRelativeAngle().getBusName1() + "'," +
                    "'" + monitorData.getRelativeAngle().getBusName2() + "'," +
                    monitorData.getRelativeAngle().getRelativeAngle() + "," +
                    "'" + monitorData.getMinBusVoltage().getName() + "'," +
                    monitorData.getMinBusVoltage().getvInPu() + "," +
                    "'" + monitorData.getMaxBusVoltage().getName() + "'," +
                    monitorData.getMaxBusVoltage().getvInPu() + "," +
                    "'" + monitorData.getMinBusFreq().getName() + "'," +
                    monitorData.getMinBusFreq().getFreq() + "," +
                    "'" + monitorData.getMaxBusFreq().getName() + "'," +
                    monitorData.getMaxBusFreq().getFreq() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);
    }
}
