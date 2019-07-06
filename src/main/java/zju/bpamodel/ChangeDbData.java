package zju.bpamodel;

import java.util.List;

public class ChangeDbData {
    public static void createTables(String dbFile) {
        BpaPfModelRw.CreateTables(dbFile);
        BpaSwiModelRw.CreateTables(dbFile);
        BpaPfResultRw.CreateTables(dbFile);
        BpaSwiOutResultRw.CreateTables(dbFile);
        BpaSwiSwxResultRw.CreateTables(dbFile);
        PlanRw.CreateTables(dbFile);
    }

    public static void truncateTable(String dbFile, String tableName) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        String sql = "delete from " + tableName;
        sqliteDb.executeSql(sql);
    }

    public static void truncateTables(String dbFile) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<String> tableNames = sqliteDb.getTableNames();
        for (String tableName : tableNames) {
            truncateTable(dbFile, tableName);
        }
    }

    public static void setShortCircuitFault(String dbFile, char busASign, String busAName, double busABaseKv,
                                            char busBSign, String busBName, double busBBaseKv, char parallelBranchCode,
                                            int mode, double startCycle, double faultR, double faultX, double posPercent) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        String sql = "insert into ShortCircuitFault values(" +
                "'LS','" + busASign + "'," +
                "'" + busAName + "'," + busABaseKv + "," +
                "'" + busBSign + "','" + busBName + "'," +
                busBBaseKv + ",'" + parallelBranchCode + "'," +
                mode + "," + startCycle + "," +
                faultR + "," + faultX + "," +
                posPercent +
                ")";
        sqliteDb.executeSql(sql);
    }

    public static void setFLTCard(String dbFile, String busAName, double busABaseKv, String busBName, double busBBaseKv,
                                  char circuitId, int fltType, int phase, int side, double tcyc0, double tcyc1,
                                  double tcyc2, double posPercent, double faultR, double faultX, double tcyc11,
                                  double tcyc21, double tcyc12, double tcyc22) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        String sql = "insert into FLTCard values(" +
                "'FLT','" + busAName + "'," + busABaseKv + "," +
                "'" + busBName + "'," + busBBaseKv + "," +
                "'" + circuitId + "'," + fltType + "," +
                phase + "," + side + "," +
                tcyc0 + "," + tcyc1 + "," +
                tcyc2 + "," + posPercent + "," +
                faultR + "," + faultX + "," +
                tcyc11 + "," + tcyc21 + "," +
                tcyc12 + "," + tcyc22 +
                ")";
        sqliteDb.executeSql(sql);
    }
}
