package zju.bpamodel;

import java.util.LinkedList;
import java.util.List;

public class ChangeDbData {
    public static void createTables(String dbFile) {
        BpaPfModelRw.CreateTables(dbFile);
        BpaSwiModelRw.CreateTables(dbFile);
        BpaPfResultRw.CreateTables(dbFile);
        BpaSwiOutResultRw.CreateTables(dbFile);
        BpaSwiSwxResultRw.CreateTables(dbFile);
    }

    public static void truncateTable(String dbFile, String tableName) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<String> sqls = new LinkedList<>();
        String sql = "delete from " + tableName;
        sqls.add(sql);
        sqliteDb.executeSqls(sqls);
    }

    public static void truncateTables(String dbFile) {
        truncateTable(dbFile, "");
    }

    public static void updataGenPlan(String dbFile, String busName, double genMw) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<String> sqls = new LinkedList<>();
        String sql = "update Bus set genMw=" + genMw + " where name='" + busName +"'";
        sqls.add(sql);
        sqliteDb.executeSqls(sqls);
    }

    public static void setShortCircuitFault(String dbFile, char busASign, String busAName, double busABaseKv,
                                            char busBSign, String busBName, double busBBaseKv, char parallelBranchCode,
                                            int mode, double startCycle, double faultR, double faultX, double posPercent) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<String> sqls = new LinkedList<>();
        String insertSql = "insert into ShortCircuitFault values(" +
                "'LS','" + busASign + "'," +
                "'" + busAName + "'," + busABaseKv + "," +
                "'" + busBSign + "','" + busBName + "'," +
                busBBaseKv + ",'" + parallelBranchCode + "'," +
                mode + "," + startCycle + "," +
                faultR + "," + faultX + "," +
                posPercent +
                ")";
        sqls.add(insertSql);
        sqliteDb.executeSqls(sqls);
    }

//    public static void setFLTCard(String dbFile, String busAName, String name2, double genMw) {
//        SqliteDb sqliteDb = new SqliteDb(dbFile);
//        List<String> sqls = new LinkedList<>();
//        String sql = "updata Bus set genMw=" + genMw + " where name='" + busName +"'";
//        sqls.add(sql);
//        sqliteDb.executeSqls(sqls);
//    }
}
