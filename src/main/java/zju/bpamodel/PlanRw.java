package zju.bpamodel;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

public class PlanRw {

    public static void CreateTables(String dbFile) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        String TABLE_DATA_NAME = "gensPlan";
        StringBuilder initSql = new StringBuilder("CREATE TABLE " + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " data     varchar(12) NOT NULL," +
                " busName     varchar(8) NOT NULL,");
        for (int i = 0; i < 95; i++) {
            initSql.append(" v").append(i).append("     decimal(5,4)     NULL,");
        }
        initSql.append(" v95     decimal(5,4)     NULL )");
        sqliteDb.initDb(initSql.toString());
    }

    public static void parseAndSave(String psId, String data, String filePath, String dbFile) {
        XMLparse parse = new XMLparse();
        HashMap hash = parse.getDailyPlanHash(filePath);
        Object[] keys = hash.keySet().toArray();
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<String> sqls = new LinkedList<>();
        String TABLE_DATA_NAME = "gensPlan";
        String sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "' and data='" + data + "'";
        sqliteDb.executeSql(sql);
        for (int i = 0; i < keys.length; i++) {
            double[] plans = (double[]) hash.get(keys[i]);
            StringBuilder insertSql = new StringBuilder("insert into " + TABLE_DATA_NAME + " values('" + psId + "','" + data + "','" + keys[i] + "',");
            for (int j = 0; j < plans.length - 1; j++) {
                insertSql.append(plans[j]).append(",");
            }
            insertSql.append(plans[plans.length - 1]).append(")");
            sqls.add(insertSql.toString());
        }
        sqliteDb.executeSqls(sqls);
    }
}
