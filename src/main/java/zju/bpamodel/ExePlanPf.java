package zju.bpamodel;

import java.util.HashMap;
import java.util.List;

public class ExePlanPf {

    private static HashMap<String, double[]> getPlan(String dbFile) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<Object> genPlans = sqliteDb.queryData("Plan");
        HashMap<String, double[]> plans = new HashMap<>();
        for (int i = 0; i < genPlans.size(); i++) {
            String[] genPlan = (String[]) genPlans.get(i);
            double[] mws = new double[96];
            for (int j = 0; j < 96; j++) {
                mws[j] = Double.parseDouble(genPlan[j + 1]);
            }
            plans.put(genPlan[0], mws);
        }
        return plans;
    }

    private static void updataGenPlan(String dbFile, String busName, double genMw) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        String sql = "update Bus set genMw=" + genMw + " where name='" + busName +"'";
        sqliteDb.executeSql(sql);
    }

    public static void doPf(String dbFile, String inputPath, String outputPath, String pfntPath, String pfoFilePath) {
        HashMap<String, double[]> plans = getPlan(dbFile);
        for (int i = 0; i < 96; i++) {
            for (String key : plans.keySet()) {
                updataGenPlan(dbFile, key, plans.get(key)[i]);
            }
            BpaPfModelRw.write(dbFile, inputPath, outputPath);
            ExeBpa.exePf(pfntPath, outputPath);
            BpaPfResultRw.parseAndSave(pfoFilePath, dbFile, "v" + i);
        }
    }
}
