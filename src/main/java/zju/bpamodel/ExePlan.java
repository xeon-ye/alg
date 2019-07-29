package zju.bpamodel;

import java.util.HashMap;
import java.util.List;

public class ExePlan {

    private static HashMap<String, double[]> getPlan(String dbFile, String psId) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<Object> genPlans = sqliteDb.queryData("Plan", psId);
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

    public static void doPf(String dbFile, String psId, String caseId, String inputPath, String outputPath, String cmdPath, String pfntPath, String pfoFilePath) {
        HashMap<String, double[]> plans = getPlan(dbFile, psId);
        for (int i = 0; i < 96; i++) {
            for (String key : plans.keySet()) {
                updataGenPlan(dbFile, key, plans.get(key)[i]);
            }
            BpaPfModelRw.write(dbFile, psId, caseId, inputPath, outputPath);
            ExeBpa.exePf(cmdPath, pfntPath, outputPath);
            BpaPfResultRw.parseAndSave(pfoFilePath, dbFile, "v" + i);
        }
    }

    public static void doSw(String dbFile, String psId, String caseId, String inputPfPath, String outputPfPath, String cmdPath, String pfntPath, String swntPath,
                            String bseFilePath, String swiFilePath, String outFilePath, String swxFilePath) {
        HashMap<String, double[]> plans = getPlan(dbFile, psId);
        for (int i = 0; i < 96; i++) {
            for (String key : plans.keySet()) {
                updataGenPlan(dbFile, key, plans.get(key)[i]);
            }
            BpaPfModelRw.write(dbFile, psId, caseId, inputPfPath, outputPfPath);
            ExeBpa.exePf(cmdPath, pfntPath, outputPfPath);
            ExeBpa.exeSw(cmdPath, swntPath, bseFilePath, swiFilePath);
            BpaSwiOutResultRw.parseAndSave(outFilePath, dbFile, "v" + i);
            BpaSwiSwxResultRw.parseAndSave(swxFilePath, dbFile, "v" + i);
        }
    }
}
