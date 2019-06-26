package zju.bpamodel;

import junit.framework.TestCase;

public class XMLparseTest  extends TestCase {

    public void testCase1() {
        String dbFile = "d:/rsa.db";
        ChangeDbData.createTables(dbFile);
        // 解析和存储发电计划
        String planFile = "C:/Users/Administrator/IdeaProjects/alg/src/test/resources/bpafiles/示范区BPA运行方式/PLANLOAD-20190614.xml";
        PlanRw.parseAndSave(planFile, "d:/rsa.db");

        String inputPfPath = "C:/Users/Administrator/IdeaProjects/alg/src/test/resources/bpafiles/示范区BPA运行方式/XIAOJIN.DAT";
        String outputPfPath = "d:/XIAOJIN.DAT";
        String pfntFile = "C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/PFNT/PFNT.exe";
        String pfoFilePath = "d:/XIAOJIN.pfo";
        // 解析和存储潮流文件
        BpaPfModelRw.parseAndSave(inputPfPath, dbFile);
        // 按照发电计划计算潮流，并将潮流计算结果存库
//        ExePlan.doPf(dbFile, inputPfPath, outputPfPath, pfntFile, pfoFilePath);
//
//        String swntPath = "C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/SWNT/SWNT.exe";
//        String bseFilePath = "C:/Users/Administrator/Desktop/BPA用户手册（PDF）/88000007.BSE";
//        String inputSwiPath = "d:/XIAOJIN.SWI";
//        String outFilePath = "d:/XIAOJIN.OUT";
//        String swxFilePath = "d:/XIAOJIN.SWX";
//        ExePlan.doSw(dbFile, inputPfPath, outputPfPath, pfntFile, swntPath, bseFilePath, inputSwiPath, outFilePath, swxFilePath);
    }

    public void testDoPf() {
        String dbFile = "d:/rsa.db";
        String inputPfPath = "C:/Users/Administrator/IdeaProjects/alg/src/test/resources/bpafiles/示范区BPA运行方式/XIAOJIN.DAT";
        String outputPfPath = "d:/XIAOJIN.DAT";
        String pfntFile = "C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/PFNT/PFNT.exe";
        String pfoFilePath = "d:/XIAOJIN.pfo";
        // 按照发电计划计算潮流，并将潮流计算结果存库
        ExePlan.doPf(dbFile, inputPfPath, outputPfPath, pfntFile, pfoFilePath);
    }
}