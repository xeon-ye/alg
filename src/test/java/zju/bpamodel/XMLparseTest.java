package zju.bpamodel;

import junit.framework.TestCase;

public class XMLparseTest  extends TestCase {

    public void testCase1() {
        String dbFile = "d:/rsa.db";
        ChangeDbData.createTables(dbFile);
        // 解析和存储发电计划
        String planFile = "D:/IdeaProjects/alg/src/test/resources/bpafiles/示范区BPA运行方式/PLANLOAD-20190614.xml";
        PlanRw.parseAndSave("1", planFile, "d:/rsa.db");

//        String inputPfPath = "C:/Users/Administrator/IdeaProjects/alg/src/test/resources/bpafiles/示范区BPA运行方式/XIAOJIN.DAT";
//        String outputPfPath = "d:/XIAOJIN.DAT";
//        String pfntFile = "C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/PFNT/PFNT.exe";
//        String pfoFilePath = "d:/XIAOJIN.pfo";
//        // 解析和存储潮流文件
//        BpaPfModelRw.parseAndSave(inputPfPath, dbFile);
    }

    public void testDoPf() {
        String dbFile = "d:/rsa.db";
        String caseId = "XIAOJIN";
        String inputPfPath = "C:/Users/Administrator/IdeaProjects/alg/src/test/resources/bpafiles/示范区BPA运行方式/XIAOJIN.DAT";
        String outputPfPath = "d:/XIAOJIN.DAT";
        String cmdPath = "d:";
        String pfntFile = "C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/PFNT/PFNT.exe";
        String pfoFilePath = "d:/XIAOJIN.pfo";
        // 按照发电计划计算潮流，并将潮流计算结果存库
        ExePlan.doPf(dbFile, caseId, inputPfPath, outputPfPath, cmdPath, pfntFile, pfoFilePath);
    }

    public void testDoSw() {
        String dbFile = "d:/rsa.db";
        String caseId = "XIAOJIN";
        String inputPfPath = "C:/Users/Administrator/IdeaProjects/alg/src/test/resources/bpafiles/示范区BPA运行方式/XIAOJIN.DAT";
        String outputPfPath = "d:/XIAOJIN.DAT";
        String cmdPath = "d:";
        String pfntFile = "C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/PFNT/PFNT.exe";
        String swntFile = "C:/Users/Administrator/Desktop/PSDEditCEPRI20180409-01/SWNT/SWNT.exe";
        String bseFile = "d:/88000007.BSE";
        String swiFilePath = "d:/XIAOJIN.SWI";
        String outFilePath = "d:/XIAOJIN.OUT";
        String swxFilePath = "d:/XIAOJIN.SWX";
        ExePlan.doSw(dbFile, caseId, inputPfPath, outputPfPath, cmdPath, pfntFile, swntFile, bseFile, swiFilePath, outFilePath, swxFilePath);
    }
}