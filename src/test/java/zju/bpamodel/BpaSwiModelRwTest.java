package zju.bpamodel;

import junit.framework.TestCase;
import zju.bpamodel.swi.*;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * BpaSwiModelWriter Tester.
 *
 * @author <Authors name>
 * @version 1.0
 * @since <pre>07/17/2012</pre>
 */
public class BpaSwiModelRwTest extends TestCase {

    public List<String> toFound;

    public BpaSwiModelRwTest(String name) {
        super(name);
        toFound = new ArrayList<String>();
        toFound.add("后石");
        toFound.add("可门");
        toFound.add("江阴");
        toFound.add("华能");
        toFound.add("鸿山");
        toFound.add("嵩屿");
        toFound.add("湄电"); //湄洲湾
        toFound.add("坑口");
        toFound.add("漳平");
        toFound.add("前云");
        toFound.add("石圳");
        toFound.add("新店");
        toFound.add("水口");
        toFound.add("安砂");
        toFound.add("周宁");

        toFound.add("池潭");
        toFound.add("棉电");//棉滩
        toFound.add("沙电");//沙溪口
        toFound.add("南埔");
        toFound.add("宁核");//宁德核电
        toFound.add("西抽");//西苑抽蓄
        toFound.add("福核");//福清核电
    }

    public void setUp() throws Exception {
        super.setUp();
    }

    public void tearDown() throws Exception {
        super.tearDown();
    }

    public void testReadAndWriter() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/fj/2012年稳定20111103.swi"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/fj/2012年稳定20111103.swi");
        //FileOutputStream out = new FileOutputStream("2012年稳定20111103-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);

        for (Exciter exciter : exciters) {
            if (exciter.getBusName().equals("闽江阴_1")) {
                exciter.setXc(-0.0396259400157145);
                System.out.println(exciter.toString());
            }
        }
    }

    public void testSwi003() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/003_bus/bpa/003bpaswi.dat"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/systemData/003_bus/bpa/003bpaswi.dat");
        //FileOutputStream out = new FileOutputStream("003bpaswi-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);
    }

    public void testSwi009() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/009_bus/bpa/009bpaswi.dat"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/systemData/009_bus/bpa/009bpaswi.dat");
        //FileOutputStream out = new FileOutputStream("009bpaswi-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);
    }

    public void testSwi039() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/039_bus/bpa/039bpaswi.dat"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/systemData/039_bus/bpa/039bpaswi.dat");
        //FileOutputStream out = new FileOutputStream("039bpaswi-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);
    }

    public void testSwi145() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/145_bus/bpa/145bpaswi.dat"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/systemData/145_bus/bpa/145bpaswi.dat");
        //FileOutputStream out = new FileOutputStream("145bpaswi-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);
    }

    public void testSwi162() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/systemData/162_bus/bpa/162bpaswi.dat"), "GBK");
        assertNotNull(model);

        List<Exciter> exciters = new ArrayList<Exciter>();
        for (String name : toFound) {
            for (Exciter exciter : model.getExciters()) {
                if (exciter.getBusName().contains(name)) {
                    exciters.add(exciter);
                }
            }
        }
        BpaSwiModel modifiedModel = new BpaSwiModel();
        modifiedModel.setGenerators(new ArrayList<Generator>());
        modifiedModel.setExciters(exciters);
        InputStream in = this.getClass().getResourceAsStream("/bpafiles/systemData/162_bus/bpa/162bpaswi.dat");
        //FileOutputStream out = new FileOutputStream("162bpaswi-opted.swi");
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
        assertTrue(r);
    }

    public void testXJ() throws IOException {
        BpaSwiModel model = BpaSwiModelParser.parse(this.getClass().getResourceAsStream("/bpafiles/示范区BPA运行方式/XIAOJIN.SWI"), "GBK");
        assertNotNull(model);
//        BpaSwiModel modifiedModel = new BpaSwiModel();
//        modifiedModel.setGenerators(new ArrayList<Generator>());
//        modifiedModel.setExciters(exciters);
//        InputStream in = this.getClass().getResourceAsStream("/bpafiles/示范区BPA运行方式/XIAOJIN.SWI");
//        //FileOutputStream out = new FileOutputStream("162bpaswi-opted.swi");
//        ByteArrayOutputStream out = new ByteArrayOutputStream();
//        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
//        assertTrue(r);
        SqliteDb sqliteDb = new SqliteDb();
        List<String> sqls = new LinkedList<>();
        String TABLE_DATA_NAME = "Generator";
        for (Generator generator : model.generators) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + generator.getType() + generator.getSubType() + "'," +
                    "'" + generator.getBusName() + "'," + generator.getBaseKv() + "," +
                    "'" + generator.getId() + "'," + generator.geteMWS() + "," +
                    generator.getpPercent() + "," + generator.getqPercent() + "," +
                    generator.getBaseMva() + "," + generator.getRa() + "," +
                    generator.getXdp() + "," + generator.getXqp() + "," +
                    generator.getXd() + "," + generator.getXq() + "," +
                    generator.getTdop() + "," + generator.getTqop() + "," +
                    generator.getXl() + "," + generator.getSg10() + "," +
                    generator.getSg12() + "," + generator.getD() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "Exciter";
        for (Exciter exciter : model.exciters) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + exciter.getType() + exciter.getSubType() + "'," +
                    "'" + exciter.getBusName() + "'," + exciter.getBaseKv() + "," +
                    "'" + exciter.getGeneratorCode() + "'," + exciter.getRc() + "," +
                    exciter.getXc() + "," + exciter.getTr() + "," +
                    exciter.getVimax() + "," + exciter.getVimin() + "," +
                    exciter.getTb() + "," + exciter.getTc() + "," +
                    exciter.getKa() + "," + exciter.getKv() + "," +
                    exciter.getTa() + "," + exciter.getTrh() + "," +
                    exciter.getVrmax() + "," + exciter.getVamax() + "," +
                    exciter.getVrmin() + "," + exciter.getVamin() + "," +
                    exciter.getKe() + "," + exciter.getKj() + "," +
                    exciter.getTe() + "," + exciter.getKf() + "," +
                    exciter.getTf() + "," + exciter.getKh() + "," +
                    exciter.getK() + "," + exciter.getT1() + "," +
                    exciter.getT2() + "," + exciter.getT3() + "," +
                    exciter.getT4() + "," + exciter.getTa1() + "," +
                    exciter.getVrminmult() + "," + exciter.getKi() + "," +
                    exciter.getKp() + "," + exciter.getSe75max() + "," +
                    exciter.getSemax() + "," + exciter.getEfdmin() + "," +
                    exciter.getVbmax() + "," + exciter.getEfdmax() + "," +
                    exciter.getXl() + "," + exciter.getTf1() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "PSS";
        for (PSS pss : model.pssList) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + pss.getType() + pss.getSubType() + "'," +
                    "'" + pss.getGenName() + "'," + pss.getBaseKv() + "," +
                    "'" + pss.getId() + "'," + pss.getKqv() + "," +
                    pss.getTqv() + "," + pss.getKqs() + "," +
                    pss.getTqs() + "," + pss.getTq() + "," +
                    pss.getTq1() + "," + pss.getTpq1() + "," +
                    pss.getTq2() + "," + pss.getTpq2() + "," +
                    pss.getTq3() + "," + pss.getTpq3() + "," +
                    pss.getMaxVs() + "," + pss.getCutoffV() + "," +
                    pss.getSlowV() + ",'" + pss.getRemoteBusName() + "'," +
                    pss.getRemoteBaseKv() + "," + pss.getKqsBaseCap() + "," +
                    pss.getTrw() + "," + pss.getT5() + "," +
                    pss.getT6() + "," + pss.getT7() + "," +
                    pss.getKr() + "," + pss.getTrp() + "," +
                    pss.getTw() + "," + pss.getTw1() + "," +
                    pss.getTw2() + "," + pss.getKs() + "," +
                    pss.getT9() + "," + pss.getT10() + "," +
                    pss.getT12() + "," + pss.getInp() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "Governor";
        for (Governor governor : model.governors) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + governor.getType() + governor.getSubType() + "'," +
                    "'" + governor.getGenName() + "'," + governor.getBaseKv() + "," +
                    "'" + governor.getGeneratorCode() + "'," + governor.getKw() + "," +
                    governor.getTr() + "," + governor.getMinusDb1() + "," +
                    governor.getDb1() + "," + governor.getKp() + "," +
                    governor.getKd() + "," + governor.getKi() + "," +
                    governor.getTd() + "," + governor.getMaxIntg() + "," +
                    governor.getMinIntg() + "," + governor.getMaxPID() + "," +
                    governor.getMinPID() + "," + governor.getDelt() + "," +
                    governor.getMaxDb() + ",'" + governor.getMinDb() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "PrimeMover";
        for (PrimeMover primeMover : model.primeMovers) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + primeMover.getType() + primeMover.getSubType() + "'," +
                    "'" + primeMover.getBusName() + "'," + primeMover.getBaseKv() + "," +
                    "'" + primeMover.getGeneratorCode() + "'," + primeMover.getMaxPower() + "," +
                    primeMover.getR() + "," + primeMover.getTg() + "," +
                    primeMover.getTp() + "," + primeMover.getTd() + "," +
                    primeMover.getTw2() + "," + primeMover.getCloseVel() + "," +
                    primeMover.getOpenVel() + "," + primeMover.getDd() + "," +
                    primeMover.getDeadZone() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "PV";
        for (PV pv : model.pvs) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + pv.getType() + "'," + "'" + pv.getBusName() + "',"
                    + pv.getBaseKv() + "," + "'" + pv.getId() + "'," +
                    pv.getT() + "," + pv.getS() + "," +
                    pv.getUoc() + "," + pv.getIsc() + "," +
                    pv.getUm() + "," + pv.getIm() + "," +
                    pv.getN1() + "," + pv.getN2() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "BC";
        for (BC bc : model.bcs) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + bc.getType() + "'," + "'" + bc.getBusName() + "',"
                    + bc.getBaseKv() + "," + "'" + bc.getId() + "'," +
                    bc.getpPercent() + "," + bc.getIpCon() + "," +
                    bc.getTma() + "," + bc.getTa1() + "," +
                    bc.getTa() + "," + bc.getKpa() + "," +
                    bc.getKia() + "," + bc.getTsa() + "," +
                    bc.getC() + "," + bc.getDcBaseKv() + "," +
                    bc.getK() + "," + bc.getMva() + "," +
                    bc.getKover() + "," + bc.getConverterNum() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);
    }

    public void testCreateTables() {
        SqliteDb sqliteDb = new SqliteDb();
        String TABLE_DATA_NAME = "Generator";
        String initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,0) NOT NULL, " +
                " id     varchar(1) DEFAULT NULL," +
                " eMWS              decimal(6,0) NULL, " +
                " pPercent           decimal(3,2)     NULL, " +
                " qPercent           decimal(3,2)     NULL, " +
                " baseMva              decimal(4,0) NULL, " +
                " ra              decimal(4,4) NULL, " +
                " xdp              decimal(5,4) NULL, " +
                " xqp              decimal(5,4) NULL, " +
                " xd              decimal(5,4) NULL, " +
                " xq              decimal(5,4) NULL, " +
                " tdop              decimal(4,2) NULL, " +
                " tqop              decimal(3,2) NULL, " +
                " xl              decimal(5,4) NULL, " +
                " sg10              decimal(5,4) NULL, " +
                " sg12              decimal(4,3) NULL, " +
                " d              decimal(3,2) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "Exciter";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,0) NOT NULL, " +
                " generatorCode     varchar(1) DEFAULT NULL," +
                " rc              decimal(5,4) NULL, " +
                " xc           decimal(5,4)     NULL, " +
                " tr           decimal(5,4)     NULL, " +
                " vimax              decimal(5,3) NULL, " +
                " vimin              decimal(5,3) NULL, " +
                " tb              decimal(5,3) NULL, " +
                " tc              decimal(5,3) NULL, " +
                " ka              decimal(5,3) NULL, " +
                " kv              decimal(5,3) NULL, " +
                " ta              decimal(5,3) NULL, " +
                " trh              decimal(5,3) NULL, " +
                " vrmax              decimal(5,3) NULL, " +
                " vamax              decimal(5,3) NULL, " +
                " vrmin              decimal(5,3) NULL, " +
                " vamin              decimal(5,3) NULL, " +
                " ke              decimal(5,3) NULL, " +
                " kj              decimal(5,3) NULL, " +
                " te              decimal(4,3) NULL, " +
                " kf              decimal(5,3) NULL, " +
                " tf              decimal(4,3) NULL, " +
                " kh              decimal(4,2) NULL, " +
                " k              decimal(5,3) NULL, " +
                " t1              decimal(5,3) NULL, " +
                " t2              decimal(5,3) NULL, " +
                " t3              decimal(5,3) NULL, " +
                " t4              decimal(5,3) NULL, " +
                " ta1              decimal(4,3) NULL, " +
                " vrminmult              decimal(4,2) NULL, " +
                " ki              decimal(4,3) NULL, " +
                " kp              decimal(4,3) NULL, " +
                " se75max              decimal(4,3) NULL, " +
                " semax              decimal(4,3) NULL, " +
                " efdmin              decimal(5,3) NULL, " +
                " vbmax              decimal(4,3) NULL, " +
                " efdmax              decimal(4,3) NULL, " +
                " xl              decimal(5,4) NULL, " +
                " tf1              decimal(5,4) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "PSS";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " type     varchar(3) NOT NULL," +
                " genName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,0) NOT NULL, " +
                " id     varchar(1) DEFAULT NULL," +
                " kqv              decimal(4,3) NULL, " +
                " tqv           decimal(3,3)     NULL, " +
                " kqs           decimal(4,3)     NULL, " +
                " tqs              decimal(3,3) NULL, " +
                " tq              decimal(4,2) NULL, " +
                " tq1              decimal(4,3) NULL, " +
                " tpq1              decimal(4,3) NULL, " +
                " tq2              decimal(4,3) NULL, " +
                " tpq2              decimal(4,3) NULL, " +
                " tq3              decimal(4,3) NULL, " +
                " tpq3              decimal(4,3) NULL, " +
                " maxVs              decimal(4,3) NULL, " +
                " cutoffV              decimal(4,3) NULL, " +
                " slowV              decimal(2,2) NULL, " +
                " remoteBusName              varchar(8) NULL, " +
                " remoteBaseKv              decimal(4,0) NULL, " +
                " kqsBaseCap              decimal(4,0) NULL, " +
                " trw              decimal(4,4) NULL, " +
                " t5              decimal(5,3) NULL, " +
                " t6              decimal(5,3) NULL, " +
                " t7              decimal(5,3) NULL, " +
                " kr              decimal(6,4) NULL, " +
                " trp              decimal(4,4) NULL, " +
                " tw              decimal(5,3) NULL, " +
                " tw1              decimal(5,3) NULL, " +
                " tw2              decimal(5,3) NULL, " +
                " ks              decimal(4,3) NULL, " +
                " t9              decimal(5,3) NULL, " +
                " t10              decimal(5,3) NULL, " +
                " t12              decimal(5,3) NULL, " +
                " inp              INTEGER NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "Governor";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " type     varchar(3) NOT NULL," +
                " genName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,0) NOT NULL, " +
                " generatorCode     varchar(1) DEFAULT NULL," +
                " kw              decimal(4,3) NULL, " +
                " tr           decimal(3,3)     NULL, " +
                " minusDb1           decimal(4,3)     NULL, " +
                " db1              decimal(3,3) NULL, " +
                " kp              decimal(4,2) NULL, " +
                " kd              decimal(4,3) NULL, " +
                " ki              decimal(4,3) NULL, " +
                " td              decimal(4,3) NULL, " +
                " maxIntg              decimal(4,3) NULL, " +
                " minIntg              decimal(4,3) NULL, " +
                " maxPID              decimal(4,3) NULL, " +
                " minPID              decimal(4,3) NULL, " +
                " delt              decimal(4,3) NULL, " +
                " maxDb              decimal(2,2) NULL, " +
                " minDb              varchar(8) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "PrimeMover";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,0) NOT NULL, " +
                " generatorCode     varchar(1) DEFAULT NULL," +
                " maxPower              decimal(6,1) NULL, " +
                " r           decimal(5.3)     NULL, " +
                " tg           decimal(5.3)     NULL, " +
                " tp              decimal(5.3) NULL, " +
                " td              decimal(5.3) NULL, " +
                " tw2              decimal(5.3) NULL, " +
                " closeVel              decimal(5.3) NULL, " +
                " openVel              decimal(5.3) NULL, " +
                " dd              decimal(5.3) NULL, " +
                " deadZone              decimal(6,5) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "PV";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,0) NOT NULL, " +
                " id     varchar(1) DEFAULT NULL," +
                " t              decimal(5,0) NULL, " +
                " s           decimal(5.0)     NULL, " +
                " uoc           decimal(5.0)     NULL, " +
                " isc              decimal(5.0) NULL, " +
                " um              decimal(5.0) NULL, " +
                " im              decimal(5.0) NULL, " +
                " n1              INTEGER NULL, " +
                " n2              INTEGER NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "BC";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,0) NOT NULL, " +
                " id     varchar(1) DEFAULT NULL," +
                " pPercent              decimal(5,0) NULL, " +
                " ipCon           INTEGER     NULL, " +
                " tma           decimal(5.0)     NULL, " +
                " ta1              decimal(5.0) NULL, " +
                " ta              decimal(5.0) NULL, " +
                " kpa              decimal(5.0) NULL, " +
                " kia              decimal(5.0) NULL, " +
                " tsa              decimal(5.0) NULL, " +
                " c              decimal(5.0) NULL, " +
                " dcBaseKv              decimal(5.0) NULL, " +
                " k              decimal(5.0) NULL, " +
                " mva              decimal(5.0) NULL, " +
                " kover              decimal(5.0) NULL, " +
                " converterNum              INTEGER NULL " +
                ")";
        sqliteDb.initDb(initSql);
    }
}
