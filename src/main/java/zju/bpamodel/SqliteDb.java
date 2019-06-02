package zju.bpamodel;

import zju.bpamodel.pf.AcLine;
import zju.bpamodel.pf.Bus;
import zju.bpamodel.pf.PowerExchange;
import zju.bpamodel.pf.Transformer;
import zju.bpamodel.swi.*;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.sql.*;
import java.util.LinkedList;
import java.util.List;

public class SqliteDb {

    String dbFile;

    public SqliteDb(String dbFile) {
        this.dbFile = dbFile;
    }

    private Connection createConn() {
        File f = new File(dbFile);
        if(!f.exists()) {
            try {
                f.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        Connection conn;
        try {
            Class.forName("org.sqlite.JDBC");
            conn = DriverManager.getConnection("jdbc:sqlite:"+dbFile);
        } catch (Exception e) {
            StringWriter w = new StringWriter();
            e.printStackTrace(new PrintWriter(w, true));
            System.out.println("111"+e.getMessage());
            return null;
        }
        return conn;
    }

    public Boolean executeSqls(List<String> sqls) {
        Connection conn = createConn();
        if(conn == null) return false;
        Statement stmt =  null;
        try {
            conn.setAutoCommit(false);
            stmt = conn.createStatement();
            for (String obj : sqls)
                stmt.addBatch(obj);
            stmt.executeBatch();
            conn.commit();
        } catch (SQLException e) {
            try {
                conn.rollback();
            } catch (SQLException e1) {
                System.out.println(e1.getMessage());
            }
            StringWriter w =new StringWriter();
            e.printStackTrace(new PrintWriter(w, true));
            System.out.println(e.getMessage());
            return false;
        } finally {
            try {
                stmt.close();
                conn.setAutoCommit(true);
                conn.close();
            } catch (SQLException e) {
            }
        }
        return true;
    }

    public void initDb(String initSql) {
        Connection conn = createConn();
        if (conn == null) {
            System.out.println("Can't connect to sqlite !!!");
            return;
        }
        Statement stmt= null;
        try {
            stmt = conn.createStatement();
            stmt.executeUpdate(initSql);
        } catch (SQLException e ) {
            if (e.getMessage().contains("table $tableName already exists")) {
                System.out.println("Table $tableName already exists");
            } else {
                StringWriter w = new StringWriter();
                e.printStackTrace(new PrintWriter(w, true));
                System.out.println(e.getMessage());
            }
        } finally {
            try {
                stmt.close();
                conn.close();
            } catch (SQLException e) {
            }
        }
    }

    public List<Object> queryData(String tableName) {
        List<Object> objs = new LinkedList<>();
        Connection conn = createConn();
        String sql = "select * from " + tableName;
        Statement stmt = null;
        ResultSet rs = null;
        try {
            stmt = conn.createStatement();
            rs = stmt.executeQuery(sql);
            while (rs.next()) {
                Object obj= createObj(tableName, rs);
                objs.add(obj);
            }
        } catch (SQLException e) {
            System.out.println(e.getMessage());
            e.printStackTrace();
        } finally {
            try {
                rs.close();
                stmt.close();
            } catch (SQLException e) {
            }
        }
        return objs;
    }

    private Object createObj(String tableName, ResultSet rs) throws SQLException {
        if (tableName.equals("Generator")) {
            Generator obj = new Generator();
            String type = rs.getString("type");
            obj.setType(type.charAt(0));
            obj.setSubType(type.charAt(1));
            obj.setBusName(rs.getString("busName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setId(rs.getString("id").charAt(0));
            obj.seteMWS(rs.getDouble("eMWS"));
            obj.setpPercent(rs.getDouble("pPercent"));
            obj.setqPercent(rs.getDouble("qPercent"));
            obj.setBaseMva(rs.getDouble("baseMva"));
            obj.setRa(rs.getDouble("ra"));
            obj.setXdp(rs.getDouble("xdp"));
            obj.setXqp(rs.getDouble("xqp"));
            obj.setXd(rs.getDouble("xd"));
            obj.setXq(rs.getDouble("xq"));
            obj.setTdop(rs.getDouble("tdop"));
            obj.setTqop(rs.getDouble("tqop"));
            obj.setXl(rs.getDouble("xl"));
            obj.setSg10(rs.getDouble("sg10"));
            obj.setSg12(rs.getDouble("sg12"));
            obj.setD(rs.getDouble("d"));
            return obj;
        } else if (tableName.equals("GeneratorDW")) {
            GeneratorDW obj = new GeneratorDW();
            obj.setBusName(rs.getString("busName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setId(rs.getString("id").charAt(0));
            obj.setBaseMva(rs.getDouble("baseMva"));
            obj.setPowerFactor(rs.getDouble("powerFactor"));
            obj.setType(rs.getString("motorType"));
            obj.setOwner(rs.getString("owner"));
            obj.setXdpp(rs.getDouble("xdpp"));
            obj.setXqpp(rs.getDouble("xqpp"));
            obj.setXdopp(rs.getDouble("xdopp"));
            obj.setXqopp(rs.getDouble("xqopp"));
            return obj;
        } else if (tableName.equals("Exciter")) {
            Exciter obj = new Exciter();
            String type = rs.getString("type");
            obj.setType(type.charAt(0));
            obj.setSubType(type.charAt(1));
            obj.setBusName(rs.getString("busName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setGeneratorCode(rs.getString("generatorCode").charAt(0));
            obj.setRc(rs.getDouble("rc"));
            obj.setXc(rs.getDouble("xc"));
            obj.setTr(rs.getDouble("tr"));
            obj.setVimax(rs.getDouble("vimax"));
            obj.setVimin(rs.getDouble("vimin"));
            obj.setTb(rs.getDouble("tb"));
            obj.setTc(rs.getDouble("tc"));
            obj.setKa(rs.getDouble("ka"));
            obj.setKv(rs.getDouble("kv"));
            obj.setTa(rs.getDouble("ta"));
            obj.setTrh(rs.getDouble("trh"));
            obj.setVrmax(rs.getDouble("vrmax"));
            obj.setVamax(rs.getDouble("vamax"));
            obj.setVrmin(rs.getDouble("vrmin"));
            obj.setVamin(rs.getDouble("vamin"));
            obj.setKe(rs.getDouble("ke"));
            obj.setKj(rs.getDouble("kj"));
            obj.setTe(rs.getDouble("te"));
            obj.setKf(rs.getDouble("kf"));
            obj.setTf(rs.getDouble("tf"));
            obj.setKh(rs.getDouble("kh"));
            obj.setK(rs.getDouble("k"));
            obj.setT1(rs.getDouble("t1"));
            obj.setT2(rs.getDouble("t2"));
            obj.setT3(rs.getDouble("t3"));
            obj.setT4(rs.getDouble("t4"));
            obj.setTa1(rs.getDouble("ta1"));
            obj.setVrminmult(rs.getDouble("vrminmult"));
            obj.setKi(rs.getDouble("ki"));
            obj.setKp(rs.getDouble("kp"));
            obj.setSe75max(rs.getDouble("se75max"));
            obj.setSemax(rs.getDouble("semax"));
            obj.setEfdmin(rs.getDouble("efdmin"));
            obj.setVbmax(rs.getDouble("vbmax"));
            obj.setEfdmax(rs.getDouble("efdmax"));
            obj.setXl(rs.getDouble("xl"));
            obj.setTf1(rs.getDouble("tf1"));
            return obj;
        } else if (tableName.equals("ExciterExtraInfo")) {
            ExciterExtraInfo obj = new ExciterExtraInfo();
            obj.setType(rs.getString("type"));
            obj.setBusName(rs.getString("busName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setGeneratorCode(rs.getString("generatorCode").charAt(0));
            obj.setVamax(rs.getDouble("vamax"));
            obj.setVamin(rs.getDouble("vamin"));
            obj.setVaimax(rs.getDouble("vaimax"));
            obj.setVaimin(rs.getDouble("vaimin"));
            obj.setKb(rs.getDouble("kb"));
            obj.setT5(rs.getDouble("t5"));
            obj.setKe(rs.getDouble("ke"));
            obj.setTe(rs.getDouble("te"));
            obj.setSe1(rs.getDouble("se1"));
            obj.setSe2(rs.getDouble("se2"));
            obj.setVrmax(rs.getDouble("vrmax"));
            obj.setVrmin(rs.getDouble("vrmin"));
            obj.setKc(rs.getDouble("kc"));
            obj.setKd(rs.getDouble("kd"));
            obj.setKli(rs.getDouble("kli"));
            obj.setVlir(rs.getDouble("vlir"));
            obj.setEfdmax(rs.getDouble("efdmax"));
            return obj;
        } else if (tableName.equals("PSS")) {
            PSS obj = new PSS();
            String type = rs.getString("type");
            obj.setType(type.charAt(0));
            obj.setSubType(type.charAt(1));
            obj.setGenName(rs.getString("genName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setId(rs.getString("id").charAt(0));
            obj.setKqv(rs.getDouble("kqv"));
            obj.setTqv(rs.getDouble("tqv"));
            obj.setKqs(rs.getDouble("kqs"));
            obj.setTqs(rs.getDouble("tqs"));
            obj.setTq(rs.getDouble("tq"));
            obj.setTq1(rs.getDouble("tq1"));
            obj.setTpq1(rs.getDouble("tpq1"));
            obj.setTq2(rs.getDouble("tq2"));
            obj.setTpq2(rs.getDouble("tpq2"));
            obj.setTq3(rs.getDouble("tq3"));
            obj.setTpq3(rs.getDouble("tpq3"));
            obj.setMaxVs(rs.getDouble("maxVs"));
            obj.setCutoffV(rs.getDouble("cutoffV"));
            obj.setSlowV(rs.getDouble("slowV"));
            obj.setRemoteBusName(rs.getString("remoteBusName"));
            obj.setRemoteBaseKv(rs.getDouble("remoteBaseKv"));
            obj.setKqsBaseCap(rs.getDouble("kqsBaseCap"));
            obj.setTrw(rs.getDouble("trw"));
            obj.setT5(rs.getDouble("t5"));
            obj.setT6(rs.getDouble("t6"));
            obj.setT7(rs.getDouble("t7"));
            obj.setKr(rs.getDouble("kr"));
            obj.setTrp(rs.getDouble("trp"));
            obj.setTw(rs.getDouble("tw"));
            obj.setTw1(rs.getDouble("tw1"));
            obj.setTw2(rs.getDouble("tw2"));
            obj.setKs(rs.getDouble("ks"));
            obj.setT9(rs.getDouble("t9"));
            obj.setT10(rs.getDouble("t10"));
            obj.setT12(rs.getDouble("t12"));
            obj.setInp(rs.getInt("inp"));
            return obj;
        } else if (tableName.equals("PSSExtraInfo")) {
            PSSExtraInfo obj = new PSSExtraInfo();
            obj.setType(rs.getString("type"));
            obj.setGenName(rs.getString("genName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setId(rs.getString("id").charAt(0));
            obj.setKp(rs.getDouble("kp"));
            obj.setT1(rs.getDouble("t1"));
            obj.setT2(rs.getDouble("t2"));
            obj.setT13(rs.getDouble("t13"));
            obj.setT14(rs.getDouble("t14"));
            obj.setT3(rs.getDouble("t3"));
            obj.setT4(rs.getDouble("t4"));
            obj.setMaxVs(rs.getDouble("maxVs"));
            obj.setMinVs(rs.getDouble("minVs"));
            obj.setIb(rs.getInt("ib"));
            obj.setBusName(rs.getString("busName"));
            obj.setBusBaseKv(rs.getDouble("busBaseKv"));
            obj.setXq(rs.getDouble("xq"));
            obj.setkMVA(rs.getDouble("kMVA"));
            return obj;
        } else if (tableName.equals("PrimeMover")) {
            PrimeMover obj = new PrimeMover();
            String type = rs.getString("type");
            obj.setType(type.charAt(0));
            obj.setSubType(type.charAt(1));
            obj.setBusName(rs.getString("busName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setGeneratorCode(rs.getString("generatorCode").charAt(0));
            obj.setMaxPower(rs.getDouble("maxPower"));
            obj.setR(rs.getDouble("r"));
            obj.setTg(rs.getDouble("tg"));
            obj.setTp(rs.getDouble("tp"));
            obj.setTd(rs.getDouble("td"));
            obj.setTw2(rs.getDouble("tw2"));
            obj.setCloseVel(rs.getDouble("closeVel"));
            obj.setOpenVel(rs.getDouble("openVel"));
            obj.setDd(rs.getDouble("dd"));
            obj.setDeadZone(rs.getDouble("deadZone"));
            return obj;
        } else if (tableName.equals("Governor")) {
            Governor obj = new Governor();
            String type = rs.getString("type");
            obj.setType(type.charAt(0));
            obj.setSubType(type.charAt(1));
            obj.setGenName(rs.getString("genName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setGeneratorCode(rs.getString("generatorCode").charAt(0));
            obj.setKw(rs.getDouble("kw"));
            obj.setTr(rs.getDouble("tr"));
            obj.setMinusDb1(rs.getDouble("minusDb1"));
            obj.setDb1(rs.getDouble("db1"));
            obj.setKp(rs.getDouble("kp"));
            obj.setKd(rs.getDouble("kd"));
            obj.setKi(rs.getDouble("ki"));
            obj.setTd(rs.getDouble("td"));
            obj.setMaxIntg(rs.getDouble("maxIntg"));
            obj.setMinIntg(rs.getDouble("minIntg"));
            obj.setMaxPID(rs.getDouble("maxPID"));
            obj.setMinPID(rs.getDouble("minPID"));
            obj.setDelt(rs.getDouble("delt"));
            obj.setMaxDb(rs.getDouble("maxDb"));
            obj.setMinDb(rs.getDouble("minDb"));
            return obj;
        } else if (tableName.equals("GovernorExtraInfo")) {
            GovernorExtraInfo obj = new GovernorExtraInfo();
            obj.setType(rs.getString("type"));
            obj.setGenName(rs.getString("genName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setGeneratorCode(rs.getString("generatorCode").charAt(0));
            obj.setDelt2(rs.getDouble("delt2"));
            obj.setTr2(rs.getDouble("tr2"));
            obj.setEp(rs.getDouble("ep"));
            obj.setMinusDb2(rs.getDouble("minusDb2"));
            obj.setDb2(rs.getDouble("db2"));
            obj.setMaxDb2(rs.getDouble("maxDb2"));
            obj.setMinDb2(rs.getDouble("minDb2"));
            obj.setItyp(rs.getInt("ityp"));
            obj.setItyp2(rs.getInt("ityp2"));
            return obj;
        } else if (tableName.equals("PV")) {
            PV obj = new PV();
            obj.setType(rs.getString("type"));
            obj.setBusName(rs.getString("busName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setId(rs.getString("Id").charAt(0));
            obj.setT(rs.getDouble("t"));
            obj.setS(rs.getDouble("s"));
            obj.setUoc(rs.getDouble("uoc"));
            obj.setIsc(rs.getDouble("isc"));
            obj.setUm(rs.getDouble("um"));
            obj.setIm(rs.getDouble("im"));
            obj.setN1(rs.getInt("n1"));
            obj.setN2(rs.getInt("n2"));
            return obj;
        } else if (tableName.equals("BC")) {
            BC obj = new BC();
            obj.setType(rs.getString("type"));
            obj.setBusName(rs.getString("busName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setId(rs.getString("Id").charAt(0));
            obj.setpPercent(rs.getDouble("pPercent"));
            obj.setIpCon(rs.getInt("ipCon"));
            obj.setTma(rs.getDouble("tma"));
            obj.setTa1(rs.getDouble("ta1"));
            obj.setTa(rs.getDouble("ta"));
            obj.setKpa(rs.getDouble("kpa"));
            obj.setKia(rs.getDouble("kia"));
            obj.setTsa(rs.getDouble("tsa"));
            obj.setC(rs.getDouble("c"));
            obj.setDcBaseKv(rs.getDouble("dcBaseKv"));
            obj.setK(rs.getDouble("k"));
            obj.setMva(rs.getDouble("mva"));
            obj.setKover(rs.getDouble("kover"));
            obj.setConverterNum(rs.getInt("converterNum"));
            return obj;
        } else if (tableName.equals("BCExtraInfo")) {
            BCExtraInfo obj = new BCExtraInfo();
            obj.setType(rs.getString("type"));
            obj.setBusName(rs.getString("busName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setId(rs.getString("Id").charAt(0));
            obj.setqPercent(rs.getDouble("qPercent"));
            obj.setRpCon(rs.getInt("rpCon"));
            obj.setTmb(rs.getDouble("tmb"));
            obj.setTb1(rs.getDouble("tb1"));
            obj.setTb(rs.getDouble("tb"));
            obj.setKpb(rs.getDouble("kpb"));
            obj.setKib(rs.getDouble("kib"));
            obj.setTsb(rs.getDouble("tsb"));
            obj.setKd(rs.getDouble("kd"));
            return obj;
        } else if (tableName.equals("Servo")) {
            Servo obj = new Servo();
            String type = rs.getString("type");
            obj.setType(type.charAt(0));
            obj.setSubType(type.charAt(1));
            obj.setGenName(rs.getString("genName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setGeneratorCode(rs.getString("generatorCode").charAt(0));
            obj.setPe(rs.getDouble("Pe"));
            obj.setTo(rs.getInt("tOpen"));
            obj.setCloseVel(rs.getDouble("closeVel"));
            obj.setOpenVel(rs.getDouble("openVel"));
            obj.setMaxPower(rs.getDouble("maxPower"));
            obj.setMinPower(rs.getDouble("minPower"));
            obj.setT1(rs.getDouble("t1"));
            obj.setKp(rs.getDouble("kp"));
            obj.setKd(rs.getDouble("kd"));
            obj.setKi(rs.getDouble("ki"));
            obj.setMaxIntg(rs.getDouble("maxIntg"));
            obj.setMinIntg(rs.getDouble("minIntg"));
            obj.setMaxPID(rs.getDouble("maxPID"));
            obj.setMinPID(rs.getDouble("minPID"));
            return obj;
        } else if (tableName.equals("Load")) {
            Load obj = new Load();
            String type = rs.getString("type");
            obj.setType(type.charAt(0));
            obj.setSubType(type.charAt(1));
            obj.setChgCode(rs.getString("chgCode").charAt(0));
            obj.setBusName(rs.getString("busName"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setZone(rs.getString("zone"));
            obj.setAreaName(rs.getString("areaName"));
            obj.setP1(rs.getDouble("p1"));
            obj.setQ1(rs.getDouble("q1"));
            obj.setP2(rs.getDouble("p2"));
            obj.setQ2(rs.getDouble("q2"));
            obj.setP3(rs.getDouble("p3"));
            obj.setQ3(rs.getDouble("q3"));
            obj.setP4(rs.getDouble("p4"));
            obj.setQ4(rs.getDouble("q4"));
            obj.setLdp(rs.getDouble("ldp"));
            obj.setLdq(rs.getDouble("ldq"));
            obj.setId(rs.getString("Id").charAt(0));
            obj.settJ(rs.getDouble("tJ"));
            obj.setPowerPercent(rs.getDouble("powerPercent"));
            obj.setLoadRate(rs.getDouble("loadRate"));
            obj.setMinPower(rs.getDouble("minPower"));
            obj.setRs(rs.getDouble("rs"));
            obj.setXs(rs.getDouble("xs"));
            obj.setXm(rs.getDouble("xm"));
            obj.setRr(rs.getDouble("rr"));
            obj.setXr(rs.getDouble("xr"));
            obj.setVi(rs.getDouble("vi"));
            obj.setTi(rs.getDouble("ti"));
            obj.setA(rs.getDouble("a"));
            obj.setB(rs.getDouble("b"));
            obj.setS(rs.getString("s").charAt(0));
            obj.setIm(rs.getInt("im"));
            return obj;
        } else if (tableName.equals("ShortCircuitFault")) {
            ShortCircuitFault obj = new ShortCircuitFault();
            obj.setBusASign(rs.getString("busASign").charAt(0));
            obj.setBusAName(rs.getString("busAName"));
            obj.setBusABaseKv(rs.getDouble("busABaseKv"));
            obj.setBusBSign(rs.getString("busBSign").charAt(0));
            obj.setBusBName(rs.getString("busBName"));
            obj.setBusBBaseKv(rs.getDouble("busBBaseKv"));
            obj.setParallelBranchCode(rs.getString("parallelBranchCode").charAt(0));
            obj.setMode(rs.getInt("mode"));
            obj.setStartCycle(rs.getDouble("startCycle"));
            obj.setFaultR(rs.getDouble("faultR"));
            obj.setFaultX(rs.getDouble("faultX"));
            obj.setPosPercent(rs.getDouble("posPercent"));
            return obj;
        } else if (tableName.equals("FLTCard")) {
            FLTCard obj = new FLTCard();
            obj.setType(rs.getString("type"));
            obj.setBusAName(rs.getString("busAName"));
            obj.setBusABaseKv(rs.getDouble("busABaseKv"));
            obj.setBusBName(rs.getString("busBName"));
            obj.setBusBBaseKv(rs.getDouble("busBBaseKv"));
            obj.setCircuitId(rs.getString("circuitId").charAt(0));
            obj.setFltType(rs.getInt("fltType"));
            obj.setPhase(rs.getInt("phase"));
            obj.setSide(rs.getInt("side"));
            obj.setTcyc0(rs.getDouble("tcyc0"));
            obj.setTcyc1(rs.getDouble("tcyc1"));
            obj.setTcyc2(rs.getDouble("tcyc2"));
            obj.setPosPercent(rs.getDouble("posPercent"));
            obj.setFaultR(rs.getDouble("faultR"));
            obj.setFaultX(rs.getDouble("faultX"));
            obj.setTcyc11(rs.getDouble("tcyc11"));
            obj.setTcyc21(rs.getDouble("tcyc21"));
            obj.setTcyc12(rs.getDouble("tcyc12"));
            obj.setTcyc22(rs.getDouble("tcyc22"));
            return obj;
        } else if (tableName.equals("FFCard")) {
            FFCard obj = new FFCard();
            obj.setType(rs.getString("type"));
            obj.setT(rs.getDouble("t"));
            obj.setDt(rs.getDouble("dt"));
            obj.setEndT(rs.getInt("endT"));
            obj.setDtc(rs.getInt("dtc"));
            obj.setIstp(rs.getInt("istp"));
            obj.setToli(rs.getDouble("toli"));
            obj.setIlim(rs.getInt("ilim"));
            obj.setDelAng(rs.getDouble("delAng"));
            obj.setDc(rs.getInt("dc"));
            obj.setDmp(rs.getDouble("dmp"));
            obj.setFrqBse(rs.getDouble("frqBse"));
            obj.setLovtex(rs.getInt("lovtex"));
            obj.setImblok(rs.getInt("imblok"));
            obj.setMfdep(rs.getInt("mfdep"));
            obj.setIgslim(rs.getInt("igslim"));
            obj.setLsolqit(rs.getInt("lsolqit"));
            obj.setNoAngLim(rs.getInt("noAngLim"));
            obj.setInfBus(rs.getInt("infBus"));
            obj.setNoPp(rs.getInt("noPp"));
            obj.setNoDq(rs.getInt("noDq"));
            obj.setNoSat(rs.getInt("noSat"));
            obj.setNoGv(rs.getInt("noGv"));
            obj.setIeqpc(rs.getInt("ieqpc"));
            obj.setNoEx(rs.getInt("noEx"));
            obj.setMftomg(rs.getInt("mftomg"));
            obj.setNoSc(rs.getInt("noSc"));
            obj.setMgtomf(rs.getInt("mgtomf"));
            obj.setNoLoad(rs.getInt("noLoad"));
            return obj;
        } else if (tableName.equals("PowerExchange")) {
            PowerExchange obj = new PowerExchange();
            String type = rs.getString("type");
            obj.setType(type.charAt(0));
            obj.setSubType(type.charAt(1));
            obj.setChgCode(rs.getString("chgCode").charAt(0));
            obj.setAreaName(rs.getString("areaName"));
            obj.setAreaBusName(rs.getString("areaBusName"));
            obj.setAreaBusBaseKv(rs.getDouble("areaBusBaseKv"));
            obj.setExchangePower(rs.getDouble("exchangePower"));
            obj.setZoneName(rs.getString("zoneName"));
            obj.setArea1Name(rs.getString("area1Name"));
            obj.setArea2Name(rs.getString("area2Name"));
            return obj;
        } else if (tableName.equals("Bus")) {
            Bus obj = new Bus();
            obj.setSubType(rs.getString("type").charAt(1));
            obj.setChgCode(rs.getString("chgCode").charAt(0));
            obj.setOwner(rs.getString("owner"));
            obj.setName(rs.getString("name"));
            obj.setBaseKv(rs.getDouble("baseKv"));
            obj.setZoneName(rs.getString("zoneName"));
            obj.setLoadMw(rs.getDouble("loadMw"));
            obj.setLoadMvar(rs.getDouble("loadMvar"));
            obj.setShuntMw(rs.getDouble("shuntMw"));
            obj.setShuntMvar(rs.getDouble("shuntMvar"));
            obj.setGenMwMax(rs.getDouble("genMwMax"));
            obj.setGenMw(rs.getDouble("genMw"));
            obj.setGenMvarShed(rs.getDouble("genMvarShed"));
            obj.setGenMvarMax(rs.getDouble("genMvarMax"));
            obj.setGenMvarMin(rs.getDouble("genMvarMin"));
            obj.setvAmplMax(rs.getDouble("vAmplMax"));
            obj.setvAmplDesired(rs.getDouble("vAmplDesired"));
            obj.setvAmplMin(rs.getDouble("vAmplMin"));
            obj.setSlackBusVAngle(rs.getDouble("slackBusVAngle"));
            obj.setRemoteCtrlBusName(rs.getString("remoteCtrlBusName"));
            obj.setRemoteCtrlBusBaseKv(rs.getDouble("remoteCtrlBusBaseKv"));
            obj.setGenMvarPercent(rs.getDouble("genMvarPercent"));
            return obj;
        } else if (tableName.equals("AcLine")) {
            AcLine obj = new AcLine();
            obj.setChgCode(rs.getString("chgCode").charAt(0));
            obj.setOwner(rs.getString("owner"));
            obj.setLinkMeterCode(rs.getInt("linkMeterCode"));
            obj.setBusName1(rs.getString("busName1"));
            obj.setBusName2(rs.getString("busName2"));
            obj.setBaseKv1(rs.getDouble("baseKv1"));
            obj.setBaseKv2(rs.getDouble("baseKv2"));
            obj.setCircuit(rs.getString("circuit").charAt(0));
            obj.setBaseI(rs.getDouble("baseI"));
            obj.setShuntLineNum(rs.getInt("shuntLineNum"));
            obj.setR(rs.getDouble("r"));
            obj.setX(rs.getDouble("x"));
            obj.setHalfG(rs.getDouble("halfG"));
            obj.setHalfB(rs.getDouble("halfB"));
            obj.setLength(rs.getDouble("length"));
            obj.setDesc(rs.getString("desc"));
            obj.setOnlineDate(rs.getString("onlineDate"));
            obj.setOfflineDate(rs.getString("offlineDate"));
            return obj;
        } else if (tableName.equals("Transformer")) {
            Transformer obj = new Transformer();
            obj.setSubType(rs.getString("type").charAt(1));
            obj.setChgCode(rs.getString("chgCode").charAt(0));
            obj.setOwner(rs.getString("owner"));
            obj.setBusName1(rs.getString("busName1"));
            obj.setBusName2(rs.getString("busName2"));
            obj.setBaseKv1(rs.getDouble("baseKv1"));
            obj.setBaseKv2(rs.getDouble("baseKv2"));
            obj.setCircuit(rs.getString("circuit").charAt(0));
            obj.setBaseMva(rs.getDouble("baseMva"));
            obj.setLinkMeterCode(rs.getInt("linkMeterCode"));
            obj.setShuntTransformerNum(rs.getInt("shuntTransformerNum"));
            obj.setR(rs.getDouble("r"));
            obj.setX(rs.getDouble("x"));
            obj.setG(rs.getDouble("g"));
            obj.setB(rs.getDouble("b"));
            obj.setTapKv1(rs.getDouble("tapKv1"));
            obj.setTapKv2(rs.getDouble("tapKv2"));
            obj.setPhaseAngle(rs.getDouble("phaseAngle"));
            obj.setOnlineDate(rs.getString("onlineDate"));
            obj.setOfflineDate(rs.getString("offlineDate"));
            obj.setDesc(rs.getString("desc"));
            return obj;
        }
        return null;
    }
}
