package zju.bpamodel;

import zju.bpamodel.swi.*;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.sql.*;
import java.util.LinkedList;
import java.util.List;

public class SqliteDb {
    private Connection createConn() {
        String dbFile = "d:/bpa.db";
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
            while (rs.next()) {
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
            }
            return obj;
        } else if (tableName.equals("Exciter")) {
            Exciter obj = new Exciter();
            while (rs.next()) {
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
            }
            return obj;
        } else if (tableName.equals("PSS")) {
            PSS obj = new PSS();
            while (rs.next()) {
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
            }
            return obj;
        } else if (tableName.equals("PrimeMover")) {
            PrimeMover obj = new PrimeMover();
            while (rs.next()) {
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
            }
            return obj;
        } else if (tableName.equals("Governor")) {
            Governor obj = new Governor();
            while (rs.next()) {
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
            }
            return obj;
        } else if (tableName.equals("PV")) {
            PV obj = new PV();
            while (rs.next()) {
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
            }
            return obj;
        } else if (tableName.equals("BC")) {
            BC obj = new BC();
            while (rs.next()) {
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
            }
            return obj;
        }
        return null;
    }
}
