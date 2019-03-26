package zju.bpamodel;

import zju.bpamodel.swi.Exciter;
import zju.bpamodel.swi.Generator;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.sql.*;
import java.util.LinkedList;
import java.util.List;

public class SqliteDb {
    private Connection createConn() {
        String dbFile = "d:/rsa.db";
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
            Exciter exciter = new Exciter();
            while (rs.next()) {

            }
        }
        return null;
    }
}
