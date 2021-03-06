package zju.bpamodel;

import zju.bpamodel.swi.*;

import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class BpaSwiModelRw {

    public static void CreateTables(String dbFile) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        String TABLE_DATA_NAME = "Generator";
        String initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,3) NOT NULL, " +
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

        TABLE_DATA_NAME = "GeneratorDW";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,3) NOT NULL, " +
                " id     varchar(1) DEFAULT NULL," +
                " baseMva              decimal(5,2) NULL, " +
                " powerFactor           decimal(3,2)     NULL, " +
                " motorType           varchar(2)     NULL, " +
                " owner              varchar(3) NULL, " +
                " xdpp              decimal(5,4) NULL, " +
                " xqpp              decimal(5,4) NULL, " +
                " xdopp              decimal(4,4) NULL, " +
                " xqopp              decimal(4,4) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "Exciter";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,3) NOT NULL, " +
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

        TABLE_DATA_NAME = "ExciterExtraInfo";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,3) NOT NULL, " +
                " generatorCode     varchar(1) DEFAULT NULL," +
                " vamax              decimal(5,3) NULL, " +
                " vamin           decimal(5,3)     NULL, " +
                " vaimax           decimal(5,0)     NULL, " +
                " vaimin              decimal(5,0) NULL, " +
                " kb              decimal(4,2) NULL, " +
                " t5              decimal(4,2) NULL, " +
                " ke              decimal(4,2) NULL, " +
                " te              decimal(4,2) NULL, " +
                " se1              decimal(5,4) NULL, " +
                " se2              decimal(5,4) NULL, " +
                " vrmax              decimal(4,2) NULL, " +
                " vrmin              decimal(4,2) NULL, " +
                " kc              decimal(4,2) NULL, " +
                " kd              decimal(4,2) NULL, " +
                " kli              decimal(4,2) NULL, " +
                " vlir              decimal(4,2) NULL, " +
                " efdmax              decimal(4,2) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "PSS";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " genName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,1) NOT NULL, " +
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
                " remoteBaseKv              decimal(4,1) NULL, " +
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

        TABLE_DATA_NAME = "PSSExtraInfo";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " genName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,1) NOT NULL, " +
                " id     varchar(1) DEFAULT NULL," +
                " kp              decimal(5,3) NULL, " +
                " t1           decimal(5,3)     NULL, " +
                " t2           decimal(5,3)     NULL, " +
                " t13              decimal(5,3) NULL, " +
                " t14              decimal(5,3) NULL, " +
                " t3              decimal(5,3) NULL, " +
                " t4              decimal(5,3) NULL, " +
                " maxVs              decimal(6,4) NULL, " +
                " minVs              decimal(6,4) NULL, " +
                " ib              INTEGER NULL, " +
                " busName              varchar(8) NULL, " +
                " busBaseKv              decimal(4,1) NULL, " +
                " xq              decimal(4,3) NULL, " +
                " kMVA              decimal(4,0) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "Governor";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " genName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,1) NOT NULL, " +
                " generatorCode     varchar(1) DEFAULT NULL," +
                " kw              decimal(5,0) NULL, " +
                " tr           decimal(4,4)     NULL, " +
                " minusDb1           decimal(4,4)     NULL, " +
                " db1              decimal(4,4) NULL, " +
                " kp              decimal(5,0) NULL, " +
                " kd              decimal(5,0) NULL, " +
                " ki              decimal(5,0) NULL, " +
                " td              decimal(4,4) NULL, " +
                " maxIntg              decimal(4,4) NULL, " +
                " minIntg              decimal(4,4) NULL, " +
                " maxPID              decimal(4,4) NULL, " +
                " minPID              decimal(4,4) NULL, " +
                " delt              decimal(4,4) NULL, " +
                " maxDb              decimal(4,4) NULL, " +
                " minDb              decimal(4,4) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "GovernorExtraInfo";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " genName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,1) NOT NULL, " +
                " generatorCode     varchar(1) DEFAULT NULL," +
                " delt2              decimal(4,3) NULL, " +
                " tr2           decimal(3,3)     NULL, " +
                " ep           decimal(4,3)     NULL, " +
                " minusDb2              decimal(3,3) NULL, " +
                " db2              decimal(4,2) NULL, " +
                " maxDb2              decimal(4,3) NULL, " +
                " minDb2              decimal(4,3) NULL, " +
                " ityp              INTEGER NULL, " +
                " ityp2              INTEGER NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "PrimeMover";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,1) NOT NULL, " +
                " generatorCode     varchar(1) DEFAULT NULL," +
                " maxPower              decimal(6,1) NULL, " +
                " r           decimal(5,4)     NULL, " +
                " tg           decimal(5,4)     NULL, " +
                " tp              decimal(5,4) NULL, " +
                " td              decimal(5,3) NULL, " +
                " tw2              decimal(5,3) NULL, " +
                " closeVel              decimal(5,3) NULL, " +
                " openVel              decimal(5,3) NULL, " +
                " dd              decimal(5,4) NULL, " +
                " deadZone              decimal(6,5) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "PV";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,1) NOT NULL, " +
                " id     varchar(1) DEFAULT NULL," +
                " t              decimal(5,2) NULL, " +
                " s           decimal(5,2)     NULL, " +
                " uoc           decimal(5,2)     NULL, " +
                " isc              decimal(5,2) NULL, " +
                " um              decimal(5,2) NULL, " +
                " im              decimal(5,2) NULL, " +
                " n1              INTEGER NULL, " +
                " n2              INTEGER NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "BC";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,1) NOT NULL, " +
                " id     varchar(1) DEFAULT NULL," +
                " pPercent              decimal(3,1) NULL, " +
                " ipCon           INTEGER     NULL, " +
                " tma           decimal(5,2)     NULL, " +
                " ta1              decimal(5,2) NULL, " +
                " ta              decimal(5,2) NULL, " +
                " kpa              decimal(5,2) NULL, " +
                " kia              decimal(5,2) NULL, " +
                " tsa              decimal(5,2) NULL, " +
                " c              decimal(5,2) NULL, " +
                " dcBaseKv              decimal(5,1) NULL, " +
                " k              decimal(5,2) NULL, " +
                " mva              decimal(5,2) NULL, " +
                " kover              decimal(5,2) NULL, " +
                " converterNum              INTEGER NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "BCExtraInfo";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " busName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,1) NOT NULL, " +
                " id     varchar(1) DEFAULT NULL," +
                " qPercent              decimal(3,1) NULL, " +
                " rpCon           INTEGER     NULL, " +
                " tmb           decimal(5,2)     NULL, " +
                " tb1              decimal(5,2) NULL, " +
                " tb              decimal(5,2) NULL, " +
                " kpb              decimal(5,2) NULL, " +
                " kib              decimal(5,2) NULL, " +
                " tsb              decimal(5,2) NULL, " +
                " kd              decimal(5,3) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "Servo";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " genName     varchar(8) NOT NULL," +
                " baseKv              decimal(4,1) NOT NULL, " +
                " generatorCode     varchar(1) DEFAULT NULL," +
                " pe              decimal(6,2) NULL, " +
                " tc           decimal(4,2)     NULL, " +
                " tOpen           decimal(4,2)     NULL, " +
                " closeVel              decimal(4,2) NULL, " +
                " openVel              decimal(4,2) NULL, " +
                " maxPower              decimal(4,2) NULL, " +
                " minPower              decimal(4,2) NULL, " +
                " t1              decimal(4,2) NULL, " +
                " kp              decimal(4,2) NULL, " +
                " kd              decimal(4,2) NULL, " +
                " ki              decimal(4,2) NULL, " +
                " maxIntg              decimal(4,2) NULL, " +
                " minIntg              decimal(4,2) NULL, " +
                " maxPID              decimal(4,2) NULL, " +
                " minPID              decimal(4,2) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "Load";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " chgCode     varchar(1) NULL," +
                " busName     varchar(8) NULL," +
                " baseKv              decimal(4,3) NULL, " +
                " zone              varchar(12) NULL, " +
                " areaName           varchar(10)     NULL, " +
                " p1           decimal(5,3)     NULL, " +
                " q1              decimal(5,3) NULL, " +
                " p2              decimal(5,3) NULL, " +
                " q2              decimal(5,3) NULL, " +
                " p3              decimal(5,3) NULL, " +
                " q3              decimal(5,3) NULL, " +
                " p4              decimal(5,3) NULL, " +
                " q4              decimal(5,3) NULL, " +
                " ldp              decimal(5,3) NULL, " +
                " ldq              decimal(5,3) NULL, " +

                " id           varchar(1)     NULL, " +
                " tJ              decimal(6,4) NULL, " +
                " powerPercent              decimal(3,3) NULL, " +
                " loadRate              decimal(4,4) NULL, " +
                " minPower              decimal(3,0) NULL, " +
                " rs              decimal(5,4) NULL, " +
                " xs              decimal(5,4) NULL, " +
                " xm              decimal(5,4) NULL, " +
                " rr              decimal(5,4) NULL, " +
                " xr              decimal(5,4) NULL, " +
                " vi              decimal(3,2) NULL, " +
                " ti              decimal(4,2) NULL, " +
                " a              decimal(5,4) NULL, " +
                " b              decimal(5,4) NULL, " +
                " s     varchar(1) NULL," +
                " im              INTEGER NULL, " +

                " total           varchar(5)     NULL, " +
                " dp              decimal(6,2) NULL, " +
                " dq              decimal(6,2) NULL, " +
                " dt              decimal(5,1) NULL, " +
                " tend              decimal(5,1) NULL, " +
                " specCode           varchar(1)     NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "ShortCircuitFault";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " busASign     varchar(1) NULL," +
                " busAName     varchar(8) NOT NULL," +
                " busABaseKv              decimal(4,3) NOT NULL, " +
                " busBSign     varchar(1) NULL," +
                " busBName     varchar(8) NOT NULL," +
                " busBBaseKv              decimal(4,3) NOT NULL, " +
                " parallelBranchCode     varchar(1) DEFAULT NULL," +
                " mode              INTEGER NULL, " +
                " startCycle           decimal(6,0)     NULL, " +
                " faultR           decimal(6,0)     NULL, " +
                " faultX              decimal(6,0) NULL, " +
                " posPercent              decimal(6,0) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "FFCard";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(2) NOT NULL," +
                " t              decimal(3,0) NOT NULL, " +
                " dt     decimal(3,1) NULL," +
                " endT              decimal(5,0) NULL, " +
                " dtc           decimal(3,1)     NULL, " +
                " istp              INTEGER NULL, " +
                " toli           decimal(5,5)     NULL, " +
                " ilim              INTEGER NULL, " +
                " delAng              decimal(4,4) NULL, " +
                " dc              INTEGER NULL, " +
                " dmp              decimal(3,3) NULL, " +
                " frqBse              decimal(2,0) NULL, " +
                " lovtex              INTEGER NULL, " +
                " imblok              INTEGER NULL, " +
                " mfdep              INTEGER NULL, " +
                " igslim              INTEGER NULL, " +
                " lsolqit              INTEGER NULL, " +
                " noAngLim              INTEGER NULL, " +
                " infBus              INTEGER NULL, " +
                " noPp              INTEGER NULL, " +
                " noDq              INTEGER NULL, " +
                " noSat              INTEGER NULL, " +
                " noGv              INTEGER NULL, " +
                " ieqpc              INTEGER NULL, " +
                " noEx              INTEGER NULL, " +
                " mftomg              INTEGER NULL, " +
                " noSc              INTEGER NULL, " +
                " mgtomf              INTEGER NULL, " +
                " noLoad              INTEGER NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "F0Card";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(2) NOT NULL," +
                " ig              INTEGER NULL, " +
                " ia              INTEGER NULL, " +
                " genName1     varchar(8) NOT NULL," +
                " genBaseKv1              decimal(4,1) NOT NULL, " +
                " id1     varchar(1) NOT NULL," +
                " genName2     varchar(8) NOT NULL," +
                " genBaseKv2              decimal(4,1) NOT NULL, " +
                " id2     varchar(1) NOT NULL," +
                " amax     decimal(5,0) NULL," +
                " amin              decimal(5,0) NULL, " +
                " iv              INTEGER NULL, " +
                " busName1     varchar(8) NOT NULL," +
                " busBaseKv1              decimal(4,1) NOT NULL, " +
                " iF              INTEGER NULL, " +
                " busName2     varchar(8) NOT NULL," +
                " busBaseKv2              decimal(4,1) NOT NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "FLTCard";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(3) NOT NULL," +
                " busAName     varchar(8) NOT NULL," +
                " busABaseKv              decimal(4,1) NOT NULL, " +
                " busBName     varchar(8) NOT NULL," +
                " busBBaseKv              decimal(4,1) NOT NULL, " +
                " circuitId     varchar(1) DEFAULT NULL," +
                " fltType              INTEGER NULL, " +
                " phase              INTEGER NULL, " +
                " side              INTEGER NULL, " +
                " tcyc0           decimal(4,0)     NULL, " +
                " tcyc1           decimal(4,0)     NULL, " +
                " tcyc2              decimal(4,0) NULL, " +
                " posPercent           decimal(2,0)     NULL, " +
                " faultR           decimal(5,0)     NULL, " +
                " faultX              decimal(5,0) NULL, " +
                " tcyc11           decimal(4,0)     NULL, " +
                " tcyc21           decimal(4,0)     NULL, " +
                " tcyc12              decimal(4,0) NULL, " +
                " tcyc22              decimal(4,0) NULL " +
                ")";
        sqliteDb.initDb(initSql);

        TABLE_DATA_NAME = "CaseCard";
        initSql = "CREATE TABLE "  + TABLE_DATA_NAME + " (" +
                " psId     varchar(10) NOT NULL," +
                " type     varchar(4) NOT NULL," +
                " caseId     varchar(10) NOT NULL," +
                " iNoPersist              INTEGER NULL, " +
                " ixo              INTEGER NULL, " +
                " dsw              INTEGER NULL, " +
                " iwscc              INTEGER NULL, " +
                " x2fac              decimal(5.5) NOT NULL, " +
                " xfact     decimal(5.5) NULL," +
                " tdodps              decimal(5.5) NULL, " +
                " tqodps           decimal(5.5)     NULL, " +
                " tdodph           decimal(5,5)     NULL, " +
                " tqodph           decimal(5,5)     NULL, " +
                " cfacl2              decimal(6,5) NULL " +
                ")";
        sqliteDb.initDb(initSql);
    }

    public static void parseAndSave(String psId, String filePath, String dbFile) {
        parseAndSave(psId, new File(filePath), dbFile);
    }

    public static void parseAndSave(String psId, File file, String dbFile) {
        try {
            parseAndSave(psId, new FileInputStream(file), dbFile);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static void parseAndSave(String psId, InputStream in, String dbFile) {
        BpaSwiModel model = BpaSwiModelParser.parse(in, "GBK");
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        List<String> sqls = new LinkedList<>();
        String TABLE_DATA_NAME = "Generator";
        String sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (Generator generator : model.getGenerators()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
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
        TABLE_DATA_NAME = "GeneratorDW";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (GeneratorDW generatorDW : model.getGeneratorDws()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
                    "'M'," + "'" + generatorDW.getBusName() + "'," +
                    generatorDW.getBaseKv() + ",'" + generatorDW.getId() + "'," +
                    generatorDW.getBaseMva() + "," + generatorDW.getPowerFactor() + "," +
                    "'" + generatorDW.getType() + "','" + generatorDW.getOwner() + "'," +
                    generatorDW.getXdpp() + "," + generatorDW.getXqpp() + "," +
                    generatorDW.getXdopp() + "," + generatorDW.getXqopp() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "Exciter";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (Exciter exciter : model.getExciters()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
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
        TABLE_DATA_NAME = "ExciterExtraInfo";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (ExciterExtraInfo exciterExtraInfo : model.getExciterExtraInfos()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
                    "'" + exciterExtraInfo.getType() + "'," + "'" + exciterExtraInfo.getBusName() + "'," +
                    exciterExtraInfo.getBaseKv() + "," + "'" + exciterExtraInfo.getGeneratorCode() + "'," +
                    exciterExtraInfo.getVamax() + "," + exciterExtraInfo.getVamin() + "," +
                    exciterExtraInfo.getVaimax() + "," + exciterExtraInfo.getVaimin() + "," +
                    exciterExtraInfo.getKb() + "," + exciterExtraInfo.getT5() + "," +
                    exciterExtraInfo.getKe() + "," + exciterExtraInfo.getTe() + "," +
                    exciterExtraInfo.getSe1() + "," + exciterExtraInfo.getSe2() + "," +
                    exciterExtraInfo.getVrmax() + "," + exciterExtraInfo.getVrmin() + "," +
                    exciterExtraInfo.getKc() + "," + exciterExtraInfo.getKd() + "," +
                    exciterExtraInfo.getKli() + "," + exciterExtraInfo.getVlir() + "," +
                    exciterExtraInfo.getEfdmax() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "PSS";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (PSS pss : model.getPssList()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
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
        TABLE_DATA_NAME = "PSSExtraInfo";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (PSSExtraInfo pssExtraInfo : model.getPssExtraInfos()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
                    "'" + pssExtraInfo.getType() + "'," + "'" + pssExtraInfo.getGenName() + "'," +
                    pssExtraInfo.getBaseKv() + ",'" + pssExtraInfo.getId() + "'," +
                    pssExtraInfo.getKp() + "," + pssExtraInfo.getT1() + "," +
                    pssExtraInfo.getT2() + "," + pssExtraInfo.getT13() + "," +
                    pssExtraInfo.getT14() + "," + pssExtraInfo.getT3() + "," +
                    pssExtraInfo.getT4() + "," + pssExtraInfo.getMaxVs() + "," +
                    pssExtraInfo.getMinVs() + "," + pssExtraInfo.getIb() + ",'" +
                    pssExtraInfo.getBusName() + "'," + pssExtraInfo.getBusBaseKv() + "," +
                    pssExtraInfo.getXq() + "," + pssExtraInfo.getkMVA() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "Governor";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (Governor governor : model.getGovernors()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
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
        TABLE_DATA_NAME = "GovernorExtraInfo";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (GovernorExtraInfo governorExtraInfo : model.getGovernorExtraInfos()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
                    "'" + governorExtraInfo.getType() + "'," + "'" + governorExtraInfo.getGenName() + "'," +
                    governorExtraInfo.getBaseKv() + "," + "'" + governorExtraInfo.getGeneratorCode() + "'," +
                    governorExtraInfo.getDelt2() + "," + governorExtraInfo.getTr2() + "," +
                    governorExtraInfo.getEp() + "," + governorExtraInfo.getMinusDb2() + "," +
                    governorExtraInfo.getDb2() + "," + governorExtraInfo.getMaxDb2() + "," +
                    governorExtraInfo.getMinDb2() + "," + governorExtraInfo.getItyp() + "," +
                    governorExtraInfo.getItyp2() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "PrimeMover";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (PrimeMover primeMover : model.getPrimeMovers()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
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
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (PV pv : model.getPvs()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
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
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (BC bc : model.getBcs()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
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

        sqls.clear();
        TABLE_DATA_NAME = "BCExtraInfo";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (BCExtraInfo bcExtraInfo : model.getBcExtraInfos()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
                    "'" + bcExtraInfo.getType() + "'," + "'" + bcExtraInfo.getBusName() + "',"
                    + bcExtraInfo.getBaseKv() + "," + "'" + bcExtraInfo.getId() + "'," +
                    bcExtraInfo.getqPercent() + "," + bcExtraInfo.getRpCon() + "," +
                    bcExtraInfo.getTmb() + "," + bcExtraInfo.getTb1() + "," +
                    bcExtraInfo.getTb() + "," + bcExtraInfo.getKpb() + "," +
                    bcExtraInfo.getKib() + "," + bcExtraInfo.getTsb() + "," +
                    bcExtraInfo.getKd() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "Servo";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (Servo servo : model.getServos()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
                    "'" + servo.getType() + servo.getSubType() + "'," +
                    "'" + servo.getGenName() + "'," + servo.getBaseKv() + "," +
                    "'" + servo.getGeneratorCode() + "'," + servo.getPe() + "," +
                    servo.getTc() + "," + servo.getTo() + "," +
                    servo.getCloseVel() + "," + servo.getOpenVel() + "," +
                    servo.getMaxPower() + "," + servo.getMinPower() + "," +
                    servo.getT1() + "," + servo.getKp() + "," +
                    servo.getKd() + "," + servo.getKi() + "," +
                    servo.getMaxIntg() + "," + servo.getMinIntg() + "," +
                    servo.getMaxPID() + ",'" + servo.getMinPID() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "Load";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (Load load : model.getLoads()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
                    "'" + load.getType() + load.getSubType() + "','" + load.getChgCode() + "'," +
                    "'" + load.getBusName() + "'," + load.getBaseKv() + "," +
                    "'" + load.getZone() + "','" + load.getAreaName() + "'," +
                    load.getP1() + "," + load.getQ1() + "," +
                    load.getP2() + "," + load.getQ2() + "," +
                    load.getP3() + "," + load.getQ3() + "," +
                    load.getP4() + "," + load.getQ4() + "," +
                    load.getLdp() + "," + load.getLdq() + "," +
                    "'" + load.getId() + "'," + load.gettJ() + "," +
                    load.getPowerPercent() + "," + load.getLoadRate() + "," +
                    load.getMinPower() + "," + load.getRs() + "," +
                    load.getXs() + "," + load.getXm() + "," +
                    load.getRr() + "," + load.getXr() + "," +
                    load.getVi() + "," + load.getTi() + "," +
                    load.getA() + "," + load.getB() + "," +
                    "'" + load.getS() + "'," + load.getIm() + "," +
                    "'" + load.getTotal() + "'," + load.getDp() + "," +
                    load.getDq() + "," + load.getDt() + "," +
                    load.getTend() + ",'" + load.getSpecCode() + "'" +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "ShortCircuitFault";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (ShortCircuitFault shortCircuitFault : model.getShortCircuitFaults()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
                    "'LS','" + shortCircuitFault.getBusASign() + "'," +
                    "'" + shortCircuitFault.getBusAName() + "'," + shortCircuitFault.getBusABaseKv() + "," +
                    "'" + shortCircuitFault.getBusBSign() + "','" + shortCircuitFault.getBusBName() + "'," +
                    shortCircuitFault.getBusBBaseKv() + ",'" + shortCircuitFault.getParallelBranchCode() + "'," +
                    shortCircuitFault.getMode() + "," + shortCircuitFault.getStartCycle() + "," +
                    shortCircuitFault.getFaultR() + "," + shortCircuitFault.getFaultX() + "," +
                    shortCircuitFault.getPosPercent() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "FLTCard";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        for (FLTCard fltCard : model.getFltCards()) {
            String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                    "'" + psId + "'," +
                    "'" + fltCard.getType() + "','" + fltCard.getBusAName() + "'," +
                    fltCard.getBusABaseKv() + ",'" + fltCard.getBusBName() + "'," +
                    fltCard.getBusBBaseKv() + ",'" + fltCard.getCircuitId() + "'," +
                    fltCard.getFltType() + "," + fltCard.getPhase() + "," +
                    fltCard.getSide() + "," + fltCard.getTcyc0() + "," +
                    fltCard.getTcyc1() + "," + fltCard.getTcyc2() + "," +
                    fltCard.getPosPercent() + "," + fltCard.getFaultR() + "," +
                    fltCard.getFaultX() + "," + fltCard.getTcyc11() + "," +
                    fltCard.getTcyc21() + "," + fltCard.getTcyc12() + "," +
                    fltCard.getTcyc22() +
                    ")";
            sqls.add(insertSql);
        }
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "FFCard";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        FFCard ffCard = model.getFf();
        String insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                "'" + psId + "'," +
                "'" + ffCard.getType() + "'," + ffCard.getT() + "," +
                ffCard.getDt() + "," + ffCard.getEndT() + "," +
                ffCard.getDtc() + "," + ffCard.getIstp() + "," +
                ffCard.getToli() + "," + ffCard.getIlim() + "," +
                ffCard.getDelAng() + "," + ffCard.getDc() + "," +
                ffCard.getDmp() + "," + ffCard.getFrqBse() + "," +
                ffCard.getLovtex() + "," + ffCard.getImblok() + "," +
                ffCard.getMfdep() + "," + ffCard.getIgslim() + "," +
                ffCard.getLsolqit() + "," + ffCard.getNoAngLim() + "," +
                ffCard.getInfBus() + "," + ffCard.getNoPp() + "," +
                ffCard.getNoDq() + "," + ffCard.getNoSat() + "," +
                ffCard.getNoGv() + "," + ffCard.getIeqpc() + "," +
                ffCard.getNoEx() + "," + ffCard.getMftomg() + "," +
                ffCard.getNoSc() + "," + ffCard.getMgtomf() + "," +
                ffCard.getNoLoad() +
                ")";
        sqls.add(insertSql);
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "F0Card";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        F0Card f0Card = model.getF0();
        insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                "'" + psId + "','" + f0Card.getType() + "'," +
                f0Card.getIg() + "," + f0Card.getIa() + "," +
                "'" + f0Card.getGenName1() + "'," + f0Card.getGenBaseKv1() + "," +
                "'" + f0Card.getId1() + "','" + f0Card.getGenName2() + "'," +
                f0Card.getGenBaseKv2() + ",'" + f0Card.getId2() + "'," +
                f0Card.getAmax() + "," + f0Card.getAmin() + "," +
                f0Card.getIv()+ ",'" + f0Card.getBusName1() + "'," +
                f0Card.getBusBaseKv1() + "," + f0Card.getiF() + "," +
                "'" + f0Card.getBusName2() + "'," + f0Card.getBusBaseKv2() +
                ")";
        sqls.add(insertSql);
        sqliteDb.executeSqls(sqls);

        sqls.clear();
        TABLE_DATA_NAME = "CaseCard";
        sql = "delete from " + TABLE_DATA_NAME + " where psId='" + psId + "'";
        sqliteDb.executeSql(sql);
        CaseCard caseCard = model.getCaseCard();
        insertSql = "insert into " + TABLE_DATA_NAME + " values(" +
                "'" + psId + "'," +
                "'" + caseCard.getType() + "','" + caseCard.getCaseId() + "'," +
                caseCard.getiNoPersist() + "," + caseCard.getIxo() + "," +
                caseCard.getDsw() + "," + caseCard.getIwscc() + "," +
                caseCard.getX2fac() + "," + caseCard.getXfact() + "," +
                caseCard.getTdodps() + "," + caseCard.getTqodps() + "," +
                caseCard.getTdodph() + "," + caseCard.getTqodph() + "," +
                caseCard.getCfacl2() +
                ")";
        sqls.add(insertSql);
        sqliteDb.executeSqls(sqls);
    }

    public static void write(String dbFile, String psId, String caseID, String inputPath, String outputPath) {
       write(dbFile, psId, caseID, new File(inputPath), new File(outputPath));
    }

    public static void write(String dbFile, String psId, String caseID, File inputFile, File outputFile) {
        try {
            write(dbFile, psId, caseID, new FileInputStream(inputFile), new FileOutputStream(outputFile));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static void write(String dbFile, String psId, String caseID, InputStream in, OutputStream out) {
        SqliteDb sqliteDb = new SqliteDb(dbFile);
        BpaSwiModel modifiedModel = new BpaSwiModel();
        List<Object> objects = sqliteDb.queryData("Generator", psId);
        List<Generator> generators = new ArrayList<>(objects.size());
        for (Object object : objects) {
            generators.add((Generator) object);
        }
        modifiedModel.setGenerators(generators);
        objects.clear();
        objects = sqliteDb.queryData("GeneratorDW", psId);
        List<GeneratorDW> generatorDWS = new ArrayList<>(objects.size());
        for (Object object : objects) {
            generatorDWS.add((GeneratorDW) object);
        }
        modifiedModel.setGeneratorDws(generatorDWS);
        objects.clear();
        objects = sqliteDb.queryData("Exciter", psId);
        List<Exciter> exciters = new ArrayList<>(objects.size());
        for (Object object : objects) {
            exciters.add((Exciter) object);
        }
        modifiedModel.setExciters(exciters);
        objects.clear();
        objects = sqliteDb.queryData("ExciterExtraInfo", psId);
        List<ExciterExtraInfo> exciterExtraInfos = new ArrayList<>(objects.size());
        for (Object object : objects) {
            exciterExtraInfos.add((ExciterExtraInfo) object);
        }
        modifiedModel.setExciterExtraInfos(exciterExtraInfos);
        objects.clear();
        objects = sqliteDb.queryData("PSS", psId);
        List<PSS> psses = new ArrayList<>(objects.size());
        for (Object object : objects) {
            psses.add((PSS) object);
        }
        modifiedModel.setPssList(psses);
        objects.clear();
        objects = sqliteDb.queryData("PSSExtraInfo", psId);
        List<PSSExtraInfo> pssExtraInfos = new ArrayList<>(objects.size());
        for (Object object : objects) {
            pssExtraInfos.add((PSSExtraInfo) object);
        }
        modifiedModel.setPssExtraInfos(pssExtraInfos);
        objects.clear();
        objects = sqliteDb.queryData("PrimeMover", psId);
        List<PrimeMover> primeMovers = new ArrayList<>(objects.size());
        for (Object object : objects) {
            primeMovers.add((PrimeMover) object);
        }
        modifiedModel.setPrimeMovers(primeMovers);
        objects.clear();
        objects = sqliteDb.queryData("Governor", psId);
        List<Governor> governors = new ArrayList<>(objects.size());
        for (Object object : objects) {
            governors.add((Governor) object);
        }
        modifiedModel.setGovernors(governors);
        objects.clear();
        objects = sqliteDb.queryData("GovernorExtraInfo", psId);
        List<GovernorExtraInfo> governorExtraInfos = new ArrayList<>(objects.size());
        for (Object object : objects) {
            governorExtraInfos.add((GovernorExtraInfo) object);
        }
        modifiedModel.setGovernorExtraInfos(governorExtraInfos);
        objects.clear();
        objects = sqliteDb.queryData("PV", psId);
        List<PV> pvs = new ArrayList<>(objects.size());
        for (Object object : objects) {
            pvs.add((PV) object);
        }
        modifiedModel.setPvs(pvs);
        objects.clear();
        objects = sqliteDb.queryData("BC", psId);
        List<BC> bcs= new ArrayList<>(objects.size());
        for (Object object : objects) {
            bcs.add((BC) object);
        }
        modifiedModel.setBcs(bcs);
        objects.clear();
        objects = sqliteDb.queryData("BCExtraInfo", psId);
        List<BCExtraInfo> bcExtraInfos = new ArrayList<>(objects.size());
        for (Object object : objects) {
            bcExtraInfos.add((BCExtraInfo) object);
        }
        modifiedModel.setBcExtraInfos(bcExtraInfos);
        objects.clear();
        objects = sqliteDb.queryData("Servo", psId);
        List<Servo> servos = new ArrayList<>(objects.size());
        for (Object object : objects) {
            servos.add((Servo) object);
        }
        modifiedModel.setServos(servos);
        objects.clear();
        objects = sqliteDb.queryData("Load", psId);
        List<Load> loads = new ArrayList<>(objects.size());
        for (Object object : objects) {
            loads.add((Load) object);
        }
        modifiedModel.setLoads(loads);
        objects.clear();
        objects = sqliteDb.queryData("ShortCircuitFault", psId);
        List<ShortCircuitFault> shortCircuitFaults = new ArrayList<>(objects.size());
        for (Object object : objects) {
            shortCircuitFaults.add((ShortCircuitFault) object);
        }
        modifiedModel.setShortCircuitFaults(shortCircuitFaults);
        objects.clear();
        objects = sqliteDb.queryData("FLTCard", psId);
        List<FLTCard> fltCards = new ArrayList<>(objects.size());
        for (Object object : objects) {
            fltCards.add((FLTCard) object);
        }
        modifiedModel.setFltCards(fltCards);
        objects.clear();
        modifiedModel.setFf((FFCard) sqliteDb.queryData("FFCard", psId).get(0));
        modifiedModel.setF0((F0Card) sqliteDb.queryData("F0Card", psId).get(0));
        CaseCard caseCard = (CaseCard) sqliteDb.queryData("CaseCard", psId).get(0);
        caseCard.setCaseId(caseID);
        modifiedModel.setCaseCard(caseCard);

        boolean r = BpaSwiModelWriter.readAndWrite(in, "GBK", out, "GBK", modifiedModel);
    }
}
