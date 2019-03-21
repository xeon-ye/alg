package zju.bpamodel.sccpc;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-11-4
 */
public class SccResult implements Serializable {
    private static Logger log = LogManager.getLogger(SccResult.class);
    private Map<String, SccBusResult> busData;

    public Map<String, SccBusResult> getBusData() {
        return busData;
    }

    public void setBusData(Map<String, SccBusResult> busData) {
        this.busData = busData;
    }

    public static SccResult parseFile(String file) {
        return parseFile(new File(file));
    }

    public static SccResult parseFile(String file, String charset) {
        return parseFile(new File(file), charset);
    }

    public static SccResult parseFile(File file, String charset) {
        try {
            return parse(new FileInputStream(file), charset);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static SccResult parseFile(File file) {
        try {
            return parse(new FileInputStream(file));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static SccResult parse(InputStream in, String charset) {
        try {
            return parse(new InputStreamReader(in, charset));
        } catch (UnsupportedEncodingException e) {
            log.warn(e);
            e.printStackTrace();
            return null;
        }
    }

    public static SccResult parse(InputStream in) {
        return parse(new InputStreamReader(in));
    }

    public static SccResult parse(Reader in) {
        try {
            BufferedReader reader = new BufferedReader(in);
            SccResult r = new SccResult();
            String strLine;
            r.setBusData(new HashMap<String, SccBusResult>());
            while ((strLine = reader.readLine()) != null) {
                strLine = strLine.trim();
                //its not a bus, but not sure the condition judgement is perfect
                if (strLine.startsWith("\"")) {
                    log.debug(strLine);
                    SccBusResult busR = new SccBusResult();
                    busR.parseString(strLine);
                    r.getBusData().put(busR.getBusName(), busR);
                }
            }

            return r;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
