package zju.dsmodel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 *
 * @author Dong Shufeng
 * Date: 2010-4-22
 */
public class FeederConfMgr implements Serializable, DsModelCons {
    private transient static Logger log = LogManager.getLogger(FeederConfMgr.class);
    private String lengthUnit;
    private Map<String, double[][]> zRealPerLen = new HashMap<>();
    private Map<String, double[][]> zImagPerLen = new HashMap<>();
    private Map<String, double[][]> bPerLen = new HashMap<>();

    public void calPara(String id, String length, String unit, double baseKva, double baseKv, Feeder feeder) {
        double[][] zReal = zRealPerLen.get(id);
        double[][] zImage = zImagPerLen.get(id);
        double[][] b = bPerLen.get(id);

        double len = 0.0;
        //换算长度
        if (lengthUnit.equals(LEN_UNIT_FEET)) {
            if (unit.equals(LEN_UNIT_FEET))
                len = Double.parseDouble(length);
            else if (unit.equals(LEN_UNIT_MILE))
                len = Double.parseDouble(length) * 5280.0;
            else if (unit.equals(LEN_UNIT_METER))
                len = Double.parseDouble(length) * 3.2808399;
            else if (unit.equals(LEN_UNIT_KILOMETER))
                len = Double.parseDouble(length) * 3280.8398950;
        } else if (lengthUnit.equals(LEN_UNIT_MILE)) {
            if (unit.equals(LEN_UNIT_FEET))
                len = Double.parseDouble(length) / 5280.0;
            else if (unit.equals(LEN_UNIT_MILE))
                len = Double.parseDouble(length);
            else if (unit.equals(LEN_UNIT_METER))
                len = Double.parseDouble(length) * 0.0006214;
            else if (unit.equals(LEN_UNIT_KILOMETER))
                len = Double.parseDouble(length) * 0.6213712;
        } else if (lengthUnit.equals(LEN_UNIT_METER)) {
            if (unit.equals(LEN_UNIT_FEET))
                len = Double.parseDouble(length) * 0.3048;
            else if (unit.equals(LEN_UNIT_MILE))
                len = Double.parseDouble(length) * 1609.344;
            else if (unit.equals(LEN_UNIT_METER))
                len = Double.parseDouble(length);
            else if (unit.equals(LEN_UNIT_KILOMETER))
                len = Double.parseDouble(length) * 1000.0;
        } else if (lengthUnit.equals(LEN_UNIT_KILOMETER)) {
            if (unit.equals(LEN_UNIT_FEET))
                len = Double.parseDouble(length) * 0.0003048;
            else if (unit.equals(LEN_UNIT_MILE))
                len = Double.parseDouble(length) * 1.609344;
            else if (unit.equals(LEN_UNIT_METER))
                len = Double.parseDouble(length) / 1000.0;
            else if (unit.equals(LEN_UNIT_KILOMETER))
                len = Double.parseDouble(length);
        }

        double baseZ = baseKv * baseKv * 1000.0 / baseKva;
        for (int i = 0; i < feeder.getZ_real().length; i++)
            for (int j = 0; j < feeder.getZ_real()[i].length; j++)
                feeder.getZ_real()[i][j] = zReal[i][j] * len / baseZ;
        for (int i = 0; i < feeder.getZ_imag().length; i++)
            for (int j = 0; j < feeder.getZ_imag()[i].length; j++)
                feeder.getZ_imag()[i][j] = zImage[i][j] * len / baseZ;
        //注意这里电纳的单位是微西门子（micro siemens）
        for (int i = 0; i < feeder.getY_imag().length; i++)
            for (int j = 0; j < feeder.getY_imag()[i].length; j++)
                feeder.getY_imag()[i][j] = b[i][j] * len * baseZ * 1e-6;
    }

    public void calPara(String id, String length, String unit, Feeder feeder) {
        calPara(id, length, unit, 1000.0, 1.0, feeder);
    }

    public void readImpedanceConf(String f) throws Exception {
        try {
            readImpedanceConf(new FileInputStream(f));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public void readImpedanceConf(File f) throws Exception {
        try {
            readImpedanceConf(new FileInputStream(f));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public String getLengthUnit() {
        return lengthUnit;
    }

    public void setLengthUnit(String lengthUnit) {
        this.lengthUnit = lengthUnit;
    }

    public void readImpedanceConf(InputStream stream) throws Exception {
        zRealPerLen.clear();
        zImagPerLen.clear();
        bPerLen.clear();
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String str;
        setLengthUnit(reader.readLine().trim());
        while ((str = reader.readLine()) != null) {
            if (str.startsWith("#"))
                continue;
            if (str.equals(""))
                continue;
            double[][] real = new double[3][3];
            double[][] imag = new double[3][3];
            double[][] b = new double[3][3];
            zRealPerLen.put(str, real);
            zImagPerLen.put(str, imag);
            bPerLen.put(str, b);

            String[] s = new String[6];
            for (int i = 0; i < s.length; i++)
                s[i] = reader.readLine();
            for (int i = 0; i < 3; i++) {
                real[i] = new double[3];
                String[] ss = s[i].split("\t");
                for (int j = 0; j < 3; j++) {
                    real[i][j] = Double.parseDouble(ss[2 * j]);
                    imag[i][j] = Double.parseDouble(ss[2 * j + 1]);
                }
            }
            for (int i = 0; i < 3; i++) {
                String[] ss = s[i + 3].split("\t");
                for (int j = 0; j < 3; j++) {
                    b[i][j] = Double.parseDouble(ss[j]);
                }
            }
        }
        reader.close();

    }

    public Map<String, double[][]> getzRealPerLen() {
        return zRealPerLen;
    }

    public void setzRealPerLen(Map<String, double[][]> zRealPerLen) {
        this.zRealPerLen = zRealPerLen;
    }

    public Map<String, double[][]> getzImagPerLen() {
        return zImagPerLen;
    }

    public void setzImagPerLen(Map<String, double[][]> zImagPerLen) {
        this.zImagPerLen = zImagPerLen;
    }

    public Map<String, double[][]> getbPerLen() {
        return bPerLen;
    }

    public void setbPerLen(Map<String, double[][]> bPerLen) {
        this.bPerLen = bPerLen;
    }

    public void clearzRealPerLen() {
        this.zRealPerLen.clear();
    }

    public void clearzImagPerLen() {
        this.zImagPerLen.clear();
    }

    public void clearbPerLen() {
        this.bPerLen.clear();
    }

}
