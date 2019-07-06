package zju.bpamodel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.bpamodel.pf.*;
import zju.bpamodel.swi.*;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Xu Chengsi
 * Date: 19-5-13
 */
public class BpaPfModelWriter {
    private static Logger log = LogManager.getLogger(BpaPfModelWriter.class);

    public static boolean writeFile(String file, String charset, ElectricIsland electricIsland) {
        return writeFile(new File(file), charset, electricIsland);
    }

    public static boolean writeFile(String file, ElectricIsland electricIsland) {
        return writeFile(new File(file), electricIsland);
    }

    public static boolean writeFile(File file, String charset, ElectricIsland electricIsland) {
        try {
            return write(new FileOutputStream(file), charset, electricIsland);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return false;
        }
    }

    public static boolean writeFile(File file, ElectricIsland electricIsland) {
        try {
            return write(new FileOutputStream(file), electricIsland);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return false;
        }
    }

    public static boolean write(OutputStream out, String charset, ElectricIsland electricIsland) {
        try {
            return write(new OutputStreamWriter(out, charset), electricIsland);
        } catch (UnsupportedEncodingException e) {
            log.warn(e);
            e.printStackTrace();
            return false;
        }
    }

    public static boolean write(OutputStream out, ElectricIsland electricIsland) {
        return write(new OutputStreamWriter(out), electricIsland);
    }

    public static boolean write(Writer out, ElectricIsland electricIsland) {
        try {
            BufferedWriter writer = new BufferedWriter(out);
            writer.write(""); //todo: not finished
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    public static boolean readAndWrite(File srcFile, String inputCharset, File targetFile, String outputCharset, String caseID, ElectricIsland electricIsland) {
        try {
            return readAndWrite(new FileInputStream(srcFile), inputCharset, new FileOutputStream(targetFile), outputCharset, caseID, electricIsland);
        } catch (IOException e) {
            log.warn(e);
            e.printStackTrace();
            return false;
        }
    }

    public static boolean readAndWrite(File srcFile, File targetFile, String caseID, ElectricIsland electricIsland) {
        try {
            return readAndWrite(new FileReader(srcFile), new FileWriter(targetFile), caseID, electricIsland);
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    public static boolean readAndWrite(InputStream in, String inputCharset, OutputStream out, String outputCharset, String caseID, ElectricIsland electricIsland) {
        try {
            return readAndWrite(new InputStreamReader(in, inputCharset), new OutputStreamWriter(out, outputCharset), caseID, electricIsland);
        } catch (UnsupportedEncodingException e) {
            log.warn("读写BPA潮流数据时发生[不支持的编码]错误:" + e.getMessage());
            return false;
        }
    }

    public static boolean readAndWrite(InputStream in, OutputStream out, String caseID, ElectricIsland electricIsland) {
        return readAndWrite(new InputStreamReader(in), new OutputStreamWriter(out), caseID, electricIsland);
    }


    public static boolean readAndWrite(Reader in, Writer out, String caseID, ElectricIsland electricIsland) {
        try {
            BufferedReader reader = new BufferedReader(in);
            String strLine;

            while ((strLine = reader.readLine()) != null) {
                if (strLine.startsWith("(POWERFLOW")) {
                    out.write("(POWERFLOW,CASEID=" + caseID + "," + strLine.split(",")[2]);
                    out.write("\n");
                    break;
                }
            }

            while ((strLine = reader.readLine()) != null) {
                if (!strLine.startsWith("/NEW_BASE,FILE=")) {
                    out.write(strLine);
                    out.write("\n");
                } else {
                    out.write("/NEW_BASE,FILE=" + caseID + "." + strLine.split("\\.")[1]);
                    out.write("\n");
                    break;
                }
            }

            while ((strLine = reader.readLine()) != null) {
                out.write(strLine);
                out.write("\n");
                if (strLine.startsWith("/NETWORK_DATA")) {
                    break;
                }
            }

            for (PowerExchange powerExchange : electricIsland.getPowerExchanges()) {
                out.write(powerExchange.toString());
                out.write("\n");
            }
            for (Bus bus : electricIsland.getBuses()) {
                out.write(bus.toString());
                out.write("\n");
            }
            out.write("\n");
            for (AcLine acLine : electricIsland.getAclines()) {
                out.write(acLine.toString());
                out.write("\n");
            }
            out.write("\n");
            for (Transformer transformer : electricIsland.getTransformers()) {
                out.write(transformer.toString());
                out.write("\n");
            }

            while ((strLine = reader.readLine()) != null) {
                if (strLine.startsWith("(END)") || strLine.startsWith("/") || strLine.startsWith(">")) {
                    out.write(strLine);
                    out.write("\n");
                }
            }
            reader.close();
            in.close();
            out.close();
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }
}
