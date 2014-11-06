package zju.bpamodel;

import org.apache.log4j.Logger;
import zju.bpamodel.swi.Exciter;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-17
 */
public class BpaSwiModelWriter {
    private static Logger log = Logger.getLogger(BpaSwiModelWriter.class);

    public static boolean writeFile(String file, String charset, BpaSwiModel swiModel) {
        return writeFile(new File(file), charset, swiModel);
    }

    public static boolean writeFile(String file, BpaSwiModel swiModel) {
        return writeFile(new File(file), swiModel);
    }

    public static boolean writeFile(File file, String charset, BpaSwiModel swiModel) {
        try {
            return write(new FileOutputStream(file), charset, swiModel);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return false;
        }
    }

    public static boolean writeFile(File file, BpaSwiModel swiModel) {
        try {
            return write(new FileOutputStream(file), swiModel);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return false;
        }
    }

    public static boolean write(OutputStream out, String charset, BpaSwiModel swiModel) {
        try {
            return write(new OutputStreamWriter(out, charset), swiModel);
        } catch (UnsupportedEncodingException e) {
            log.warn(e);
            e.printStackTrace();
            return false;
        }
    }

    public static boolean write(OutputStream out, BpaSwiModel swiModel) {
        return write(new OutputStreamWriter(out), swiModel);
    }

    public static boolean write(Writer out, BpaSwiModel swiModel) {
        try {
            BufferedWriter writer = new BufferedWriter(out);
            writer.write(""); //todo: not finished
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    public static boolean readAndWrite(File srcFile, String inputCharset, File targetFile, String outputCharset, BpaSwiModel swiModel) {
        try {
            return readAndWrite(new FileInputStream(srcFile), inputCharset, new FileOutputStream(targetFile), outputCharset, swiModel);
        } catch (IOException e) {
            log.warn(e);
            e.printStackTrace();
            return false;
        }
    }

    public static boolean readAndWrite(File srcFile, File targetFile, BpaSwiModel swiModel) {
        try {
            return readAndWrite(new FileReader(srcFile), new FileWriter(targetFile), swiModel);
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    public static boolean readAndWrite(InputStream in, String inputCharset, OutputStream out, String outputCharset, BpaSwiModel swiModel) {
        try {
            return readAndWrite(new InputStreamReader(in, inputCharset), new OutputStreamWriter(out, outputCharset), swiModel);
        } catch (UnsupportedEncodingException e) {
            log.warn("读写BPA稳定数据时发生[不支持的编码]错误:" + e.getMessage());
            return false;
        }
    }

    public static boolean readAndWrite(InputStream in, OutputStream out, BpaSwiModel swiModel) {
        return readAndWrite(new InputStreamReader(in), new OutputStreamWriter(out), swiModel);
    }


    public static boolean readAndWrite(Reader in, Writer out, BpaSwiModel swiModel) {
        try {
            BufferedReader reader = new BufferedReader(in);
            String strLine;
            //Map<String, Generator> genMap = new HashMap<String, Generator>(swiModel.getGenerators().size());
            //for(Generator gen : swiModel.getGenerators()) {
            //    String id = gen.getType() + "_" + gen.getBusName() + "_" + gen.getId();
            //    genMap.put(id, gen);
            //}
            Map<String, Exciter> exciterMap = new HashMap<String, Exciter>(swiModel.getExciters().size());
            for (Exciter exciter : swiModel.getExciters()) {
                String id = exciter.getType() + exciter.getSubType() + "_" + exciter.getBusName() + "_" + exciter.getGeneratorCode();
                exciterMap.put(id, exciter);
            }

            while ((strLine = reader.readLine()) != null) {
                //log.debug(strLine);
                //if (strLine.startsWith("MC") || strLine.startsWith("MF")) {
                //    try {
                //        Generator gen = Generator.createGen(strLine);
                //        String id = gen.getType() + "_" + gen.getBusName() + "_" + gen.getId();
                //        if(genMap.containsKey(id))
                //            out.write(genMap.get(id).toString());
                //        else
                //            out.write(strLine);
                //    } catch (NumberFormatException ex) {
                //        log.warn("Failed to parse because NumberFormatException is found:");
                //        log.warn(strLine);
                //    } catch (NegativeArraySizeException ex) {
                //        log.warn("Failed to parse because NegativeArraySizeException is found:");
                //        log.warn(strLine);
                //    }
                //} else
                if (strLine.startsWith("FF  ")) {
                    out.write(strLine);//todo: not perfect
                } else if (strLine.startsWith("F")) {
                    if ((strLine.charAt(1) >= 'A' && strLine.charAt(1) <= 'H')
                            || (strLine.charAt(1) >= 'J' && strLine.charAt(1) <= 'L')
                            || (strLine.charAt(1) >= 'M' && strLine.charAt(1) <= 'V')) {
                        try {
                            Exciter exciter = Exciter.createExciter(strLine);
                            String id = exciter.getType() + exciter.getSubType() + "_" + exciter.getBusName() + "_" + exciter.getGeneratorCode();
                            if (exciterMap.containsKey(id))
                                out.write(exciterMap.get(id).toString());
                            else
                                out.write(strLine);
                        } catch (NumberFormatException ex) {
                            ex.printStackTrace();
                            log.warn("Failed to parse because NumberFormatException is found:");
                            log.warn(strLine);
                            return false;
                        } catch (NegativeArraySizeException ex) {
                            ex.printStackTrace();
                            log.warn("Failed to parse because NegativeArraySizeException is found:");
                            log.warn(strLine);
                            return false;
                        }
                    } else
                        out.write(strLine);
                } else if (strLine.startsWith("E")) {
                    if ((strLine.charAt(1) >= 'A' && strLine.charAt(1) <= 'G')
                            || (strLine.charAt(1) >= 'J' && strLine.charAt(1) <= 'K')) {
                        try {
                            Exciter exciter = Exciter.createExciter(strLine);
                            String id = exciter.getType() + exciter.getSubType() + "_" + exciter.getBusName() + "_" + exciter.getGeneratorCode();
                            if (exciterMap.containsKey(id))
                                out.write(exciterMap.get(id).toString());
                            else
                                out.write(strLine);
                        } catch (NumberFormatException ex) {
                            ex.printStackTrace();
                            log.warn("Failed to parse because NumberFormatException is found:");
                            log.warn(strLine);
                            return false;
                        } catch (NegativeArraySizeException ex) {
                            ex.printStackTrace();
                            log.warn("Failed to parse because NegativeArraySizeException is found:");
                            log.warn(strLine);
                            return false;
                        }
                    } else
                        out.write(strLine);
                } else
                    out.write(strLine);
                out.write("\n");
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
