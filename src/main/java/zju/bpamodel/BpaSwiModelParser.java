package zju.bpamodel;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.bpamodel.swi.*;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by IntelliJ IDEA.
 * User: Dong Shufeng
 * Date: 12-7-11
 */
public class BpaSwiModelParser {
    private static Logger log = LogManager.getLogger(BpaSwiModelParser.class);

    public static BpaSwiModel parseFile(String file) {
        return parseFile(new File(file));
    }

    public static BpaSwiModel parseFile(String file, String charset) {
        return parseFile(new File(file), charset);
    }

    public static BpaSwiModel parseFile(File file, String charset) {
        try {
            return parse(new FileInputStream(file), charset);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return null;
        }
    }

    public static BpaSwiModel parseFile(File file) {
        try {
            return parse(new FileInputStream(file));
        } catch (FileNotFoundException e) {
            log.warn(e);
            e.printStackTrace();
            return null;
        }
    }


    public static BpaSwiModel parse(InputStream in, String charset) {
        try {
            return parse(new InputStreamReader(in, charset));
        } catch (UnsupportedEncodingException e) {
            log.warn(e);
            e.printStackTrace();
        }
        return null;
    }

    public static BpaSwiModel parse(InputStream in) {
        return parse(new InputStreamReader(in));
    }

    public static BpaSwiModel parse(Reader in) {
        try {
            BufferedReader reader = new BufferedReader(in);
            BpaSwiModel model = new BpaSwiModel();
            String strLine;
            List<GeneratorDW> generatorDws = new ArrayList<GeneratorDW>();
            List<Generator> generators = new ArrayList<Generator>();
            List<Exciter> exciters = new ArrayList<Exciter>();
            List<ExciterExtraInfo> exciterExtraInfos = new ArrayList<ExciterExtraInfo>();
            List<ShortCircuitFault> shortCircuitFaults = new ArrayList<>();
            List<FLTCard> fltCards = new ArrayList<>();
            List<Load> loads = new ArrayList<>();
            List<InductionMotor> inductionMotors = new ArrayList<>();
            FFCard ffCard = null;
            while ((strLine = reader.readLine()) != null) {
                try {
                    if (strLine.startsWith("M")) {
                        if (strLine.charAt(1) == 'C' || strLine.charAt(1) == 'F' || strLine.charAt(1) == 'G')
                            generators.add(Generator.createGen(strLine));
                        else if (strLine.charAt(1) == 'H')
                            ;//todo: no aciton is done for MH data card
                        else if (strLine.charAt(1) == 'L' || strLine.charAt(1) == 'J' || strLine.charAt(1) == 'K')
                            inductionMotors.add(InductionMotor.createInductionMotor(strLine));
                        else
                            generatorDws.add(GeneratorDW.createGenDampingWinding(strLine));
                    } else if (strLine.startsWith("F+") || strLine.startsWith("F#")) {
                        exciterExtraInfos.add(ExciterExtraInfo.createExciterExtraInfo(strLine));
                    } else if (strLine.startsWith("F")) {
                        if ((strLine.charAt(1) >= 'A' && strLine.charAt(1) <= 'H')
                                || (strLine.charAt(1) >= 'J' && strLine.charAt(1) <= 'L')
                                || (strLine.charAt(1) >= 'M' && strLine.charAt(1) <= 'V')) {
                            if (strLine.charAt(1) == 'F' && strLine.charAt(3) == ' ' && strLine.charAt(7) == ' ' && strLine.charAt(11) == ' ') {
                                ffCard = FFCard.createFF(strLine);
                            } else if (strLine.charAt(1) == 'L' && strLine.charAt(2) == 'T') {
                                fltCards.add(FLTCard.createFault(strLine));
                            } else {
                                exciters.add(Exciter.createExciter(strLine));
                            }
                        }
                    } else if (strLine.startsWith("E")) {
                        if ((strLine.charAt(1) >= 'A' && strLine.charAt(1) <= 'G')
                                || (strLine.charAt(1) >= 'J' && strLine.charAt(1) <= 'K'))
                            exciters.add(Exciter.createExciter(strLine));
                    } else if (strLine.startsWith("L")) {
                        if (strLine.charAt(1) == 'A' || strLine.charAt(1) == 'B') {
                            loads.add(Load.createLoad(strLine));
                        } else if (strLine.charAt(1) == 'S') {
                            shortCircuitFaults.add(ShortCircuitFault.createFault(strLine));
                        }
                    }
                } catch (NumberFormatException ex) {
                    log.warn("Failed to parse because NumberFormatException is found:");
                    log.warn(strLine);
                } catch (NegativeArraySizeException ex) {
                    log.warn("Failed to parse because NegativeArraySizeException is found:");
                    log.warn(strLine);
                }
            }
            model.setGenerators(generators);
            model.setGeneratorDws(generatorDws);
            model.setExciters(exciters);
            model.setExciterExtraInfos(exciterExtraInfos);
            model.setShortCircuitFaults(shortCircuitFaults);
            model.setFltCards(fltCards);
            model.setLoads(loads);
            model.setInductionMotors(inductionMotors);
            model.setFf(ffCard);
            model.buildMaps();
            return model;
        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}
