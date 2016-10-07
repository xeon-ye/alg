package zju.ieeeformat;

import org.apache.log4j.Logger;
import sun.util.locale.StringTokenIterator;

import java.io.*;
import java.util.ArrayList;

/**
 * Class DefaultIcfParser
 * <p> parse and rebuild file of ieee common data format</P>
 *
 * @author Dong Shufeng
 * @version 1.0
 * Date: 2006-7-16
 */

/**
 * Class DefaultIcfParser
 * <p> parse and rebuild file of ieee common data format</P>
 *
 * @author Dong Shufeng
 * @version 1.0
 * @Date: 2006-7-16
 */
public class DefaultIcfParser {
    private static Logger log = Logger.getLogger(DefaultIcfParser.class);
    private String charset = null;
    private boolean isHexBase = false;
    public StringBuilder busNumberStr;

    public IEEEDataIsland parseString(String ieeeDataString) {
        return this.parse(new StringReader(ieeeDataString));
    }

    public IEEEDataIsland parse(String filePath) {
        return this.parse(new File(filePath));
    }

    public IEEEDataIsland parse(File file) {
        try {
            return this.parse(new BufferedReader(new FileReader(file)));
        } catch (FileNotFoundException e) {
            return null;
        }
    }

    public IEEEDataIsland parse(String filePath, String charset) {
        return this.parse(new File(filePath), charset);
    }

    public IEEEDataIsland parse(File file, String charset) {
        try {
            return this.parse(new FileInputStream(file), charset);
        } catch (FileNotFoundException e) {
            return null;
        }
    }

    public IEEEDataIsland parse(InputStream in, String charset) {
        try {
            this.charset = charset;
            return parse(new InputStreamReader(in, charset));
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
            return null;
        }
    }

    public IEEEDataIsland parse(InputStream in) {
        return parse(new InputStreamReader(in));
    }

    public IEEEDataIsland parse(Reader in) {
        BufferedReader reader = new BufferedReader(in);
        TitleData title = new TitleData();

        ArrayList<BusData> buses = new ArrayList<>();
        ArrayList<BranchData> branches = new ArrayList<>();
        ArrayList<LossZoneData> lossZones = new ArrayList<>();
        ArrayList<InterchangeData> interchanges = new ArrayList<>();
        ArrayList<TieLineData> tieLines = new ArrayList<>();

        busNumberStr = new StringBuilder(50000);
        try {
            String strLine = reader.readLine();
            title.setDate(strLine.substring(1, 9));
            title.setOriginatorName(strLine.substring(10, 30));
            title.setMvaBase(Double.parseDouble(strLine.substring(31, 37).trim()));
            title.setYear(Integer.parseInt(strLine.substring(38, 42).trim()));
            title.setSeason(strLine.substring(43, 44).trim().toCharArray()[0]);
            title.setCaseIdentification(strLine.substring(45));
            strLine = reader.readLine().trim();
            while (!strLine.startsWith("BUS")) {
                strLine = reader.readLine().trim().substring(0, 3);
            }
            log.debug("Bus data line is: [ " + strLine + " ]");
            // deal bus data
            while (true) {
                strLine = reader.readLine();
                log.debug(strLine);
                if (strLine == null || strLine.trim().equalsIgnoreCase("-999"))
                    break;
                if (strLine.trim().equals(""))
                    continue;
                buses.add(buildIeeeBusData(strLine));
            }
            if (buses.size() > 9999)
                isHexBase = true;
            StringTokenIterator iter = new StringTokenIterator(busNumberStr.toString(), ";");
            int i = 0;
            while (iter.hasNext()) {
                if (isHexBase)
                    buses.get(i).setBusNumber(Integer.parseInt(iter.current(), 16));
                else
                    buses.get(i).setBusNumber(Integer.parseInt(iter.current()));
                iter.next();
                i++;
            }

            strLine = reader.readLine().trim();
            while (!strLine.startsWith("BRA")) {
                strLine = reader.readLine().trim().substring(0, 3);
            }
            log.debug("Branch data line is: [ " + strLine + " ]");
            // deal branch data
            while (true) {
                strLine = reader.readLine();
                log.debug(strLine);
                if (strLine == null || strLine.trim().equalsIgnoreCase("-999"))
                    break;
                if (strLine.trim().equals(""))
                    continue;
                branches.add(buildIeeeBranchData(strLine));
            }

            strLine = reader.readLine();
            while (strLine != null && !strLine.trim().startsWith("LOS"))
                strLine = reader.readLine().trim().substring(0, 3);
            log.debug("Loss zone data line is: [ " + strLine + " ]");
            // deal losszone data
            while (true) {
                strLine = reader.readLine();
                if (strLine == null || strLine.trim().equalsIgnoreCase("-99"))
                    break;
                if (strLine.trim().equals(""))
                    continue;
                lossZones.add(buildIeeeLossZoneData(strLine));
            }

            strLine = reader.readLine();
            while (strLine != null && !strLine.trim().startsWith("INT"))
                strLine = reader.readLine().trim().substring(0, 3);
            log.debug("Interchange data line is: [ " + strLine + " ]");
            // deal intercharge data
            while (true) {
                strLine = reader.readLine();
                log.debug(strLine);
                if (strLine == null || strLine.trim().equalsIgnoreCase("-9"))
                    break;
                if (strLine.trim().equals(""))
                    continue;
                interchanges.add(buildIeeeInterchangeData(strLine));
            }

            strLine = reader.readLine();
            while (strLine != null && !strLine.trim().startsWith("TIE"))
                strLine = reader.readLine().trim().substring(0, 3);
            log.debug("Tie line data line is: [ " + strLine + " ]");
            // deal tieline data
            while (true) {
                strLine = reader.readLine();
                log.debug(strLine);
                if (strLine == null || strLine.trim().equalsIgnoreCase("-999"))
                    break;
                if (strLine.trim().equals(""))
                    continue;
                tieLines.add(buildIeeeTieLineData(strLine));
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
            log.error("io exception occured when trying to parse ieee data file!");
            return null;
        }
        return new IEEEDataIsland(title, buses, branches, lossZones, interchanges, tieLines);
    }

    public BusData buildIeeeBusData(String strLine) throws UnsupportedEncodingException {
        byte[] src = charset != null ? strLine.getBytes(charset) : strLine.getBytes();
        BusData data = new BusData();
        busNumberStr.append(new String(getTarget(src, 0, 4)).trim()).append(";");
        // bus number
        //data.setBusNumber(Integer.parseInt(new String(getTarget(src, 0, 4)).trim()));
        // bus name
        data.setName(new String(getTarget(src, 5, 17)).trim());
        // area number
        data.setArea(Integer.parseInt(new String(getTarget(src, 18, 20)).trim()));
        // loss zone number
        data.setLossZone(Integer.parseInt(new String(getTarget(src, 20, 23)).trim()));
        // bus type
        String type = new String(getTarget(src, 24, 26)).trim();
        if (type.equals(""))  //todo_Arno check this is right
            data.setType(0);
        else
            data.setType(Integer.parseInt(new String(getTarget(src, 24, 26)).trim()));
        // current voltage
        data.setFinalVoltage(Double.parseDouble(new String(getTarget(src, 27, 33)).trim()));
        // current angle
        data.setFinalAngle(Double.parseDouble(new String(getTarget(src, 33, 40)).trim()));
        // P load
        data.setLoadMW(Double.parseDouble(new String(getTarget(src, 40, 49)).trim()));
        // Q load
        data.setLoadMVAR(Double.parseDouble(new String(getTarget(src, 49, 59)).trim()));
        // P gen
        data.setGenerationMW(Double.parseDouble(new String(getTarget(src, 59, 67)).trim()));
        // Qgen
        data.setGenerationMVAR(Double.parseDouble(new String(getTarget(src, 67, 75)).trim()));
        // baseKV
        data.setBaseVoltage(Double.parseDouble(new String(getTarget(src, 76, 83)).trim()));
        // Desired volts(p.u.)
        data.setDesiredVolt(Double.parseDouble(new String(getTarget(src, 84, 90)).trim()));
        // Qmax or Vmax
        data.setMaximum(Double.parseDouble(new String(getTarget(src, 90, 98)).trim()));
        // Qmin or Vmin
        data.setMinimum(Double.parseDouble(new String(getTarget(src, 98, 106)).trim()));
        // G shunt
        data.setShuntConductance(Double.parseDouble(new String(getTarget(src, 106, 114)).trim()));
        // B shunt
        data.setShuntSusceptance(Double.parseDouble(new String(getTarget(src, 114, 122)).trim()));
        // remote control bus number
        if (src.length >= 127)
            data.setRemoteControlBusNumber(Integer.parseInt(new String(getTarget(src, 123, 127)).trim()));
        else
            data.setRemoteControlBusNumber(Integer.parseInt(strLine.substring(123)));
        return data;
    }

    /**
     * build line/transformer from ieee branch data
     */
    public BranchData buildIeeeBranchData(String strLine) {
        BranchData data = new BranchData();
        // from bus id
        if (isHexBase)
            data.setTapBusNumber(Integer.parseInt(strLine.substring(0, 4).trim(), 16));
        else
            data.setTapBusNumber(Integer.parseInt(strLine.substring(0, 4).trim()));
        // to bus id
        if (isHexBase)
            data.setZBusNumber(Integer.parseInt(strLine.substring(5, 9).trim(), 16));
        else
            data.setZBusNumber(Integer.parseInt(strLine.substring(5, 9).trim()));
        // area
        data.setArea(Integer.parseInt(strLine.substring(10, 12).trim()));
        // loss zone
        data.setLossZone(Integer.parseInt(strLine.substring(12, 15).trim()));
        // circuit
        String s = strLine.substring(16, 17).trim();
        if (s.equals(""))
            data.setCircuit(1);
        else
            data.setCircuit(Integer.parseInt(s));

        // branch Type
        s = strLine.substring(18, 19).trim();
        if (s.equals(""))
            data.setType(0);
        else
            data.setType(Integer.parseInt(strLine.substring(18, 19).trim()));

        // r
        data.setBranchR(Double.parseDouble(strLine.substring(19, 29).trim()));
        // x
        data.setBranchX(Double.parseDouble(strLine.substring(29, 40).trim()));
        // b
        data.setLineB(Double.parseDouble(strLine.substring(40, 50).trim()));
        // line MVA rating 1
        data.setMvaRating1(parseInt(strLine.substring(50, 55).trim()));
        // line MVA rating 2
        data.setMvaRating2(parseInt(strLine.substring(56, 61).trim()));
        // line MVA rating 3
        data.setMvaRating3(parseInt(strLine.substring(62, 67).trim()));
        // ctrl bus number
        data.setControlBusNumber(Integer.parseInt(strLine.substring(68, 72).trim()));
        // side
        data.setSide(Integer.parseInt(strLine.substring(73, 74).trim()));
        // ************* following data is about trans former *******
        // current ratio
        data.setTransformerRatio(Double.parseDouble(strLine.substring(76, 82).trim()));
        // curret angle
        data.setTransformerAngle(Double.parseDouble(strLine.substring(83, 90).trim()));
        // min tap or phase shift
        data.setMinimumTap(Double.parseDouble(strLine.substring(90, 97).trim()));
        // max tap or phase shift
        data.setMaximumTap(Double.parseDouble(strLine.substring(97, 104).trim()));
        // step size
        data.setStepSize(Double.parseDouble(strLine.substring(105, 111).trim()));
        // ************* above data is about trans former *******

        // min v, MVAR or MW
        data.setMinimum(Double.parseDouble(strLine.substring(112, 119).trim()));
        // max v, MVAR or MW
        if (strLine.length() >= 126)
            data.setMaximum(Double.parseDouble(strLine.substring(119, 126).trim()));
        else
            data.setMaximum(Double.parseDouble(strLine.substring(119).trim()));
        return data;
    }

    public LossZoneData buildIeeeLossZoneData(String strLine) {
        LossZoneData data = new LossZoneData();
        //loss zone number
        data.setLossZoneNumber(Integer.parseInt(strLine.substring(0, 3).trim()));
        //loss zone name
        if (strLine.length() >= 16)
            data.setLossZoneName(strLine.substring(4, 16).trim());
        else
            data.setLossZoneName(strLine.substring(4).trim());
        return data;
    }

    public InterchangeData buildIeeeInterchangeData(String strLine) {
        InterchangeData data = new InterchangeData();
        // area number
        data.setAreaNumber(Integer.parseInt(strLine.substring(0, 2).trim()));
        //interchange slack bus number
        data.setSlackBusNumber(Integer.parseInt(strLine.substring(3, 7).trim()));
        //alternate swing bus number
        data.setAlternateSwingBusName(strLine.substring(8, 20).trim());
        //area interchange export, MW
        data.setAreaExport(Double.parseDouble(strLine.substring(20, 28).trim()));
        //area interchange tolerance, MW
        data.setAreaTolerance(Double.parseDouble(strLine.substring(29, 35)));
        //area code (abbreviated name)
        data.setAreaCode(strLine.substring(37, 43).trim());
        //area name
        data.setAreaName(strLine.substring(45).trim());

        return data;
    }

    public TieLineData buildIeeeTieLineData(String strLine) {
        TieLineData data = new TieLineData();
        //metered bus number
        data.setMeteredBusNum(Integer.parseInt(strLine.substring(0, 4).trim()));
        //metered area number
        data.setMeteredAreaNum(Integer.parseInt(strLine.substring(6, 8).trim()));
        //non-metered bus number
        data.setNonmeteredBusNum(Integer.parseInt(strLine.substring(10, 14).trim()));
        //non-metered area number
        data.setNonmeteredAreaNum(Integer.parseInt(strLine.substring(16, 18).trim()));
        //circuit number
        data.setCircuitNum(Integer.parseInt(strLine.substring(20).trim()));

        return data;
    }

    public int parseInt(String s) {
        if (s.equals(""))
            return 0;
        return Integer.parseInt(s);
    }

    private byte[] getTarget(byte[] src, int start, int end) {
        byte[] target = new byte[end - start];
        System.arraycopy(src, start, target, 0, end - start);
        return target;
    }
}