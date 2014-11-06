package zju.ieeeformat;

import org.apache.log4j.Logger;

import java.io.*;
import java.util.List;

/**
 * Class IcfWriter
 * <p> write file in ieee common format </P>
 * Copyright (c) Dong Shufeng
 *
 * @author Dong Shufeng
 * @version 1.0
 * @Date: 2006-8-29
 */
public class IcfWriter {

    private static Logger log = Logger.getLogger(IcfWriter.class);
    private IEEEDataIsland island;

    private DataOutputFormat format = new DataOutputFormat();

    public IcfWriter() {
    }

    public IcfWriter(IEEEDataIsland island) {
        this();
        this.island = island;
    }

    public void setIsland(IEEEDataIsland island) {
        this.island = island;
    }

    public boolean write(String filePath) {
        return write(new File(filePath));
    }

    public boolean write(String filePath, String charset) {
        return write(new File(filePath), charset);
    }

    public boolean write(File file, String charset) {
        try {
            return write(new FileOutputStream(file), charset);
        } catch (FileNotFoundException e) {
            log.warn("输出IEEE标准数据文件时发生[找不到文件]的错误:" + e.getMessage());
            return false;
        }
    }

    public boolean write(File file) {
        try {
            return write(new FileOutputStream(file));
        } catch (FileNotFoundException e) {
            log.warn("输出IEEE标准数据文件时发生[找不到文件]的错误:" + e.getMessage());
            return false;
        }
    }

    public boolean write(OutputStream stream) {
        format.setCharset(System.getProperty("file.encoding"));
        return write(new OutputStreamWriter(stream));
    }

    public boolean write(OutputStream stream, String charset) {
        try {
            format.setCharset(charset);
            return write(new OutputStreamWriter(stream, charset));
        } catch (UnsupportedEncodingException e) {
            log.warn("输出IEEE标准数据文件时发生[不支持的编码]的错误:" + e.getMessage());
            return false;
        }
    }

    public boolean write(Writer writer) {
        if (island == null)
            return false;
        TitleData title = island.getTitle();
        List<BusData> buses = island.getBuses();
        List<BranchData> branches = island.getBranches();
        List<LossZoneData> lossZones = island.getLossZones();
        List<InterchangeData> interchanges = island.getInterchanges();
        List<TieLineData> tieLines = island.getTieLines();
        if (title == null) {
            log.info("no ieee common format data!");
            return false;
        }
        try {
            BufferedWriter w = new BufferedWriter(writer);
            w.write(createTitleStr(title));
            w.newLine();
            w.write("BUS DATA FOLLOWS                    " + buses.size() + " ITEMS");
            for (BusData data : buses) {
                w.newLine();
                w.write(createBusStr(data));
            }
            w.newLine();
            w.write("-999");
            w.newLine();
            w.write("BRANCH DATA FOLLOWS                    " + branches.size() + " ITEMS");
            for (BranchData data : branches) {
                w.newLine();
                w.write(createBranchString(data));
            }
            w.newLine();
            w.write("-999");
            w.newLine();
            w.write("LOSS ZONES FOLLOWS   " + lossZones.size() + " ITEMS");
            for (LossZoneData data : lossZones) {
                w.newLine();
                w.write(createLossZoneStr(data));
            }
            w.newLine();
            w.write("-99");
            w.newLine();
            w.write("INTERCHANGE DATA FOLLOWS      " + interchanges.size() + " ITEMS");
            for (InterchangeData data : interchanges) {
                w.newLine();
                w.write(createInterchangeStr(data));
            }
            w.newLine();
            w.write("-9");
            w.newLine();
            w.write("TIE LINES FOLLOWS             " + tieLines.size() + "ITEMS");
            for (TieLineData data : tieLines) {
                w.newLine();
                w.write(createTieLineStr(data));
            }
            w.newLine();
            w.write("-999");
            w.newLine();
            w.write("END OF DATA");
            w.close();
            writer.close();
        } catch (IOException e) {
            log.error("error occured when trying to write stream to file !!!");
            return false;
        }
        return true;
    }

    private String createTitleStr(TitleData title) {
        String strLine = " ";
        strLine += format.getFormatStr(title.getDate(), "8L") + " ";
        strLine += format.getFormatStr(title.getOriginatorName(), "20L") + " ";
        strLine += format.getFormatStr(Double.toString(title.getMvaBase()), "6L") + " ";
        strLine += format.getFormatStr(Integer.toString(title.getYear()), "4L") + " ";
        strLine += format.getFormatStr(title.getSeason(), "1") + " ";
        strLine += format.getFormatStr(title.getCaseIdentification(), "28");
        return strLine;
    }

    public String createBusStr(BusData b) {
        String strLine = "";
        if (island.getBuses().size() > 9999)
            strLine += format.getFormatStr(b.getBusNumber(), "4X") + " ";
        else
            strLine += format.getFormatStr(b.getBusNumber(), "4") + " ";
        strLine += format.getFormatStr(b.getName(), "12L") + " ";
        strLine += format.getFormatStr(b.getArea(), "2L") + "  ";
        strLine += format.getFormatStr(b.getLossZone(), "3L") + "";
        strLine += format.getFormatStr(b.getType(), "2L") + "";
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getFinalVoltage()), "6L");
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getFinalAngle()), "7L");
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getLoadMW()), "9L");
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getLoadMVAR()), "10L");
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getGenerationMW()), "8L");
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getGenerationMVAR()), "8L")+" ";
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getBaseVoltage()), "7L")+" ";
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getDesiredVolt()), "6L");
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getMaximum()), "8L");
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getMinimum()), "8L");
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getShuntConductance()), "8L");
        //strLine += DataOutputFormat.getFormatStr(Double.toString(getShuntSusceptance()), "8L")+" ";
        //strLine += DataOutputFormat.getFormatStr(Integer.toString(getRemoteControlBusNumber()), "4L");
        strLine += format.getFormatStr(b.getFinalVoltage(), "6.6") + "  ";
        strLine += format.getFormatStr(b.getFinalAngle(), "5.5") + "   ";
        strLine += format.getFormatStr(b.getLoadMW(), "6.6") + "  ";
        strLine += format.getFormatStr(b.getLoadMVAR(), "7.7") + "  ";
        strLine += format.getFormatStr(b.getGenerationMW(), "7.7") + "  ";
        strLine += format.getFormatStr(b.getGenerationMVAR(), "6.6") + "  ";
        strLine += format.getFormatStr(b.getBaseVoltage(), "6.6") + " ";
        strLine += format.getFormatStr(b.getDesiredVolt(), "6.6") + " ";
        strLine += format.getFormatStr(b.getMaximum(), "7.7") + " ";
        strLine += format.getFormatStr(b.getMinimum(), "7.7") + "  ";
        strLine += format.getFormatStr(b.getShuntConductance(), "6.6") + "  ";
        strLine += format.getFormatStr(b.getShuntSusceptance(), "6.6") + " ";
        strLine += format.getFormatStr(Integer.toString(b.getRemoteControlBusNumber()), "4L");
        return strLine;
    }

    public String createBranchString(BranchData b) {
        String strLine = "";
        if (island.getBuses().size() > 9999) {
            strLine += format.getFormatStr(b.getTapBusNumber(), "4X") + " ";
            strLine += format.getFormatStr(b.getZBusNumber(), "4X") + " ";
        } else {
            strLine += format.getFormatStr(b.getTapBusNumber(), "4") + " ";
            strLine += format.getFormatStr(b.getZBusNumber(), "4") + " ";
        }
        strLine += format.getFormatStr(b.getArea(), "2L");
        strLine += format.getFormatStr(b.getLossZone(), "3L") + " ";
        strLine += format.getFormatStr(b.getCircuit(), "1L") + " ";
        strLine += format.getFormatStr(b.getType(), "1L") + "  ";
        strLine += format.getFormatStr(b.getBranchR(), "8.8") + "  ";
        strLine += format.getFormatStr(b.getBranchX(), "8.8") + "   ";
        strLine += format.getFormatStr(b.getLineB(), "7.7") + "  ";
        strLine += format.getFormatStr(b.getMvaRating1(), "5L") + " ";
        strLine += format.getFormatStr(b.getMvaRating2(), "5L") + " ";
        strLine += format.getFormatStr(b.getMvaRating3(), "5L") + "";
        strLine += format.getFormatStr(b.getControlBusNumber(), "4L") + " ";
        strLine += format.getFormatStr(b.getSide(), "1L") + "  "; // 2 blanks
        strLine += format.getFormatStr(b.getTransformerRatio(), "6.6") + "  ";
        strLine += format.getFormatStr(b.getTransformerAngle(), "4.4") + "   ";
        strLine += format.getFormatStr(b.getMinimumTap(), "6.6") + " ";
        strLine += format.getFormatStr(b.getMaximumTap(), "7.7") + " ";
        strLine += format.getFormatStr(b.getStepSize(), "6.6") + " ";
        strLine += format.getFormatStr(b.getMinimum(), "6.6") + " ";
        strLine += format.getFormatStr(b.getMaximum(), "7.7");
        return strLine;
    }

    public String createLossZoneStr(LossZoneData l) {
        String str = "";
        str += format.getFormatStr(l.getLossZoneNumber(), "3L") + " ";
        str += format.getFormatStr(l.getLossZoneName(), "12L");
        return str;
    }

    public String createInterchangeStr(InterchangeData i) {
        String strLine = "";
        strLine += format.getFormatStr(Integer.toString(i.getAreaNumber()), "2") + " ";
        strLine += format.getFormatStr(Integer.toString(i.getSlackBusNumber()), "4L") + " ";
        strLine += format.getFormatStr(i.getAlternateSwingBusName(), "12L");
        strLine += format.getFormatStr(Double.toString(i.getAreaExport()), "8L") + " ";
        strLine += format.getFormatStr(Double.toString(i.getAreaTolerance()), "6L") + "  ";
        strLine += format.getFormatStr(i.getAreaCode(), "6L") + "  ";
        strLine += format.getFormatStr(i.getAreaName(), "30L");

        return strLine;
    }

    private String createTieLineStr(TieLineData data) {
        //todo: not finished..
        return "";
    }

    public String toString() {
        StringWriter w = new StringWriter();
        if (write(w))
            return w.getBuffer().toString();
        return super.toString();
    }
}
