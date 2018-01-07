package zju.dsieeeformat;

import com.csvreader.CsvReader;
import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;
import org.apache.poi.poifs.filesystem.POIFSFileSystem;
import org.apache.poi.ss.usermodel.CellType;
import zju.dsmodel.DsModelCons;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;

/**
 * @Description: 处理8500节点原始数据形成Java程序的可读版本。
 * 根据说明文件中所述，可忽略120V低压配电网复杂模型，
 * 将低压配电变压器后的等值网络用一次绕组的等效负荷进行替换，
 * 等效负荷数据采用Solutions Files中的计算结果，
 * 该处理方式会减少较多节点数，所形成的文件中拓扑结构只含有中压配电网数据。
 * @Author: Fang Rui
 * @Date: 2018/1/5
 * @Time: 11:02
 */
public class DsIeee8500NodeWriter implements DsModelCons {

    private String path;
    private String fileName;
    private File outputFile;

    private ArrayList<String[]> linesList = new ArrayList<String[]>();
    private ArrayList<String[]> capacitorsList = new ArrayList<String[]>();
    private ArrayList<String[]> transformersList = new ArrayList<String[]>();
    private ArrayList<String[]> loadXfmrsList = new ArrayList<String[]>();
    private ArrayList<String[]> solutionBalancedList = new ArrayList<String[]>();

    public DsIeee8500NodeWriter(String path, String fileName) {
        this.path = path;
        this.fileName = fileName;
    }

    public DsIeee8500NodeWriter() {
    }

    //获取原始数据
    public void readRawData(InputStream linesResource, InputStream capacitorsResource, InputStream transformersResource, InputStream loadXfmrsResource) {
        System.out.println("########################### 读取原始数据开始 ###########################");
        try {
            CsvReader linesReader = new CsvReader(linesResource, Charset.forName("UTF-8"));
            linesReader.readHeaders();
            linesReader.readHeaders();
            while (linesReader.readRecord()) {
                linesList.add(linesReader.getValues());
                System.out.println(linesReader.getRawRecord());
            }
            linesReader.close();

            CsvReader capacitorsReader = new CsvReader(capacitorsResource, Charset.forName("UTF-8"));
            capacitorsReader.readHeaders();
            capacitorsReader.readHeaders();
            capacitorsReader.readHeaders();
            while (capacitorsReader.readRecord()) {
                capacitorsList.add(capacitorsReader.getValues());
                System.out.println(capacitorsReader.getRawRecord());
            }
            capacitorsReader.close();

            CsvReader transformersReader = new CsvReader(transformersResource, Charset.forName("UTF-8"));
            transformersReader.readHeaders();
            transformersReader.readHeaders();
            while (transformersReader.readRecord()) {
                transformersList.add(transformersReader.getValues());
                System.out.println(transformersReader.getRawRecord());
            }
            transformersReader.close();

            CsvReader loadXfmrsReader = new CsvReader(loadXfmrsResource, Charset.forName("UTF-8"));
            loadXfmrsReader.readHeaders();
            loadXfmrsReader.readHeaders();
            while (loadXfmrsReader.readRecord()) {
                loadXfmrsList.add(loadXfmrsReader.getValues());
                System.out.println(loadXfmrsReader.getRawRecord());
            }
            loadXfmrsReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("########################### 读取原始数据结束 ###########################");
    }

    public void readSolutionBalancedData(InputStream solutionBalancedResource) {
        try {
            POIFSFileSystem fileSystem = new POIFSFileSystem(solutionBalancedResource);
            HSSFWorkbook workbook = new HSSFWorkbook(fileSystem);
            HSSFSheet sheet = workbook.getSheetAt(2);
            int rowNum = sheet.getLastRowNum() + 1;
            for (int i = 1; i < rowNum; i++) {
                HSSFRow row = sheet.getRow(i);
                String[] rowString = new String[4];
                for (int j = 0; j < 4; j++) {
                    row.getCell(j).setCellType(CellType.STRING);
                    rowString[j] = row.getCell(j).getStringCellValue();
                }
                solutionBalancedList.add(rowString);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void createFile() {
        String fullPath = path + fileName + ".txt";//文件路径+名称+文件类型
        outputFile = new File(fullPath);
        try {
            //如果文件不存在，则创建新的文件
            if (!outputFile.exists()) {
                outputFile.createNewFile();
                System.out.println("创建文件成功，路径名为" + fullPath);
            }
            FileWriter fileWriter = new FileWriter(outputFile);
            fileWriter.write("");
            fileWriter.flush();
            fileWriter.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void writeLineSegmentData() {
        int lineSegmentDataLength = linesList.size();
        int transformerDataLength = transformersList.size();

        try {
            FileWriter fileWriter = new FileWriter(outputFile);

            for (String[] strings : linesList) {
                if (strings[7].contains("Connector")) {
                    lineSegmentDataLength++;
                }
            }
            lineSegmentDataLength += transformerDataLength;

            fileWriter.write("Line Segment Data\t" + lineSegmentDataLength + " items\t" + LEN_UNIT_KILOMETER + "\n");
            for (String[] strings : transformersList) {
                // 去除regxfmr的前缀，这是因为忽略了调压器，将两个节点收缩为一个节点
                if (strings[3].startsWith("regxfmr")) {
                    String removePrefixString = strings[3].replaceAll("regxfmr", "");
                    fileWriter.write(strings[2] + "\t" + removePrefixString + "\t" + "0" + "\t" + strings[0] + "\n");
                } else {
                    fileWriter.write(strings[2] + "\t" + strings[3] + "\t" + "0" + "\t" + strings[0] + "\n");
                }
            }
            for (String[] strings : linesList) {
                // 去除regxfmr_的前缀，这是因为忽略了调压器，将两个节点收缩为一个节点
                if (strings[3].startsWith("regxfmr_")) {
                    strings[3] = strings[3].replaceAll("regxfmr_", "");
                }

                if (strings[7].contains("Connector")) {
                    // 对开关进行特殊处理，增加一个虚拟节点，将阻抗和纯开关分开
                    fileWriter.write(strings[1] + "\t" + strings[1] + "--" + strings[3] + "\t" + strings[5] + "\t" + strings[7] + "-" + strings[2] + "\n");
                    fileWriter.write(strings[1] + "--" + strings[3] + "\t" + strings[3] + "\t" + "0" + "\t" + "Switch" + "\n");
                } else {
                    fileWriter.write(strings[1] + "\t" + strings[3] + "\t" + strings[5] + "\t" + strings[7] + "-" + strings[2] + "\n");
                }
            }
            fileWriter.write("-999\n");
            fileWriter.flush();
            fileWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeSpotLoadData() {
        try {
            FileWriter fileWriter = new FileWriter(outputFile, true);
            int spotLoadDataLength = loadXfmrsList.size();
            fileWriter.write("Spot Loads\t" + spotLoadDataLength + " items\n");
            for (String[] strings : loadXfmrsList) {
                for (String[] solutionStrings : solutionBalancedList) {
                    if (solutionStrings[0].toLowerCase().contains(strings[0].toLowerCase()) && solutionStrings[1].equals("1")) {
                        if (strings[3].equals("A")) {
                            fileWriter.write(strings[2] + "\t" + "Y-PQ" + "\t" + solutionStrings[2] + "\t" + solutionStrings[3] + "\t0\t0\t0\t0\n");
                        }
                        if (strings[3].equals("B")) {
                            fileWriter.write(strings[2] + "\t" + "Y-PQ" + "\t" + "0\t0\t" + solutionStrings[2] + "\t" + solutionStrings[3] + "\t0\t0\n");
                        }
                        if (strings[3].equals("C")) {
                            fileWriter.write(strings[2] + "\t" + "Y-PQ" + "\t0\t0\t0\t0\t" + solutionStrings[2] + "\t" + solutionStrings[3] + "\n");
                        }
                    }
                }
            }
            fileWriter.write("-999\n");
            fileWriter.flush();
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public void writeDistributedLoadData() {
        try {
            FileWriter fileWriter = new FileWriter(outputFile, true);
            fileWriter.write("Distributed Loads\t0 items\n");
            fileWriter.write("-999\n");
            fileWriter.flush();
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeCapacitorData() {
        try {
            FileWriter fileWriter = new FileWriter(outputFile, true);
            int capacitorDataLength = capacitorsList.size();
            fileWriter.write("Shunt Capacitors\t" + capacitorDataLength + " items\n");
            for (String[] strings : capacitorsList) {
                if (strings[2].equals("A") || strings[2].equals("ABC")) {
                    fileWriter.write(strings[1] + "\t" + strings[4] + "\t" + strings[4] + "\t" + strings[4] + "\n");
                }
            }
            fileWriter.write("-999\n");
            fileWriter.flush();
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeTransformerData() {
        try {
            FileWriter fileWriter = new FileWriter(outputFile, true);
            int transformerDataLength = transformersList.size();
            fileWriter.write("Transformer\t" + transformerDataLength + " items\n");
            for (String[] strings : transformersList) {
                String Kva = String.valueOf((int) (Double.parseDouble(strings[6]) * 1000));
                String connPri = "";
                String connSec = "";
                // 低压侧中性点应该不接地，但这里可能没法写成D-Y
                if (strings[7].equals("Delta")) {
                    connPri = "D";
                } else if (strings[7].equals("Wye")) {
                    connPri = "Gr.Y";
                }
                if (strings[8].equals("Delta")) {
                    connSec = "D";
                } else if (strings[8].equals("Wye")) {
                    connSec = "Gr.Y";
                }
                fileWriter.write(strings[0] + "\t" + Kva + "\t" + connPri + "-" + connSec + "\t" +
                        strings[4] + "\t" + strings[5] + "\t" + strings[10] + "\t" + strings[9] + "\n");
            }
            fileWriter.write("-999\n");
            fileWriter.flush();
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void writeRegulatorData() {
        try {
            FileWriter fileWriter = new FileWriter(outputFile, true);
            fileWriter.write("Regulator\t0 item\n");
            fileWriter.write("-999\n");
            fileWriter.flush();
            fileWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }


}


