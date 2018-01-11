package zju.dsieeeformat;

import java.io.*;
import java.util.ArrayList;

/**
 * @Description:
 * @Author: Fang Rui
 * @Date: 2018/1/8
 * @Time: 9:38
 */
public class DsIeee8500NodeConfigWriter {

    private String path;
    private String fileName;
    private File outputFile;

    private ArrayList<String[]> configList = new ArrayList<String[]>();

    public DsIeee8500NodeConfigWriter(String path, String fileName) {
        this.path = path;
        this.fileName = fileName;
    }

    public DsIeee8500NodeConfigWriter() {
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

    public void readRawData(InputStream configResource) {
        System.out.println("########################### 读取原始数据开始 ###########################");
        BufferedReader reader = new BufferedReader(new InputStreamReader(configResource));
        String strLine;
        try {
            strLine = reader.readLine(); //光标在第一行
            while (strLine != null) {
                while (!strLine.startsWith("[Linecode]")) {
                    strLine = reader.readLine();
                    if (strLine == null)
                        break;
                }
                // 读完后光标位于[Linecode]
                if (strLine == null)
                    break;
                readOneRawData(strLine, reader);
                strLine = reader.readLine(); //读完一组后光标往下移动一行，该行为空白行
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("########################### 读取原始数据结束 ###########################");
    }

    private void readOneRawData(String strLine, BufferedReader reader) {
        String[] oneRawData = new String[28];
        String[] decoupledConfig = new String[3];
        for (int i = 0; i < oneRawData.length; i++) {
            oneRawData[i] = "0.0";
        }
        for (int i = 0; i < decoupledConfig.length; i++) {
            decoupledConfig[i] = "0.0";
        }
        int phase;
        String unit;

        try {
            // 读入name
            strLine = reader.readLine();
            oneRawData[0] = strLine.trim().replace("name=", "").toUpperCase();
            // 读入phase
            strLine = reader.readLine();
            phase = Integer.parseInt(strLine.trim().replace("nphases=", ""));
            // 读入unit
            strLine = reader.readLine();
            unit = strLine.trim().replace("units=", "");
            if (!unit.equals("km")) {
                System.out.println(oneRawData[0] + "单位不为km");
                return;
            }

            // 读入数据开始
            strLine = reader.readLine();
            if (!strLine.trim().equals("[Rmatrix]")) {
                // 拿到r、x、b，光标此时已经在r1上
                for (int i = 0; i < 6; i++) {
                    if (strLine.trim().contains("r1")) {
                        decoupledConfig[0] = strLine.trim().replace("r1=", "");
                    } else if (strLine.trim().contains("x1")) {
                        decoupledConfig[1] = strLine.trim().replace("x1=", "");
                    } else if (strLine.trim().contains("c1")) {
                        decoupledConfig[2] = strLine.trim().replace("c1=", "");
                    }
                    if (i == 5)
                        break;
                    strLine = reader.readLine(); // 最后一次循环时光标在c0处
                }
                if (phase == 3) {
                    // 修改name
                    oneRawData[0] += "-ABC";
                    oneRawData[1] = oneRawData[9] = oneRawData[17] = decoupledConfig[0];
                    oneRawData[2] = oneRawData[10] = oneRawData[18] = decoupledConfig[1];
                    // 电纳的单位是微西门子，电容的单位是纳法
                    oneRawData[19] = oneRawData[23] = oneRawData[27] = getSusceptanceByCapacitor(decoupledConfig[2]);
                    configList.add(oneRawData);
                }
                if (phase == 1) {
                    String[] oneRawDataA = oneRawData.clone();
                    oneRawDataA[0] += "-A";
                    oneRawDataA[1] = decoupledConfig[0];
                    oneRawDataA[2] = decoupledConfig[1];
                    // 电纳的单位是微西门子，电容的单位是纳法
                    oneRawDataA[19] = getSusceptanceByCapacitor(decoupledConfig[2]);
                    configList.add(oneRawDataA);
                    String[] oneRawDataB = oneRawData.clone();
                    oneRawDataB[0] += "-B";
                    oneRawDataB[9] = decoupledConfig[0];
                    oneRawDataB[10] = decoupledConfig[1];
                    // 电纳的单位是微西门子，电容的单位是纳法
                    oneRawDataB[23] = getSusceptanceByCapacitor(decoupledConfig[2]);
                    configList.add(oneRawDataB);
                    String[] oneRawDataC = oneRawData.clone();
                    oneRawDataC[0] += "-C";
                    oneRawDataC[17] = decoupledConfig[0];
                    oneRawDataC[18] = decoupledConfig[1];
                    // 电纳的单位是微西门子，电容的单位是纳法
                    oneRawDataC[27] = getSusceptanceByCapacitor(decoupledConfig[2]);
                    configList.add(oneRawDataC);
                }
            } else {
                // 此时光标在[Rmatrix]上
                if (phase == 3) {
                    // 修改name
                    oneRawData[0] += "-ABC";

                    // 一行
                    strLine = reader.readLine();
                    oneRawData[1] = strLine.trim();
                    // 二行
                    strLine = reader.readLine();
                    String[] element = strLine.trim().split("\\s+"); //正则表达式，切割中间的一个或多个空格
                    oneRawData[3] = oneRawData[7] = element[0];
                    oneRawData[9] = element[1];
                    // 三行
                    strLine = reader.readLine();
                    element = strLine.trim().split("\\s+"); //正则表达式，切割中间的一个或多个空格
                    oneRawData[5] = oneRawData[13] = element[0];
                    oneRawData[11] = oneRawData[15] = element[1];
                    oneRawData[17] = element[2];

                    strLine = reader.readLine();
                    // 一行
                    strLine = reader.readLine();
                    oneRawData[2] = strLine.trim();
                    // 二行
                    strLine = reader.readLine();
                    element = strLine.trim().split("\\s+"); //正则表达式，切割中间的一个或多个空格
                    oneRawData[4] = oneRawData[8] = element[0];
                    oneRawData[10] = element[1];
                    // 三行
                    strLine = reader.readLine();
                    element = strLine.trim().split("\\s+"); //正则表达式，切割中间的一个或多个空格
                    oneRawData[6] = oneRawData[14] = element[0];
                    oneRawData[12] = oneRawData[16] = element[1];
                    oneRawData[18] = element[2];

                    strLine = reader.readLine();
                    // 一行
                    strLine = reader.readLine();
                    oneRawData[19] = getSusceptanceByCapacitor(strLine.trim());
                    // 二行
                    strLine = reader.readLine();
                    element = strLine.trim().split("\\s+"); //正则表达式，切割中间的一个或多个空格
                    oneRawData[20] = oneRawData[22] = getSusceptanceByCapacitor(element[0]);
                    oneRawData[23] = getSusceptanceByCapacitor(element[1]);
                    // 三行
                    strLine = reader.readLine();
                    element = strLine.trim().split("\\s+"); //正则表达式，切割中间的一个或多个空格
                    oneRawData[21] = oneRawData[25] = getSusceptanceByCapacitor(element[0]);
                    oneRawData[24] = oneRawData[26] = getSusceptanceByCapacitor(element[1]);
                    oneRawData[27] = getSusceptanceByCapacitor(element[2]);
                    configList.add(oneRawData);
                    //此时光标位于C矩阵最后一行
                } else if (phase == 1) {
                    // 一行
                    strLine = reader.readLine();
                    decoupledConfig[0] = strLine.trim();
                    // 二行
                    strLine = reader.readLine();
                    strLine = reader.readLine();
                    decoupledConfig[1] = strLine.trim();
                    // 三行
                    strLine = reader.readLine();
                    strLine = reader.readLine();
                    decoupledConfig[2] = strLine.trim();

                    String[] oneRawDataA = oneRawData.clone();
                    oneRawDataA[0] += "-A";
                    oneRawDataA[1] = decoupledConfig[0];
                    oneRawDataA[2] = decoupledConfig[1];
                    // 电纳的单位是微西门子，电容的单位是纳法
                    oneRawDataA[19] = getSusceptanceByCapacitor(decoupledConfig[2]);
                    configList.add(oneRawDataA);
                    String[] oneRawDataB = oneRawData.clone();
                    oneRawDataB[0] += "-B";
                    oneRawDataB[9] = decoupledConfig[0];
                    oneRawDataB[10] = decoupledConfig[1];
                    // 电纳的单位是微西门子，电容的单位是纳法
                    oneRawDataB[23] = getSusceptanceByCapacitor(decoupledConfig[2]);
                    configList.add(oneRawDataB);
                    String[] oneRawDataC = oneRawData.clone();
                    oneRawDataC[0] += "-C";
                    oneRawDataC[17] = decoupledConfig[0];
                    oneRawDataC[18] = decoupledConfig[1];
                    // 电纳的单位是微西门子，电容的单位是纳法
                    oneRawDataC[27] = getSusceptanceByCapacitor(decoupledConfig[2]);
                    configList.add(oneRawDataC);
                } else if (phase == 2) {
                    // 原数据文件中只存在AC相
                    // 修改name
                    oneRawData[0] += "-AC";

                    // 一行
                    strLine = reader.readLine();
                    oneRawData[1] = strLine.trim();
                    // 二行
                    strLine = reader.readLine();
                    String[] element = strLine.trim().split("\\s+"); //正则表达式，切割中间的一个或多个空格
                    oneRawData[5] = oneRawData[13] = element[0];
                    oneRawData[17] = element[1];

                    strLine = reader.readLine();
                    // 一行
                    strLine = reader.readLine();
                    oneRawData[2] = strLine.trim();
                    // 二行
                    strLine = reader.readLine();
                    element = strLine.trim().split("\\s+"); //正则表达式，切割中间的一个或多个空格
                    oneRawData[6] = oneRawData[14] = element[0];
                    oneRawData[18] = element[1];

                    strLine = reader.readLine();
                    // 一行
                    strLine = reader.readLine();
                    oneRawData[19] = getSusceptanceByCapacitor(strLine.trim());
                    // 二行
                    strLine = reader.readLine();
                    element = strLine.trim().split("\\s+"); //正则表达式，切割中间的一个或多个空格
                    oneRawData[21] = oneRawData[25] = getSusceptanceByCapacitor(element[0]);
                    oneRawData[27] = getSusceptanceByCapacitor(element[1]);
                    configList.add(oneRawData);
                    //此时光标位于C矩阵最后一行

                }

            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String getSusceptanceByCapacitor(String capacitor) {
        return String.valueOf(Double.parseDouble(capacitor) * 2 * Math.PI * 60 * 1e-3);
    }

    public void writeAll() {
        try {
            FileWriter fileWriter = new FileWriter(outputFile);
            fileWriter.write("kilometer\n");
            fileWriter.write("################################### feeder 8500 ###########################\n");
            String[] sourceData = new String[28];
            sourceData[0] = "X_Source";
            for (int i = 1; i < sourceData.length; i++) {
                sourceData[i] = "0.0";
            }
            sourceData[2] = "14.798";
            sourceData[10] = "14.798";
            sourceData[18] = "14.798";
            fileWriter.write(sourceData[0] + "\n");
            for (int i = 1; i < 28; i++) {
                if (i == 6 || i == 12 || i == 18 || i == 21 || i == 24 || i == 27) {
                    fileWriter.write(sourceData[i] + "\n");
                } else {
                    fileWriter.write(sourceData[i] + "\t");
                }
            }

            for (String[] strings : configList) {
                fileWriter.write(strings[0] + "\n");
                for (int i = 1; i < 28; i++) {
                    if (i == 6 || i == 12 || i == 18 || i == 21 || i == 24 || i == 27) {
                        fileWriter.write(strings[i] + "\n");
                    } else {
                        fileWriter.write(strings[i] + "\t");
                    }
                }
            }
            fileWriter.flush();
            fileWriter.close();

        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
