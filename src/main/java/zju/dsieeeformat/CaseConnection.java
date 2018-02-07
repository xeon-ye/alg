package zju.dsieeeformat;

import java.io.*;

/**
 * Created by Fang Rui on 2017/5/7.
 */
public class CaseConnection {

    private static final String SAVE_PATH = "./src/main/resources/dsieee/case_connection";

    public static void parse(int caseNum, int count, String slackCnId) {
        String parentPath = "/dsieee/";
        InputStream stream;
        if (caseNum != 8500) {
            stream = CaseConnection.class.getResourceAsStream(parentPath + "case" + caseNum + "/case" + caseNum + ".txt");
        } else {
            stream = CaseConnection.class.getResourceAsStream(parentPath + "8500node/case8500.txt");
        }

        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String strLine;
        String strLineWriter;

        File saveFile = new File(SAVE_PATH + "/case" + caseNum + "x" + count + ".txt");
        if (saveFile.exists()) {
            saveFile.delete();
        } else {
            try {
                saveFile.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        try {
            FileWriter fileWriter = new FileWriter(saveFile);

            strLine = reader.readLine();
            while (!strLine.startsWith("Line Segment Data"))
                strLine = reader.readLine();

            String[] strings = strLine.split("\t");
            String s = strings[1];
            strings = s.split(" ");
            s = strings[0];
            int num = Integer.parseInt(s);
            num = num * count;
            strLine = "Line Segment Data\t" + num + " items\tfeet";
            fileWriter.write(strLine);
            fileWriter.write("\r\n");

            while (true) {
                strLine = reader.readLine();
                if (strLine.trim().equalsIgnoreCase("-999")) {
                    fileWriter.write(strLine);
                    fileWriter.write("\r\n");
                    break;
                }

                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;
                for (int i = 1; i < count + 1; i++) {

                    StringBuilder builder = new StringBuilder();
                    builder.append(strLine);
                    int firstT = strLine.indexOf("\t");
                    builder.insert(firstT + 1, i + "-");
                    if (!strLine.startsWith("150"))
                        builder.insert(0, i + "-");
                    strLineWriter = builder.toString();
                    fileWriter.write(strLineWriter);
                    fileWriter.write("\r\n");
                }
            }

            strLine = reader.readLine();
            while (!strLine.startsWith("Spot Loads"))
                strLine = reader.readLine();

            strings = strLine.split("\t");
            s = strings[1];
            strings = s.split(" ");
            s = strings[0];
            num = Integer.parseInt(s);
            num = num * count;
            strLine = "Spot Loads\t" + num + " items";
            fileWriter.write(strLine);
            fileWriter.write("\r\n");

            while (true) {
                strLine = reader.readLine();
                if (strLine.trim().equalsIgnoreCase("-999")) {
                    fileWriter.write(strLine);
                    fileWriter.write("\r\n");
                    break;
                }

                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;
                for (int i = 1; i < count + 1; i++) {

                    StringBuilder builder = new StringBuilder();
                    builder.append(strLine);
                    if (!strLine.startsWith("150"))
                        builder.insert(0, i + "-");
                    strLineWriter = builder.toString();
                    fileWriter.write(strLineWriter);
                    fileWriter.write("\r\n");
                }
            }

            strLine = reader.readLine();
            while (!strLine.startsWith("Distributed Loads"))
                strLine = reader.readLine();

            strings = strLine.split("\t");
            s = strings[1];
            strings = s.split(" ");
            s = strings[0];
            num = Integer.parseInt(s);
            num = num * count;
            strLine = "Distributed Loads\t" + num + " items";
            fileWriter.write(strLine);
            fileWriter.write("\r\n");

            while (true) {
                strLine = reader.readLine();
                if (strLine.trim().equalsIgnoreCase("-999")) {
                    fileWriter.write(strLine);
                    fileWriter.write("\r\n");
                    break;
                }

                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;
                for (int i = 1; i < count + 1; i++) {

                    StringBuilder builder = new StringBuilder();
                    builder.append(strLine);
                    if (!strLine.startsWith("150"))
                        builder.insert(0, i + "-");
                    strLineWriter = builder.toString();
                    fileWriter.write(strLineWriter);
                    fileWriter.write("\r\n");
                }
            }

            strLine = reader.readLine();
            while (!strLine.startsWith("Shunt Capacitors"))
                strLine = reader.readLine();

            strings = strLine.split("\t");
            s = strings[1];
            strings = s.split(" ");
            s = strings[0];
            num = Integer.parseInt(s);
            num = num * count;
            strLine = "Shunt Capacitors\t" + num + " items";
            fileWriter.write(strLine);
            fileWriter.write("\r\n");

            while (true) {
                strLine = reader.readLine();
                if (strLine.trim().equalsIgnoreCase("-999")) {
                    fileWriter.write(strLine);
                    fileWriter.write("\r\n");
                    break;
                }

                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;
                for (int i = 1; i < count + 1; i++) {

                    StringBuilder builder = new StringBuilder();
                    builder.append(strLine);
                    if (!strLine.startsWith("150"))
                        builder.insert(0, i + "-");
                    strLineWriter = builder.toString();
                    fileWriter.write(strLineWriter);
                    fileWriter.write("\r\n");
                }
            }

            strLine = reader.readLine();
            while (!strLine.startsWith("Transformer"))
                strLine = reader.readLine();

            fileWriter.write(strLine);
            fileWriter.write("\r\n");
            while (true) {
                strLine = reader.readLine();
                if (strLine.trim().equalsIgnoreCase("-999")) {
                    fileWriter.write(strLine);
                    fileWriter.write("\r\n");
                    break;
                }

                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;

                fileWriter.write(strLine);
                fileWriter.write("\r\n");
            }

            strLine = reader.readLine();
            while (!strLine.startsWith("Regulator"))
                strLine = reader.readLine();

            fileWriter.write(strLine);
            fileWriter.write("\r\n");
            while (true) {
                strLine = reader.readLine();
                if (strLine.trim().equalsIgnoreCase("-999")) {
                    fileWriter.write(strLine);
                    fileWriter.write("\r\n");
                    break;
                }

                if (strLine.trim().equals("") || strLine.startsWith("#")) //Debug Mode,you can disable elements with"//"
                    continue;

                fileWriter.write(strLine);
                fileWriter.write("\r\n");
            }


            fileWriter.flush();
            fileWriter.close(); // 关闭数据流
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        parse(123, 100, "150");
    }
}
