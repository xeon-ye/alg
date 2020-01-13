package zju.forecast;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class CsvReader {

    public double[][] read(String filePath){
        List<double[]> dataList = new ArrayList<>();

        File csv = new File(filePath);
        BufferedReader bufferedReader = null;
        try {
            bufferedReader = new BufferedReader(new FileReader(csv));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        int columnsNum = 0;
        try {
            String line = bufferedReader.readLine();
            String[] columns = line.split(",");
            columnsNum = columns.length-1;
            double[] temp = null;

            line = bufferedReader.readLine();
            while(line!=null){
                temp = new double[columnsNum];
                columns = line.split(",");
                for(int i = 1;i<columns.length;i++){
                    temp[i-1] = Double.valueOf(columns[i]);
                }
                dataList.add(temp);
                line = bufferedReader.readLine();
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        double[][] dataArray = new double[dataList.size()][columnsNum];
        for(int i = 0;i<dataList.size();i++){
            for(int j =0;j<columnsNum;j++) dataArray[i][j] = dataList.get(i)[j];
        }

        return dataArray;
    }

    public static void main(String[] args) {
        CsvReader csvReader = new CsvReader();
        csvReader.read(CsvReader.class.getResource("data.csv").getPath());
    }

}
