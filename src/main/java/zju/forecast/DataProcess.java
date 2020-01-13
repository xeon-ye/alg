package zju.forecast;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class DataProcess {
    private int[] columnsInd;
    private int labelInd;

    private double[] featureAverages;
    private double[] featureStds;
    private double labelAverage;
    private double labelStd;

    private INDArray features;
    private INDArray label;

    private DataSet dataSet;

    public void preprocess(double[][] data, int[] columnsToReg, int labelToReg) {
        columnsInd = columnsToReg;
        labelInd = labelToReg;

        featureAverages = new double[columnsToReg.length];
        featureStds = new double[columnsToReg.length];

        for (int i = 0; i < columnsToReg.length; i++) {
            double average = getAverage(data, columnsToReg[i]);
            featureAverages[i] = average;
            double std = getStd(data, columnsToReg[i], average);
            featureStds[i] = std;
            for (int row = 0; row < data.length; row++) {
                data[row][columnsToReg[i]] = (data[row][columnsToReg[i]] - average) / std;
            }
        }

        labelAverage = getAverage(data, labelToReg);
        labelStd = getStd(data, labelToReg, labelAverage);
        for (int row = 0; row < data.length; row++) {
            data[row][labelToReg] = (data[row][labelToReg] - labelAverage) / labelStd;
        }


        double[][] featuresArray = new double[data.length][data[0].length - 1];
        double[][] labelArray = new double[data.length][1];

        for (int row = 0; row < data.length; row++) {
            int i = 0;
            for (int column = 0; column < data[0].length; column++) {
                if (column != labelToReg) {
                    featuresArray[row][i++] = data[row][column];
                } else {
                    labelArray[row][0] = data[row][column];
                }
            }
        }

        features = Nd4j.create(featuresArray);
        label = Nd4j.create(labelArray);

        dataSet = new DataSet(features, label);
    }

    public void transformFeature(double[][] data) {
        for (int value : columnsInd) {
            double average = featureAverages[value];
            double std = featureStds[value];
            for (int row = 0; row < data.length; row++) {
                data[row][value] = (data[row][value] - average) / std;
            }
        }
    }

    public void transformLabel(double[] data) {
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] * labelStd + labelAverage;
        }
    }

    private double getAverage(double[][] data, int column) {
        double sum = 0;
        for (double[] datum : data) {
            sum += datum[column];
        }
        return sum / data.length;
    }

    private double getStd(double[][] data, int column, double average) {
        double sum = 0;
        for (double[] datum : data) {
            sum += Math.pow(datum[column] - average, 2);
        }
        return Math.sqrt(sum / data.length);
    }

    public static void main(String[] args) {
        double[][] data = new double[][]{{1, 1}, {3, 4}, {1, 2}};
        DataProcess dataPreprocess = new DataProcess();
        dataPreprocess.preprocess(data, new int[]{0, 1}, 1);
    }

    public INDArray getFeatures() {
        return features;
    }

    public INDArray getLabel() {
        return label;
    }

    public DataSet getDataSet() {
        return dataSet;
    }
}
