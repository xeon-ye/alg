package zju.forecast;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class model {
    private static final int seed = 12345;
    private static final int iterations = 10;
    private static final int nEpochs = 100;
    private static final double learningRate = 0.01;
    private static final int batchSize = 1000;

    public static void main(String[] args) {
        int numInput = 101;
        int numNeuron = 100;
        int numOutput = 1;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(learningRate))
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(numInput)
                        .nOut(numNeuron)
                        .build())
                .layer(1,new DenseLayer.Builder()
                        .nOut(numNeuron)
                        .build())
                .layer(2,new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nOut(numOutput)
                        .build())
                .build();


        MultiLayerNetwork net = new MultiLayerNetwork(conf);


        net.init();

        System.out.println(net.summary());

        net.setListeners(new ScoreIterationListener(10));

        CsvReader csvReader = new CsvReader();
        double[][] rawData = csvReader.read(CsvReader.class.getResource("/forecast/data.csv").getPath());

        DataProcess dataProcess = new DataProcess();
        dataProcess.preprocess(rawData,new int[]{0,1},2);
        DataSet dataSet = dataProcess.getDataSet();
        List<DataSet> dataSetList = dataSet.asList();
        ListDataSetIterator listDataSetIterator = new ListDataSetIterator(dataSetList, batchSize);

        for( int i=0; i<nEpochs; i++ ){
            listDataSetIterator.reset();
            net.fit(listDataSetIterator);
        }


        double[][] testData = new double[rawData.length][rawData[0].length - 1];
        for (int row = 0; row < testData.length; row++) {
            int j = 0;
            for (int column = 0; column < rawData[0].length; column++) {
                if (column != 2) testData[row][j++] = rawData[row][column];
            }
        }

        INDArray input = Nd4j.create(testData);
        INDArray out = net.output(input, false);

        double[] result = new double[testData.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = out.getDouble(i, 0);
        }

        dataProcess.transformLabel(result);

        FileWriter writer = null;
        try {
            writer = new FileWriter("G:/result.csv");
        } catch (IOException e) {
            e.printStackTrace();
        }
        for(int i = 0;i<result.length;i++){
            try {
                writer.write(String.valueOf(result[i])+"\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }



}
