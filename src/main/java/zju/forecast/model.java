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
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

public class model {
    private static final int seed = 12345;
    private static final int iterations = 10;
    private static final int nEpochs = 50;
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

        DataProcess dataPreprocess = new DataProcess();
        dataPreprocess.preprocess(rawData,new int[]{0,1,2},2);
        DataSet dataSet = dataPreprocess.getDataSet();
        List<DataSet> dataSetList = dataSet.asList();
        ListDataSetIterator listDataSetIterator = new ListDataSetIterator(dataSetList, batchSize);

        for( int i=0; i<nEpochs; i++ ){
            listDataSetIterator.reset();
            net.fit(listDataSetIterator);
        }

    }



}
