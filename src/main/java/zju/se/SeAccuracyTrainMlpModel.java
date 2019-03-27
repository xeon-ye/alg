package zju.se;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Created by meditation on 2019/3/26.
 */
public class SeAccuracyTrainMlpModel implements SeAccuracyTrainModel{
    private int seed = 123;
    //训练次数
    private int nEpochs = 20;
    //批数量
    private int batchSize = 100;
    //学习率
    private double learningRate = 0.01;


    @Override
    public void fit(List<double[]> attributes, List<Double> labels) {
        int numInput = attributes.get(0).length;
        int numLayer1 = 100;
        int numOutput = 1;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                //找方向
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                //迈步子
                .updater(new Sgd(learningRate))
                .list()
                .layer(0,new DenseLayer.Builder().nIn(numInput).nOut(numLayer1)
                        .activation(Activation.TANH).build())
                .layer(1,new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numLayer1)
                        .nOut(numOutput).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);

        net.init();

        System.out.println(net.summary());

        //监听器
//        net.setListeners(new ScoreIterationListener(1));

        List<DataSet> xTrain = new LinkedList<>();
        for(int i=0;i<attributes.size();i++){
            INDArray x = Nd4j.create(attributes.get(i));
            INDArray y = Nd4j.create(new double[]{labels.get(i)});
            DataSet sample = new DataSet(x,y);
            xTrain.add(sample);
        }

        DataSetIterator iterator = new ListDataSetIterator(xTrain,batchSize);

        for(int i=0;i<nEpochs;i++){
            iterator.reset();
            net.fit(iterator);

            Map<String,INDArray> params = net.paramTable();

            params.forEach((key,value)-> System.out.println("key:"+key+",value = "+value));
            System.out.println(net.score());
        }

        final INDArray input = Nd4j.create(new double[]{10,100},new int[]{2,1});
        INDArray out = net.output(input,false);
        System.out.println(out);
    }

    @Override
    public double predict(double[] attribute) {
        return 0;
    }

    @Override
    public double score(List<double[]> attributes, List<Double> labels) {
        return 0;
    }


}
