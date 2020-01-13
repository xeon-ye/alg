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

import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

public class DeepLearningPvForecast implements PhotovoltaicForecastHandler {
    private static final int seed = 12345;
    private static final int iterations = 10;
    private static final int nEpochs = 50;
    private static final double learningRate = 0.01;
    private static final int batchSize = 1000;

    private MultiLayerNetwork model;

    private DataProcess dataProcess;

    @Override
    public double[] predictPhotovoltaic(List<Weather> weathers) {
        double[][] encodings = new double[weathers.size()][102]; // 编码后的向量
        Calendar c = Calendar.getInstance();
        for (int i = 0; i < encodings.length; i++) {
            double[] encoding = encodings[i];
            Weather weather = weathers.get(i);
            encoding[0] = weather.getWindSpeed();
            encoding[1] = weather.getTemperature();
            switch (weather.getWeather()) {
                case "clear":
                    encoding[3] = 1;
                    break;
                case "cloud":
                    encoding[4] = 1;
                    break;
                case "rain":
                    encoding[5] = 1;
                    break;
            }
            c.setTime(weather.getDtime());
            encoding[c.get(Calendar.HOUR_OF_DAY) * 4 + c.get(Calendar.MINUTE) / 15 + 6] = 1;
        }

        if(encodings.length==0) return null;

        dataProcess.transformFeature(encodings);
        double[][] testData = new double[encodings.length][encodings[0].length - 1];
        for (int row = 0; row < testData.length; row++) {
            int j = 0;
            for (int column = 0; column < encodings[0].length; column++) {
                if (column != 2) testData[row][j++] = encodings[row][column];
            }
        }

        INDArray input = Nd4j.create(testData);
        INDArray out = model.output(input, false);

        double[] result = new double[testData.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = out.getDouble(i, 0);
        }

        dataProcess.transformLabel(result);

        return result;
    }

    @Override
    public void fitPhotovoltaic(List<Weather> weathers, List<Measurement> measurements) {
        List<double[]> encodings = new ArrayList<>(Math.min(weathers.size(), measurements.size()));
        Calendar c = Calendar.getInstance();
        for (int i = 0, j = 0; i < weathers.size() && j < measurements.size(); ) {
            Weather weather = weathers.get(i);
            Measurement measurement = measurements.get(j);
            if (weather.getDtime().getTime() > measurement.getDtime().getTime())
                ++j;
            else if (weather.getDtime().getTime() < measurement.getDtime().getTime())
                ++i;
            else {
                double[] encoding = new double[102];
                encoding[0] = weather.getWindSpeed();
                encoding[1] = weather.getTemperature();
                encoding[2] = measurement.getVal();
                switch (weather.getWeather()) {
                    case "clear":
                        encoding[3] = 1;
                        break;
                    case "cloud":
                        encoding[4] = 1;
                        break;
                    case "rain":
                        encoding[5] = 1;
                        break;
                }
                c.setTime(weather.getDtime());
                encoding[c.get(Calendar.HOUR_OF_DAY) * 4 + c.get(Calendar.MINUTE) / 15 + 6] = 1;
                encodings.add(encoding);
                ++i;
                ++j;
            }
        }

        if (model == null) {
            model = buildModel();
        }

        double[][] rawData = new double[encodings.size()][encodings.get(0).length];
        for (int i = 0; i < rawData.length; i++) {
            for (int j = 0; j < rawData[0].length; j++) {
                rawData[i][j] = encodings.get(i)[j];
            }
        }

        dataProcess = new DataProcess();
        dataProcess.preprocess(rawData, new int[]{0, 1}, 2);
        DataSet dataSet = dataProcess.getDataSet();
        List<DataSet> dataSetList = dataSet.asList();
        ListDataSetIterator listDataSetIterator = new ListDataSetIterator(dataSetList, batchSize);

        for (int i = 0; i < nEpochs; i++) {
            listDataSetIterator.reset();
            model.fit(listDataSetIterator);
        }

    }

    public MultiLayerNetwork buildModel() {
        int numInput = 101;
        int numNeuron = 100;
        int numOutput = 1;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(learningRate))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInput)
                        .nOut(numNeuron)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nOut(numNeuron)
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nOut(numOutput)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        System.out.println(net.summary());
        net.setListeners(new ScoreIterationListener(10));
        return net;
    }

}
