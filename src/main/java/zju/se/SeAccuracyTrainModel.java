package zju.se;

import java.util.List;

public interface SeAccuracyTrainModel {

    /**
     * 有监督学习
     *
     * @param attributes 输入数据集
     * @param labels     输出标签集
     */
    void train(List<double[]> attributes, List<Double> labels);

    /**
     * 对新数据进行预测
     *
     * @param attribute 输入一组数据
     * @return 预测结果
     */
    double forecast(double[] attribute);
}
