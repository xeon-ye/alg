package zju.se;

import Jama.Matrix;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * 粒子类
 * 求解函数 f(x)=x1^2+(x2-x3)^2 的最大值
 *
 * @author
 */
class Particle {
    double[] position;//粒子的位置，求解问题多少维，则此数组为多少维
    double[] velocity;//粒子的速度，维数同位置
    double fitness;//粒子的适应度

    double[] pBestPositon;//粒子的历史最好位置
    double pBestFitness;//历史最优适应值

    static int maxNumber;//粒子为1的最大个数限制
    static int dimension;//声明维数

    static Random rnd;//产生随机变量
    static double w;//声明权重
    static double c1;//声明学习系数
    static double c2;

    /**
     * 初始化粒子
     *
     * @param dimension 表示粒子的维数,即可安装量测的位置
     */
    public void initial(int dimension) {
        //
        position = new double[dimension];//开辟存储位置的数组的内存
        velocity = new double[dimension];//开辟存储位置的数组的内存
        fitness = 1e6;//初始粒子适应值是一个1*10的6次方，赋值的过程

        pBestPositon = new double[dimension];//开辟存储历史最优适应值的内存
        pBestFitness = 1e6;//初始历史最优的适应值是1*10的6次方，赋值的过程

        Particle.dimension = dimension;//维数=输入的维数，赋值的过程

        for (int i = 0; i < dimension; ++i) {
            position[i] = getRandomPostion();//位置初始化
            pBestPositon[i] = position[i];//将初始化后的位置赋值给“粒子的历史最好位置”，取缔之前的-1*10的6次方
            velocity[i] = getRandomValue(-1, 1);//速度的初始化，取（-1，1）之间的随机数
        }
        //位置为1的总数
        int sum = 0;
        Random rd = new Random();
        for (double i : position) {
            sum += i;
        }
        //去1操作
        if (sum > this.maxNumber) {
            int[] oneSetPlace = new int[sum];
            int index = 0;
            for (int i = 0; i < Particle.dimension; i++) {
                if (position[i] == 1) {
                    oneSetPlace[index] = i;
                    index++;
                }
            }

            Set toSetZero = new HashSet();
            for (int i = 0; i < (sum - maxNumber); i++) {
                int rdNumber;
                do {
                    rdNumber = oneSetPlace[rd.nextInt(sum)];
                } while (toSetZero.contains(rdNumber));
                toSetZero.add(rdNumber);
            }
            for (Object i : toSetZero) {
                position[(int) i] = 0;
            }
        }
    }

    /**
     * 评估函数值,同时记录历史最优位置
     */
    public void evaluate() {
        //todo:修改适应度的计算公式
        fitness = 0;
        Matrix matrixSum = Parameter.getMatrixList().get(0);
        for (int i = 0; i < position.length; i++) {
            if(position[i] ==1 ){
                matrixSum = matrixSum.plus(Parameter.getMatrixList().get(i+1));
            }
        }
        if(matrixSum.det() != 0) {
            matrixSum = matrixSum.inverse();
            for (int i = 0; i < matrixSum.getColumnDimension(); i++) {
                fitness += matrixSum.get(i, i);
            }
        }else{
            fitness = 1e6;
        }
        if (fitness < pBestFitness | fitness != 0) {
            pBestFitness = fitness;//假如计算出的适应值比历史最优适应值好，则用新计算出的适应值函数替代历史最优适应值
            for (int i = 0; i < dimension; ++i) {
                pBestPositon[i] = position[i];
            }//这里是方法中的，记录最优的历史位置，只有历史最优适应值被替代，历史最优位置才会进行更新
        }
    }

    /**
     * 更新速度和位置
     */
    public void updatePosAndVel() {
        for (int j = 0; j < dimension; ++j) {
            velocity[j] = w * velocity[j] + c1 * rnd.nextDouble() * (pBestPositon[j] - position[j])
                    + c2 * rnd.nextDouble() * (DPSO.globalBestPosition[j] - position[j]);//速度更新

            double threshold = 1 / (1 + Math.exp(-velocity[j]));//sig函数
            //System.out.println(threshold);
            if (rnd.nextDouble() < threshold) {//若随机数小于速度的sig函数值
                position[j] = 1;//位置等于1
            } else {
                position[j] = 0;//位置等于0
            }
        }

        int sum = 0;
        Random rd = new Random();
        for (double i : position) {
            sum += i;
        }
        //去1操作
        if (sum > this.maxNumber) {
            int[] oneSetPlace = new int[sum];
            int index = 0;
            for (int i = 0; i < dimension; i++) {
                if (position[i] == 1) {
                    oneSetPlace[index] = i;
                    index++;
                }
            }
            Set toSetZero = new HashSet();
            for (int i = 0; i < (sum - maxNumber); i++) {
                int rdNumber;
                do {
                    rdNumber = oneSetPlace[rd.nextInt(sum)];
                } while (toSetZero.contains(rdNumber));
                toSetZero.add(rdNumber);
            }
            for (Object i : toSetZero) {
                position[(int) i] = 0;
            }
        }
    }

    /**
     * 返回low—uper之间的数
     *
     * @param low  下限
     * @param uper 上限
     * @return 返回low—uper之间的数
     */
    double getRandomValue(double low, double uper) {//一个方法
        rnd = new Random();//一个对象
        return rnd.nextDouble() * (uper - low) + low;//返回一个随机数
    }

    double getRandomPostion() {
        rnd = new Random();
        if (rnd.nextDouble() < 0.5) {
            return 0;
        } else {
            return 1;
        }
    }//一个方法，关于获取一个随机的位置。

    public static void setDimension(int[] candPos) {
        Particle.dimension = candPos.length;
    }//一个设置维度的方法，粒子类的维度

    public double[] getPosition() {
        return position;
    }

    public void setPosition(double[] position) {
        this.position = position;
    }

    public double[] getVelocity() {
        return velocity;
    }

    public void setVelocity(double[] velocity) {
        this.velocity = velocity;
    }

    public double getFitness() {
        return fitness;
    }

    public void setFitness(double fitness) {
        this.fitness = fitness;
    }

    public double[] getpBestPositon() {
        return pBestPositon;
    }

    public void setpBestPositon(double[] pBestPositon) {
        this.pBestPositon = pBestPositon;
    }

    public double getpBestFitness() {
        return pBestFitness;
    }

    public void setpBestFitness(double pBestFitness) {
        this.pBestFitness = pBestFitness;
    }

    public static int getMaxNumber() {
        return maxNumber;
    }

    public static void setMaxNumber(int maxNumber) {
        Particle.maxNumber = maxNumber;
    }

    public static int getDimension() {
        return dimension;
    }

    public static void setDimension(int dimension) {
        Particle.dimension = dimension;
    }

    public static Random getRnd() {
        return rnd;
    }

    public static void setRnd(Random rnd) {
        Particle.rnd = rnd;
    }

    public static double getW() {
        return w;
    }

    public static void setW(double w) {
        Particle.w = w;
    }

    public static double getC1() {
        return c1;
    }

    public static void setC1(double c1) {
        Particle.c1 = c1;
    }

    public static double getC2() {
        return c2;
    }

    public static void setC2(double c2) {
        Particle.c2 = c2;
    }
}