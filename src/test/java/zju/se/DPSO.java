package zju.se;

/**
 *粒子群类
 * @author
 */
class DPSO {
    Particle[] particles;//粒子群，一个Particle的对象
    double globalBestFitness;//全局最优解
    public static double[] globalBestPosition;//所有粒子找到的最好位置
    int particlesAmount;//粒子的数量

    /**
     * 粒子群初始化
     */
    public void initial() {
        //todo:修改参数的地方
        //类的静态成员的初始化
        Particle.setC1(2);
        Particle.setC2(2);
        Particle.setW(0.8);
        //todo:维度
        Particle.setDimension(3);
        //todo:最大为1的个数
        Particle.setMaxNumber(0);
        //todo:粒子个数
        this.particlesAmount = 1;

        particles = new Particle[this.particlesAmount];//开辟内存，确定该数组对象的大小
        globalBestFitness = 1e6;//全局最优适应值
        globalBestPosition = new double[Particle.dimension];//开辟内存，全局最优位置，存储全局最优位置的数组长度应等于粒子的维数.
        int index = -1;//记录最优位置

        for (int i = 0; i < particlesAmount; ++i) {
            particles[i] = new Particle();//particles中每个元素，都是Particle类型的对象
            particles[i].initial(Particle.dimension);//调用Particle中的方法，对每个粒子进行维度的初始化

            particles[i].evaluate();//调用评估适应值的方法
            if (globalBestFitness > particles[i].getFitness()) {//将求得的每个粒子更新的适应值比原来的全局变量大
                globalBestFitness = particles[i].getFitness();//则更新全局变量
                index = i;//记录全局最优是哪个粒子
            }
        }

        for (int i = 0; i < Particle.dimension; ++i) {
            globalBestPosition[i] = particles[index].position[i];//更新全局最优位置，根据index
        }
    }
    /**
     * 粒子群的运行
     */
    public void run() {
        int runTimes = 1;//运行次数是1
        int index;//index
        //todo:最大迭代次数，当前值为500
        while (runTimes <= 20) {//确定迭代次数
            index = -1;
            //每个粒子更新位置和适应值
            for (int i = 0; i < particlesAmount; ++i) {
                particles[i].updatePosAndVel();//更新每个粒子的位置和速度
                particles[i].evaluate();//更新每个粒子的适应值
                if (globalBestFitness > particles[i].fitness) {
                    globalBestFitness = particles[i].fitness;
                    index = i;//更新全局最优适应值，并记录最优的粒子
                }
            }
            //发现更好的解
            if (index != -1) {
                for (int i = 0; i < Particle.dimension; ++i) {
                    globalBestPosition[i] = particles[index].position[i];//若index不是-1，则记录位置
                }
                //打印结果
                System.out.println("第"+runTimes+"次迭代发现更好解：");
                System.out.println("globalBestFitness:"+globalBestFitness);
                for (int i =0 ; i < Particle.dimension; i++){
                    System.out.println("position[" + i + "]:" +particles[index].position[i]);//输出最优的粒子的位置
                }
            }
            //System.out.println(runTimes);
            runTimes++;
        }
    }
    /**
     * 显示程序求解结果
     */
    public void showResult() {
        System.out.println("状态估计误差最小值" + globalBestFitness);
        System.out.println("状态估计误差最小时：");
        for (int i = 0; i < Particle.dimension; ++i) {
            System.out.println(globalBestPosition[i]);
        }
    }
}