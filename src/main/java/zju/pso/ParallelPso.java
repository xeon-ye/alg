package zju.pso;

import jcuda.*;
import jcuda.driver.*;
import jcuda.jcurand.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.jcuda.JCudaUtil;

import java.io.IOException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

import static jcuda.driver.CUstream_flags.*;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.*;


/**
 * 使用Cuda进行并行加速
 *
 * @Author: Fang Rui
 * @Date: 2018/8/27
 * @Time: 18:54
 */
public class ParallelPso extends AbstractPso implements PsoConstants {

    private static Logger logger = LogManager.getLogger(ParallelPso.class);

    private ParallelOptModel optModel;
    private double[] fitness; // 各粒子当前适应度值
    private boolean isGBestfeasible = false; // gBest是否在可行域内

    private int numElements;
    private int n;
    private int numBytes;
    private int numBytesOfOneParticle;
    private int blockSizeX;
    private int gridSizeX;

    private CUfunction function_initialize = new CUfunction();
    private CUfunction function_warmInitialize = new CUfunction();
    private CUfunction function_update = new CUfunction();
    private CUfunction function_calBusV = new CUfunction();

    private CUdeviceptr deviceInputLocation = new CUdeviceptr();
    private CUdeviceptr devicePBestLocation = new CUdeviceptr();
    private CUdeviceptr deviceGBestLocation = new CUdeviceptr();
    private CUdeviceptr deviceInputVelocity = new CUdeviceptr();
    private CUdeviceptr deviceInputMinLocation = new CUdeviceptr();
    private CUdeviceptr deviceInputMinVelocity = new CUdeviceptr();
    private CUdeviceptr deviceInputMaxLocation = new CUdeviceptr();
    private CUdeviceptr deviceInputMaxVelocity = new CUdeviceptr();

    private CUdeviceptr deviceRandom1 = new CUdeviceptr();
    private CUdeviceptr deviceRandom2 = new CUdeviceptr();
    private curandGenerator generator = new curandGenerator();

    private CUstream stream = new CUstream();

    private float[] gpuLocation;

    class FitnessCalTask extends RecursiveTask<Integer> {

        public int threshold = swarmSize / 4; // 每个任务处理的粒子数
        private int from;
        private int to;
        private boolean isFirst;

        public FitnessCalTask(int from, int to, boolean isFirst) {
            this.from = from;
            this.to = to;
            this.isFirst = isFirst;
        }

        public FitnessCalTask(int from, int to, boolean isFirst, int threshold) {
            this.from = from;
            this.to = to;
            this.isFirst = isFirst;
            this.threshold = threshold;
        }

        @Override
        protected Integer compute() {
            int bestParticleIndex = from;
            if (to - from < threshold) {
                for (int i = from; i < to; i++) {

                    double violation = optModel.paraEvalConstr(gpuLocation, i * n);
                    if (violation > 0) {
                        fitness[i] = violation + PUNISHMENT;
                    } else {
                        fitness[i] = optModel.paraEvalObj(gpuLocation, i * n);
                        isGBestfeasible = true;
                    }

                    if (isFirst) {
                        pBest[i] = fitness[i];
                    } else if (fitness[i] < pBest[i]) {
                        pBest[i] = fitness[i];
                        cuMemcpyDtoDAsync(devicePBestLocation.withByteOffset(i * numBytesOfOneParticle),
                                deviceInputLocation.withByteOffset(i * numBytesOfOneParticle),
                                numBytesOfOneParticle, stream);
                    }
                    if (fitness[i] < fitness[bestParticleIndex])
                        bestParticleIndex = i;
                }
                return bestParticleIndex;
            } else {
                int mid = (from + to) / 2;
                FitnessCalTask front = new FitnessCalTask(from, mid, isFirst);
                FitnessCalTask behind = new FitnessCalTask(mid, to, isFirst);
                invokeAll(front, behind);
                int i = front.join();
                int j = behind.join();
                return fitness[i] < fitness[j] ? i : j;
            }
        }
    }

    public ParallelPso(ParallelOptModel optModel) {
        this(optModel, (int) (10 + 2 * Math.sqrt(optModel.getDimentions())));
    }

    public ParallelPso(ParallelOptModel optModel, int swarmSize) {
        super(swarmSize);
        this.optModel = optModel;
        this.fitness = new double[swarmSize];
    }

    public ParallelPso(ParallelOptModel optModel, double[] initVariableState) {
        this(optModel, (int) (10 + 2 * Math.sqrt(optModel.getDimentions())), initVariableState);
    }

    public ParallelPso(ParallelOptModel optModel, int swarmSize, double[] initVariableState) {
        super(swarmSize, initVariableState);
        this.optModel = optModel;
        this.fitness = new double[swarmSize];
    }

    private void initializeCuda() {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);

        // Create the PTX file by calling the NVCC
        String ptxFileName = null;
        try {
            ptxFileName = JCudaUtil.preparePtxFile("CudaPso.cu");
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Obtain the number of devices
        cuInit(0);
        try {
            JCudaUtil.queryDeviceInfo();
        } catch (CudaException e) {
            e.printStackTrace();
        }

        // Initialize the driver and create a context for the first device.
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "add" function.
        cuModuleGetFunction(function_initialize, module, "initialize");
        cuModuleGetFunction(function_warmInitialize, module, "warmInitialize");
        cuModuleGetFunction(function_update, module, "update");
        cuModuleGetFunction(function_calBusV, module, "calBusV");
    }

    @Override
    protected void initializeSwarm() {
        n = optModel.getDimentions();
        numElements = n * swarmSize;
        numBytes = numElements * Sizeof.FLOAT;
        numBytesOfOneParticle = n * Sizeof.FLOAT;
        blockSizeX = 512;
        gridSizeX = (int) Math.ceil((double) numElements / blockSizeX);

        float[] minLoc = optModel.paraGetMinLoc();
        float[] maxLoc = optModel.paraGetMaxLoc();
        float[] minVel = optModel.paraGetMinVel();
        float[] maxVel = optModel.paraGetMaxVel();

        gpuLocation = new float[numElements];

        cuMemAlloc(deviceInputLocation, numBytes);
        cuMemAlloc(devicePBestLocation, numBytes);
        cuMemAlloc(deviceGBestLocation, numBytesOfOneParticle);
        cuMemAlloc(deviceInputVelocity, numBytes);
        cuMemAlloc(deviceRandom1, numBytes);
        cuMemAlloc(deviceRandom2, numBytes);
        cuMemAlloc(deviceInputMinLocation, numBytesOfOneParticle);
        cuMemAlloc(deviceInputMinVelocity, numBytesOfOneParticle);
        cuMemAlloc(deviceInputMaxLocation, numBytesOfOneParticle);
        cuMemAlloc(deviceInputMaxVelocity, numBytesOfOneParticle);

        cuMemcpyHtoD(deviceInputMinLocation, Pointer.to(minLoc), numBytesOfOneParticle);
        cuMemcpyHtoD(deviceInputMaxLocation, Pointer.to(maxLoc), numBytesOfOneParticle);
        cuMemcpyHtoD(deviceInputMinVelocity, Pointer.to(minVel), numBytesOfOneParticle);
        cuMemcpyHtoD(deviceInputMaxVelocity, Pointer.to(maxVel), numBytesOfOneParticle);

        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(generator, System.currentTimeMillis());
        curandGenerateUniform(generator, deviceRandom1, numElements);
        curandSetPseudoRandomGeneratorSeed(generator, System.currentTimeMillis() + 1);
        curandGenerateUniform(generator, deviceRandom2, numElements);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        if (isWarmStart) {
            float[] initArray = new float[initVariableState.length];
            for (int i = 0; i < initArray.length; i++) {
                initArray[i] = (float) initVariableState[i];
            }
            for (int i = 0; i < swarmSize; i++) {
                System.arraycopy(initArray, 0, gpuLocation, i * initArray.length, initArray.length);
            }
            cuMemcpyHtoD(deviceInputLocation, Pointer.to(gpuLocation), numBytes);
            Pointer kernel_warmInitializeParameters = Pointer.to(
                    Pointer.to(deviceRandom2),
                    Pointer.to(deviceInputVelocity),
                    Pointer.to(deviceInputMinVelocity),
                    Pointer.to(deviceInputMaxVelocity),
                    Pointer.to(new int[]{numElements}),
                    Pointer.to(new int[]{n})
            );
            cuLaunchKernel(function_warmInitialize,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernel_warmInitializeParameters, null // Kernel- and extra parameters
            );
        } else {
            Pointer kernel_initializeParameters = Pointer.to(
                    Pointer.to(deviceRandom1),
                    Pointer.to(deviceRandom2),
                    Pointer.to(deviceInputLocation),
                    Pointer.to(deviceInputVelocity),
                    Pointer.to(deviceInputMinLocation),
                    Pointer.to(deviceInputMaxLocation),
                    Pointer.to(deviceInputMinVelocity),
                    Pointer.to(deviceInputMaxVelocity),
                    Pointer.to(new int[]{numElements}),
                    Pointer.to(new int[]{n})
            );
            cuLaunchKernel(function_initialize,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernel_initializeParameters, null // Kernel- and extra parameters
            );
        }
        cuCtxSynchronize();

        // 把初始化后的位置拷出来
        if (!isWarmStart)
            cuMemcpyDtoH(Pointer.to(gpuLocation), deviceInputLocation, numBytes);

        FitnessCalTask task = new FitnessCalTask(0, swarmSize, true);
        ForkJoinPool pool = new ForkJoinPool();
        pool.invoke(task);
        cuStreamSynchronize(stream);
        int bestParticleIndex = task.join();
        gBest = fitness[bestParticleIndex];

        cuMemcpyDtoD(devicePBestLocation, deviceInputLocation, numBytes);
        cuMemcpyDtoD(deviceGBestLocation, deviceInputLocation.withByteOffset(bestParticleIndex * numBytesOfOneParticle), numBytesOfOneParticle);
    }


    @Override
    public void execute() {
        long start = System.currentTimeMillis();
        initializeCuda();
        initializeSwarm();
        logger.info("初始化用时: " + (System.currentTimeMillis() - start) + "ms");

        cuStreamCreate(stream, CU_STREAM_DEFAULT);

        int iterNum = 0;
        int maxIter = optModel.getMaxIter();

        double tol = 1e6;
        float w; // 惯性权重

        while (iterNum < maxIter && tol > 0) {
            w = (float) (W_UPPERBOUND - (((double) iterNum) / maxIter) * (W_UPPERBOUND - W_LOWERBOUND)); // 惯性逐渐减小

//            curandSetPseudoRandomGeneratorSeed(generator, System.currentTimeMillis());
//            curandGenerateUniform(generator, deviceRandom1, numElements);
//            curandSetPseudoRandomGeneratorSeed(generator, System.currentTimeMillis() + 1);
//            curandGenerateUniform(generator, deviceRandom2, numElements);
//            logger.info("随机数产生用时: " + (System.currentTimeMillis() - start) + "ms");

            cuStreamSynchronize(stream);

            Pointer kernel_updateParameters = Pointer.to(
                    Pointer.to(new float[]{w}),
                    Pointer.to(deviceRandom1),
                    Pointer.to(deviceRandom2),
                    Pointer.to(deviceInputLocation),
                    Pointer.to(devicePBestLocation),
                    Pointer.to(deviceGBestLocation),
                    Pointer.to(deviceInputVelocity),
                    Pointer.to(deviceInputMinVelocity),
                    Pointer.to(deviceInputMaxVelocity),
                    Pointer.to(deviceInputMinLocation),
                    Pointer.to(deviceInputMaxLocation),
                    Pointer.to(new int[]{numElements}),
                    Pointer.to(new int[]{n})
            );
            cuLaunchKernel(function_update,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernel_updateParameters, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();

            cuMemcpyDtoH(Pointer.to(gpuLocation), deviceInputLocation, numBytes);

            FitnessCalTask task = new FitnessCalTask(0, swarmSize, false);
            ForkJoinPool pool = new ForkJoinPool();
            pool.invoke(task);
            int bestParticleIndex = task.join();
            if (fitness[bestParticleIndex] < gBest) {
                gBest = fitness[bestParticleIndex];
                cuMemcpyDtoDAsync(deviceGBestLocation,
                        deviceInputLocation.withByteOffset(bestParticleIndex * numBytesOfOneParticle),
                        numBytesOfOneParticle, stream);
            }
            // 如果全局粒子在可行域内，如果已经达到模型的要求，是一个足够好的适应度值那么就结束寻优
            if (isGBestfeasible)
                tol = gBest - optModel.getTolFitness();
            logger.debug("ITERATION " + iterNum + ": Value: " + gBest + "  " + isGBestfeasible);
            iterNum++;
        }

        logger.info("PSO执行过程用时: " + (System.currentTimeMillis() - start) + "ms");

        if (isGBestfeasible) {
            logger.info("Solution found at iteration " + iterNum + ", best fitness value: " + gBest);
        } else {
            logger.warn("Solution not found");
        }

        // Clean up.
        cuMemFree(deviceInputLocation);
        cuMemFree(devicePBestLocation);
        cuMemFree(deviceGBestLocation);
        cuMemFree(deviceInputVelocity);
        cuMemFree(deviceRandom1);
        cuMemFree(deviceRandom2);
        cuMemFree(deviceInputMinLocation);
        cuMemFree(deviceInputMinVelocity);
        cuMemFree(deviceInputMaxLocation);
        cuMemFree(deviceInputMaxVelocity);
        cuStreamDestroy(stream);
        curandDestroyGenerator(generator);
    }

    public boolean isGBestfeasible() {
        return isGBestfeasible;
    }


}
