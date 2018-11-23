package zju.pso;

import jcuda.*;
import jcuda.driver.*;
import jcuda.jcurand.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import zju.jcuda.JCudaUtil;
import zju.measure.MeasVector;
import zju.util.YMatrixGetter;

import java.io.IOException;
import java.util.List;

import static jcuda.driver.CUstream_flags.*;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.*;
import static jcuda.runtime.JCuda.*;
import static zju.pso.PsoUtil.*;


/**
 * 使用Cuda进行并行加速
 *
 * @Author: Fang Rui
 * @Date: 2018/8/27
 * @Time: 18:54
 */
public class ParallelPso extends AbstractPso implements PsoConstants {

    private static Logger logger = LogManager.getLogger(ParallelPso.class);

    private OptModel optModel;
    private float[] fitness; // 各粒子当前适应度值
    private float[] pBest; // 各粒子当前适应度值
    private float[] gBestFitness;
    private float[] gBestLoc;
    private boolean isGBestfeasible = false; // gBest是否在可行域内

    private MeasVector meas;
    private YMatrixGetter Y;
    private double[] threshold;
    private int[] zeroPBuses;
    private int[] zeroQBuses;
    private float tol_p;
    private float tol_q;
    private boolean isConstrained = false;

    private int numElements;
    private int n;
    private int numBytes;
    private int numBytesOfOneParticle;
    private int blockSizeX;

    private CUfunction function_initialize = new CUfunction();
    private CUfunction function_warmInitialize = new CUfunction();
    private CUfunction function_update = new CUfunction();
    private CUfunction function_calBusV = new CUfunction();
    private CUfunction function_calBusP = new CUfunction();
    private CUfunction function_calBusQ = new CUfunction();
    private CUfunction function_calLinePFrom = new CUfunction();
    private CUfunction function_calLineQFrom = new CUfunction();
    private CUfunction function_calLinePTo = new CUfunction();
    private CUfunction function_calLineQTo = new CUfunction();
    private CUfunction function_calZeroBusP = new CUfunction();
    private CUfunction function_calZeroBusQ = new CUfunction();
    private CUfunction function_findThisBest = new CUfunction();
    private CUfunction function_findBest = new CUfunction();

    private CUdeviceptr devicePBest = new CUdeviceptr();
    private CUdeviceptr deviceInputLocation = new CUdeviceptr();
    private CUdeviceptr devicePBestLocation = new CUdeviceptr();
    private CUdeviceptr deviceGBestLocation = new CUdeviceptr();
    private CUdeviceptr deviceGBestObj = new CUdeviceptr();
    private CUdeviceptr deviceInputVelocity = new CUdeviceptr();
    private CUdeviceptr deviceInputMinLocation = new CUdeviceptr();
    private CUdeviceptr deviceInputMinVelocity = new CUdeviceptr();
    private CUdeviceptr deviceInputMaxLocation = new CUdeviceptr();
    private CUdeviceptr deviceInputMaxVelocity = new CUdeviceptr();

    private CUdeviceptr deviceRandom1 = new CUdeviceptr();
    private CUdeviceptr deviceRandom2 = new CUdeviceptr();
    private curandGenerator generator = new curandGenerator();

    private CUdeviceptr deviceBusVPosition = new CUdeviceptr();
    private CUdeviceptr deviceBusPPosition = new CUdeviceptr();
    private CUdeviceptr deviceBusQPosition = new CUdeviceptr();
    private CUdeviceptr deviceLinePFromPosition = new CUdeviceptr();
    private CUdeviceptr deviceLineQFromPosition = new CUdeviceptr();
    private CUdeviceptr deviceLinePToPosition = new CUdeviceptr();
    private CUdeviceptr deviceLineQToPosition = new CUdeviceptr();
    private CUdeviceptr deviceZeroBusPPosition = new CUdeviceptr();
    private CUdeviceptr deviceZeroBusQPosition = new CUdeviceptr();

    private CUdeviceptr deviceZ = new CUdeviceptr();
    private CUdeviceptr deviceThreshold = new CUdeviceptr();
    private CUdeviceptr deviceObj = new CUdeviceptr();
    private CUdeviceptr deviceG = new CUdeviceptr();
    private CUdeviceptr deviceB = new CUdeviceptr();
    private CUdeviceptr deviceJA = new CUdeviceptr();
    private CUdeviceptr deviceIA = new CUdeviceptr();
    private CUdeviceptr deviceLink = new CUdeviceptr();

    private CUdeviceptr deviceLinePFromI = new CUdeviceptr();
    private CUdeviceptr deviceLinePFromJ = new CUdeviceptr();
    private CUdeviceptr deviceLinePFromG = new CUdeviceptr();
    private CUdeviceptr deviceLinePFromB = new CUdeviceptr();
    private CUdeviceptr deviceLinePFromG1 = new CUdeviceptr();
    private CUdeviceptr deviceLinePFromB1 = new CUdeviceptr();

    private CUdeviceptr deviceLineQFromI = new CUdeviceptr();
    private CUdeviceptr deviceLineQFromJ = new CUdeviceptr();
    private CUdeviceptr deviceLineQFromG = new CUdeviceptr();
    private CUdeviceptr deviceLineQFromB = new CUdeviceptr();
    private CUdeviceptr deviceLineQFromG1 = new CUdeviceptr();
    private CUdeviceptr deviceLineQFromB1 = new CUdeviceptr();

    private CUdeviceptr deviceLinePToI = new CUdeviceptr();
    private CUdeviceptr deviceLinePToJ = new CUdeviceptr();
    private CUdeviceptr deviceLinePToG = new CUdeviceptr();
    private CUdeviceptr deviceLinePToB = new CUdeviceptr();
    private CUdeviceptr deviceLinePToG1 = new CUdeviceptr();
    private CUdeviceptr deviceLinePToB1 = new CUdeviceptr();

    private CUdeviceptr deviceLineQToI = new CUdeviceptr();
    private CUdeviceptr deviceLineQToJ = new CUdeviceptr();
    private CUdeviceptr deviceLineQToG = new CUdeviceptr();
    private CUdeviceptr deviceLineQToB = new CUdeviceptr();
    private CUdeviceptr deviceLineQToG1 = new CUdeviceptr();
    private CUdeviceptr deviceLineQToB1 = new CUdeviceptr();

    private CUdeviceptr deviceViolation = new CUdeviceptr();
    private CUdeviceptr deviceIsGBestfeasible = new CUdeviceptr();

    private CUstream[] streamArray = new CUstream[9];

    private float[] gpuLocation;

    public ParallelPso(OptModel optModel) {
        this(optModel, (int) (10 + 2 * Math.sqrt(optModel.getDimentions())));
    }

    public ParallelPso(OptModel optModel, int swarmSize) {
        super(swarmSize);
        this.optModel = optModel;
        this.fitness = new float[swarmSize];
    }

    public ParallelPso(OptModel optModel, double[] initVariableState) {
        this(optModel, (int) (10 + 2 * Math.sqrt(optModel.getDimentions())), initVariableState);
    }

    public ParallelPso(OptModel optModel, int swarmSize, double[] initVariableState) {
        super(swarmSize, initVariableState);
        this.optModel = optModel;
        this.fitness = new float[swarmSize];
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
        cuModuleGetFunction(function_calBusP, module, "calBusP");
        cuModuleGetFunction(function_calBusQ, module, "calBusQ");
        cuModuleGetFunction(function_calLinePFrom, module, "calLinePFrom");
        cuModuleGetFunction(function_calLineQFrom, module, "calLineQFrom");
        cuModuleGetFunction(function_calLinePTo, module, "calLinePTo");
        cuModuleGetFunction(function_calLineQTo, module, "calLineQTo");
        cuModuleGetFunction(function_calZeroBusP, module, "calZeroBusP");
        cuModuleGetFunction(function_calZeroBusQ, module, "calZeroBusQ");
        cuModuleGetFunction(function_findThisBest, module, "findThisBest");
        cuModuleGetFunction(function_findBest, module, "findBest");
    }

    @Override
    protected void initializeSwarm() {
        n = optModel.getDimentions();
        numElements = n * swarmSize;
        numBytes = numElements * Sizeof.FLOAT;
        numBytesOfOneParticle = n * Sizeof.FLOAT;
        blockSizeX = 512;

        float[] minLoc = doubleArr2floatArr(optModel.getMinLoc());
        float[] maxLoc = doubleArr2floatArr(optModel.getMaxLoc());
        float[] minVel = doubleArr2floatArr(optModel.getMinVel());
        float[] maxVel = doubleArr2floatArr(optModel.getMaxVel());

        gpuLocation = new float[numElements];
        gBestFitness = new float[1];
        gBestLoc = new float[n];

        cuMemAlloc(devicePBest, swarmSize * Sizeof.FLOAT);
        cuMemAlloc(deviceInputLocation, numBytes);
        cuMemAlloc(devicePBestLocation, numBytes);
        cuMemAlloc(deviceGBestLocation, numBytesOfOneParticle);
        cuMemAlloc(deviceInputVelocity, numBytes);
        cuMemAlloc(deviceGBestObj, Sizeof.FLOAT);

        cuMemAlloc(deviceRandom1, numBytes);
        cuMemAlloc(deviceRandom2, numBytes);
        cuMemAlloc(deviceInputMinLocation, numBytesOfOneParticle);
        cuMemAlloc(deviceInputMinVelocity, numBytesOfOneParticle);
        cuMemAlloc(deviceInputMaxLocation, numBytesOfOneParticle);
        cuMemAlloc(deviceInputMaxVelocity, numBytesOfOneParticle);

        initCalFitness();
        if (zeroPBuses.length != 0 && zeroQBuses.length != 0) {
            isConstrained = true;
            initCalViolation();
        }

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
            float[] initArray = doubleArr2floatArr(initVariableState);
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
                    (int) Math.ceil((double) numElements / blockSizeX), 1, 1,      // Grid dimension
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
                    (int) Math.ceil((double) numElements / blockSizeX), 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernel_initializeParameters, null // Kernel- and extra parameters
            );
        }
        cuCtxSynchronize();

        for (int i = 0; i < streamArray.length; i++) {
            streamArray[i] = new CUstream();
            cuStreamCreate(streamArray[i], CU_STREAM_DEFAULT);
        }
        calFitness();
        if (isConstrained) {
            cuMemcpyHtoD(deviceIsGBestfeasible, Pointer.to(new int[]{0}), Sizeof.INT);
            calViolation();
        } else {
            isGBestfeasible = true;
        }
        cuMemcpyDtoD(devicePBest, deviceObj, swarmSize * Sizeof.FLOAT);
        cuMemcpyDtoD(devicePBestLocation, deviceInputLocation, numBytes);
        cuMemcpyHtoD(deviceGBestObj, Pointer.to(new float[]{1e7f}), Sizeof.FLOAT);

        cudaDeviceSynchronize();
        findBest();
    }

    @Override
    public void execute() {
        long start = System.currentTimeMillis();
        initializeCuda();
        initializeSwarm();
        logger.info("并行粒子群初始化用时: " + (System.currentTimeMillis() - start) + "ms");


        int iterNum = 0;
        int maxIter = optModel.getMaxIter();

        double tol = 1e6;
        float w; // 惯性权重

        while (iterNum < maxIter && tol > 0) {
            start = System.currentTimeMillis();
            w = (float) (W_UPPERBOUND - (((double) iterNum) / maxIter) * (W_UPPERBOUND - W_LOWERBOUND)); // 惯性逐渐减小

            curandSetPseudoRandomGeneratorSeed(generator, System.currentTimeMillis());
            curandGenerateUniform(generator, deviceRandom1, numElements);
            curandSetPseudoRandomGeneratorSeed(generator, System.currentTimeMillis() + 1);
            curandGenerateUniform(generator, deviceRandom2, numElements);
            logger.info("随机数产生用时: " + (System.currentTimeMillis() - start) + "ms");

            updateSwarm(w);
            calFitness();
            if (isConstrained) {
                calViolation();
            }

            cudaDeviceSynchronize();
            findBest();

            // 如果全局粒子在可行域内，如果已经达到模型的要求，是一个足够好的适应度值那么就结束寻优
            if (isGBestfeasible) {
                cuMemcpyDtoH(Pointer.to(gBestFitness), deviceGBestObj, Sizeof.FLOAT);
                tol = gBestFitness[0] - optModel.getTolFitness();
            }
            logger.info("ITERATION " + iterNum + ": " + (System.currentTimeMillis() - start) + "ms");
            iterNum++;
        }

        logger.info("PSO执行过程用时: " + (System.currentTimeMillis() - start) + "ms");
        cuMemcpyDtoH(Pointer.to(gBestFitness), deviceGBestObj, Sizeof.FLOAT);
        cuMemcpyDtoH(Pointer.to(gBestLoc), deviceGBestLocation, n * Sizeof.FLOAT);

        gBest = gBestFitness[0];
        gBestLocation = new Location(floatArr2doubleArr(gBestLoc));

        if (isGBestfeasible) {
            logger.info("Solution found at iteration " + iterNum + ", best fitness value: " + gBest);
        } else {
            logger.warn("Solution not found");
        }

        // Clean up.
        clean();
    }

    private void initCalFitness() {
        List<Double> G = Y.getAdmittance()[0].getVA();
        List<Double> B = Y.getAdmittance()[1].getVA();
        List<Integer> JA = Y.getAdmittance()[0].getJA();
        int[] IA = Y.getAdmittance()[0].getIA();
        List<Integer> LINK = Y.getAdmittance()[0].getLINK();

        cuMemAlloc(deviceZ, meas.getZ().getN() * Sizeof.FLOAT);
        cuMemAlloc(deviceThreshold, threshold.length * Sizeof.FLOAT);
        cuMemAlloc(deviceObj, swarmSize * Sizeof.FLOAT);

        cuMemAlloc(deviceBusVPosition, meas.getBus_v_pos().length * Sizeof.INT);
        cuMemAlloc(deviceBusPPosition, meas.getBus_p_pos().length * Sizeof.INT);
        cuMemAlloc(deviceBusQPosition, meas.getBus_q_pos().length * Sizeof.INT);

        cuMemAlloc(deviceG, G.size() * Sizeof.FLOAT);
        cuMemAlloc(deviceB, B.size() * Sizeof.FLOAT);
        cuMemAlloc(deviceJA, JA.size() * Sizeof.INT);
        cuMemAlloc(deviceIA, IA.length * Sizeof.INT);
        cuMemAlloc(deviceLink, LINK.size() * Sizeof.INT);

        cuMemcpyHtoD(deviceZ, Pointer.to(doubleArr2floatArr(meas.getZ().getValues())), meas.getZ().getN() * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceThreshold, Pointer.to(doubleArr2floatArr(threshold)), threshold.length * Sizeof.FLOAT);

        cuMemcpyHtoD(deviceBusVPosition, Pointer.to(meas.getBus_v_pos()), meas.getBus_v_pos().length * Sizeof.INT);
        cuMemcpyHtoD(deviceBusPPosition, Pointer.to(meas.getBus_p_pos()), meas.getBus_p_pos().length * Sizeof.INT);
        cuMemcpyHtoD(deviceBusQPosition, Pointer.to(meas.getBus_q_pos()), meas.getBus_q_pos().length * Sizeof.INT);

        cuMemcpyHtoD(deviceG, Pointer.to(doubleArr2floatArr(G.stream().mapToDouble(Double::doubleValue).toArray())), G.size() * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceB, Pointer.to(doubleArr2floatArr(B.stream().mapToDouble(Double::doubleValue).toArray())), B.size() * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceJA, Pointer.to(JA.stream().mapToInt(Integer::intValue).toArray()), JA.size() * Sizeof.INT);
        cuMemcpyHtoD(deviceIA, Pointer.to(IA), IA.length * Sizeof.INT);
        cuMemcpyHtoD(deviceLink, Pointer.to(LINK.stream().mapToInt(Integer::intValue).toArray()), LINK.size() * Sizeof.INT);

        initialLineVar(meas.getLine_from_p_pos(), deviceLinePFromPosition, deviceLinePFromI, deviceLinePFromJ,
                deviceLinePFromG, deviceLinePFromB, deviceLinePFromG1, deviceLinePFromB1, YMatrixGetter.LINE_FROM);
        initialLineVar(meas.getLine_from_q_pos(), deviceLineQFromPosition, deviceLineQFromI, deviceLineQFromJ,
                deviceLineQFromG, deviceLineQFromB, deviceLineQFromG1, deviceLineQFromB1, YMatrixGetter.LINE_FROM);
        initialLineVar(meas.getLine_to_p_pos(), deviceLinePToPosition, deviceLinePToI, deviceLinePToJ,
                deviceLinePToG, deviceLinePToB, deviceLinePToG1, deviceLinePToB1, YMatrixGetter.LINE_TO);
        initialLineVar(meas.getLine_to_q_pos(), deviceLineQToPosition, deviceLineQToI, deviceLineQToJ,
                deviceLineQToG, deviceLineQToB, deviceLineQToG1, deviceLineQToB1, YMatrixGetter.LINE_TO);
    }


    private void initCalViolation() {
        cuMemAlloc(deviceZeroBusPPosition, zeroPBuses.length * Sizeof.INT);
        cuMemAlloc(deviceZeroBusQPosition, zeroQBuses.length * Sizeof.INT);
        cuMemAlloc(deviceViolation, swarmSize * Sizeof.FLOAT);
        cuMemAlloc(deviceIsGBestfeasible, Sizeof.INT);

        cuMemcpyHtoD(deviceZeroBusPPosition, Pointer.to(zeroPBuses), zeroPBuses.length * Sizeof.INT);
        cuMemcpyHtoD(deviceZeroBusQPosition, Pointer.to(zeroQBuses), zeroQBuses.length * Sizeof.INT);
    }


    private void initialLineVar(int[] pos, CUdeviceptr devicePosition, CUdeviceptr deviceI, CUdeviceptr deviceJ,
                                CUdeviceptr deviceG, CUdeviceptr deviceB,
                                CUdeviceptr deviceG1, CUdeviceptr deviceB1, int type) {
        int length = pos.length;
        if (length == 0)
            return;
        cuMemAlloc(devicePosition, length * Sizeof.INT);
        cuMemcpyHtoD(devicePosition, Pointer.to(pos), length * Sizeof.INT);
        cuMemAlloc(deviceI, length * Sizeof.INT);
        cuMemAlloc(deviceJ, length * Sizeof.INT);
        cuMemAlloc(deviceG, length * Sizeof.FLOAT);
        cuMemAlloc(deviceB, length * Sizeof.FLOAT);
        cuMemAlloc(deviceG1, length * Sizeof.FLOAT);
        cuMemAlloc(deviceB1, length * Sizeof.FLOAT);

        int[] iBus = new int[length];
        int[] jBus = new int[length];
        float[] g = new float[length];
        float[] b = new float[length];
        float[] g1 = new float[length];
        float[] b1 = new float[length];
        for (int i = 0; i < length; i++) {
            int branchId = meas.getLine_from_q_pos()[i];//num starts from 1
            int[] ij = Y.getFromTo(branchId);
            iBus[i] = ij[0] - 1;
            jBus[i] = ij[1] - 1;
            double[] gbg1b1 = Y.getLineAdmittance(branchId, type);
            g[i] = (float) gbg1b1[0];
            b[i] = (float) gbg1b1[1];
            g1[i] = (float) gbg1b1[2];
            b1[i] = (float) gbg1b1[3];
        }
        cuMemcpyHtoD(deviceI, Pointer.to(iBus), length * Sizeof.INT);
        cuMemcpyHtoD(deviceJ, Pointer.to(jBus), length * Sizeof.INT);
        cuMemcpyHtoD(deviceG, Pointer.to(g), length * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceB, Pointer.to(b), length * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceG1, Pointer.to(g1), length * Sizeof.FLOAT);
        cuMemcpyHtoD(deviceB1, Pointer.to(b1), length * Sizeof.FLOAT);
    }

    private void updateSwarm(float w) {
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
                (int) Math.ceil((double) numElements / blockSizeX), 1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernel_updateParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

    }

    private void calFitness() {
        // 计算电压量测
        int offset = 0;
        Pointer kernel_calBusVParameters = Pointer.to(
                Pointer.to(deviceInputLocation),
                Pointer.to(deviceBusVPosition),
                Pointer.to(deviceZ),
                Pointer.to(deviceThreshold),
                Pointer.to(deviceObj),
                Pointer.to(new int[]{swarmSize}),
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{meas.getBus_v_pos().length}),
                Pointer.to(new int[]{offset})
        );
        cuLaunchKernel(function_calBusV,
                (int) Math.ceil(meas.getBus_v_pos().length * swarmSize / 512f), 1, 1,      // Grid dimension
                512, 1, 1,      // Block dimension
                0, streamArray[0],               // Shared memory size and stream
                kernel_calBusVParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        offset += meas.getBus_v_pos().length;

        // 计算有功注入
        Pointer kernel_calBusPParameters = Pointer.to(
                Pointer.to(deviceInputLocation),
                Pointer.to(deviceBusPPosition),
                Pointer.to(deviceZ),
                Pointer.to(deviceThreshold),
                Pointer.to(deviceObj),
                Pointer.to(deviceG),
                Pointer.to(deviceB),
                Pointer.to(deviceJA),
                Pointer.to(deviceIA),
                Pointer.to(deviceLink),
                Pointer.to(new int[]{swarmSize}),
                Pointer.to(new int[]{Y.getAdmittance()[0].getM()}),
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{meas.getBus_p_pos().length}),
                Pointer.to(new int[]{offset})
        );
        cuLaunchKernel(function_calBusP,
                (int) Math.ceil(meas.getBus_p_pos().length * swarmSize / 512f), 1, 1,      // Grid dimension
                512, 1, 1,      // Block dimension
                0, streamArray[1],               // Shared memory size and stream
                kernel_calBusPParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        offset += meas.getBus_p_pos().length;

        // 计算无功注入
        Pointer kernel_calBusQParameters = Pointer.to(
                Pointer.to(deviceInputLocation),
                Pointer.to(deviceBusQPosition),
                Pointer.to(deviceZ),
                Pointer.to(deviceThreshold),
                Pointer.to(deviceObj),
                Pointer.to(deviceG),
                Pointer.to(deviceB),
                Pointer.to(deviceJA),
                Pointer.to(deviceIA),
                Pointer.to(deviceLink),
                Pointer.to(new int[]{swarmSize}),
                Pointer.to(new int[]{Y.getAdmittance()[0].getM()}),
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{meas.getBus_q_pos().length}),
                Pointer.to(new int[]{offset})
        );
        cuLaunchKernel(function_calBusQ,
                (int) Math.ceil(meas.getBus_q_pos().length * swarmSize / 512f), 1, 1,      // Grid dimension
                512, 1, 1,      // Block dimension
                0, streamArray[2],               // Shared memory size and stream
                kernel_calBusQParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
        offset += meas.getBus_q_pos().length;

        // 计算LinePFrom
        if (meas.getLine_from_p_pos().length != 0) {
            Pointer kernel_calLinePFromParameters = Pointer.to(
                    Pointer.to(deviceInputLocation),
                    Pointer.to(deviceZ),
                    Pointer.to(deviceThreshold),
                    Pointer.to(deviceObj),
                    Pointer.to(deviceLinePFromI),
                    Pointer.to(deviceLinePFromJ),
                    Pointer.to(deviceLinePFromG),
                    Pointer.to(deviceLinePFromB),
                    Pointer.to(deviceLinePFromG1),
                    Pointer.to(deviceLinePFromB1),
                    Pointer.to(new int[]{swarmSize}),
                    Pointer.to(new int[]{Y.getAdmittance()[0].getM()}),
                    Pointer.to(new int[]{n}),
                    Pointer.to(new int[]{meas.getLine_from_p_pos().length}),
                    Pointer.to(new int[]{offset})
            );
            cuLaunchKernel(function_calLinePFrom,
                    (int) Math.ceil(meas.getLine_from_p_pos().length * swarmSize / 512f), 1, 1,      // Grid dimension
                    512, 1, 1,      // Block dimension
                    0, streamArray[3],               // Shared memory size and stream
                    kernel_calLinePFromParameters, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            offset += meas.getLine_from_p_pos().length;
        }

        // 计算LineQFrom
        if (meas.getLine_from_q_pos().length != 0) {
            Pointer kernel_calLineQFromParameters = Pointer.to(
                    Pointer.to(deviceInputLocation),
                    Pointer.to(deviceZ),
                    Pointer.to(deviceThreshold),
                    Pointer.to(deviceObj),
                    Pointer.to(deviceLineQFromI),
                    Pointer.to(deviceLineQFromJ),
                    Pointer.to(deviceLineQFromG),
                    Pointer.to(deviceLineQFromB),
                    Pointer.to(deviceLineQFromG1),
                    Pointer.to(deviceLineQFromB1),
                    Pointer.to(new int[]{swarmSize}),
                    Pointer.to(new int[]{Y.getAdmittance()[0].getM()}),
                    Pointer.to(new int[]{n}),
                    Pointer.to(new int[]{meas.getLine_from_q_pos().length}),
                    Pointer.to(new int[]{offset})
            );
            cuLaunchKernel(function_calLineQFrom,
                    (int) Math.ceil(meas.getLine_from_q_pos().length * swarmSize / 512f), 1, 1,      // Grid dimension
                    512, 1, 1,      // Block dimension
                    0, streamArray[4],               // Shared memory size and stream
                    kernel_calLineQFromParameters, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            offset += meas.getLine_from_q_pos().length;
        }

        // 计算LinePTo
        if (meas.getLine_to_p_pos().length != 0) {
            Pointer kernel_calLinePToParameters = Pointer.to(
                    Pointer.to(deviceInputLocation),
                    Pointer.to(deviceZ),
                    Pointer.to(deviceThreshold),
                    Pointer.to(deviceObj),
                    Pointer.to(deviceLinePToI),
                    Pointer.to(deviceLinePToJ),
                    Pointer.to(deviceLinePToG),
                    Pointer.to(deviceLinePToB),
                    Pointer.to(deviceLinePToG1),
                    Pointer.to(deviceLinePToB1),
                    Pointer.to(new int[]{swarmSize}),
                    Pointer.to(new int[]{Y.getAdmittance()[0].getM()}),
                    Pointer.to(new int[]{n}),
                    Pointer.to(new int[]{meas.getLine_to_p_pos().length}),
                    Pointer.to(new int[]{offset})
            );
            cuLaunchKernel(function_calLinePTo,
                    (int) Math.ceil(meas.getLine_to_p_pos().length * swarmSize / 512f), 1, 1,      // Grid dimension
                    512, 1, 1,      // Block dimension
                    0, streamArray[5],               // Shared memory size and stream
                    kernel_calLinePToParameters, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            offset += meas.getLine_to_p_pos().length;
        }

        // 计算LineQTo
        if (meas.getLine_to_q_pos().length != 0) {
            Pointer kernel_calLineQToParameters = Pointer.to(
                    Pointer.to(deviceInputLocation),
                    Pointer.to(deviceZ),
                    Pointer.to(deviceThreshold),
                    Pointer.to(deviceObj),
                    Pointer.to(deviceLineQToI),
                    Pointer.to(deviceLineQToJ),
                    Pointer.to(deviceLineQToG),
                    Pointer.to(deviceLineQToB),
                    Pointer.to(deviceLineQToG1),
                    Pointer.to(deviceLineQToB1),
                    Pointer.to(new int[]{swarmSize}),
                    Pointer.to(new int[]{Y.getAdmittance()[0].getM()}),
                    Pointer.to(new int[]{n}),
                    Pointer.to(new int[]{meas.getLine_to_q_pos().length}),
                    Pointer.to(new int[]{offset})
            );
            cuLaunchKernel(function_calLineQTo,
                    (int) Math.ceil(meas.getLine_to_q_pos().length * swarmSize / 512f), 1, 1,      // Grid dimension
                    512, 1, 1,      // Block dimension
                    0, streamArray[6],               // Shared memory size and stream
                    kernel_calLineQToParameters, null // Kernel- and extra parameters
            );
            cuCtxSynchronize();
            offset += meas.getLine_to_q_pos().length;
        }
        assert offset == meas.getZ().getN();
    }

    private void calViolation() {
        Pointer kernel_calZeroBusPParameters = Pointer.to(
                Pointer.to(deviceInputLocation),
                Pointer.to(deviceZeroBusPPosition),
                Pointer.to(deviceG),
                Pointer.to(deviceB),
                Pointer.to(deviceJA),
                Pointer.to(deviceIA),
                Pointer.to(deviceLink),
                Pointer.to(deviceViolation),
                Pointer.to(new int[]{swarmSize}),
                Pointer.to(new int[]{Y.getAdmittance()[0].getM()}),
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{zeroPBuses.length}),
                Pointer.to(new float[]{tol_p})
        );
        cuLaunchKernel(function_calZeroBusP,
                (int) Math.ceil(zeroPBuses.length * swarmSize / 512f), 1, 1,      // Grid dimension
                512, 1, 1,      // Block dimension
                0, streamArray[7],               // Shared memory size and stream
                kernel_calZeroBusPParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        Pointer kernel_calZeroBusQParameters = Pointer.to(
                Pointer.to(deviceInputLocation),
                Pointer.to(deviceZeroBusQPosition),
                Pointer.to(deviceObj),
                Pointer.to(deviceG),
                Pointer.to(deviceB),
                Pointer.to(deviceJA),
                Pointer.to(deviceIA),
                Pointer.to(deviceLink),
                Pointer.to(deviceViolation),
                Pointer.to(deviceIsGBestfeasible),
                Pointer.to(new int[]{swarmSize}),
                Pointer.to(new int[]{Y.getAdmittance()[0].getM()}),
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{zeroQBuses.length}),
                Pointer.to(new float[]{tol_q})
        );
        cuLaunchKernel(function_calZeroBusQ,
                (int) Math.ceil(zeroQBuses.length * swarmSize / 512f), 1, 1,      // Grid dimension
                512, 1, 1,      // Block dimension
                0, streamArray[8],               // Shared memory size and stream
                kernel_calZeroBusQParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        int[] bool = new int[1];
        cuMemcpyDtoH(Pointer.to(bool), deviceIsGBestfeasible, Sizeof.INT);
        if (bool[0] == 1)
            isGBestfeasible = true;
    }

    private void findBest() {
        Pointer kernel_findThisBestParameters = Pointer.to(
                Pointer.to(deviceObj),
                Pointer.to(deviceGBestObj),
                Pointer.to(new int[]{swarmSize})
        );
        cuLaunchKernel(function_findThisBest,
                1, 1, 1,      // Grid dimension
                32, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernel_findThisBestParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();

        Pointer kernel_findBestParameters = Pointer.to(
                Pointer.to(deviceInputLocation),
                Pointer.to(devicePBest),
                Pointer.to(devicePBestLocation),
                Pointer.to(deviceGBestLocation),
                Pointer.to(deviceObj),
                Pointer.to(deviceGBestObj),
                Pointer.to(new int[]{swarmSize}),
                Pointer.to(new int[]{n}),
                Pointer.to(new int[]{numElements})
        );
        cuLaunchKernel(function_findBest,
                (int) Math.ceil((double) numElements / 512), 1, 1,      // Grid dimension
                512, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernel_findBestParameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
    }

    private void clean() {

        cuMemFree(devicePBest);
        cuMemFree(deviceInputLocation);
        cuMemFree(devicePBestLocation);
        cuMemFree(deviceGBestLocation);
        cuMemFree(deviceGBestObj);
        cuMemFree(deviceInputVelocity);
        cuMemFree(deviceInputMinLocation);
        cuMemFree(deviceInputMinVelocity);
        cuMemFree(deviceInputMaxLocation);
        cuMemFree(deviceInputMaxVelocity);

        cuMemFree(deviceRandom1);
        cuMemFree(deviceRandom2);
        curandDestroyGenerator(generator);

        cuMemFree(deviceBusVPosition);
        cuMemFree(deviceBusPPosition);
        cuMemFree(deviceBusQPosition);
        cuMemFree(deviceLinePFromPosition);
        cuMemFree(deviceLineQFromPosition);
        cuMemFree(deviceLinePToPosition);
        cuMemFree(deviceLineQToPosition);
        cuMemFree(deviceZeroBusPPosition);
        cuMemFree(deviceZeroBusQPosition);

        cuMemFree(deviceZ);
        cuMemFree(deviceThreshold);
        cuMemFree(deviceObj);
        cuMemFree(deviceG);
        cuMemFree(deviceB);
        cuMemFree(deviceJA);
        cuMemFree(deviceIA);
        cuMemFree(deviceLink);

        cuMemFree(deviceLinePFromI);
        cuMemFree(deviceLinePFromJ);
        cuMemFree(deviceLinePFromG);
        cuMemFree(deviceLinePFromB);
        cuMemFree(deviceLinePFromG1);
        cuMemFree(deviceLinePFromB1);

        cuMemFree(deviceLineQFromI);
        cuMemFree(deviceLineQFromJ);
        cuMemFree(deviceLineQFromG);
        cuMemFree(deviceLineQFromB);
        cuMemFree(deviceLineQFromG1);
        cuMemFree(deviceLineQFromB1);

        cuMemFree(deviceLinePToI);
        cuMemFree(deviceLinePToJ);
        cuMemFree(deviceLinePToG);
        cuMemFree(deviceLinePToB);
        cuMemFree(deviceLinePToG1);
        cuMemFree(deviceLinePToB1);

        cuMemFree(deviceLineQToI);
        cuMemFree(deviceLineQToJ);
        cuMemFree(deviceLineQToG);
        cuMemFree(deviceLineQToB);
        cuMemFree(deviceLineQToG1);
        cuMemFree(deviceLineQToB1);

        cuMemFree(deviceViolation);
        cuMemFree(deviceIsGBestfeasible);
        for (int i = 0; i < streamArray.length; i++) {
            cuStreamDestroy(streamArray[i]);
        }
    }


    public boolean isGBestfeasible() {
        return isGBestfeasible;
    }

    public MeasVector getMeas() {
        return meas;
    }

    public void setMeas(MeasVector meas) {
        this.meas = meas;
    }

    public YMatrixGetter getY() {
        return Y;
    }

    public void setY(YMatrixGetter y) {
        Y = y;
    }

    public double[] getThreshold() {
        return threshold;
    }

    public void setThreshold(double[] threshold) {
        this.threshold = threshold;
    }

    public int[] getZeroPBuses() {
        return zeroPBuses;
    }

    public void setZeroPBuses(int[] zeroPBuses) {
        this.zeroPBuses = zeroPBuses;
    }

    public int[] getZeroQBuses() {
        return zeroQBuses;
    }

    public void setZeroQBuses(int[] zeroQBuses) {
        this.zeroQBuses = zeroQBuses;
    }

    public float getTol_p() {
        return tol_p;
    }

    public void setTol_p(float tol_p) {
        this.tol_p = tol_p;
    }

    public float getTol_q() {
        return tol_q;
    }

    public void setTol_q(float tol_q) {
        this.tol_q = tol_q;
    }
}
