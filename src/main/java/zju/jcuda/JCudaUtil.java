package zju.jcuda;

import jcuda.CudaException;
import jcuda.driver.CUdevice;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import static jcuda.driver.CUdevice_attribute.*;
import static jcuda.driver.JCudaDriver.*;

/**
 * @Author: Fang Rui
 * @Date: 2018/8/8
 * @Time: 15:47
 */
public class JCudaUtil {

    private static Logger logger = LogManager.getLogger(JCudaUtil.class);

    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is
     * compiled from the given file using NVCC. The name of the
     * PTX file is returned.
     *
     * @param inputFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    public static String preparePtxFile(String inputFileName) throws IOException {
        String cuFileName = "kernel/" + inputFileName;
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1) {
            endIndex = cuFileName.length() - 1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex + 1) + "ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists()) {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists()) {
            throw new IOException("Input file not found: " + cuFileName);
        }
        String modelString = "-m" + System.getProperty("sun.arch.data.model");
        String command =
                "nvcc " + modelString + " -ptx " +
                        cuFile.getPath() + " -o " + ptxFileName;

        logger.info("Executing\n" + command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
                new String(toByteArray(process.getErrorStream()));
        String outputMessage =
                new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try {
            exitValue = process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(
                    "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0) {
            logger.info("nvcc process exitValue " + exitValue);
            logger.info("errorMessage:\n" + errorMessage);
            logger.info("outputMessage:\n" + outputMessage);
            throw new IOException(
                    "Could not create .ptx file: " + errorMessage);
        }

        logger.info("Finished creating PTX file");
        return ptxFileName;
    }

    public static float[] double2float(double[] doubles) {
        float[] floats = new float[doubles.length];
        for (int i = 0; i < floats.length; i++) {
            floats[i] = (float) doubles[i];
        }
        return floats;
    }

    /**
     * Queries and prints all attributes of all available devices
     */
    public static void queryDeviceInfo() throws CudaException {
        // Obtain the number of devices
        int deviceCountArray[] = {0};
        cuDeviceGetCount(deviceCountArray);
        int deviceCount = deviceCountArray[0];
        if (deviceCount == 0) {
            throw new CudaException("Found no devices");
        }
        logger.info("Found " + deviceCount + " devices.");

        for (int i = 0; i < deviceCount; i++) {
            CUdevice device = new CUdevice();
            cuDeviceGet(device, i);

            // Obtain the device name
            byte deviceName[] = new byte[1024];
            cuDeviceGetName(
                    deviceName, deviceName.length, device);
            String name = createString(deviceName);

            logger.info("Device " + i + ": " + name);

            // Obtain the device attributes
            int array[] = {0};
            List<Integer> attributes = getAttributes();
            for (Integer attribute : attributes) {
                String description = getAttributeDescription(attribute);
                cuDeviceGetAttribute(array, attribute, device);
                int value = array[0];

                System.out.printf("    %-52s : %d\n", description, value);
            }
        }
    }

    private static String getAttributeDescription(int attribute) {
        switch (attribute) {
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
                return "Maximum number of threads per block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X:
                return "Maximum x-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y:
                return "Maximum y-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z:
                return "Maximum z-dimension of a block";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X:
                return "Maximum x-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y:
                return "Maximum y-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z:
                return "Maximum z-dimension of a grid";
            case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK:
                return "Maximum shared memory per thread block in bytes";
            case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY:
                return "Total constant memory on the device in bytes";
            case CU_DEVICE_ATTRIBUTE_WARP_SIZE:
                return "Warp size in threads";
            case CU_DEVICE_ATTRIBUTE_MAX_PITCH:
                return "Maximum pitch in bytes allowed for memory copies";
            case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK:
                return "Maximum number of 32-bit registers per thread block";
            case CU_DEVICE_ATTRIBUTE_CLOCK_RATE:
                return "Clock frequency in kilohertz";
            case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT:
                return "Alignment requirement";
            case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
                return "Number of multiprocessors on the device";
            case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT:
                return "Whether there is a run time limit on kernels";
            case CU_DEVICE_ATTRIBUTE_INTEGRATED:
                return "Device is integrated with host memory";
            case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY:
                return "Device can map host memory into CUDA address space";
            case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE:
                return "Compute mode";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH:
                return "Maximum 1D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH:
                return "Maximum 2D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT:
                return "Maximum 2D texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH:
                return "Maximum 3D texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT:
                return "Maximum 3D texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH:
                return "Maximum 3D texture depth";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH:
                return "Maximum 2D layered texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT:
                return "Maximum 2D layered texture height";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS:
                return "Maximum layers in a 2D layered texture";
            case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT:
                return "Alignment requirement for surfaces";
            case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS:
                return "Device can execute multiple kernels concurrently";
            case CU_DEVICE_ATTRIBUTE_ECC_ENABLED:
                return "Device has ECC support enabled";
            case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID:
                return "PCI bus ID of the device";
            case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID:
                return "PCI device ID of the device";
            case CU_DEVICE_ATTRIBUTE_TCC_DRIVER:
                return "Device is using TCC driver model";
            case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE:
                return "Peak memory clock frequency in kilohertz";
            case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH:
                return "Global memory bus width in bits";
            case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE:
                return "Size of L2 cache in bytes";
            case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR:
                return "Maximum resident threads per multiprocessor";
            case CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT:
                return "Number of asynchronous engines";
            case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING:
                return "Device shares a unified address space with the host";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH:
                return "Maximum 1D layered texture width";
            case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS:
                return "Maximum layers in a 1D layered texture";
            case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID:
                return "PCI domain ID of the device";
        }
        return "(UNKNOWN ATTRIBUTE)";
    }

    private static List<Integer> getAttributes() {
        List<Integer> list = new ArrayList<Integer>();
        list.add(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
        list.add(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_PITCH);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
        list.add(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
        list.add(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT);
        list.add(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        list.add(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT);
        list.add(CU_DEVICE_ATTRIBUTE_INTEGRATED);
        list.add(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY);
        list.add(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT);
        list.add(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS);
        list.add(CU_DEVICE_ATTRIBUTE_ECC_ENABLED);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);
        list.add(CU_DEVICE_ATTRIBUTE_TCC_DRIVER);
        list.add(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
        list.add(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
        list.add(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
        list.add(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);
        list.add(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH);
        list.add(CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS);
        list.add(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID);
        return list;
    }

    private static String createString(byte bytes[]) {
        StringBuilder sb = new StringBuilder();
        for (byte aByte : bytes) {
            char c = (char) aByte;
            if (c == 0) {
                break;
            }
            sb.append(c);
        }
        return sb.toString();
    }

    private static byte[] toByteArray(InputStream inputStream) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true) {
            int read = inputStream.read(buffer);
            if (read == -1) {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }
}
