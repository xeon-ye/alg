package zju.jcuda;

import static jcuda.driver.JCudaDriver.*;

import java.io.*;

import jcuda.*;
import jcuda.driver.*;

/**
 * This is a sample class demonstrating how to use the JCuda driver
 * bindings to load a CUDA kernel in form of an PTX file and execute 
 * the kernel. The sample reads a CUDA file, compiles it to a PTX 
 * file using NVCC, and loads the PTX file as a module. <br />
 * <br />
 * The the sample creates a 2D float array and passes it to the kernel 
 * that sums up the elements of each row of the array (each in its 
 * own thread) and writes the sums into an 1D output array.
 */
public class JCudaDriverSample
{
    /**
     * Entry point of this sample
     * 
     * @param args Not used
     * @throws IOException If an IO error occurs
     */
    public static void main(String args[]) throws IOException
    {
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        
        // Create the PTX file by calling the NVCC
        String ptxFileName = JCudaUtil.preparePtxFile("JCudaSampleKernel.cu");
        
        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUcontext pctx = new CUcontext();
        CUdevice dev = new CUdevice();
        cuDeviceGet(dev, 0);
        cuCtxCreate(pctx, 0, dev);

        // Load the ptx file.
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        // Obtain a function pointer to the "sampleKernel" function.
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "sampleKernel");

        int numThreads = 8;
        int size = 128;

        // Allocate and fill host input memory: A 2D float array with
        // 'numThreads' rows and 'size' columns, each row filled with
        // the values from 0 to size-1.
        float hostInput[][] = new float[numThreads][size];
        for(int i = 0; i < numThreads; i++)
        {
            for (int j=0; j<size; j++)
            {
                hostInput[i][j] = (float)j;
            }
        }

        // Allocate arrays on the device, one for each row. The pointers
        // to these array are stored in host memory.
        CUdeviceptr hostDevicePointers[] = new CUdeviceptr[numThreads];
        for(int i = 0; i < numThreads; i++)
        {
            hostDevicePointers[i] = new CUdeviceptr();
            cuMemAlloc(hostDevicePointers[i], size * Sizeof.FLOAT);
        }

        // Copy the contents of the rows from the host input data to
        // the device arrays that have just been allocated.
        for(int i = 0; i < numThreads; i++)
        {
            cuMemcpyHtoD(hostDevicePointers[i],
                Pointer.to(hostInput[i]), size * Sizeof.FLOAT);
        }

        // Allocate device memory for the array pointers, and copy
        // the array pointers from the host to the device.
        CUdeviceptr deviceInput = new CUdeviceptr();
        cuMemAlloc(deviceInput, numThreads * Sizeof.POINTER);
        cuMemcpyHtoD(deviceInput, Pointer.to(hostDevicePointers),
            numThreads * Sizeof.POINTER);

        // Allocate device output memory: A single column with
        // height 'numThreads'.
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, numThreads * Sizeof.FLOAT);

        // Set up the kernel parameters: A pointer to an array
        // of pointers which point to the actual values.
        Pointer kernelParams = Pointer.to(
            Pointer.to(deviceInput), 
            Pointer.to(new int[]{size}), 
            Pointer.to(deviceOutput)
        );
        
        // Call the kernel function.
        cuLaunchKernel(function, 
            1, 1, 1,           // Grid dimension 
            numThreads, 1, 1,  // Block dimension
            0, null,           // Shared memory size and stream 
            kernelParams, null // Kernel- and extra parameters
        ); 
        cuCtxSynchronize();

        // Allocate host output memory and copy the device output
        // to the host.
        float hostOutput[] = new float[numThreads];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
            numThreads * Sizeof.FLOAT);

        // Verify the result
        boolean passed = true;
        for(int i = 0; i < numThreads; i++)
        {
            float expected = 0;
            for(int j = 0; j < size; j++)
            {
                expected += hostInput[i][j];
            }
            if (Math.abs(hostOutput[i] - expected) > 1e-5)
            {
                passed = false;
                break;
            }
        }
        System.out.println("Test "+(passed?"PASSED":"FAILED"));

        // Clean up.
        for(int i = 0; i < numThreads; i++)
        {
            cuMemFree(hostDevicePointers[i]);
        }
        cuMemFree(deviceInput);
        cuMemFree(deviceOutput);
    }
}