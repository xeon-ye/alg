#include <math.h>

extern "C"
__global__ void initialize(float* r1, float* r2, float* loc, float* vel, float* minLoc, float* maxLoc, float* minVel, float* maxVel, int numElements, int n)
{ 
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements)
    {
        int threadId = tid % n;
        loc[tid] = minLoc[threadId] + r1[tid] * (maxLoc[threadId] - minLoc[threadId]);
        vel[tid] = minVel[threadId] + r2[tid] * (maxVel[threadId] - minVel[threadId]);
        // loc[tid] = maxLoc[threadId];    
    }
}

extern "C"
__global__ void warmInitialize(float* r2, float* vel, float* minVel, float* maxVel, int numElements, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements)
    {
        int threadId = tid % n;
        vel[tid] = minVel[threadId] + r2[tid] * (maxVel[threadId] - minVel[threadId]);
    }
}

extern "C"
__global__ void update(float w, float* r1, float* r2, float* loc, float* pBest, float* best, float* vel, float* minVel, float* maxVel, float* minLoc, float* maxLoc, int numElements, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements)
    {
        int threadId = tid % n;

        float tempVel = (w * vel[tid]) + (r1[tid] * 1.4961) * (pBest[tid] - loc[tid]) + (r2[tid] * 1.4961) * (best[threadId] - loc[tid]);
        if (tempVel < minVel[threadId]){
            tempVel = minVel[threadId];
        } else if (tempVel > maxVel[threadId]){
            tempVel = maxVel[threadId];
        }
        vel[tid] = tempVel;
        
        float tempLoc = loc[tid] + tempVel;
        if (tempLoc < minLoc[threadId]){
            tempLoc = minLoc[threadId];
        } else if (tempLoc > maxLoc[threadId]){
            tempLoc = maxLoc[threadId];
        }
        loc[tid] = tempLoc;
    }
}

extern "C"
__global__ void calBusV(float* loc, int* pos, float* z, int* threshold, double* obj, int swarmSize, int n, int posSize, int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // 一个线程计算一个bus
    int threadId = threadIdx.x;

    // 先清空共享内存
    extern __shared__ float obj_shared[];
    if (threadId < swarmSize)
    {
        obj_shared[threadId] = 0;
    }
    __syncthreads;

    if (tid < posSize)
    {
        int num = pos[tid] - 1;
        float z_value = z[offset + tid];
        float threshold_value = threshold[offset + tid];
        for (int i = 0; i < swarmSize; i++) 
        {            
            float result = loc[num + i * n];
            float d = (result - z_value) / threshold_value;
            if (fabsf(d) > 1)
            {
                atomicAdd(&obj_shared[i], 1); // 可能会造成较大的冲突问题
            }            
        }
    }

    // 再把共享内存的数据写入全局内存
    __syncthreads;
    if (threadId < swarmSize)
    {
        atomicAdd(&obj[threadId], obj_shared[threadId]);
    }
}
