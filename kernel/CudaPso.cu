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
__global__ void calBusV(float* loc, int* pos, float* z, float* threshold, float* obj, int swarmSize, int n, int posSize, int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    int threadId = threadIdx.x;

    // 先清空全局内存，calBusV是第一步
    if (tid < swarmSize)
    {
        obj[tid] = 0;
    }

    // 清空共享内存
    // extern __shared__ float obj_shared[];
    // if (threadId < swarmSize)
    // {
    //     obj_shared[threadId] = 0;
    // }
    // __syncthreads;

    // if (tid < posSize)
    // {
    //     int num = pos[tid] - 1;
    //     float z_value = z[offset + tid];
    //     float threshold_value = threshold[offset + tid];
    //     for (int i = 0; i < swarmSize; i++) 
    //     {
    //         float result = loc[num + i * n];
    //         float d = (result - z_value) / threshold_value;
    //         if (fabsf(d) > 1)
    //         {
    //             atomicAdd(&obj_shared[i], 1); // 可能会造成较大的冲突问题
    //         }            
    //     }
    // }

    if (tid < posSize * swarmSize)
    {
        int swarmIndex = tid / posSize;
        int posIndex = tid % posSize;
        int num = pos[posIndex] - 1;        
        float z_value = z[offset + posIndex];
        float threshold_value = threshold[offset + posIndex];
        float result = loc[num + swarmIndex * n];
        float d = (result - z_value) / threshold_value;
        if (fabsf(d) > 1)
        {
            atomicAdd(&obj[swarmIndex], 1); // 不使用共享内存
        }            
    }
    // __syncthreads;
    
    // 再把共享内存的数据写入全局内存
    
    // if (threadId < swarmSize)
    // {
    //     atomicAdd(&obj[threadId], obj_shared[threadId]);
    // }    
}

extern "C"
__global__ void calBusP(float* loc, int* pos, float* z, float* threshold, float* obj, float* g, float* b, int* ja, int* ia, int* link, int swarmSize, int m, int n, int posSize, int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    int threadId = threadIdx.x;

    // // 清空共享内存
    // extern __shared__ float obj_shared[];
    // if (threadId < swarmSize)
    // {
    //     obj_shared[threadId] = 0;
    // }
    // __syncthreads;

    if (tid < posSize * swarmSize)
    {
        int swarmIndex = tid / posSize;
        int posIndex = tid % posSize;

        int num = pos[posIndex] - 1;        
        float z_value = z[offset + posIndex];
        float threshold_value = threshold[offset + posIndex];

        float vi = loc[num + swarmIndex * n];
        float thetaI = loc[num + m + swarmIndex * n];
        float result = 0;

        int k = ia[num];
        while (k != -1) {
            int j = ja[k]; // 列号
            float thetaIJ = thetaI - loc[j + m + swarmIndex * n];
            float gij = g[k];
            float bij = b[k];
            result += (vi * loc[j + swarmIndex * n] * (gij * cosf(thetaIJ) + bij * sinf(thetaIJ)));
            k = link[k];
        }

        float d = (result - z_value) / threshold_value;
        if (fabsf(d) > 1)
        {
            atomicAdd(&obj[swarmIndex], 1); // 可能会造成较大的冲突问题
        }            
    }

    // 再把共享内存的数据写入全局内存
    // __syncthreads;
    // if (threadId < swarmSize)
    // {
    //     atomicAdd(&obj[threadId], obj_shared[threadId]);
    // }
}

extern "C"
__global__ void calBusQ(float* loc, int* pos, float* z, float* threshold, float* obj, float* g, float* b, int* ja, int* ia, int* link, int swarmSize, int m, int n, int posSize, int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    if (tid < posSize * swarmSize)
    {
        int swarmIndex = tid / posSize;
        int posIndex = tid % posSize;

        int num = pos[posIndex] - 1;        
        float z_value = z[offset + posIndex];
        float threshold_value = threshold[offset + posIndex];

        float vi = loc[num + swarmIndex * n];
        float thetaI = loc[num + m + swarmIndex * n];
        float result = 0;

        int k = ia[num];
        while (k != -1) {
            int j = ja[k]; // 列号
            float thetaIJ = thetaI - loc[j + m + swarmIndex * n];
            float gij = g[k];
            float bij = b[k];
            result += (vi * loc[j + swarmIndex * n] * (gij * sinf(thetaIJ) - bij * cosf(thetaIJ)));
            k = link[k];
        }

        float d = (result - z_value) / threshold_value;
        if (fabsf(d) > 1)
        {
            atomicAdd(&obj[swarmIndex], 1); // 可能会造成较大的冲突问题
        }            
    }
}

extern "C"
__global__ void calLinePFrom(float* loc, float* z, float* threshold, float* obj, int* iBus, int* jBus, float* g, float* b, float* g1, float* b1, int swarmSize, int m, int n, int posSize, int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    if (tid < posSize * swarmSize)
    {
        int swarmIndex = tid / posSize;
        int posIndex = tid % posSize;

        float z_value = z[offset + posIndex];
        float threshold_value = threshold[offset + posIndex];

        int i = iBus[posIndex] - 1;
        int j = jBus[posIndex] - 1;
        float gij = g[posIndex];
        float bij = b[posIndex];
        float gc = g1[posIndex];
        float bc = b1[posIndex];
        float vi = loc[i + swarmIndex * n];
        float vj = loc[j + swarmIndex * n];
        float thetaIJ = loc[i + m + swarmIndex * n] - loc[j + m + swarmIndex * n];
        float result = vi * vi * (gij + gc) - vi * vj * (gij * cosf(thetaIJ) + bij * sinf(thetaIJ));

        float d = (result - z_value) / threshold_value;
        if (fabsf(d) > 1)
        {
            atomicAdd(&obj[swarmIndex], 1); // 可能会造成较大的冲突问题
        }            
    }
}

extern "C"
__global__ void calLineQFrom(float* loc, float* z, float* threshold, float* obj, int* iBus, int* jBus, float* g, float* b, float* g1, float* b1, int swarmSize, int m, int n, int posSize, int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    if (tid < posSize * swarmSize)
    {
        int swarmIndex = tid / posSize;
        int posIndex = tid % posSize;

        float z_value = z[offset + posIndex];
        float threshold_value = threshold[offset + posIndex];

        int i = iBus[posIndex] - 1;
        int j = jBus[posIndex] - 1;
        float gij = g[posIndex];
        float bij = b[posIndex];
        float gc = g1[posIndex];
        float bc = b1[posIndex];
        float vi = loc[i + swarmIndex * n];
        float vj = loc[j + swarmIndex * n];
        float thetaIJ = loc[i + m + swarmIndex * n] - loc[j + m + swarmIndex * n];
        float result = -vi * vi * (bij + bc) - vi * vj * (gij * sinf(thetaIJ) - bij * cosf(thetaIJ));
        
        float d = (result - z_value) / threshold_value;
        if (fabsf(d) > 1)
        {
            atomicAdd(&obj[swarmIndex], 1); // 可能会造成较大的冲突问题
        }            
    }
}


extern "C"
__global__ void calLinePTo(float* loc, float* z, float* threshold, float* obj, int* iBus, int* jBus, float* g, float* b, float* g1, float* b1, int swarmSize, int m, int n, int posSize, int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    if (tid < posSize * swarmSize)
    {
        int swarmIndex = tid / posSize;
        int posIndex = tid % posSize;

        float z_value = z[offset + posIndex];
        float threshold_value = threshold[offset + posIndex];

        int i = iBus[posIndex] - 1;
        int j = jBus[posIndex] - 1;
        float gij = g[posIndex];
        float bij = b[posIndex];
        float gc = g1[posIndex];
        float bc = b1[posIndex];
        float vi = loc[i + swarmIndex * n];
        float vj = loc[j + swarmIndex * n];
        float thetaJI = loc[j + m + swarmIndex * n] - loc[i + m + swarmIndex * n];
        float result = vj * vj * (gij + gc) - vi * vj * (gij * cosf(thetaJI) + bij * sinf(thetaJI));

        float d = (result - z_value) / threshold_value;
        if (fabsf(d) > 1)
        {
            atomicAdd(&obj[swarmIndex], 1); // 可能会造成较大的冲突问题
        }            
    }
}

extern "C"
__global__ void calLineQTo(float* loc, float* z, float* threshold, float* obj, int* iBus, int* jBus, float* g, float* b, float* g1, float* b1, int swarmSize, int m, int n, int posSize, int offset)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    if (tid < posSize * swarmSize)
    {
        int swarmIndex = tid / posSize;
        int posIndex = tid % posSize;

        float z_value = z[offset + posIndex];
        float threshold_value = threshold[offset + posIndex];

        int i = iBus[posIndex] - 1;
        int j = jBus[posIndex] - 1;
        float gij = g[posIndex];
        float bij = b[posIndex];
        float gc = g1[posIndex];
        float bc = b1[posIndex];
        float vi = loc[i + swarmIndex * n];
        float vj = loc[j + swarmIndex * n];
        float thetaJI = loc[j + m + swarmIndex * n] - loc[i + m + swarmIndex * n];
        float result = -vj * vj * (bij + bc) - vi * vj * (gij * sinf(thetaJI) - bij * cosf(thetaJI));
        
        float d = (result - z_value) / threshold_value;
        if (fabsf(d) > 1)
        {
            atomicAdd(&obj[swarmIndex], 1); // 可能会造成较大的冲突问题
        }            
    }
}


extern "C"
__global__ void calZeroBusP(float* loc, int* pos, float* g, float* b, int* ja, int* ia, int* link, float* violation, int swarmSize, int m, int n, int posSize, float tol_p)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    // 先清空violation
    if (tid < swarmSize)
    {
        violation[tid] = 0;
    }
    __syncthreads;

    if (tid < posSize * swarmSize)
    {
        int swarmIndex = tid / posSize;
        int posIndex = tid % posSize;

        int num = pos[posIndex] - 1;

        float vi = loc[num + swarmIndex * n];
        float thetaI = loc[num + m + swarmIndex * n];
        float result = 0;

        int k = ia[num];
        while (k != -1) {
            int j = ja[k]; // 列号
            float thetaIJ = thetaI - loc[j + m + swarmIndex * n];
            float gij = g[k];
            float bij = b[k];
            result += (vi * loc[j + swarmIndex * n] * (gij * cosf(thetaIJ) + bij * sinf(thetaIJ)));
            k = link[k];
        }

        float d = fabsf(result) - tol_p;
        if (d > 0)
        {
            atomicAdd(&violation[swarmIndex], d); // 可能会造成较大的冲突问题
        }
    }
}


extern "C"
__global__ void calZeroBusQ(float* loc, int* pos, float* obj, float* g, float* b, int* ja, int* ia, int* link, float* violation, int* isGBestfeasible, int swarmSize, int m, int n, int posSize, float tol_q)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 

    if (tid < posSize * swarmSize)
    {
        int swarmIndex = tid / posSize;
        int posIndex = tid % posSize;
        
        int num = pos[posIndex] - 1;        

        float vi = loc[num + swarmIndex * n];
        float thetaI = loc[num + m + swarmIndex * n];
        float result = 0;

        int k = ia[num];
        while (k != -1) {
            int j = ja[k]; // 列号
            float thetaIJ = thetaI - loc[j + m + swarmIndex * n];
            float gij = g[k];
            float bij = b[k];
            result += (vi * loc[j + swarmIndex * n] * (gij * sinf(thetaIJ) - bij * cosf(thetaIJ)));
            k = link[k];
        }

        float d = fabsf(result) - tol_q;
        if (d > 0)
        {            
            atomicAdd(&violation[swarmIndex], d); // 可能会造成较大的冲突问题
        }
    }
    __syncthreads;

    if (tid < swarmSize)
    {
        if (violation[tid] > 0)
        {
            atomicAdd(&obj[tid], violation[tid] + 1e6);
        } else
        {
            *isGBestfeasible = 1;
        }
    }
}

extern "C"
__global__ void findThisBest(float* obj, float* bestObj, int swarmSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0)
    {
        float obj_value = *bestObj;
        for (int i = 0; i < swarmSize; i++) {
            obj_value = fminf(obj[i], obj_value);
        }
        *bestObj = obj_value;
    }
}

extern "C"
__global__ void findBest(float* loc, float* pBest, float* pBestLoc, float* bestLoc, float* obj, float* bestObj, int swarmSize, int n, int numElements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // __shared__ float min_val;
    // __shared__ int min_tid;

    // if (tid == 0)
    // {
    //     min_val = 1e6;
    //     min_tid = 1e6;       
    // }
    // __syncthreads;

    // int data;
    // if (tid < swarmSize)
    // {
    //     data = obj[tid];
    //     atomicMin(&min_val, data);
    // }
    // __syncthreads;

    // if (tid < swarmSize)
    // {
    //     if (min_val == data)
    //     {
    //         atomicMin(&min_tid, tid);
    //     }
    // }
    // __syncthreads;

    if (tid < numElements)
    {
        int swarmIndex = tid / n;
        int index = tid % n;
        if (obj[swarmIndex] == *bestObj)
        {
            bestLoc[index] = loc[tid];
        }
        if (obj[swarmIndex] < pBest[swarmIndex])
        {
            pBest[swarmIndex] = obj[swarmIndex];
            pBestLoc[tid] = loc[tid];
        }
    }
}