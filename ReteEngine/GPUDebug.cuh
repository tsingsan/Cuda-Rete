#include "GPUDebug.h"

#define  GPUDEBUGPOOLSIZE 1024 * 1024

int g_nDebugThreadsNum = 0;

CUDASTL::SimpleMemPool g_hDebugPool;
CUDASTL::SimpleMemPool* g_dpDebugPool;

__device__ CUDASTL::SimpleMemPool* g_pDeviceDebugPool = NULL;

#if defined(__ENABLE_GPU_DEBUG__)

void InitGPUDebug()
{
	CUDASTL_SAFECALL(cudaHostAlloc((void**)&g_hDebugPool.base_ptr, GPUDEBUGPOOLSIZE, cudaHostAllocMapped));
	memset(g_hDebugPool.base_ptr, 0, GPUDEBUGPOOLSIZE);

	CUDASTL::SimpleMemPool dTempPool;
	dTempPool.base_ptr = (char*)GetDevicePointer((void*)g_hDebugPool.base_ptr);
	dTempPool.bytes_used = 0;
	dTempPool.total_size = GPUDEBUGPOOLSIZE;
	
	CUDASTL_SAFECALL(cudaMalloc((void**)&g_dpDebugPool, sizeof(CUDASTL::SimpleMemPool)));
	CUDASTL_SAFECALL(cudaMemcpy(g_dpDebugPool, &dTempPool, sizeof(CUDASTL::SimpleMemPool), cudaMemcpyHostToDevice));
	CUDASTL_SAFECALL(cudaMemcpyToSymbol(g_pDeviceDebugPool, (const char*)&g_dpDebugPool, sizeof(CUDASTL::SimpleMemPool*)));
}

void DestroyGPUDebug()
{
	if (g_dpDebugPool)
	{
		CUDASTL_SAFECALL(cudaFree(g_dpDebugPool));
		CUDASTL_SAFECALL(cudaFreeHost(g_hDebugPool.base_ptr));
	}
}

void StartGPUDebug(int threadnums)
{
	if (threadnums > GPUDEBUGPOOLSIZE)
	{
		printf("Threads number bigger than Debug Pool\n");
	}

	g_nDebugThreadsNum = threadnums;
	memset(g_hDebugPool.base_ptr, 0, GPUDEBUGPOOLSIZE);
}

void EndGPUDebug(int blockthreadnums)
{
	char* base_ptr = g_hDebugPool.base_ptr;
	for(int i = 0; i < g_nDebugThreadsNum; i++)
	{
		if(base_ptr[i] != 0)
		{
			printf("{%d , %d}:%d\t", i / blockthreadnums, i % blockthreadnums, base_ptr[i]);
		}
	}
}

__device__ void gpuDebugCheckPoint_start(byte id)
{
	*(g_pDeviceDebugPool->base_ptr + threadIdx.x + blockDim.x * blockIdx.x) += id;
	__threadfence_system();
}

__device__ void gpuDebugCheckPoint_end(byte id)
{
	*(g_pDeviceDebugPool->base_ptr + threadIdx.x + blockDim.x * blockIdx.x) -= id;
	__threadfence_system();
}

#else

void InitGPUDebug()
{
}
void DestroyGPUDebug()
{}
void StartGPUDebug(int threadnums)
{
}
void EndGPUDebug(int blockthreadnums)
{
}
__device__ void gpuDebugCheckPoint_start(byte id)
{
}
__device__ void gpuDebugCheckPoint_end(byte id)
{
}
#endif