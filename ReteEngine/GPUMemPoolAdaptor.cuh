__device__ CUDASTL::SimpleMemPool* g_pDeviceTempPool = 0;
__device__ CUDASTL::SimpleMemPool* g_pDeviceResultPool = 0;


__device__ CUDASTL::SimpleMemPool* GetDeviceTempPool()
{
	return g_pDeviceTempPool;
}

__host__ void SetDeviceTempPool(CUDASTL::SimpleMemPool* pMemPool)
{
	CUDASTL_SAFECALL(cudaMemcpyToSymbol(g_pDeviceTempPool, (const char*)&pMemPool, sizeof(CUDASTL::SimpleMemPool*)));
}

__global__ void GPUClearDeviceTempPool()
{
	if(g_pDeviceTempPool)
		g_pDeviceTempPool->bytes_used = 0;
}

__host__ void ClearDeviceTempPool()
{
	GPUClearDeviceTempPool<<<1,1>>>();
	CUDASTL_SAFECALL(cudaThreadSynchronize());
}

__device__ CUDASTL::SimpleMemPool* GetDeviceResultPool()
{
	return g_pDeviceResultPool;
}

__host__ void SetDeviceResultPool(CUDASTL::SimpleMemPool* pMemPool)
{
	CUDASTL_SAFECALL(cudaMemcpyToSymbol(g_pDeviceResultPool, (const char*)&pMemPool, sizeof(CUDASTL::SimpleMemPool*)));
}

__global__ void GPUClearDeviceResultPool()
{
	if(g_pDeviceResultPool)
		g_pDeviceResultPool->bytes_used = 0;
}

__host__ void ClearDeviceResultPool()
{
	GPUClearDeviceResultPool<<<1,1>>>();
	CUDASTL_SAFECALL(cudaThreadSynchronize());
}