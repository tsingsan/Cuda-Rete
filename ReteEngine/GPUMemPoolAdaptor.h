#pragma  once

#include "GPUMemPool.h"

__device__ CUDASTL::SimpleMemPool* GetDeviceTempPool();
__host__ void SetDeviceTempPool(CUDASTL::SimpleMemPool* pMemPool);
__host__ void ClearDeviceTempPool();

__device__ CUDASTL::SimpleMemPool* GetDeviceResultPool();
__host__ void SetDeviceResultPool(CUDASTL::SimpleMemPool* pMemPool);
__host__ void ClearDeviceResultPool();