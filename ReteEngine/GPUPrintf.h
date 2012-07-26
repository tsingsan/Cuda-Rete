#pragma  once

void InitGPUPrintf();
void DestroyGPUPrintf();

bool DumpGPUPrintf();
void ResetGPUPrintf();

__device__ void gpuprintf(LPCTSTR value);
__device__ void gpuprintf(int value);
__device__ void gpuprintf(uint32_t value);
__device__ void gpuprintf(float value);
__device__ void gpuprintf(double value);
__device__ void gpuprintf(void* value);