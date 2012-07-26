#pragma  once

void InitGPUDebug();
void DestroyGPUDebug();

__device__ void gpuDebugCheckPoint_start(byte id);
__device__ void gpuDebugCheckPoint_end(byte id);

void StartGPUDebug(int threadnums);
void EndGPUDebug(int blockthreadnums);