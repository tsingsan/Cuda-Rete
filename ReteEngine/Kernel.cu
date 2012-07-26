#include "common.h"

#include "GlobalMemPool.h"

#include "common.cuh"
#include "GPUMemPool.cuh"
#include "VFTBMap.cuh"
#include "GPUPrintf.cuh"
#include "GPUDebug.cuh"
#include "GPUMemPoolAdaptor.cuh"

#include "ReteRuntime.h"

#include "ClauseFilter.h"
#include "ReteQueue.h"
#include "ReteTerminal.h"

__global__ void Kernel(FireInfo* pVecFireInfo, size_t nCountFireInfo)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < nCountFireInfo)
	{
		FireInfo& myFireInfo = pVecFireInfo[idx];
		CTriplePattern* pMyTriple = myFireInfo.pTriple;
		CClauseFilter* pMyFilter = myFireInfo.pClauseFilter;

		VFCheck(pMyFilter)->Start(pMyTriple);
	}
}

__global__ void ClearQueue(CReteQueue* pReteQueue)
{
	pReteQueue->ClearQueue();
}

void ReteQueueClear(CReteQueue* pReteQueue)
{
	ClearQueue<<<1,1>>>(pReteQueue);
	if(cudaThreadSynchronize())
	{
		printf("KERNEL FAILED: %s\n",cudaGetErrorString(cudaGetLastError()));
	}
}

void KernelCall(FireInfo* pVecFireInfo, size_t nCountFireInfo)
{
	const size_t numThreads = 64;
	const size_t numBlocks = (nCountFireInfo + numThreads - 1) / numThreads;

	StartGPUDebug(numThreads * numBlocks);
	Kernel<<<numBlocks, numThreads>>>(pVecFireInfo, nCountFireInfo);
	if(cudaThreadSynchronize())
	{
		printf("KERNEL FAILED: %s\n",cudaGetErrorString(cudaGetLastError()));
	}

	EndGPUDebug(numThreads);
	if(DumpGPUPrintf())
	{
		ResetGPUPrintf();
	}
}
