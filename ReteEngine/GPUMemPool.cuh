#ifndef VM_CUDA_GPUMEMPOOL_CUH
#define VM_CUDA_GPUMEMPOOL_CUH

#include "GPUMemPool.h"
#include "GPUPrintf.h"
#include <stdio.h>

namespace CUDASTL{

//	__device__ int MemMutex = 0;

	// General Method
	__device__ __host__ bool MemAlloc(void** ppMem, uint32_t size, uint32_t mode)
	{
#if !defined(__CUDA_ARCH__)
		switch(mode)
		{
		case GPUPool:
			return cudaMalloc(ppMem, size) == cudaSuccess;
		case HostPool:
			*ppMem = malloc(size);
			return *ppMem != NULL;
		case MapHostPool:
			return cudaHostAlloc(ppMem, size, cudaHostAllocMapped) == cudaSuccess;
		default:
			return false;
		}
#else
		if (mode != KernelPool)
		{
			return false;
		}
		*ppMem = malloc(size);
		return *ppMem != NULL;
#endif
	}

	__device__ __host__ void MemFree(void* pMem, uint32_t mode)
	{
#if !defined(__CUDA_ARCH__)
		switch(mode)
		{
		case GPUPool:
			cudaFree(pMem);
		case HostPool:
			free(pMem);
		case MapHostPool:
			cudaFreeHost(pMem);
		}
#else
		if (mode == KernelPool)
		{
			free(pMem);
		}
#endif
	}

	__device__ void * SimpleMemPool::malloc(uint32_t size)
	{
		uint32_t offset = atomicAdd((unsigned int*)&bytes_used, (unsigned int)size);
		if(bytes_used > total_size)
		{
			gpuprintf("Simple MemPool Overflow!");
		}
		return base_ptr + offset;
	}

	//Create a pool block
	__device__ __host__ bool MemPool::CreatePoolBlock(uint32_t minSize)
	{
		MemPoolBlock* newMPB = new MemPoolBlock;
		uint32_t allocSize = minSize > ublocksize ? minSize : ublocksize;
		if(MemAlloc((void**)&newMPB->base_ptr, allocSize, uMode))
		{
			newMPB->bytes_used = 0;
			newMPB->total_size = allocSize;
			newMPB->next_block = pMPBHead;
			pMPBHead = newMPB;
			return true;
		}
		else
		{
			delete newMPB;
			return false;
		}
	}

	//Get Memory From Memory Pool.If there is no enough space, alloc a new pool block.
	__device__ __host__ void * MemPool::malloc(uint32_t size, uint32_t align)
	{
		char* result = (char*)0xffffffff;

// #if defined(__CUDA_ARCH__)
// 		int bfinished = 0;
// 		while (!__all(bfinished))
// 		{
// 			if (!bfinished && !atomicCAS(&MemMutex,0,1))
// 			{
// #endif

		uint32_t alignment = align ? align : uAlignMent;
		uint32_t off = 0;
		if (pMPBHead && pMPBHead->bytes_used)
		{
			off = pMPBHead->bytes_used;
			off = (off + alignment - 1) & ~(alignment - 1);
		}

		if(pMPBHead == NULL || off + size > pMPBHead->total_size)
		{
			if(!bAutoAlloc || !CreatePoolBlock(size))
			{
#if !defined(__CUDA_ARCH__)
				printf("MemPool Alloc Error: %s\n",cudaGetErrorString(cudaGetLastError()));
#endif
				result = NULL;
			}
			else
			{
				off = pMPBHead->bytes_used;
			}
		}

		if (result != NULL)
		{
			result = pMPBHead->base_ptr + off;
			pMPBHead->bytes_used = off + size;
		}

// #if defined(__CUDA_ARCH__)
// 		MemMutex = 0;
// 		bfinished = 1;
// 			}
// 		}
// #endif
		return result;
	}

	__device__ __host__ bool MemPool::attach(void* ptr, uint32_t size)
	{
		if (pMPBHead == NULL && ptr && size)
		{
			MemPoolBlock* newMPB = new MemPoolBlock;
			newMPB->base_ptr = (char*)ptr;
			newMPB->bytes_used = 0;
			newMPB->total_size = size;
			newMPB->next_block = NULL;
			pMPBHead = newMPB;
			
			bAutoAlloc = false;
			return true;
		}
		return false;
	}

	__device__ __host__ MemPool::MemPool()
	:pMPBHead(NULL),
	ublocksize(POOLBLOCKDEFAULTSIZE),
	uAlignMent(__alignof(float)),
	bAutoAlloc(true)
	{
#if !defined(__CUDA_ARCH__)	
		uMode = GPUPool;
#else
		uMode = KernelPool;
#endif
	}

	//Free memory
	__device__ __host__ MemPool::~MemPool()
	{
		clear();
	}

	__device__ __host__ void MemPool::clear()
	{
		MemPoolBlock* pBlock = pMPBHead;
		while(pBlock)
		{
			MemPoolBlock* curBlock = pBlock;		
			pBlock = curBlock->next_block;
			MemFree(curBlock->base_ptr, uMode);
			delete curBlock;
		}
		pMPBHead = NULL;
	}
}

#endif //VM_CUDA_GPUMEMPOOL_CUH