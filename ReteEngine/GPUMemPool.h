#ifndef VM_CUDA_GPUMEMPOOL_H
#define VM_CUDA_GPUMEMPOOL_H

#include <cuda_runtime.h>
#define POOLBLOCKDEFAULTSIZE 512 * 1024

enum
{
//Host Support
	GPUPool,	
	HostPool,
	MapHostPool,
//Kernel Support
	KernelPool
};

typedef unsigned int uint32_t;

namespace CUDASTL{

	__device__ __host__ bool MemAlloc(void** ppMem, uint32_t size);
	__device__ __host__ void MemFree(void* pMem);

	class SimpleMemPool{
	public:
		SimpleMemPool():base_ptr(NULL),total_size(0),bytes_used(0){}
		__device__ void * malloc(uint32_t size);
	public:
		char * base_ptr;
		uint32_t total_size;
		uint32_t bytes_used;
	};

	class MemPoolBlock{
	public:
		char * base_ptr;
		uint32_t total_size;
		uint32_t bytes_used;
		MemPoolBlock* next_block;
	};

	class MemPool{
	private:
		__device__ __host__ bool CreatePoolBlock(uint32_t minSize);
	public:		
		__device__ __host__ MemPool();
		__device__ __host__ ~MemPool();
		__device__ __host__ void clear();
		__device__ __host__ void* malloc(uint32_t size, uint32_t align = 0);
		__device__ __host__ bool attach(void* ptr, uint32_t size);
		__device__ __host__ void setdefaultalignment(uint32_t alignment){uAlignMent = alignment;}
		__device__ __host__ void setdefaultsize(uint32_t size){ublocksize = size;}
		__device__ __host__ bool setmode(uint32_t mode)
		{
			if (pMPBHead == NULL)
			{
				uMode = mode;
				return true;
			}
			return false;
		}
		__device__ __host__ uint32_t getallocsize()
		{
			uint32_t uSize = 0;
			MemPoolBlock* curBlock = pMPBHead;
			while (curBlock)
			{
				uSize += curBlock->bytes_used;
				curBlock = curBlock->next_block;
			}
			return uSize;
		}
	private:
		MemPoolBlock* pMPBHead;	//PoolBlock Header
		uint32_t ublocksize;	//Default Block Size
		uint32_t uMode;	//Pool Mode
		uint32_t uAlignMent; //Default Alignment Requirement;
		bool bAutoAlloc;
	};
}

#endif //VM_CUDA_GPUMEMPOOL_H
