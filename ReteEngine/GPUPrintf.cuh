#include "GPUPrintf.h"

#define  GPUPRINTPOOLSIZE 32 * 1024 * 1024

CUDASTL::SimpleMemPool g_hPrintPool;
CUDASTL::SimpleMemPool* g_dpPrintPool;

__device__ CUDASTL::SimpleMemPool* g_pDevicePrintPool = NULL;

#define GPPRINTSTR 's'
#define GPPRINTINT 'd'
#define GPPRINTUINT 'u'
#define GPPRINTFLOAT 'f'
#define GPPRINTDOUBLE 'b'
#define GPPRINTPOINTER 'p'

void InitGPUPrintf()
{
	CUDASTL_SAFECALL(cudaHostAlloc((void**)&g_hPrintPool.base_ptr, GPUPRINTPOOLSIZE, cudaHostAllocMapped));
	memset(g_hPrintPool.base_ptr, 0, GPUPRINTPOOLSIZE);

	CUDASTL::SimpleMemPool dTempPool;
	dTempPool.base_ptr = (char*)GetDevicePointer((void*)g_hPrintPool.base_ptr);
	dTempPool.bytes_used = 0;
	dTempPool.total_size = GPUPRINTPOOLSIZE;
	
	CUDASTL_SAFECALL(cudaMalloc((void**)&g_dpPrintPool, sizeof(CUDASTL::SimpleMemPool)));
	CUDASTL_SAFECALL(cudaMemcpy(g_dpPrintPool, &dTempPool, sizeof(CUDASTL::SimpleMemPool), cudaMemcpyHostToDevice));
	CUDASTL_SAFECALL(cudaMemcpyToSymbol(g_pDevicePrintPool, (const char*)&g_dpPrintPool, sizeof(CUDASTL::SimpleMemPool*)));
}

void DestroyGPUPrintf()
{
	if (g_dpPrintPool)
	{
		CUDASTL_SAFECALL(cudaFree(g_dpPrintPool));
		CUDASTL_SAFECALL(cudaFreeHost(g_hPrintPool.base_ptr));
	}
}

__global__ void resetprintfpool()
{
	if(g_pDevicePrintPool == NULL)
	{
		return;
	}

	g_pDevicePrintPool->bytes_used = 0;
}

bool DumpGPUPrintf()
{
	char* pCurHead = g_hPrintPool.base_ptr;
	while(*pCurHead)
	{
		switch(*pCurHead)
		{
		case GPPRINTSTR:
			{
				pCurHead += 1;
				printf("%s ",pCurHead);
				pCurHead += StringLength(pCurHead) + 1;
				break;
			}
		case GPPRINTINT:
			{
				pCurHead += 1;
				printf("%d ", *(int*)pCurHead);
				pCurHead += sizeof(int);
				break;
			}
		case GPPRINTUINT:
			{
				pCurHead += 1;
				printf("%u ", *(uint32_t*)pCurHead);
				pCurHead += sizeof(uint32_t);
				break;
			}
		case GPPRINTFLOAT:
			{
				pCurHead += 1;
				printf("%f ", *(float*)pCurHead);
				pCurHead += sizeof(float);
				break;
			}
		case GPPRINTDOUBLE:
			{
				pCurHead += 1;
				printf("%lf ", *(double*)pCurHead);
				pCurHead += sizeof(double);
				break;
			}
		case GPPRINTPOINTER:
			{
				pCurHead += 1;
				printf("%p ", (void*)pCurHead);
				pCurHead += sizeof(void*);
				break;
			}
		default:
			{
				printf("unknown prefix\n");
				return true;
			}
		}
	}

	if(pCurHead != g_hPrintPool.base_ptr)
	{
		printf("\n");
		return true;
	}
	return false;
}

void ResetGPUPrintf()
{
	memset(g_hPrintPool.base_ptr, 0, GPUPRINTPOOLSIZE);
	resetprintfpool<<<1,1>>>();
	CUDASTL_SAFECALL(cudaThreadSynchronize());
}

__device__ void gpuprintf(LPCTSTR value)
{
	if(g_pDevicePrintPool == NULL)
	{
		return;
	}
	
	int n = StringLength(value);
	if(n == 0)
	{
		return;
	}
	char* pAlloc = (char*)g_pDevicePrintPool->malloc((n + 2) * sizeof(TCHAR));
	pAlloc[0] = GPPRINTSTR;
	pAlloc += 1;
	memcpy((void*)pAlloc, value, n+1);

	__threadfence_system();
}

__device__ void gpuprintf(int value)
{
	if(g_pDevicePrintPool == NULL)
	{
		return;
	}
	char* pAlloc = (char*)g_pDevicePrintPool->malloc(1 + sizeof(int));
	pAlloc[0] = GPPRINTINT;
	pAlloc += 1;
	memcpy((void*)pAlloc, &value, sizeof(int));
}

__device__ void gpuprintf(uint32_t value)
{
	if(g_pDevicePrintPool == NULL)
	{
		return;
	}
	char* pAlloc = (char*)g_pDevicePrintPool->malloc(1 + sizeof(uint32_t));
	pAlloc[0] = GPPRINTUINT;
	pAlloc += 1;
	memcpy((void*)pAlloc, &value, sizeof(uint32_t));
}

__device__ void gpuprintf(float value)
{
	if(g_pDevicePrintPool == NULL)
	{
		return;
	}
	char* pAlloc = (char*)g_pDevicePrintPool->malloc(1 + sizeof(float));
	pAlloc[0] = GPPRINTFLOAT;
	pAlloc += 1;
	memcpy((void*)pAlloc, &value, sizeof(float));
}

__device__ void gpuprintf(double value)
{
	if(g_pDevicePrintPool == NULL)
	{
		return;
	}
	char* pAlloc = (char*)g_pDevicePrintPool->malloc(1 + sizeof(double));
	pAlloc[0] = GPPRINTDOUBLE;
	pAlloc += 1;
	memcpy((void*)pAlloc, &value, sizeof(double));
}

__device__ void gpuprintf(void* value)
{
	if(g_pDevicePrintPool == NULL)
	{
		return;
	}
	char* pAlloc = (char*)g_pDevicePrintPool->malloc(1 + sizeof(void*));
	pAlloc[0] = GPPRINTPOINTER;
	pAlloc += 1;
	memcpy((void*)pAlloc, &value, sizeof(void*));
}