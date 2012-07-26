#include "common.h"
#include <stdlib.h>

void* GetDevicePointer(void* pHost)
{
	void* pDevice = NULL;
	if(!cudaHostGetDevicePointer(&pDevice,pHost,0))
	{
		return pDevice;
	}
	return NULL;
};

void KernelSafeCall( cudaError err, const char *file, const int line )
{
	if( cudaSuccess != err) {
		printf("%s(%i) : Kernel Runtime API error : %s.\n", file, line, cudaGetErrorString( err));
		exit(-1);
	}
}