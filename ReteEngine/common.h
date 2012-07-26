#ifndef __RULEREASON_COMMON_H__
#define __RULEREASON_COMMON_H__

#ifdef _WIN32
#ifdef RULEREASON_EXPORTS
#define EXPORTDLL __declspec(dllexport)
#else
#define EXPORTDLL __declspec(dllimport)
#endif
#else
#define EXPORTDLL
#endif

#ifndef MAX_PATH
#define MAX_PATH 260
#endif

#ifndef NULL
#ifdef __cplusplus
#define NULL    0
#else
#define NULL    ((void *)0)
#endif
#endif

typedef char TCHAR;
typedef const TCHAR* LPCTSTR;

typedef unsigned char byte;

#include <cuda_runtime_api.h>
#include <stdio.h>
#include "GPUMemPool.h"

void KernelSafeCall( cudaError err, const char *file, const int line );
void* GetDevicePointer(void* pHost);

__device__ __host__ bool StringEqual(LPCTSTR p, LPCTSTR c);
__device__ __host__ int StringLength(LPCTSTR p);

#ifdef __SUPPORT_UVA__

#define UVADevice(y) \
	y

#define UVAMember(x,y) \
	x y;

#define UVADevicePointer(x,z) \
	z

#define UVAInit(x,y,z) \
	y(z)

#define UVAAssign(x,y,z) \
	y = z

#define UVAMemberAssign(x,y,z,v) \
	v->y = z

#else

#define UVADevice(y) \
	d##y

#define UVAMember(x,y) \
	x y;\
	x d##y

#define UVADevicePointer(x,z) \
	z?(x)GetDevicePointer((void*)z):z

#define UVAInit(x,y,z) \
	y(z),\
	UVADevice(y)(UVADevicePointer(x,z))

#define UVAAssign(x,y,z) \
	y = z; \
	UVADevice(y) = UVADevicePointer(x,z)

#define UVAMemberAssign(x,y,z,v) \
	v->y = z; \
	v->UVADevice(y) = UVADevicePointer(x,z)

#endif //__SUPPORT_UVA__

#define DEFAULT_DEVICE_STRUCTION_FUNCTION(x) __device__ x(){}

#define CUDASTL_SAFECALL(err) KernelSafeCall(err, __FILE__, __LINE__)

//GPU使用内存池
#define __ENABLE_DEVICE_POOL__

#define __SUPPORT_UVA__

//启用GPU DEBUG
//#define __ENABLE_GPU_DEBUG__

#endif //__RULEREASON_COMMON_H__
