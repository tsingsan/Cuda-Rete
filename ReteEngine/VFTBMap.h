#pragma once

#include "common.h"

class VFTBMap
{
public:
	VFTBMap(void);
	~VFTBMap(void);

	static void InitMap();

	__device__ __host__ static int* GetRelatedVFTB(int* pVF);

	__device__ __host__ static int GetVFTBCount();
};

template<class T>
__device__ __host__ T* VFCheck(T* pClass)
{
	int* pTmp = VFTBMap::GetRelatedVFTB((int*)*(int*)pClass);
	if(pTmp != NULL)
	{
		*(int**)pClass = pTmp;
	}
	return pClass;
}