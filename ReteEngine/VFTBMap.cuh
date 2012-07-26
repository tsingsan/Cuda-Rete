#include "VFTBMap.h"

#include "GlobalMemPool.h"

#include "Node.h"
#include "NodeURI.h"
#include "NodeVariable.h"

#include "ReteNode.h"
#include "ClauseFilter.h"
#include "ReteQueue.h"
#include "ReteTerminal.h"

#define CONST_VIRTUAL_CLASS_COUNTS 7

int** g_pVFTable = 0;
__device__ int** UVADevice(g_pVFTable) = 0;

unsigned int g_CPUVFMask = 0xFFFFFFFF;
__device__ unsigned int g_GPUVFMask = 0xFFFFFFFF;

__host__ void CPURegisterClassVFTB()
{
	int offset = -2;
	int** pReturn = g_pVFTable;

	CNode* pNode = new CNode("");
	CNodeVariable* pNodeVariable = new CNodeVariable("", 0);
	CNodeURI* pNodeURI = new CNodeURI("");

	CReteNode* pReteNode = new CReteNode;
	CClauseFilter* pClauseFilter = new CClauseFilter(NULL, NULL, 0, 0);
	CReteQueue* pReteQueue = new CReteQueue(NULL, 0);
	CReteTerminal* pReteTerminal = new CReteTerminal(NULL, 0);

	pReturn[offset += 2] = (int*)*(int*)pNode;
	g_CPUVFMask = g_CPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pNodeVariable;
	g_CPUVFMask = g_CPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pNodeURI;
	g_CPUVFMask = g_CPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pReteNode;
	g_GPUVFMask = g_GPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pClauseFilter;
	g_GPUVFMask = g_GPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pReteQueue;
	g_GPUVFMask = g_GPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pReteTerminal;
	g_GPUVFMask = g_GPUVFMask & (unsigned int)pReturn[offset];

	delete pNode;
	delete pNodeVariable;
	delete pNodeURI;

	delete pReteNode;
	delete pClauseFilter;
	delete pReteQueue;
	delete pReteTerminal;
}

__global__ void GPURegisterClassVFTB()
{
	int offset = -1;
	int** pReturn = UVADevice(g_pVFTable);

	CNode* pNode = new CNode;
	CNodeVariable* pNodeVariable = new CNodeVariable;
	CNodeURI* pNodeURI = new CNodeURI;

	CReteNode* pReteNode = new CReteNode;
	CClauseFilter* pClauseFilter = new CClauseFilter;
	CReteQueue* pReteQueue = new CReteQueue;
	CReteTerminal* pReteTerminal = new CReteTerminal;

	pReturn[offset += 2] = (int*)*(int*)pNode;
	g_GPUVFMask = g_GPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pNodeVariable;
	g_GPUVFMask = g_GPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pNodeURI;
	g_GPUVFMask = g_GPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pReteNode;
	g_GPUVFMask = g_GPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pClauseFilter;
	g_GPUVFMask = g_GPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pReteQueue;
	g_GPUVFMask = g_GPUVFMask & (unsigned int)pReturn[offset];

	pReturn[offset += 2] = (int*)*(int*)pReteTerminal;
	g_GPUVFMask = g_GPUVFMask & (unsigned int)pReturn[offset];


	delete pNode;
	delete pNodeVariable;
	delete pNodeURI;

	delete pReteNode;
	delete pClauseFilter;
	delete pReteQueue;
	delete pReteTerminal;
}

VFTBMap::VFTBMap(void)
{
}

VFTBMap::~VFTBMap(void)
{
}

void VFTBMap::InitMap()
{
	g_pVFTable = (int**)GlobalMemPool::GetVFTBPool().malloc(sizeof(int*) * 2 * CONST_VIRTUAL_CLASS_COUNTS);
	int** tmpDeviceTable = UVADevicePointer(int**, g_pVFTable);
	cudaMemcpyToSymbol(UVADevice(g_pVFTable), &tmpDeviceTable, sizeof(int**));

	CPURegisterClassVFTB();
	GPURegisterClassVFTB<<<1, 1>>>();

	if (cudaThreadSynchronize())
	{
		printf("Register Virtual Function Map Error");
	}
}

__device__ __host__ int* VFTBMap::GetRelatedVFTB(int* pVF)
{
#if defined(__CUDA_ARCH__)

// 	if(pVF & g_GPUVFMask == g_GPUVFMask)
// 	{
// 		return NULL;
// 	}
	for(int i=0; i < GetVFTBCount(); i++)
	{
		int* pCurPointer = UVADevice(g_pVFTable)[i*2];
		if(pCurPointer == pVF)
		{
			return UVADevice(g_pVFTable)[i*2 + 1];
		}
	}
	return NULL;

#else

// 	if(pVF & g_CPUVFMask == g_CPUVFMask)
// 	{
// 		return NULL;
// 	}
	for(int i=0; i < GetVFTBCount(); i++)
	{
		int* pCurPointer = g_pVFTable[i*2 + 1];
		if(pCurPointer == pVF)
		{
			return g_pVFTable[i*2];
		}
	}
	return NULL;

#endif
}

__device__ __host__ int VFTBMap::GetVFTBCount()
{
	return CONST_VIRTUAL_CLASS_COUNTS;
}