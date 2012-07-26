#include "ReteRuntime.h"
#include "common.h"

#include "GlobalMemPool.h"
#include "GPUMemPoolAdaptor.h"
#include "NodeFactory.h"
#include "TripleContainer.h"
#include "TriplePattern.h"

#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <iostream>
using namespace std;

CUDASTL::SimpleMemPool CReteRuntime::m_hTempPool;
CUDASTL::SimpleMemPool* CReteRuntime::m_dpTempPool;

CUDASTL::SimpleMemPool CReteRuntime::m_hResultPool;
CUDASTL::SimpleMemPool* CReteRuntime::m_dpResultPool;

#define  TEMPPOOLSIZE 256 * 1024 * 1024
#define  RESULTPOOLSIZE 10 * 1024 * 1024

CReteRuntime::CReteRuntime(void)
{
	if (m_hTempPool.base_ptr == NULL && m_hResultPool.base_ptr == NULL)
	{
		CreateDeviceMemPool();	
	}
	else if(m_hTempPool.base_ptr != NULL && m_hResultPool.base_ptr != NULL)
	{
		return;
	}
	else
	{
		cout <<"Runtime Error: Result and Temp Pool are not both Uncreated or Created." << endl;
		exit(-1);
	}
}

CReteRuntime::~CReteRuntime(void)
{
}

extern void KernelCall(FireInfo* pVecFireInfo, size_t nCountFireInfo);

void CReteRuntime::go(FireInfo* pVecFireInfo, size_t nCountFireInfo, CTripleContainer* pResultContainer)
{
	FireInfo* pFire;
	pFire = (FireInfo*)GlobalMemPool::GetHostTempPoolRef()->malloc(sizeof(FireInfo) * nCountFireInfo);
	assert(pFire);
	memcpy(pFire, pVecFireInfo, sizeof(FireInfo) * nCountFireInfo);
	pFire = UVADevicePointer(FireInfo*, pFire);
	//CUDASTL_SAFECALL(cudaMalloc((void**)&pFire, sizeof(FireInfo) * nCountFireInfo));
	//CUDASTL_SAFECALL(cudaMemcpy(pFire, pVecFireInfo, sizeof(FireInfo) * nCountFireInfo, cudaMemcpyHostToDevice));

	KernelCall(pFire, nCountFireInfo);

	//CUDASTL_SAFECALL(cudaFree(pFire));

	UpdateHostPoolInfo();

	uint32_t uNewTripleCounts = m_hResultPool.bytes_used / sizeof(LPCTSTR) / 3 ;
	LPCTSTR* pResultArray = (LPCTSTR*)m_hResultPool.base_ptr;
	for (uint32_t i = 0; i < uNewTripleCounts; i ++ )
	{
		LPCTSTR sub = pResultArray[ i * 3];
		LPCTSTR pre = pResultArray[ i * 3 + 1];
		LPCTSTR obj = pResultArray[ i * 3 + 2];
		pResultContainer->ParseTriple(sub, pre, obj);
	}

	ClearDeviceResultPool();
}

void CReteRuntime::UpdateHostPoolInfo()
{
	if (m_dpTempPool)
	{
		CUDASTL::SimpleMemPool hTempPool;
		CUDASTL_SAFECALL(cudaMemcpy(&hTempPool, m_dpTempPool, sizeof(CUDASTL::SimpleMemPool), cudaMemcpyDeviceToHost));
		m_hTempPool.bytes_used = hTempPool.bytes_used;
	}
	if (m_dpResultPool)
	{
		CUDASTL::SimpleMemPool hTempPool;
		CUDASTL_SAFECALL(cudaMemcpy(&hTempPool, m_dpResultPool, sizeof(CUDASTL::SimpleMemPool), cudaMemcpyDeviceToHost));
		m_hResultPool.bytes_used = hTempPool.bytes_used;
	}
}

void CReteRuntime::UpdateDevicePoolInfo(CUDASTL::SimpleMemPool** ppDstPool, CUDASTL::SimpleMemPool& dSrcPool)
{
	if (*ppDstPool == NULL)
	{
		CUDASTL_SAFECALL(cudaMalloc((void**)ppDstPool, sizeof(CUDASTL::SimpleMemPool)));
	}
	
	CUDASTL_SAFECALL(cudaMemcpy(*ppDstPool, &dSrcPool, sizeof(CUDASTL::SimpleMemPool), cudaMemcpyHostToDevice));
}

void CReteRuntime::ClearDeviceRuntime()
{
	if (m_hTempPool.base_ptr != NULL)
	{
		ClearDeviceTempPool();
		memset(m_hTempPool.base_ptr, 0, m_hTempPool.total_size);
	}
	if (m_hResultPool.base_ptr != NULL)
	{
		ClearDeviceResultPool();
		memset(m_hResultPool.base_ptr, 0, m_hResultPool.total_size);
	}
}

void CReteRuntime::DestroyDeviceMemPool()
{
	if (m_hTempPool.base_ptr != NULL)
  	{
		CUDASTL_SAFECALL(cudaFree(m_dpTempPool));
  		CUDASTL_SAFECALL(cudaFreeHost(m_hTempPool.base_ptr));
 		m_hTempPool.base_ptr = NULL;
		m_dpTempPool = NULL;
 	}
	if (m_hResultPool.base_ptr != NULL)
	{
		CUDASTL_SAFECALL(cudaFree(m_dpResultPool));
		CUDASTL_SAFECALL(cudaFreeHost(m_hResultPool.base_ptr));
		m_hResultPool.base_ptr = NULL;
		m_dpResultPool = NULL;
	}
}

void CReteRuntime::CreateDeviceMemPool()
{	
	CUDASTL::SimpleMemPool dTempPool;

	if (m_hTempPool.base_ptr == NULL)
	{
		CUDASTL_SAFECALL(cudaHostAlloc((void**)&m_hTempPool.base_ptr, TEMPPOOLSIZE, cudaHostAllocMapped));
		dTempPool.base_ptr = (char*)GetDevicePointer((void*)m_hTempPool.base_ptr);
		dTempPool.bytes_used = 0;
		dTempPool.total_size = TEMPPOOLSIZE;
		UpdateDevicePoolInfo(&m_dpTempPool, dTempPool);
		SetDeviceTempPool(m_dpTempPool);
	}
	
	if (m_hResultPool.base_ptr == NULL)
	{
		CUDASTL_SAFECALL(cudaHostAlloc((void**)&m_hResultPool.base_ptr, RESULTPOOLSIZE, cudaHostAllocMapped));
		dTempPool.base_ptr = (char*)GetDevicePointer((void*)m_hResultPool.base_ptr);
		dTempPool.bytes_used = 0;
		dTempPool.total_size = RESULTPOOLSIZE;
		UpdateDevicePoolInfo(&m_dpResultPool, dTempPool);
		SetDeviceResultPool(m_dpResultPool);
	}
}

void CReteRuntime::PrintAllocInfo()
{
	cout.setf(ios::fixed);
	cout.precision(3);
	cout<<"DeviceTempPool: "<<(double)m_hTempPool.bytes_used/1024<<" KB\t";
	cout<<"HostTempPool: "<<(double)GlobalMemPool::GetHostTempPoolRef()->getallocsize()/1024<<" KB"<<endl;

	cout<<"NodePool: "<<(double)GlobalMemPool::GetNodePool().getallocsize()/1024<<" KB"<<"\t";
	cout<<"StringPool: "<<(double)GlobalMemPool::GetStringPool().getallocsize()/1024<<" KB"<<endl;

	cout<<"RetePool: "<<(double)GlobalMemPool::GetReteNodePool().getallocsize()/1024<<" KB"<<"\t";
	cout<<"VFTBPool: "<<(double)GlobalMemPool::GetVFTBPool().getallocsize()/1024<<" KB"<<endl;
}
