#include "ReteQueue.h"

#include <memory.h>
#include <new>

CReteQueue::CReteQueue(byte* pMatchIndices, size_t nCountIndices)
:m_nCountIndices(nCountIndices),
m_pMatchIndices(NULL)
{
	if (nCountIndices)
	{
		m_pMatchIndices = new byte[nCountIndices];
		memcpy(m_pMatchIndices, pMatchIndices, sizeof(byte)* nCountIndices);
	}
}

CReteQueue::CReteQueue(byte* pMatchIndices, size_t nCountIndices, CUDASTL::MemPool* pMemPool)
:m_nCountIndices(nCountIndices)
{
	if (nCountIndices)
	{
		byte* pTmpMatchIndices = (byte*)pMemPool->malloc(sizeof(byte) * nCountIndices);
		UVAAssign(byte*, m_pMatchIndices, pTmpMatchIndices);
		memcpy(m_pMatchIndices, pMatchIndices, sizeof(byte)* nCountIndices);
	}	
}

CReteQueue::~CReteQueue(void)
{
	if (m_pMatchIndices)
	{
		delete[] m_pMatchIndices;
		m_pMatchIndices = NULL;
	}	
}

CReteQueue* CReteQueue::ConstructFromPool(byte* pMatchIndices, size_t nCountIndices)
{
 	CReteQueue* pResult = (CReteQueue*)GlobalMemPool::GetReteNodePool().malloc(sizeof(CReteQueue));

	new(pResult) CReteQueue(pMatchIndices, nCountIndices, &GlobalMemPool::GetReteNodePool());
	return pResult;
}
