#pragma once

#include <memory.h>
#include "GPUMemPool.h"
#include "GPUMemPoolAdaptor.h"

#include "GPUList.h"
#include "GPUDebug.h"
#include "GPUPrintf.h"

class CBindingVector
{
public:
	__device__ CBindingVector(size_t nCount)
	{
		m_nNodesCount = nCount;
		m_ppNodesVec = new CNode*[nCount];
		memset(m_ppNodesVec, 0, sizeof(CNode*) * nCount);
	}

	__device__ CBindingVector* Clone()
	{
		CBindingVector* pNewBV = new CBindingVector(m_nNodesCount);
		memcpy(pNewBV->m_ppNodesVec, m_ppNodesVec, sizeof(CNode*) * m_nNodesCount);
		return pNewBV;
	}

	__device__ CBindingVector* CloneFromPool()
	{	
		CBindingVector* pNewBV = (CBindingVector*)GetDeviceTempPool()->malloc(sizeof(CBindingVector));
		pNewBV->m_nNodesCount = m_nNodesCount;
		pNewBV->m_ppNodesVec = (CNode**)GetDeviceTempPool()->malloc(sizeof(CNode*) * m_nNodesCount);
		memcpy(pNewBV->m_ppNodesVec, m_ppNodesVec, sizeof(CNode*) * m_nNodesCount);
		return pNewBV;
	}

	__device__ static CBindingVector* ConstructFromPool(size_t nCount)
	{
		CBindingVector* pNewBV = (CBindingVector*)GetDeviceTempPool()->malloc(sizeof(CBindingVector));
		pNewBV->m_nNodesCount = nCount;
		pNewBV->m_ppNodesVec = (CNode**)GetDeviceTempPool()->malloc(sizeof(CNode*) * nCount);
		memset(pNewBV->m_ppNodesVec, 0, sizeof(CNode*) * nCount);
		return pNewBV;
	}

	__device__ ~CBindingVector(void)
	{
		delete[] m_ppNodesVec;
	}
	__device__ bool bind(size_t i, CNode* pNode)
	{
		if (i >= m_nNodesCount)
		{
			gpuprintf("bindoverflow");
			return false;
		}
		CNode* pCurNode = m_ppNodesVec[i];
		if (pCurNode == NULL)
		{
			m_ppNodesVec[i] = pNode;
			return true;
		}
		else
		{
			gpuprintf("bindexist");
			return pCurNode->sameValueAs(pNode);
		}
	}

	__device__ bool equal(CBindingVector* pRhs)
	{
		if (pRhs->m_nNodesCount != m_nNodesCount)
		{
			return false;
		}
		CNode* pRhsNode = NULL;
		CNode* pLhsNode = NULL;
		for (size_t i = 0; i < m_nNodesCount; i++)
		{
			pLhsNode = m_ppNodesVec[i];
			pRhsNode = pRhs->m_ppNodesVec[i];
			if (pLhsNode != pRhsNode)
			{
// 				if (pLhsNode != NULL && pRhsNode != NULL && pLhsNode->sameValueAs(pRhsNode))
// 				{
// 					continue;
// 				}
				return false;
			}
		}
		return true;
	}

	__device__ CNode* getNode(size_t index)
	{
		if (index >= m_nNodesCount)
		{
			return NULL;
		}
		else
		{
			return m_ppNodesVec[index];
		}
	}

	__device__ size_t length()
	{
		return m_nNodesCount;
	}
private:
	CNode** m_ppNodesVec;
	size_t m_nNodesCount;
};

#ifndef __cplusplus
template<> __device__ CListElement<CBindingVector*>::~CListElement()
{
	delete (CBindingVector*)m_Elem;
}
#endif
/*
#if defined(__CUDA_ARCH__)
__device__ CListElement<CBindingVector*>* CGPUList<CBindingVector*>::find(CBindingVector* pValue)
{
	CListElement<CBindingVector*>* pCurElem = m_pHead;
	while(pCurElem)
	{
		if (pValue->equal(pCurElem->m_Elem))
		{
			return pCurElem; 
		}			
		pCurElem = pCurElem->m_Next;			
	}
	return NULL;
}
#endif
*/
