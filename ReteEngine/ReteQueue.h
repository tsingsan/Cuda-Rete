#pragma once
#include "common.h"
#include "ReteNode.h"

#include "GlobalMemPool.h"
#include "GPUMemPoolAdaptor.h"

#include "BindingVector.h"
#include "GPUList.h"

#include "VFTBMap.h"

class CReteQueue :
	public CReteNode
{
public:
	__device__ CReteQueue():m_pMatchIndices(NULL){}

	CReteQueue(byte* pMatchIndices, size_t nCountIndices);
#if defined(__CUDA_ARCH__)
	__device__ ~CReteQueue(void){}
#else
	~CReteQueue(void);
#endif

	void SetSibling(CReteQueue* pSibling)
	{
		UVAAssign(CReteQueue*, m_pSibling, pSibling);
	}
	void SetContinuation(CReteNode* pNode)
	{
		UVAAssign(CReteNode*, m_continuation, pNode);
		if (m_pSibling)
		{
			UVAMemberAssign(CReteNode*, m_continuation, pNode, m_pSibling);
		}
	}

	__device__ void ClearQueue()
	{
		m_queue.clear();
	}

	__device__ void fire(CBindingVector* pBindVec)
	{
		//printf("Queue Fire ");
		//return;

// 		CBindingVector* pMyBindVec;
// 		if (m_queue.find(pBindVec) != NULL)
// 		{
// 			pMyBindVec = m_queue.find(pBindVec)->m_Elem;
// 		}
// 		else
// 		{
// 			pMyBindVec = pBindVec->CloneFromPool(GetDeviceTempPool());
// 			m_queue.put(pMyBindVec);
// 		}
// 		delete pBindVec;

		CBindingVector* pMyBindVec = pBindVec;	
		m_queue.put(pMyBindVec);

		size_t index = 0;
		bool matchOK = true;
		CBindingVector* pSiblingBind = NULL;
		CNode* pMyNode =  NULL;
		CNode* pHisNode = NULL;
		volatile CListElement<CBindingVector*>* pBindVecIterator = UVADevice(m_pSibling)->m_queue.m_pHead;
		for ( ; pBindVecIterator != NULL; pBindVecIterator = pBindVecIterator->m_Next)
		{
			pSiblingBind = pBindVecIterator->m_Elem;
			matchOK = true;

			for (size_t j = 0; j < m_nCountIndices; j++)
			{	
				index = (size_t)UVADevice(m_pMatchIndices)[j];
				pMyNode = pMyBindVec->getNode(index);
				pHisNode = pSiblingBind->getNode(index);
				if (pMyNode == NULL || pHisNode == NULL)
				{
					gpuprintf("ReteQueueError: MatchNULLNode;");
					gpuprintf(index);
/*
					gpuprintf((void*)pMyBindVec->getNode(0));
					gpuprintf((void*)pMyBindVec->getNode(1));
					gpuprintf((void*)pMyBindVec->getNode(2));
					gpuprintf((void*)pSiblingBind->getNode(0));
					gpuprintf((void*)pSiblingBind->getNode(1));
					gpuprintf((void*)pSiblingBind->getNode(2));*/
					matchOK = false;
					break;
				}
				if (!VFCheck(pMyNode)->sameValueAs(VFCheck(pHisNode)))
				{
					matchOK = false;
					break;
				}
			}

			if (matchOK)
			{
#if defined(__ENABLE_DEVICE_POOL__)
				CBindingVector* pMatchBindVec = pMyBindVec->CloneFromPool();
#else
				CBindingVector* pMatchBindVec = pMyBindVec->Clone();
#endif
				for(size_t j = 0; j < pMatchBindVec->length(); j++)
				{
					if (pMatchBindVec->getNode(j) == NULL && pSiblingBind->getNode(j) != NULL)
					{
						pMatchBindVec->bind(j, pSiblingBind->getNode(j));
					}
				}

				VFCheck(UVADevice(m_continuation))->fire(pMatchBindVec);
			}

		}		
	}

	static CReteQueue* ConstructFromPool(byte* pMatchIndices, size_t nCountIndices);

	CGPUList<CBindingVector*> m_queue;

protected:
	CReteQueue(byte* pMatchIndices, size_t nCountIndices, CUDASTL::MemPool* pMemPool);

private:
	UVAMember(byte*, m_pMatchIndices);
	UVAMember(CReteQueue*, m_pSibling);
	size_t m_nCountIndices;
};
