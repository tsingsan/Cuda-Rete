#pragma once
#include "ReteNode.h"

#include "GlobalMemPool.h"
#include "GPUMemPoolAdaptor.h"

#include "BindingVector.h"
#include "TriplePattern.h"

#include "VFTBMap.h"
#include "GPUPrintf.h"

class CRule;
class CReteTerminal :
	public CReteNode
{
public:
	__device__ CReteTerminal():m_ppTriplePattern(NULL){}

	CReteTerminal(CTriplePattern** ppTriplePattern, size_t nCountTriples);
#if defined(__CUDA_ARCH__)
	__device__ ~CReteTerminal(void){}
#else
	~CReteTerminal(void);
#endif

	static CReteTerminal* ConstructFromPool(CRule* pRule);

	__device__ void fire(CBindingVector* pBindVec)
	{
		//printf("Terminal Reach ");

		for (size_t i = 0; i < m_nCountTriples; i++)
		{
			CTriplePattern* pHeadTriple = UVADevice(m_ppTriplePattern)[i];
			CNode *pSub, *pPre, *pObj;

			CNode* pHeadSub = pHeadTriple->getSubject();
			if (VFCheck(pHeadSub)->isVariable())
			{
				pSub = pBindVec->getNode(((CNodeVariable*)pHeadSub)->getIndex());
			}
			else
			{
				pSub = pHeadSub;
			}

			CNode* pHeadPre = pHeadTriple->getPredicate();
			if (VFCheck(pHeadPre)->isVariable())
			{
				pPre = pBindVec->getNode(((CNodeVariable*)pHeadPre)->getIndex());
			}
			else
			{
				pPre = pHeadPre;
			}

			CNode* pHeadObj = pHeadTriple->getObject();
			if (VFCheck(pHeadObj)->isVariable())
			{
				pObj = pBindVec->getNode(((CNodeVariable*)pHeadObj)->getIndex());
			}
			else
			{
				pObj = pHeadObj;
			}

			LPCTSTR* pResult = (LPCTSTR*)GetDeviceResultPool()->malloc(sizeof(LPCTSTR) * 3);
			pResult[0] = pSub->getLocalString();
			pResult[1] = pPre->getLocalString();
			pResult[2] = pObj->getLocalString();
			
		//	printf("\nCreate A new Triple by Thread (%d, %d):\n\t%s\n\t%s\n\t%s\n", blockIdx.x, threadIdx.x, pSub->toString(), pPre->toString(), pObj->toString());

		}

#if !defined(__ENABLE_DEVICE_POOL__)
		delete pBindVec;
#endif
	}

protected:
	CReteTerminal(CTriplePattern** ppTriplePattern, size_t nCountTriples, CUDASTL::MemPool* pMemPool);

private:
	UVAMember(CTriplePattern**, m_ppTriplePattern);
	size_t m_nCountTriples;
};
