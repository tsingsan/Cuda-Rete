#pragma once
#include "common.h"
#include "ReteNode.h"

#include "GlobalMemPool.h"

#include "TriplePattern.h"
#include "BindingVector.h"

#include "VFTBMap.h"

#include <stdio.h>

enum InstructionCode
{
	TestValue,
	TestFunctorNum,
	TestIntraMatch,
	CreateToken,
	Bind,
	End,
	AddSubject,
	AddPredicate,
	AddObject,
	AddFunctorNode
};

class CTriplePattern;
class CClauseFilter :
	public CReteNode
{
public:
	__device__ CClauseFilter():m_pInstructions(NULL),m_ppArgs(NULL){}

	CClauseFilter(byte* pInstruction, CNode** ppArgs, size_t nCountInst, size_t nCountArgs);
#if defined(__CUDA_ARCH__)
	__device__ ~CClauseFilter(void){}
#else
	~CClauseFilter(void);
#endif


	static CClauseFilter* compile(CTriplePattern* pClause, int nEnvlength, CNode** ppClauseVars, bool bFromPool = true);

	__device__ CNode* getTripleValue(CTriplePattern* pTriple, byte address)
	{
		switch(address & 0x0f)
		{
		case AddSubject:
			return pTriple->getSubject();
		case AddPredicate:
			return pTriple->getPredicate();
		case AddObject:
			return pTriple->getObject();
		}

		return NULL;
	}

	__device__ void Start(CTriplePattern* pTriple)
	{
		//printf("clause Start ");

		CBindingVector* bindVec = NULL;
		CNode* pTmpNode = NULL;
		byte* instruction = UVADevice(m_pInstructions);
		for(size_t pc = 0; pc < m_nCountInst; )
		{
			switch(instruction[pc++])
			{
			case TestValue:
			// Check triple entry (arg1) against literal value (arg2)				
				if(!VFCheck(getTripleValue(pTriple,instruction[pc++]))->sameValueAs(VFCheck(UVADevice(m_ppArgs)[instruction[pc++]])))
				{
					return;
				}
				break;
			case CreateToken:	
#if defined(__ENABLE_DEVICE_POOL__)
				bindVec = CBindingVector::ConstructFromPool(instruction[pc++]);
#else
				bindVec = new CBindingVector(instruction[pc++]);
#endif
				break;
			case Bind:
				pTmpNode = getTripleValue(pTriple, instruction[pc++]);
				VFCheck(pTmpNode);
				if (bindVec == NULL || !bindVec->bind(instruction[pc++], pTmpNode))
				{
					gpuprintf("Bind Error ");
					return;
				}
				break;
			case End:
				VFCheck(UVADevice(m_continuation))->fire(bindVec);
				break;
			}
		}
	}

protected:
	CClauseFilter(byte* pInstruction, CNode** ppArgs, size_t nCountInst, size_t nCountArgs, CUDASTL::MemPool* pMemPool);

private:
	UVAMember(byte*, m_pInstructions);
	UVAMember(CNode**, m_ppArgs);
	size_t m_nCountInst;
	size_t m_nCountArgs;
};
