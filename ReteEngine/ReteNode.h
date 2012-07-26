#pragma once

#include "common.h"
#include "Node.h"
#include "NodeVariable.h"
#include "NodeURI.h"

#include "VFTBMap.h"
#include "GPUPrintf.h"

class CBindingVector;
class CReteNode
{
public:
	__device__ __host__ CReteNode(void):m_continuation(NULL){}
	virtual __device__ __host__ ~CReteNode(void){}

	virtual void SetContinuation(CReteNode* pNode)
	{
		UVAAssign(CReteNode*, m_continuation, pNode);
	}

	virtual __device__ void fire(CBindingVector* pBindVec){}

// 	__device__ CReteNode* GetContinuation()
// 	{
// 		return UVADevice(m_continuation);
// 	}

protected:
	UVAMember(CReteNode*,m_continuation);
};