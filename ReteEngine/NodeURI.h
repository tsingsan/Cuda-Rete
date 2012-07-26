#pragma once
#include "Node.h"

class CNodeURI :
	public CNode
{
public:
	DEFAULT_DEVICE_STRUCTION_FUNCTION(CNodeURI)

	__host__	CNodeURI(LPCTSTR szStr):CNode(szStr){}
	__device__ __host__ ~CNodeURI(void){}

	__device__ __host__ bool isURI(){return true;}

	__device__ __host__ bool sameValueAs(CNode* pNode)
	{
		if (this == pNode)
		{
			return true;
		}
		if (pNode->isURI() && StringEqual(((CNodeURI*)pNode)->toString(), toString()))
		{
			return true;
		}
		return false;
	}
};
