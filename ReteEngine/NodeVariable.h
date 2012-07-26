#pragma once
#include "Node.h"

class CNodeVariable :
	public CNode
{
public:
	DEFAULT_DEVICE_STRUCTION_FUNCTION(CNodeVariable)

	__host__	CNodeVariable(LPCTSTR szStr, int nIndex):CNode(szStr),m_nIndex(nIndex){}
	__device__ __host__ ~CNodeVariable(void){}

	__device__ __host__ bool isVariable(){return true;}

	__device__ __host__ int getIndex(){return m_nIndex;}

	__device__ __host__ bool sameValueAs(CNode* pNode)
	{
		if (this == pNode)
		{
			return true;
		}
		if (pNode->isVariable() && StringEqual(((CNodeVariable*)pNode)->toString(), toString())
			&& ((CNodeVariable*)pNode)->m_nIndex == m_nIndex)
		{
			return true;
		}
		return false;
	}

private:
	int m_nIndex;	//index in varmap
};
