#pragma once

#include "common.h"

class CNode
{
public:
	DEFAULT_DEVICE_STRUCTION_FUNCTION(CNode)

	__host__	CNode(LPCTSTR szStr):UVAInit(LPCTSTR, m_szLabel, szStr){}
	virtual __device__ __host__ ~CNode(void){}

	virtual __device__ __host__ bool isURI(){return false;}

	virtual __device__ __host__ bool isVariable(){return false;}

	virtual __device__ __host__ bool sameValueAs(CNode* pNode){return this == pNode;}

	virtual __device__ __host__ LPCTSTR toString(){
#if defined(__CUDA_ARCH__)
		return UVADevice(m_szLabel);
#else
		return m_szLabel;
#endif
	}

	__device__ __host__ LPCTSTR getLocalString()
	{
		return m_szLabel;
	}

protected:
	UVAMember(LPCTSTR, m_szLabel);
};
