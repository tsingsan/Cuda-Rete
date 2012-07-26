#pragma once

#include "common.h"

class CNode;
class CTriplePattern
{
public:
	CTriplePattern(void);
	CTriplePattern(CTriplePattern* pTriple);
	CTriplePattern(CNode* sub, CNode* pre, CNode* obj);
	~CTriplePattern(void);

	__device__ __host__ CNode* getSubject()
	{
#if !defined(__CUDA_ARCH__)
		return m_pSubject;
#else
		return UVADevice(m_pSubject);
#endif
	}
	__device__ __host__ CNode* getPredicate()
	{
#if !defined(__CUDA_ARCH__)
		return m_pPredicate;
#else
		return UVADevice(m_pPredicate);
#endif
	}
	__device__ __host__ CNode* getObject()
	{
#if !defined(__CUDA_ARCH__)
		return m_pObject;
#else
		return UVADevice(m_pObject);
#endif
	}

	bool equal(CTriplePattern* rhs); 
	void trace();

private:
	UVAMember(CNode*, m_pSubject);
	UVAMember(CNode*, m_pPredicate);
	UVAMember(CNode*, m_pObject);
};
