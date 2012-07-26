#pragma once

#include "common.h"
#include <vector>

//Usage: Put Triple In MemPool.

class CTriplePattern;
class CTripleContainer
{
public:
	CTripleContainer(void);
	~CTripleContainer(void);

	void ParseTriple(CTriplePattern* pTriple, CUDASTL::MemPool* pMemPool = NULL);
	void ParseTriple(LPCTSTR sub, LPCTSTR pre, LPCTSTR obj);

	size_t GetTriplesCount(){return m_TripleVec.size();}
	CTriplePattern* GetTriple(size_t i)
	{
		if (GetTriplesCount() > i)
		{
			return m_TripleVec[i];
		}
		return NULL;
	}
	CTriplePattern** GetTripleVector()
	{
		if (GetTriplesCount() > 0)
		{
			return &m_TripleVec[0];
		}
		return NULL;
	}
	
	bool HasTriple(CTriplePattern* pTriple);

	bool Empty(){return m_TripleVec.empty();}

	void Simplification();

private:
	std::vector<CTriplePattern*> m_TripleVec; 
};
