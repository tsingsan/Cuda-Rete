#include "TripleContainer.h"

#include "Node.h"
#include "NodeURI.h"
#include "NodeFactory.h"
#include "TriplePattern.h"

#include "GlobalMemPool.h"

CTripleContainer::CTripleContainer(void)
{
}

CTripleContainer::~CTripleContainer(void)
{
	m_TripleVec.clear();
}

void CTripleContainer::ParseTriple(LPCTSTR sub, LPCTSTR pre, LPCTSTR obj)
{
	CNodeFactory nodefactory(GlobalMemPool::GetHostTempPoolRef());
	CNode* pSub = nodefactory.CreateNodeURI(sub);
	CNode* pPre = nodefactory.CreateNodeURI(pre);
	CNode* pObj = nodefactory.CreateNodeURI(obj);

	CTriplePattern* pTempTriple = new CTriplePattern(pSub, pPre, pObj);

	ParseTriple(pTempTriple, GlobalMemPool::GetHostTempPoolRef());
	
	delete pTempTriple;
}

void CTripleContainer::ParseTriple(CTriplePattern* pTriple, CUDASTL::MemPool* pMemPool)
{
	CTriplePattern* pMyTriple;
	if (pMemPool)
	{
		pMyTriple = (CTriplePattern*)pMemPool->malloc(sizeof(CTriplePattern));
		new(pMyTriple) CTriplePattern(pTriple);
	}
	else
	{
		pMyTriple = pTriple;
	}
	
	m_TripleVec.push_back(pMyTriple);
}

bool CTripleContainer::HasTriple(CTriplePattern* pTriple)
{
	std::vector<CTriplePattern*>::const_iterator it = m_TripleVec.begin();
	for (; it != m_TripleVec.end(); it++)
	{
		CTriplePattern* curTriple = *it;
		if (pTriple->equal(curTriple))
		{
			break;
		}
	}
	if (it == m_TripleVec.end())
	{
		return false;
	}
	else
	{
		return true;
	}
}

void CTripleContainer::Simplification()
{
	CTripleContainer temp;
	for (size_t i = 0; i < GetTriplesCount(); i++)
	{
		CTriplePattern* pTriple = GetTriple(i);
		if (!temp.HasTriple(pTriple))
		{
			temp.ParseTriple(pTriple);
		}
	}
	m_TripleVec = temp.m_TripleVec;
}