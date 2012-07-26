#include "ReteTerminal.h"

#include "Rule.h"
#include "TripleContainer.h"
#include <new>

CReteTerminal::CReteTerminal(CTriplePattern** ppTriplePattern, size_t nCountTriples)
:m_nCountTriples(nCountTriples),
m_ppTriplePattern(NULL)
{
	if(nCountTriples)
	{
		m_ppTriplePattern = new CTriplePattern*[nCountTriples];
		memcpy(m_ppTriplePattern, ppTriplePattern, sizeof(CTriplePattern*) * nCountTriples);
	}
}

CReteTerminal::CReteTerminal(CTriplePattern** ppTriplePattern, size_t nCountTriples, CUDASTL::MemPool* pMemPool)
:m_nCountTriples(nCountTriples),
UVAInit(CTriplePattern**, m_ppTriplePattern, NULL)
{
	if(nCountTriples)
	{
		CTriplePattern** ppTmpTriplePattern = (CTriplePattern**)pMemPool->malloc(sizeof(CTriplePattern*) * nCountTriples);
		UVAAssign(CTriplePattern**, m_ppTriplePattern, ppTmpTriplePattern);
		memcpy(m_ppTriplePattern, ppTriplePattern, sizeof(CTriplePattern*) * nCountTriples);
	}	
}

CReteTerminal::~CReteTerminal(void)
{
	if (m_ppTriplePattern)
	{
		delete[] m_ppTriplePattern;
		m_ppTriplePattern = NULL;
	}
}

CReteTerminal* CReteTerminal::ConstructFromPool(CRule* pRule)
{
	CTripleContainer container;
	for (size_t i = 0; i < pRule->getHeadSize(); i++)
	{
		CTriplePattern* pTriplePattern = pRule->getHeadElement(i);
		container.ParseTriple(UVADevicePointer(CTriplePattern*, pTriplePattern));
	}

	CReteTerminal* pResult = (CReteTerminal*)GlobalMemPool::GetReteNodePool().malloc(sizeof(CReteTerminal));
	new(pResult) CReteTerminal(container.GetTripleVector(), container.GetTriplesCount(), &GlobalMemPool::GetReteNodePool());

	return pResult;
}
