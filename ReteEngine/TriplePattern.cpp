#include "TriplePattern.h"
#include "VFTBMap.h"
#include "Node.h"
#include "ctrans.h"

CTriplePattern::CTriplePattern(void)
:UVAInit(CNode*, m_pSubject, NULL),
UVAInit(CNode*, m_pPredicate, NULL),
UVAInit(CNode*, m_pObject, NULL)
{
}

CTriplePattern::CTriplePattern(CNode* sub, CNode* pre, CNode* obj)
:UVAInit(CNode*, m_pSubject, sub),
UVAInit(CNode*, m_pPredicate, pre),
UVAInit(CNode*, m_pObject, obj)
{
}

CTriplePattern::CTriplePattern(CTriplePattern* pTriple)
{
	UVAAssign(CNode*, m_pSubject, pTriple->m_pSubject);
	UVAAssign(CNode*, m_pPredicate, pTriple->m_pPredicate);
	UVAAssign(CNode*, m_pObject, pTriple->m_pObject);
}

CTriplePattern::~CTriplePattern(void)
{
}

bool CTriplePattern::equal(CTriplePattern* rhs)
{
	return VFCheck(m_pSubject)->sameValueAs(VFCheck(rhs->m_pSubject)) && VFCheck(m_pPredicate)->sameValueAs(VFCheck(rhs->m_pPredicate)) && VFCheck(m_pObject)->sameValueAs(VFCheck(rhs->m_pObject));
}

void CTriplePattern::trace()
{
	CTrans trans;
	printf("\t%s\n\t%s\n\t%s\n\n", trans.UTF2ACP(m_pSubject->getLocalString()), trans.UTF2ACP(m_pPredicate->getLocalString()), trans.UTF2ACP(m_pObject->getLocalString()));
}