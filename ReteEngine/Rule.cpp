#include "Rule.h"
#include "TriplePattern.h"

CRule::CRule(void)
:m_nNumVar(0),
m_bBackward(false)
{
}

CRule::~CRule(void)
{
	if (!m_body.empty())
	{
		m_body.clear();
	}
	if (!m_head.empty())
	{
		m_head.clear();
	}
}

CTriplePattern* CRule::getBodyElement(size_t index)
{
	if (index >= getBodySize())
	{
		return NULL;
	}
	return m_body[index];
}

CTriplePattern* CRule::getHeadElement(size_t index)
{
	if (index >= getHeadSize())
	{
		return NULL;
	}
	return m_head[index];
}