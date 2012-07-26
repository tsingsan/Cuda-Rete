#include "RuleReason.h"
#include "RuleParser.h"
#include "Compiler.h"

CRuleReason::CRuleReason()
:m_pParser(NULL),
m_pCompiler(NULL)
{
}

CRuleReason::~CRuleReason()
{
	if (m_pParser)
	{
		delete m_pParser;
		m_pParser = NULL;
	}
	if (m_pCompiler)
	{
		delete m_pCompiler;
		m_pCompiler = NULL;
	}
}

void CRuleReason::parserule(const char* filename)
{
	if (!m_pParser)
	{
		m_pParser = new CRuleParser;
	}
	if (m_pParser->parserule(filename))
	{
		if (!m_pCompiler)
		{
			m_pCompiler = new CCompiler;
		}
		m_pCompiler->compile(m_pParser->getRules(), m_pParser->getRulesCount());
	}
}

CClauseFilter** CRuleReason::GetMatchedFilter(CTriplePattern* pTriple, size_t& nCounts)
{
	return m_pCompiler->GetMatchedFilter(pTriple, nCounts);
}

void CRuleReason::ClearReteNode()
{
	if (m_pCompiler)
	{
		m_pCompiler->ClearReteNode();
	}
}