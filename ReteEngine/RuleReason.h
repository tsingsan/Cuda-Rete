#ifndef __RULEREASON_CRULEREASON_H__
#define __RULEREASON_CRULEREASON_H__

#include "common.h"

class CRuleParser;
class CCompiler;
class CClauseFilter;
class CTriplePattern;
class EXPORTDLL CRuleReason
{
public:
	CRuleReason();
	~CRuleReason();

	void parserule(const char* filename);

	CClauseFilter** GetMatchedFilter(CTriplePattern* pTriple, size_t& nCounts);

	void ClearReteNode();

private:
	
	CRuleParser* m_pParser;
	CCompiler* m_pCompiler;
};

#endif