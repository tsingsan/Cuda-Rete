#pragma once

#include "OneToManyMap.h"
#include <string>

class CRule;
class CClauseFilter;
class CReteQueue;
class CTriplePattern;
class CCompiler
{
public:
	CCompiler(void);
	~CCompiler(void);

	void compile(CRule** pRule, size_t nCount);

	CClauseFilter** GetMatchedFilter(CTriplePattern* pTriple, size_t& nCounts);

	void ClearReteNode();

private:
	OneToManyMap<std::string, CClauseFilter*> m_clauseIndex;
	std::vector<CReteQueue*> m_queueNode;
	bool m_bWildcardRule;
};
