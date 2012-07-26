#pragma once

#include <string>
#include <vector>
#include <map>

#include "GPUMemPool.h"

class CNode;
class CRule;
class CRuleParser
{
public:
	CRuleParser(void);
	~CRuleParser(void);

	bool parserule(const char* filename);

	CRule** getRules(){return &m_vecCRule[0];}
	size_t getRulesCount(){return m_vecCRule.size();}

protected:
	void parseclause(std::string& str, CRule* pCurRule, bool bClauseBody = true);
	CNode* parsenode(std::string& str);

private:
	std::map<std::string, std::string> m_prefixes;
	std::map<std::string, int> m_varmap;

	std::vector<CRule*> m_vecCRule;
};
