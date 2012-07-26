#pragma once

#include <string>
#include <vector>

class CTriplePattern;
class CRule
{
public:
	CRule(void);
	~CRule(void);

	void addBody(CTriplePattern* lhs){m_body.push_back(lhs);}
	void addHead(CTriplePattern* lhs){m_head.push_back(lhs);}

	void SetName(std::string str){m_szName = str;}
	void SetNumVar(int num){m_nNumVar = num;}
	void SetBackward(bool bBackward){m_bBackward = bBackward;}

	CTriplePattern* getBodyElement(size_t index);
	size_t getBodySize(){return m_body.size();}

	CTriplePattern* getHeadElement(size_t index);
	size_t getHeadSize(){return m_head.size();}

	int getNumVar(){return m_nNumVar;}
	bool isBackWard(){return m_bBackward;}

	bool empty(){return m_body.empty()||m_head.empty();}

private:
	 /** Rule body */
	std::vector<CTriplePattern*> m_body;

	/** Rule head or set of heads */
	std::vector<CTriplePattern*> m_head;
	
	/** Optional name for the rule */
	std::string m_szName;
	
	/** The number of distinct variables used in the rule */
	int m_nNumVar;

	/** Flags whether the rule was written as a forward or backward rule */
	bool m_bBackward;
};
