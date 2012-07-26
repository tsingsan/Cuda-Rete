#pragma once

#include "common.h"
#include <vector>
#include <string>

struct FireInfo;
class CRuleReason;
class CRDFAdaptor;
class CTripleContainer;
class EXPORTDLL CReteEngine
{
public:
	CReteEngine(void);
	~CReteEngine(void);

	void SetRDFSourceName(std::string& strRDF, std::string& strRule);
	void SetRuleReason(CRuleReason* pRuleReason){m_pRuleReason = pRuleReason;}
	void ParseNamespace(LPCTSTR name, LPCTSTR value);
	void ParseTriple(LPCTSTR sub, LPCTSTR pre, LPCTSTR obj);

	void Run();
	void ClearTemporary();

	void SetResultDifference(bool bDiff){m_bDifferenceModel = bDiff;}

protected:
	void ResetContainer();
	void ProcessFireVec(CTripleContainer* pTripleContainer, std::vector<FireInfo>& fireVec);

private:
	CRDFAdaptor* m_pRDFAdaptor;
	CRuleReason* m_pRuleReason;
	CTripleContainer* m_pTripleContainer;
	bool m_bDifferenceModel;
};
