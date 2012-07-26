#pragma once

#include "common.h"
#include "tinyxml.h"
#include <string>
#include <map>


class CRDFAdaptor
{
public:
	CRDFAdaptor(void);
	~CRDFAdaptor(void);

	void ParseNamespace(LPCTSTR name, LPCTSTR value);
	void ParseTriple(LPCTSTR sub, LPCTSTR pre, LPCTSTR obj);
	void SetRDFSourceName(std::string& strRDF, std::string& strRule);
	void CompleteRDF();

protected:
	void CreateRDFDocument();

	void ParsePredicate(std::string& strPredicate);
	bool IsURI(std::string& strValue);

private:
	TiXmlDocument* m_pInferedDocument;
	TiXmlElement* m_pRDFRoot;
	std::string m_strOutputName;
	std::map<std::string, TiXmlElement*> m_objectmap;
	std::map<std::string, std::string> m_namespacemap; 
	std::map<std::string, std::string> m_addednsmap;
};
