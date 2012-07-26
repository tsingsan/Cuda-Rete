#include "RDFAdaptor.h"

const char* INFEREDPATHPREFIX = "../Infered/";

CRDFAdaptor::CRDFAdaptor(void)
:m_pInferedDocument(NULL),
m_strOutputName(std::string(INFEREDPATHPREFIX) + "Default_Infered.rdf")
{
	m_namespacemap["http://www.w3.org/1999/02/22-rdf-syntax-ns#"] = "xmlns:rdf";
}

CRDFAdaptor::~CRDFAdaptor(void)
{
	CompleteRDF();

	m_namespacemap.clear();
	m_addednsmap.clear();
	m_objectmap.clear();
}

void CRDFAdaptor::SetRDFSourceName(std::string& strRDF, std::string& strRule)
{
	size_t rdffilepos = strRDF.rfind('/') == std::string::npos ? 0 : strRDF.rfind('/') + 1;
	size_t rulefilepos = strRule.rfind('/') == std::string::npos ? 0 : strRule.rfind('/') + 1;
	m_strOutputName = strRDF.substr(rdffilepos, strRDF.rfind('.') - rdffilepos);
	m_strOutputName += "_" + strRule.substr(rulefilepos, strRule.rfind('.') - rulefilepos);
	m_strOutputName = std::string(INFEREDPATHPREFIX) + m_strOutputName + ".rdf";
}

void CRDFAdaptor::CreateRDFDocument()
{
	if (m_pInferedDocument)
	{
		return;
	}
	m_pInferedDocument = new TiXmlDocument();
	TiXmlDeclaration* pDeclaration = new TiXmlDeclaration("1.0", "", "");
	m_pInferedDocument->LinkEndChild(pDeclaration);
	m_pRDFRoot = new TiXmlElement("rdf:RDF");
	m_pInferedDocument->LinkEndChild(m_pRDFRoot);
	//m_pRDFRoot->SetAttribute("xmlns:rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#");
}

bool CRDFAdaptor::IsURI(std::string& strValue)
{
	return strValue.compare(0, 7, "http://") == 0;
}

void CRDFAdaptor::ParsePredicate(std::string& strPredicate)
{
	std::map<std::string, std::string>::iterator it = m_namespacemap.begin();
	for (; it != m_namespacemap.end(); it++)
	{
		if(strPredicate.compare(0, it->first.length(), it->first) == 0)
		{
			if (m_addednsmap.find(it->second) == m_addednsmap.end())
			{
				m_pRDFRoot->SetAttribute(it->second, it->first);
				m_addednsmap.insert(make_pair(it->second, it->first));
			}
			std::string replaceprefix = it->second.find(':') == std::string::npos ? "" : it->second.substr(it->second.find(':') + 1) + ':';
			strPredicate.replace(0, it->first.length(), replaceprefix);
			break;
		}
	}
}

void CRDFAdaptor::ParseTriple(LPCTSTR sub, LPCTSTR pre, LPCTSTR obj)
{
	if (m_pInferedDocument == NULL)
	{
		CreateRDFDocument();
	}
	TiXmlElement* pElement;
	std::string strObject = sub;
	std::string strProperty = pre;
	std::string strValue = obj;
	if (m_objectmap.find(strObject) == m_objectmap.end())
	{
		pElement = new TiXmlElement("rdf:Description");
		pElement->SetAttribute("rdf:about", sub);
		m_pRDFRoot->LinkEndChild(pElement);
		m_objectmap.insert(make_pair(strObject, pElement));
	}
	else
	{
		pElement = m_objectmap[strObject];
	}	
	
	ParsePredicate(strProperty);
	TiXmlElement* pProperty = new TiXmlElement(strProperty);
	if (IsURI(strValue))
	{
		pProperty->SetAttribute("rdf:resource", strValue);
	}
	else
	{
		TiXmlText* pText = new TiXmlText(strValue);
		pProperty->LinkEndChild(pText);
	}
	
	pElement->LinkEndChild(pProperty);
}

void CRDFAdaptor::ParseNamespace(LPCTSTR name, LPCTSTR value)
{
	m_namespacemap[value] = name;
	//printf("%s\t%s\n",name,value);
}

void CRDFAdaptor::CompleteRDF()
{
	if (m_pInferedDocument)
	{
		m_pInferedDocument->SaveFile(m_strOutputName);
	}
}
