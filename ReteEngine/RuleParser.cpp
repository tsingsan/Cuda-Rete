#include "RuleParser.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "common.h"
#include "StringUtil.h"

#include "TriplePattern.h"
#include "Rule.h"

#include "Node.h"
#include "NodeURI.h"
#include "NodeVariable.h"
#include "NodeFactory.h"

#include "GlobalMemPool.h"

using namespace std;

CRuleParser::CRuleParser(void)
{
	m_prefixes["rdf"] = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
}

CRuleParser::~CRuleParser(void)
{
	m_prefixes.clear();
	m_varmap.clear();
	for(vector<CRule*>::iterator it = m_vecCRule.begin(); it != m_vecCRule.end(); it++)
	{
		delete (*it);
	}
	m_vecCRule.clear();
}

bool CRuleParser::parserule(const char* filename)
{
	ifstream fs(filename);
	if (!fs)
	{
		cout<<"Can't open rule file."<<endl;
		exit(-1);
	}
	char linebuffer[MAX_PATH];

	char bomtest[3];
	fs.read(bomtest, 3);
	if (!(bomtest[0] == (char)0xef && bomtest[1] == (char)0xbb && bomtest[2] == (char)0xbf))
	{
		//Not UTF8 BOM SeekBack;
		fs.seekg(0, ios_base::beg);
	}

	
	CRule* pCurRule = NULL;
	bool bRuleStarted = false;
	bool bClauseBody = true;
	while (fs.getline(linebuffer, MAX_PATH))
	{
		string line(linebuffer);
		trim(line);
		if (line.empty()) continue;

		if (line[0] == '#')	continue;	//skip comment lines;
		if (line.compare(0, 2, "//") == 0)	continue;	//skip comment lines;
		if (line.compare(0, 7, "@prefix") == 0)
		{
			line.erase(0, 8);
			string content[2];
			split(line, content, 2);

			string &prefix = content[0];
			string &rest = content[1];

			if (prefix.empty() || rest.empty())
			{
				cout<<"empty prefix!"<<endl;
				continue;
			}

			if(prefix[prefix.length()-1] == ':')	prefix.erase(prefix.length()-1);
			if (rest[0] == '<')
			{
				string::size_type splitpos = rest.find('>');
				rest = rest.substr(1, splitpos - 1);
			}
			m_prefixes[prefix] = rest;
		}
		else if(line[0] == '[')
		{
			if (bRuleStarted)
			{
				cout<<"unexpected '[' before a rule creation ends."<<endl;
				exit(-1);
			}
			bRuleStarted = true;
			bClauseBody = true;
			pCurRule = new CRule;			

			line.erase(0, 1);
			trim(line);
			if (!line.empty())
			{
				if (line[line.length()-1] == ':')	line.erase(line.length()-1);
				pCurRule->SetName(line);
			}
		}
		else if (line[0] == ']')
		{
			if (!bRuleStarted)
			{
				cout<<"unexpected ']' before a rule creation starts."<<endl;
				exit(-1);
			}

			bRuleStarted = false;
			pCurRule->SetNumVar(m_varmap.size());
			m_varmap.clear();

			if (pCurRule->empty())
			{
				delete pCurRule;
				pCurRule = NULL;
				continue;
			}
			m_vecCRule.push_back(pCurRule);
		}
		else if (line.compare(0, 2, "->") == 0)
		{
			if (!bRuleStarted)
			{
				cout<<"unexpected '->' before a rule creation starts."<<endl;
				exit(-1);
			}
			bClauseBody = false;
		}
		else if (line.compare(0, 2, "<-") == 0)
		{
			if (!bRuleStarted)
			{
				cout<<"unexpected '<-' before a rule creation starts."<<endl;
				exit(-1);
			}
			bClauseBody = false;
			pCurRule->SetBackward(true);
		}
		else if (line.find('(') != string::npos && line.find(')') != string::npos)
		{
			if (!bRuleStarted)
			{
				cout<<"unexpected Clause Entry before a rule creation starts."<<endl;
				exit(-1);
			}
			string::size_type lpos = line.find('(');
			string::size_type rpos = line.find(')');
			string clause = line.substr(lpos + 1, rpos - lpos - 1);
			parseclause(clause, pCurRule, bClauseBody);
		}
		else
		{
			cout<<"unprocessed line: "<<line<<endl;
			continue;
		}
	}
	return !m_vecCRule.empty();
}

void CRuleParser::parseclause(std::string& str, CRule* pCurRule, bool bClauseBody)
{
	string sznode[3];
	if(split(str, sznode, 3) != 3) return;

	CNode* sub = parsenode(sznode[0]);
	CNode* pre = parsenode(sznode[1]);
	CNode* obj = parsenode(sznode[2]);
	
	CTriplePattern* newTriplePattern = (CTriplePattern*)GlobalMemPool::GetNodePool().malloc(sizeof(CTriplePattern));
	new(newTriplePattern) CTriplePattern(sub, pre, obj);

	if (bClauseBody)
	{
		pCurRule->addBody(newTriplePattern);
	}
	else
	{
		pCurRule->addHead(newTriplePattern);
	}
}

CNode* CRuleParser::parsenode(std::string& str)
{
	CNodeFactory nodefactory;
	if (str[0] == '?')
	{
		CNode* newnode;
		if (m_varmap.find(str) == m_varmap.end())
		{
			newnode =  nodefactory.CreateNodeVariable(str, m_varmap.size());//new CNodeVariable(str, m_varmap.size());
			m_varmap.insert(make_pair(str, m_varmap.size()));
		}
		else
		{
			newnode =  nodefactory.CreateNodeVariable(str, m_varmap[str]);//new CNodeVariable(str, m_varmap[str]);
		}
		return newnode;
	}
	if (str[0] == '<' && str[str.length()-1] == '>')
	{
		return nodefactory.CreateNodeURI(str.substr(1, str.length()-2).c_str());//new CNodeURI(str.substr(1, str.length()-2));
	}
	if (str.find("^^") != string::npos)	//DataType Ignore
	{
		if (str[0] == '"')
		{
			if (str.find('"', 1) != string::npos)
			{
				return nodefactory.CreateNodeURI(str.substr(1, str.find('"', 1)  - 1).c_str());
			}
		}
	}
	if (str.find(':') != string::npos)
	{
		string newstr;
		string slice[4]; //at most support three prefixes.
		size_t nCount = split(str, slice, 4, ":");
		for (size_t i = 0; i < nCount - 1; i++)
		{
			if (m_prefixes.find(slice[i]) != m_prefixes.end())
			{
				newstr += m_prefixes[slice[i]];
			}
			else
			{
				if (slice[i] != "http" && slice[i] != "urn" && slice[i] != "file" && slice[i] != "ftp" && slice[i] != "mailto")
				{
					cout<<"Unrecognized prefix "<<slice[i]<<" in rule"<<endl;
				}
				newstr += slice[i] + ":";
			}
		}
		newstr += slice[nCount - 1];
		return nodefactory.CreateNodeURI(newstr);//new CNodeURI(newstr);
	}
	return nodefactory.CreateNodeURI(str);//new CNodeURI(str);
}
