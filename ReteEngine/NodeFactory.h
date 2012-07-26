#pragma once

#include "common.h"
#include <string>

class CNode;
class CNodeFactory
{
public:
	CNodeFactory();
	CNodeFactory(CUDASTL::MemPool* pMemPool);
	~CNodeFactory(void);

	CNode* CreateNodeURI(std::string& str);
	CNode* CreateNodeURI(LPCTSTR str);
	CNode* CreateNodeVariable(std::string& str, int nIndex);
	CNode* CreateNodeVariable(LPCTSTR str, int nIndex);

protected:

	LPCTSTR ParseString(std::string& str);
	LPCTSTR ParseString(LPCTSTR str);

private:
	CUDASTL::MemPool* m_pMemPool;
};
