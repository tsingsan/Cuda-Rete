#include "NodeFactory.h"
#include "GlobalMemPool.h"

#include "Node.h"
#include "NodeURI.h"
#include "NodeVariable.h"

#ifdef _WIN32
#include <tchar.h>
#else
#include <memory.h>
#define TCHAR char
#define _tcsclen(x) strlen(x)
#endif

CNodeFactory::CNodeFactory()
:m_pMemPool(NULL)
{
}

CNodeFactory::CNodeFactory(CUDASTL::MemPool* pMemPool)
:m_pMemPool(pMemPool)
{
}

CNodeFactory::~CNodeFactory(void)
{
}

CNode* CNodeFactory::CreateNodeURI(std::string& str)
{
	LPCTSTR szStr = ParseString(str);
	CUDASTL::MemPool* pMemPool = m_pMemPool ? m_pMemPool : &GlobalMemPool::GetNodePool();
	CNodeURI* pNewNode = (CNodeURI*)pMemPool->malloc(sizeof(CNodeURI));
	new(pNewNode) CNodeURI(szStr);
	return pNewNode;
}

CNode* CNodeFactory::CreateNodeURI(LPCTSTR str)
{
	LPCTSTR szStr = ParseString(str);
	CUDASTL::MemPool* pMemPool = m_pMemPool ? m_pMemPool : &GlobalMemPool::GetNodePool();
	CNodeURI* pNewNode = (CNodeURI*)pMemPool->malloc(sizeof(CNodeURI));
	new(pNewNode) CNodeURI(szStr);
	return pNewNode;
}

CNode* CNodeFactory::CreateNodeVariable(std::string& str, int nIndex)
{
	LPCTSTR szStr = ParseString(str);
	CUDASTL::MemPool* pMemPool = m_pMemPool ? m_pMemPool : &GlobalMemPool::GetNodePool();
	CNodeVariable* pNewNode = (CNodeVariable*)pMemPool->malloc(sizeof(CNodeVariable));
	new(pNewNode) CNodeVariable(szStr, nIndex);
	return pNewNode;
}

CNode* CNodeFactory::CreateNodeVariable(LPCTSTR str, int nIndex)
{
	LPCTSTR szStr = ParseString(str);
	CUDASTL::MemPool* pMemPool = m_pMemPool ? m_pMemPool : &GlobalMemPool::GetNodePool();
	CNodeVariable* pNewNode = (CNodeVariable*)pMemPool->malloc(sizeof(CNodeVariable));
	new(pNewNode) CNodeVariable(szStr, nIndex);
	return pNewNode;
}

LPCTSTR CNodeFactory::ParseString(std::string& str)
{
	CUDASTL::MemPool* pMemPool = m_pMemPool ? m_pMemPool : &GlobalMemPool::GetStringPool();
	LPCTSTR szStr = (LPCTSTR)pMemPool->malloc(sizeof(TCHAR)*(str.length() + 1), __alignof(TCHAR));
	memcpy((void*)szStr, str.c_str(), sizeof(TCHAR)*(str.length() + 1));
	return szStr;
}

LPCTSTR CNodeFactory::ParseString(LPCTSTR str)
{
	CUDASTL::MemPool* pMemPool = m_pMemPool ? m_pMemPool : &GlobalMemPool::GetStringPool();
	LPCTSTR szStr = (LPCTSTR)pMemPool->malloc(sizeof(TCHAR)*(_tcsclen(str)+1), __alignof(TCHAR));
	memcpy((void*)szStr, str, sizeof(TCHAR)*(_tcsclen(str)+1));
	return szStr;
}
