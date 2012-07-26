#pragma once

#include "common.h"

class CTripleContainer;
class CTriplePattern;
class CClauseFilter;
struct FireInfo
{
	CTriplePattern* pTriple;
	CClauseFilter* pClauseFilter;
};

class CReteRuntime
{
public:
	CReteRuntime(void);
	~CReteRuntime(void);

	void go(FireInfo* pVecFireInfo, size_t nCountFireInfo, CTripleContainer* pResultContainer);

	static void ClearDeviceRuntime();
	static void DestroyDeviceMemPool();

	static void PrintAllocInfo();

protected:
	static void UpdateHostPoolInfo();
	static void UpdateDevicePoolInfo(CUDASTL::SimpleMemPool** ppDstPool, CUDASTL::SimpleMemPool& dSrcPool);

	static void CreateDeviceMemPool();
private:
	static CUDASTL::SimpleMemPool m_hTempPool;
	static CUDASTL::SimpleMemPool* m_dpTempPool;

	static CUDASTL::SimpleMemPool m_hResultPool;
	static CUDASTL::SimpleMemPool* m_dpResultPool;
};
