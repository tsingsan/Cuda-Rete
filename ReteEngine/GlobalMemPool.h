#pragma once

#include "GPUMemPool.h"

class GlobalMemPool
{
public:
	GlobalMemPool(void);
	~GlobalMemPool(void);

	static CUDASTL::MemPool& GetNodePool(){return m_NodePool;}
	static CUDASTL::MemPool& GetStringPool(){return m_StringPool;}
	static CUDASTL::MemPool& GetReteNodePool(){return m_ReteNodePool;}

	static CUDASTL::MemPool& GetVFTBPool(){return m_VFTBPool;}

	static CUDASTL::MemPool* GetHostTempPoolRef(){return &m_HostTempPool;}

	static void Init();
	static void Destroy();

private:
	static CUDASTL::MemPool m_NodePool;
	static CUDASTL::MemPool m_StringPool;
	static CUDASTL::MemPool m_ReteNodePool;

	static CUDASTL::MemPool m_VFTBPool;

	static CUDASTL::MemPool m_HostTempPool;
};
