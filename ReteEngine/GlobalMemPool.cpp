#include "GlobalMemPool.h"

#include <stdlib.h>

CUDASTL::MemPool GlobalMemPool::m_NodePool;
CUDASTL::MemPool GlobalMemPool::m_StringPool;
CUDASTL::MemPool GlobalMemPool::m_ReteNodePool;

CUDASTL::MemPool GlobalMemPool::m_VFTBPool;

CUDASTL::MemPool GlobalMemPool::m_HostTempPool;

GlobalMemPool::GlobalMemPool(void)
{
}

GlobalMemPool::~GlobalMemPool(void)
{
}

void GlobalMemPool::Init()
{
	m_NodePool.setmode(MapHostPool);
	m_StringPool.setmode(MapHostPool);
	m_ReteNodePool.setmode(MapHostPool);

	m_VFTBPool.setmode(MapHostPool);

	m_HostTempPool.setmode(MapHostPool);
}

void GlobalMemPool::Destroy()
{
	m_NodePool.clear();
	m_StringPool.clear();
	m_ReteNodePool.clear();

	m_VFTBPool.clear();
	m_HostTempPool.clear();
}