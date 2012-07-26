#include "ClauseMap.h"
#include "Node.h"

std::vector<CClauseFilter*>* CClauseMap::get(CNode* key)
{
	if (m_map.find(key) != m_map.end())
	{
		return &m_map[key];
	}
	for (std::map<CNode*,listvalue>::iterator it = m_map.begin(); it != m_map.end(); it++)
	{
		CNode* curNode = it->first;
		if (key->equal(curNode))
		{
			return &it->second;
		}
	}
	return NULL;
}