#pragma once
#include "OneToManyMap.h"

class CNode;
class CClauseFilter;
class CClauseMap :
	public OneToManyMap<CNode*,CClauseFilter*>
{
public:
	CClauseMap(void){}
	~CClauseMap(void){}

	listvalue* get(CNode* key);
};
