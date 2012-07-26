#pragma once
#include "Node.h"

class CNodeAny :
	public CNode
{
private:
	CNodeAny(void);
	~CNodeAny(void);

	static CNode* m_pSelf;

public:

	static CNode* getSingleton()
	{
		if (m_pSelf == NULL)
		{
			m_pSelf = new CNodeAny;
		}
		return m_pSelf;
	}
};
