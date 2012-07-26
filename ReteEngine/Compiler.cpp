#include "Compiler.h"

#include "Rule.h"
#include "TriplePattern.h"

#include "Node.h"
#include "NodeVariable.h"
#include "NodeAny.h"
#include "NodeFactory.h"

#include "ReteNode.h"
#include "ClauseFilter.h"
#include "ReteQueue.h"
#include "ReteTerminal.h"

CCompiler::CCompiler(void)
:m_bWildcardRule(false)
{
}

CCompiler::~CCompiler(void)
{
	m_clauseIndex.clear();
	m_queueNode.clear();
}

void CCompiler::compile(CRule** pRule, size_t nCount)
{
	if (!m_clauseIndex.empty())
	{
		m_clauseIndex.clear();
	}
	m_bWildcardRule = false;

	for (size_t i=0; i < nCount; i++)
	{
		CRule* pCurRule = pRule[i];
		if (pCurRule->isBackWard())
		{
			continue;	//currently not support backward rule.
		}
		int numVars = pCurRule->getNumVar();
		bool* bSeenVar = new bool[numVars];
		memset(bSeenVar, false, sizeof(bool)*numVars);
		CReteNode* pPriornode = NULL;

		for (size_t k=0; k < pCurRule->getBodySize(); k++)
		{
			CNode** ppClauseVars = new CNode*[numVars];
			memset(ppClauseVars, NULL, sizeof(CNode*)*numVars);
			CTriplePattern* pTriPat = pCurRule->getBodyElement(k);
			CClauseFilter* clauseNode = CClauseFilter::compile(pTriPat, numVars, ppClauseVars);

			CNode* pPredicate = pTriPat->getPredicate();
			CNode* pObject = pTriPat->getObject();

			if (pPredicate->isVariable())
			{
				m_clauseIndex.put(CNodeAny::getSingleton()->toString(), UVADevicePointer(CClauseFilter*, clauseNode));
				m_bWildcardRule = true;
			} 
			else
			{
				m_clauseIndex.put(pPredicate->toString(), UVADevicePointer(CClauseFilter*, clauseNode));
				/*if (!m_bWildcardRule)
				{
					if (pObject->isVariable())
					{
						pObject = CNode::ANY;
					}
				}*/
			}

			std::vector<byte> matchIndices;
			for (int m = 0; m < numVars; m++)
			{
				if (ppClauseVars[m] == NULL)
				{
					break;
				}
				int varIndex = ((CNodeVariable*)ppClauseVars[m])->getIndex();
				if (bSeenVar[varIndex])
				{
					matchIndices.push_back((byte)varIndex);
				}
				bSeenVar[varIndex] = true;
			}

			if (pPriornode == NULL)
			{
				pPriornode = (CReteNode*)clauseNode;
			}
			else
			{
				//CReteQueue* leftQ = new CReteQueue(&matchIndices[0], matchIndices.size());
				//CReteQueue* rightQ = new CReteQueue(&matchIndices[0], matchIndices.size());
				CReteQueue* leftQ = CReteQueue::ConstructFromPool(&matchIndices[0], matchIndices.size());
				CReteQueue* rightQ = CReteQueue::ConstructFromPool(&matchIndices[0], matchIndices.size());
				m_queueNode.push_back(leftQ);
				m_queueNode.push_back(rightQ);
				leftQ->SetSibling(rightQ);
				rightQ->SetSibling(leftQ);
				clauseNode->SetContinuation(rightQ);
				pPriornode->SetContinuation(leftQ);
				pPriornode = leftQ;
			}


			delete[] ppClauseVars;
		}

		delete[] bSeenVar;

		if(pPriornode)
		{
			CReteTerminal* pEnd = CReteTerminal::ConstructFromPool(pCurRule);
			pPriornode->SetContinuation(pEnd);
		}
	}
}

CClauseFilter** CCompiler::GetMatchedFilter(CTriplePattern* pTriple, size_t& nCounts)
{
	CNode* pPredicate = pTriple->getPredicate();
	std::vector<CClauseFilter*>* pFound = m_clauseIndex.get(pPredicate->toString());
	if (pFound)
	{
		nCounts = pFound->size();
		return &(*pFound)[0];
	}
	else
	{
		nCounts = 0;
		return NULL;
	}
}

extern void ReteQueueClear(CReteQueue* pReteQueue);

void CCompiler::ClearReteNode()
{
	std::vector<CReteQueue*>::iterator it = m_queueNode.begin();
	for (; it != m_queueNode.end(); it++)
	{
		CReteQueue* pQueueNode = *it;
		ReteQueueClear(UVADevicePointer(CReteQueue*, pQueueNode));
	}
}
