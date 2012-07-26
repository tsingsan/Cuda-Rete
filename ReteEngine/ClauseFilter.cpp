#include "ClauseFilter.h"

#include "Node.h"
#include "NodeVariable.h"
#include <vector>
#include <assert.h>

using namespace std;

CClauseFilter::CClauseFilter(byte* pInstruction, CNode** ppArgs, size_t nCountInst, size_t nCountArgs)
:m_nCountInst(nCountInst),
m_nCountArgs(nCountArgs),
m_pInstructions(NULL),
m_ppArgs(NULL)
{
	if (nCountInst)
	{
		m_pInstructions = new byte[nCountInst];
		memcpy(m_pInstructions, pInstruction, sizeof(byte)*nCountInst);
	}
	if (nCountArgs)
	{
		m_ppArgs = new CNode*[nCountArgs];
		memcpy(m_ppArgs, ppArgs, sizeof(CNode*)*nCountArgs);
	}
}

CClauseFilter::CClauseFilter(byte* pInstruction, CNode** ppArgs, size_t nCountInst, size_t nCountArgs, CUDASTL::MemPool* pMemPool)
:m_nCountInst(nCountInst),
m_nCountArgs(nCountArgs)
{
	if (nCountInst)
	{
		byte* pTmpInstruction = (byte*)pMemPool->malloc(sizeof(byte) * nCountInst);
		UVAAssign(byte*, m_pInstructions, pTmpInstruction);
		memcpy(m_pInstructions, pInstruction, sizeof(byte)*nCountInst);
	}
	if (nCountArgs)
	{
		CNode** ppTmpArgs = (CNode**)pMemPool->malloc(sizeof(CNode*) * nCountArgs);
		UVAAssign(CNode**, m_ppArgs, ppTmpArgs);
		memcpy(m_ppArgs, ppArgs, sizeof(CNode*)*nCountArgs);
	}
}

CClauseFilter::~CClauseFilter(void)
{
	if (m_pInstructions)
	{
		delete[] m_pInstructions;
		m_pInstructions = NULL;
	}
	if (m_ppArgs)
	{
		delete[] m_ppArgs;
		m_ppArgs = NULL;
	}
}

CClauseFilter* CClauseFilter::compile(CTriplePattern* pClause, int nEnvlength, CNode** ppClauseVars, bool bFromPool/* = true*/)
{
	byte* instructions = new byte[300];
	byte* bindInstrction = new byte[100];
	vector<CNode*> args;
	
	int pc = 0;
	int bpc = 0;
	int nVarIndex = 0;

	bindInstrction[bpc++] = CreateToken;
	bindInstrction[bpc++] = (byte)nEnvlength;

	CNode* pNode = pClause->getSubject();
	assert(pNode);
	if (!pNode->isVariable())
	{
		instructions[pc++] = TestValue;
		instructions[pc++] = AddSubject;
		instructions[pc++] = (byte)args.size();
		args.push_back(UVADevicePointer(CNode*, pNode));
	}
	else
	{
		bindInstrction[bpc++] = Bind;
		bindInstrction[bpc++] = AddSubject;
		bindInstrction[bpc++] = (byte)((CNodeVariable*)pNode)->getIndex();
		ppClauseVars[nVarIndex++] = pNode;
	}

	pNode = pClause->getPredicate();
	assert(pNode);
	if (!pNode->isVariable())
	{
		instructions[pc++] = TestValue;
		instructions[pc++] = AddPredicate;
		instructions[pc++] = (byte)args.size();
		args.push_back(UVADevicePointer(CNode*, pNode));
	}
	else
	{
		bindInstrction[bpc++] = Bind;
		bindInstrction[bpc++] = AddPredicate;
		bindInstrction[bpc++] = (byte)((CNodeVariable*)pNode)->getIndex();
		ppClauseVars[nVarIndex++] = pNode;
	}

	//Currently Not Suppport Functor Node
	pNode = pClause->getObject();
	assert(pNode);
	if (!pNode->isVariable())
	{
		instructions[pc++] = TestValue;
		instructions[pc++] = AddObject;
		instructions[pc++] = (byte)args.size();
		args.push_back(UVADevicePointer(CNode*, pNode));
	}
	else
	{
		bindInstrction[bpc++] = Bind;
		bindInstrction[bpc++] = AddObject;
		bindInstrction[bpc++] = (byte)((CNodeVariable*)pNode)->getIndex();
		ppClauseVars[nVarIndex++] = pNode;
	}
	bindInstrction[bpc++] = End;

	byte* packed = new byte[pc+bpc];
	memcpy(packed, instructions, sizeof(byte)*pc);
	memcpy(&packed[pc], bindInstrction, sizeof(byte)*bpc);
	delete[] instructions;
	delete[] bindInstrction;

	CClauseFilter* pResult;

	if (bFromPool)
	{
		pResult = (CClauseFilter*)GlobalMemPool::GetReteNodePool().malloc(sizeof(CClauseFilter));
		new(pResult) CClauseFilter(packed, args.size()?&args[0]:NULL, pc+bpc, args.size(), &GlobalMemPool::GetReteNodePool());
	}
	else
	{
		pResult = new CClauseFilter(packed, args.size()?&args[0]:NULL, pc+bpc, args.size());
	}

	delete[] packed;

	return pResult;
}
