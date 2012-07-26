#include "ReteEngine.h"
#include "GlobalMemPool.h"

#include "ReteRuntime.h"
#include "RuleReason.h"

#include "NodeAny.h"

#include "RDFAdaptor.h"
#include "TripleContainer.h"
#include "TriplePattern.h"

#include "VFTBMap.h"
#include "GPUPrintf.h"
#include "GPUDebug.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
using namespace std;

#define MIN(a,b) ((a < b) ? a : b)
#define MAX(a,b) ((a > b) ? a : b)

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = 
	{ { 0x10,  8 },
	{ 0x11,  8 },
	{ 0x12,  8 },
	{ 0x13,  8 },
	{ 0x20, 32 },
	{ 0x21, 48 },
	{   -1, -1 } 
	};

	int index = 0;
	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) ) {
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	printf("MapSMtoCores undefined SMversion %d.%d!\n", major, minor);
	return -1;
}

inline int cutGetMaxGflopsDeviceId()
{
	int current_device   = 0, sm_per_multiproc = 0;
	int max_compute_perf = 0, max_perf_device  = 0;
	int device_count     = 0, best_SM_arch     = 0;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount( &device_count );
	// Find the best major SM Architecture GPU device
	while ( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major > 0 && deviceProp.major < 9999) {
			best_SM_arch = MAX(best_SM_arch, deviceProp.major);
		}
		current_device++;
	}

	// Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count ) {
		cudaGetDeviceProperties( &deviceProp, current_device );
		if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
			sm_per_multiproc = 1;
		} else {
			sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		}

		int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		if( compute_perf  > max_compute_perf ) {
			// If we find GPU with SM major > 2, search only these
			if ( best_SM_arch > 2 ) {
				// If our device==dest_SM_arch, choose this, or else pass
				if (deviceProp.major == best_SM_arch) {	
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			} else {
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}

CReteEngine::CReteEngine(void)
:m_pRuleReason(NULL),
m_bDifferenceModel(true)
{
	CUDASTL_SAFECALL(cudaSetDevice(cutGetMaxGflopsDeviceId()));	
	CUDASTL_SAFECALL(cudaSetDeviceFlags(cudaDeviceMapHost));
	
	GlobalMemPool::Init();
	VFTBMap::InitMap();
	InitGPUPrintf();
	InitGPUDebug();

	m_pTripleContainer = new CTripleContainer;
	m_pRDFAdaptor = new CRDFAdaptor;
}

CReteEngine::~CReteEngine(void)
{
	if (m_pTripleContainer)
	{
		delete m_pTripleContainer;
		m_pTripleContainer = NULL;
	}
	if (m_pRDFAdaptor)
	{
		delete m_pRDFAdaptor;
		m_pRDFAdaptor = NULL;
	}

	//清主机段内存
	GlobalMemPool::Destroy();
	//清设备端内存
	CReteRuntime::DestroyDeviceMemPool();
	DestroyGPUDebug();
	DestroyGPUPrintf();
	
	cudaThreadExit();
}

void CReteEngine::SetRDFSourceName(std::string& strRDF, std::string& strRule)
{
	if (m_pRDFAdaptor)
	{
		m_pRDFAdaptor->SetRDFSourceName(strRDF, strRule);
	}
}

void CReteEngine::ParseNamespace(LPCTSTR name, LPCTSTR value)
{
	if (m_pRDFAdaptor)
	{
		m_pRDFAdaptor->ParseNamespace(name, value);
	}
}

void CReteEngine::ParseTriple(LPCTSTR sub, LPCTSTR pre, LPCTSTR obj)
{
	if (m_pTripleContainer)
	{
		m_pTripleContainer->ParseTriple(sub, pre, obj);
	}
	if (m_pRDFAdaptor && !m_bDifferenceModel)
	{
		m_pRDFAdaptor->ParseTriple(sub, pre, obj);
	}	
}

void CReteEngine::ResetContainer()
{
	if (m_pTripleContainer)
	{
		delete m_pTripleContainer;
		m_pTripleContainer = new CTripleContainer;
	}
	if (m_pRDFAdaptor)
	{
		delete m_pRDFAdaptor;
		m_pRDFAdaptor = new CRDFAdaptor;
	}
}

void CReteEngine::ClearTemporary()
{
	ResetContainer();

	//清HostMemPool
	GlobalMemPool::GetHostTempPoolRef()->clear();
	//清NodeQueue上的m_queue链表
	if (m_pRuleReason)
	{
		m_pRuleReason->ClearReteNode();
	}
	//清DeviceMemPool
	CReteRuntime::ClearDeviceRuntime();
}

void CReteEngine::ProcessFireVec(CTripleContainer* pTripleContainer, std::vector<FireInfo>& fireVec)
{
	for (size_t i = 0; i < pTripleContainer->GetTriplesCount(); i++)
	{
		CTriplePattern* pTriple = pTripleContainer->GetTriple(i);
		size_t nMatchedFilter = 0;
		CClauseFilter** ppFilter = m_pRuleReason->GetMatchedFilter(pTriple, nMatchedFilter);
		if (nMatchedFilter && ppFilter)
		{
			for(size_t j = 0; j < nMatchedFilter; j++)
			{
				FireInfo finfo;
				finfo.pClauseFilter = ppFilter[j];
				finfo.pTriple = UVADevicePointer(CTriplePattern*, pTriple);
				fireVec.push_back(finfo);
			}
		}

		CTriplePattern AnyTriple(NULL, CNodeAny::getSingleton(), NULL);
		ppFilter = m_pRuleReason->GetMatchedFilter(&AnyTriple, nMatchedFilter);

		if (nMatchedFilter && ppFilter)
		{
			for(size_t j = 0; j < nMatchedFilter; j++)
			{
				FireInfo finfo;
				finfo.pClauseFilter = ppFilter[j];
				finfo.pTriple = UVADevicePointer(CTriplePattern*, pTriple);
				fireVec.push_back(finfo);
			}
		}
	}
}

void CReteEngine::Run()
{
	if (m_pTripleContainer == NULL || m_pTripleContainer->Empty())
	{
		cout<<"ReteEngine Error: No Triple"<<endl;
		return;
	}
	if (m_pRuleReason == NULL)
	{
		cout<<"ReteEngine Error: No RuleReason"<<endl;
		return;
	}

	//Get Ready For Fire 
	std::vector<FireInfo> fireVec;
	ProcessFireVec(m_pTripleContainer, fireVec);
	
	CTripleContainer TotalContainer;
	while(!fireVec.empty())
	{
		CTripleContainer container;
		CReteRuntime runtime;
		runtime.go(&fireVec[0], fireVec.size(), &container);

		container.Simplification();
		for (size_t i = 0; i < container.GetTriplesCount(); i++)
		{
			CTriplePattern* pTriple = container.GetTriple(i);
			if (!TotalContainer.HasTriple(pTriple))
			{
				TotalContainer.ParseTriple(pTriple);
			}			
		}

		fireVec.clear();
		ProcessFireVec(&container, fireVec);
	}

	for (size_t i = 0; i < TotalContainer.GetTriplesCount(); i++)
	{
		if (m_pRDFAdaptor)
		{
			CTriplePattern* pTriple = TotalContainer.GetTriple(i);
			m_pRDFAdaptor->ParseTriple(pTriple->getSubject()->getLocalString(), pTriple->getPredicate()->getLocalString(), pTriple->getObject()->getLocalString());
		}
	}
	
#if _DEBUG
	for (size_t i = 0; i < TotalContainer.GetTriplesCount(); i++)
	{
		TotalContainer.GetTriple(i)->trace();
	}
	CReteRuntime::PrintAllocInfo();
#endif

	ResetContainer();
}

