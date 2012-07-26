#pragma once

template<class T>
struct CListElement
{
	T m_Elem;
	volatile CListElement* m_Next;
	__device__ ~CListElement(){}
};

template<class T>
class CGPUList
{
public:

	__device__ CGPUList(void)
	:m_pHead(NULL)
	{
	}

	__device__  ~CGPUList(void)
	{
		clear();
	}

	__device__ void clear()
	{
//没有使用设备端内存池时清HEAP
#if !defined(__ENABLE_DEVICE_POOL__)
		volatile CListElement<T>* pCurElem = m_pHead;
		while(pCurElem)
		{
			volatile CListElement<T>* pTmpElem = pCurElem;
			pCurElem = pCurElem->m_Next;
			delete pTmpElem;
		}
#endif
		m_pHead = NULL;
	}
/*
	__device__ CListElement<T>* find(T pValue)
	{
		CListElement<T>* pCurElem = m_pHead;
		while(pCurElem)
		{
			if (pValue == pCurElem->m_Elem)
			{
				return pCurElem; 
			}			
			pCurElem = pCurElem->m_Next;			
		}
		return NULL;
	}
*/
	__device__ void put(T pValue)
	{
#if defined(__ENABLE_DEVICE_POOL__)
		CListElement<T>* pElem = (CListElement<T>*)GetDeviceTempPool()->malloc(sizeof(CListElement<T>));
#else
		CListElement<T>* pElem = (CListElement<T>*)new CListElement<T>;
#endif
		pElem->m_Elem = pValue;
		pElem->m_Next = m_pHead;

#if defined(__CUDA_ARCH__)
/*		
		CListElement<T>* pCurElem = m_pHead;
		if (pCurElem == NULL)
		{
			if (NULL == atomicCAS((unsigned int*)&m_pHead, NULL, (unsigned int)pElem))
			{
				return;
			}
		}
		
		pCurElem = m_pHead;
		while (1)
		{
			if (pCurElem->m_Next != NULL)
			{
				pCurElem = pCurElem->m_Next;
				continue;
			}
			if (NULL == atomicCAS((unsigned int*)&pCurElem->m_Next, NULL, (unsigned int)pElem))
			{
				return;
			}
		}*/

 		pElem->m_Next = (CListElement<T>*)atomicExch((unsigned int*)&m_pHead, (unsigned int)pElem);

#endif
/*
#if defined(__CUDA_ARCH__)
		int bfinished = 0;
		while (!__all(bfinished))
		{
			if (!bfinished && !atomicCAS(getGPUListMutexRef(), 0, 1))
			{
				pElem->m_Next  = m_pHead;
				m_pHead = pElem;
				setGPUListMutexZero();
				bfinished = 1;
			}
		}
#endif
*/
	}

	volatile CListElement<T>* m_pHead;
};
