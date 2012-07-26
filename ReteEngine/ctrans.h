#ifndef __CUDARETE_UTF2ACP_H__
#define __CUDARETE_UTF2ACP_H__

#include <vector>

#ifdef __GNUC__  
#define CSET_GBK    "GBK"  
#define CSET_UTF8   "UTF-8"  
#define LC_NAME_zh_CN   "zh_CN"  
// ifdef __GNUC__  
#elif defined(_MSC_VER)  
#define CSET_GBK    "936"  
#define CSET_UTF8   "65001"  
#define LC_NAME_zh_CN   "Chinese_People's Republic of China"  
// ifdef _MSC_VER  
#endif

#define LC_NAME_zh_CN_GBK       LC_NAME_zh_CN "." CSET_GBK  
#define LC_NAME_zh_CN_UTF8      LC_NAME_zh_CN "." CSET_UTF8  
#define LC_NAME_zh_CN_DEFAULT   LC_NAME_zh_CN_GBK

class CTrans
{
public:
	CTrans(){m_VecTmpStr.clear();}
	~CTrans()
	{
		for(std::vector<const char*>::iterator it = m_VecTmpStr.begin(); it!= m_VecTmpStr.end(); it++)
		{
			delete[] *it;
		}
		m_VecTmpStr.clear();
	}
	char* UTF2ACP(const char* szUTF8);
private:
	std::vector<const char*> m_VecTmpStr;
};

#endif