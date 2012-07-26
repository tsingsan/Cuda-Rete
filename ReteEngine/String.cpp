#include <vector>
#include "String.h"

using namespace std;

const char* commonspace = " \r\n\t";

void trim(std::string& str)
{	
	string::size_type pos = str.find_last_not_of(commonspace);
	if(pos!=string::npos)
	{
		str.erase(pos+1);
		pos = str.find_first_not_of(commonspace);
		if (pos != string::npos)
		{
			str.erase(0,pos);
		}
	}
	else
		str.erase(str.begin(),str.end());
}

size_t split(std::string& strIn, std::string* strOut, size_t nCount, const char* separator /*= NULL*/)
{
	const char* myseparator = separator ? separator : commonspace;
	int nCurPos = 0, nPrevPos = -1;
	size_t nCurCount = 0;
	while(nCurCount < nCount)	 
	{
		nCurPos = strIn.find_first_of(myseparator, nPrevPos + 1);
		if (nCurPos != string::npos)
		{
			if(nCurPos == nPrevPos + 1)
			{
				nPrevPos = nCurPos;
				continue;
			}
			string prevStr = strIn.substr(nPrevPos + 1, nCurPos - nPrevPos - 1);
			trim(prevStr);
			if (!prevStr.empty())
			{
				strOut[nCurCount++] = prevStr;
			}
			nPrevPos = nCurPos;
		}
		else
		{
			break;
		}
	}
	if (nCurCount < nCount && nPrevPos != strIn.length() - 1)
	{
		string prevStr = strIn.substr(nPrevPos + 1);
		trim(prevStr);
		if (!prevStr.empty())
		{
			strOut[nCurCount++] = prevStr;
		}
	}
	return nCurCount;
}