#include "ctrans.h"

#ifdef WIN32
#include <Windows.h>// for MultiByteToWideChar, WideCharToMultiByte
#else
#include <string.h>
#include <cstdlib> // for mbstocws, cwstombs
#endif

#ifndef UINT
#define UINT unsigned int
#endif

#ifndef CP_UTF8
#define CP_UTF8 0
#endif

#ifndef CP_ACP
#define CP_ACP 0
#endif

wchar_t* HTmb2wc(const char* mbstr, UINT CodePage)
{
	wchar_t* wcstr = NULL;

	// Get the size of wchar_t after converted
#ifdef WIN32
	int size = MultiByteToWideChar(CodePage, 0, mbstr, -1, NULL, 0);
#else
	size_t size = mbstowcs(NULL, mbstr, 0);
#endif

	wcstr = new wchar_t[size+1];
	if (wcstr)
	{
		memset(wcstr, 0, (size + 1) * sizeof(wchar_t));
#ifdef WIN32
		int ret = MultiByteToWideChar(CodePage, 0, mbstr, -1, wcstr, size);
		if (ret == 0) // MultiByteToWideChar returns 0 if it does not succeed.
#else
		size_t ret = mbstowcs(wcstr, mbstr, size+1);
		if (ret == -1)
#endif
		{
			delete[] wcstr;
			wcstr = NULL;
		}
	}

	return wcstr;
}

char* HTwc2mb(const wchar_t* wcstr, UINT CodePage)
{
	char* mbstr = NULL;

	// Get the size of char after converted
#ifdef WIN32
	int size = WideCharToMultiByte(CodePage, 0, wcstr, -1, NULL, 0, NULL, NULL);
#else
	size_t size = wcstombs(NULL, wcstr, 0);
#endif

	mbstr = new char[size+1];
	if (mbstr)
	{
		memset(mbstr, 0, size * sizeof(char));
#ifdef WIN32
		int ret = WideCharToMultiByte(CodePage, 0, wcstr, -1, mbstr, size, NULL, NULL);
		if (ret == 0) // MultiByteToWideChar returns 0 if it does not succeed.
#else
		size_t ret = wcstombs(mbstr, wcstr, size+1);
		if (ret == -1)
#endif
		{
			delete[] mbstr;
			mbstr = NULL;
		}
	}

	return mbstr;
}

char* CTrans::UTF2ACP(const char* szUTF8)
{
	wchar_t* szUnicode = HTmb2wc(szUTF8, CP_UTF8);
	char* szACP = HTwc2mb(szUnicode, CP_ACP);
	delete[] szUnicode;
	m_VecTmpStr.push_back(szACP);
	return szACP;
}
