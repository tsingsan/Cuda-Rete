#ifndef __RULEREASON_STRING_H__
#define __RULEREASON_STRING_H__

#include <string>

void trim(std::string& str);
size_t split(std::string& strIn, std::string* strOut, size_t nCount, const char* separator = NULL);

#endif