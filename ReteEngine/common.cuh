#include "common.h"

__device__ __host__ bool StringEqual(LPCTSTR p, LPCTSTR c)
{
	if (p == c)
	{
		return true;
	}
	for(; *p != '\0' && *c != '\0' && *p == *c; p++, c++);
	if(*p == '\0' && *c == '\0')
	{
		return true;
	}
	else
	{
		return false;
	}
}

__device__ __host__ int StringLength(LPCTSTR p)
{
	int n = 0;
	for(; *p != 0; n++, p++);
	return n;
}