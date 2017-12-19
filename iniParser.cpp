#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "linkedlist.h"
#include "enumerator.h"
#include "iniParser.h"

#ifndef WIN32
__inline void strcpy_s(char *buf, size_t bufSize, LPCSTR src)

{

 size_t len = strlen(src);

 if(len >= (size_t)bufSize)

  len = bufSize-1;

 strncpy(buf, src, len);

 buf[len] = 0;

}
#endif

__inline bool	StringCompareNoCase(LPCSTR str1, LPCSTR str2)
{
	size_t len = strlen(str1);

	if (strlen(str2) != len)
		return false;

	for (size_t i = 0;i<len;i++)
	{
		char c1, c2;
		c1 = str1[i];
		c2 = str2[i];

		if (c1 >= 'A' && c1 <= 'Z')
			c1 = c1 - 'A' + 'a';

		if (c2 >= 'A' && c2 <= 'Z')
			c2 = c2 - 'A' + 'a';

		if (c1 != c2)
			return false;
	}

	return true;
}

CIniParser::CIniParser()
{
}
CIniParser::~CIniParser()
{
	Clear();
}

bool		CIniParser::GetValueAndName(LPCSTR line, LPSTR name, LPSTR value)
{
	char		tmpLine[4096];
	strcpy_s(tmpLine, 4096, line);
	int			len = (int)strlen(tmpLine);

	int p1=0, p2=0;
	for(int i=0;i<len;i++)
	{
		if(tmpLine[i] == '=')
		{
			tmpLine[i] = 0;
			p2 = i+1;
			break;
		}
	}

	if(p2 == 0)
		return false;	

	ClearLine(&tmpLine[p1], name);
	if(p2 >= len)
	{
		value[0] = 0;
	}else
	{
		ClearLine(&tmpLine[p2], value);
	}	

	if(strlen(name) == 0)
		return false;

	return true;
}
void		CIniParser::ClearLine(LPCSTR input, LPSTR output)
{
	int startIdx = 0, endIdx = 0;
	int len = (int)strlen(input);

	for(int i=0;i<len;i++)
	{
		char c = input[i];

		if(c == ' ' || c == '\t' || c == 13)
			continue;

		startIdx = i;
		break;
	}
	endIdx = startIdx;

	for(int i=startIdx;i<len;i++)
	{
		char c = input[i];

		if(c == ' ' || c == '\t' || c == 13)
			continue;

		if(c == '#')
			break;

		endIdx = i+1;
	}

	int toCopy = endIdx - startIdx;
	output[0] = 0;

	for(int i=0;i<toCopy;i++)
	{
		output[i] = input[i + startIdx];
		output[i+1] = 0;
	}
}

bool		CIniParser::Parse(LPCSTR inhalt)
{
	bool				running = true;
	int					idx = 0;
	int					len = (int)strlen(inhalt);
	CIniParserGroup		*actGroup = NULL;

	while(running)
	{
		if(idx >= len)
		{
			running = false;
			break;
		}

		char	tmpLine[4096];
		tmpLine[0] = 0;

		for(int i=idx;i<len;i++)
		{
			char c = inhalt[i];
			int lIdx = i-idx;
			if(lIdx >= 4090)
			{
				Clear();
				return false;
			}

			if(c == '\n')
			{
				idx = i+1;
				break;
			}			

			tmpLine[lIdx] = c;
			tmpLine[lIdx+1] = 0;

			if(inhalt[i+1] == 0)
			{
				idx = i+1;
				break;
			}
		}

		char	clearedLine[4096];
		ClearLine(tmpLine, clearedLine);

		if(strlen(clearedLine) == 0)
		{
			continue;
		}

		int firstChar = clearedLine[0];
		if(firstChar == '[')
		{
			int clearedLineLen = (int)strlen(clearedLine);
			char sectionName[4096];
			sectionName[0] = 0;
			for(int i=1;i<clearedLineLen;i++)
			{
				char c = clearedLine[i];
				if(c == ']')
					break;
				sectionName[i-1] = c;
				sectionName[i] = 0;
			}

			actGroup = new CIniParserGroup();
			groups.AddItemLast(actGroup);

			strcpy_s(actGroup->groupName, 4096, sectionName);
		}else
		{
			char	value[4096];
			char	name[4096];

			bool res = GetValueAndName(clearedLine, name, value);
			if(res == true && actGroup != NULL)
			{
				CIniParserElement *el = new CIniParserElement();
				
				strcpy_s(el->name, 4096, name);
				strcpy_s(el->value, 4096, value);

				if(actGroup->elemente.AddItemLast(el) == false)
				{
					delete el;
				}
			}
		}
	}

	return true;
}
void		CIniParser::Clear()
{
	groups.Clear();
}

bool		CIniParser::GetValueAsString(LPCSTR section, LPCSTR value, LPSTR dest, int maxDestSize)
{
	CIniParserGroup *ptrGroup = NULL;

	FOREACH_REF(CIniParserGroup, ptr, groups)
	{
		if(StringCompareNoCase(ptr->groupName, section) == true)
		{
			ptrGroup = ptr;
			break;
		}		
	}

	if(ptrGroup == NULL)
		return false;

	CIniParserElement *ptrElement = NULL;

	FOREACH_REF(CIniParserElement, ptr, ptrGroup->elemente)
	{
		if(StringCompareNoCase(ptr->name, value) == true)
		{
			ptrElement = ptr;
			break;
		}
	}

	if(ptrElement == NULL)
		return false;

	strcpy_s(dest, maxDestSize, ptrElement->value);
	return true;
}
bool		CIniParser::GetValueAsInteger(LPCSTR section, LPCSTR value, int *dest)
{
	char tmp[512];

	if(GetValueAsString(section, value, tmp, 512) == false)
		return false;

	*dest = atoi(tmp);
	return true;
}
bool		CIniParser::GetValueAsDouble(LPCSTR section, LPCSTR value, double *dest)
{
	char tmp[512];

	if(GetValueAsString(section, value, tmp, 512) == false)
		return false;

	*dest = atof(tmp);
	return true;
}

CIniParserGroup::CIniParserGroup()
{
}
CIniParserGroup::~CIniParserGroup()
{
	this->elemente.Clear(true);
}

