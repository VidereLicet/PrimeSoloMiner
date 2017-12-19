#ifndef _INI_PARSER_H
#define _INI_PARSER_H

#include "linkedlist.h"
typedef char* LPSTR;
typedef const char* LPCSTR;

class CIniParserElement
{
public:
	char		value[4096];
	char		name[4096];
};

class CIniParserGroup
{
public:
	char		groupName[4096];

	CIniParserGroup();
	~CIniParserGroup();	

	CLinkedListByRef<CIniParserElement> elemente;	
};

class CIniParser
{
	CLinkedListByRef<CIniParserGroup>	groups;

	void		ClearLine(LPCSTR input, LPSTR output);
	bool		GetValueAndName(LPCSTR line, LPSTR name, LPSTR value);
public:

	CIniParser();
	~CIniParser();

	bool		Parse(LPCSTR inhalt);
	void		Clear();

	bool		GetValueAsString(LPCSTR section, LPCSTR value, LPSTR dest, int maxDestSize);
	bool		GetValueAsInteger(LPCSTR section, LPCSTR value, int *dest);
	bool		GetValueAsDouble(LPCSTR section, LPCSTR value, double *dest);
};

#endif