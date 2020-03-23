#ifndef _FRM_ENUMERATOR_H
#define _FRM_ENUMERATOR_H

template <class T>
class CEnumerateByValue
{
	typedef struct __wraper__
	{
		struct __wraper__	*next, *prev;
		T ptr;
	}WRAPER, *LPWRAPER;

	CLinkedListByValue<T>		*liste;
	LPWRAPER					actWraper;
public:
	CEnumerateByValue(CLinkedListByValue<T> *list)
	{
		this->liste = list;

		this->actWraper = (LPWRAPER)this->liste->firstElement;
	}

	~CEnumerateByValue()
	{
	}

	void	Reset()
	{
		this->actWraper = (LPWRAPER)this->liste->firstElement;
	}

	T	GetActValue()
	{
		if(actWraper == NULL)
			return (T)0;

		return actWraper->ptr;
	}

	bool	SetToNextElement()
	{
		if(actWraper == NULL)
			return false;

		actWraper = actWraper->next;
		return true;
	}

	bool	SetToPrevElement()
	{
		if(actWraper == NULL)
			return false;

		actWraper = actWraper->prev;
		return true;
	}

	bool	IsCurrentValidValue()
	{
		if(actWraper == NULL)
			return false;
		return true;
	}
};

template <class T>
class CEnumerateByRef
{
	typedef struct __wraper__
	{
		struct __wraper__	*next, *prev;
		T* ptr;
	}WRAPER, *LPWRAPER;

	CLinkedListByRef<T>			*liste;
	LPWRAPER					actWraper;
public:
	CEnumerateByRef(CLinkedListByRef<T> *list)
	{
		this->liste = list;

		this->actWraper = (LPWRAPER)this->liste->firstElement;
	}

	~CEnumerateByRef()
	{
	}

	void	Reset()
	{
		this->actWraper = (LPWRAPER)this->liste->firstElement;
	}

	T*	GetActValue()
	{
		if(actWraper == NULL)
			return NULL;

		return actWraper->ptr;
	}

	bool	SetToNextElement()
	{
		if(actWraper == NULL)
			return false;

		actWraper = actWraper->next;
		return true;
	}

	bool	SetToPrevElement()
	{
		if(actWraper == NULL)
			return false;

		actWraper = actWraper->prev;
		return true;
	}

	bool	IsCurrentValidValue()
	{
		if(actWraper == NULL)
			return false;
		return true;
	}
};

#define FOREACH_REF(type, pointer, list) for(bool __int__oneTimeLoop__=true;__int__oneTimeLoop__;)for(CEnumerateByRef<type> __int__enMine__(&(list));__int__oneTimeLoop__;__int__oneTimeLoop__=false)for(type* pointer=__int__enMine__.GetActValue();__int__enMine__.IsCurrentValidValue() && pointer != NULL;__int__enMine__.SetToNextElement(),pointer=__int__enMine__.GetActValue())
#define FOREACH_VALUE(type, valName, list) for(bool __int__oneTimeLoop__=true;__int__oneTimeLoop__;)for(CEnumerateByValue<type> __int__enMine__(&(list));__int__oneTimeLoop__;__int__oneTimeLoop__=false)for(type valName=__int__enMine__.GetActValue();__int__enMine__.IsCurrentValidValue();__int__enMine__.SetToNextElement(),valName=__int__enMine__.GetActValue())

#endif
