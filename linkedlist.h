#ifndef _LINKEDLIST_H
#define _LINKEDLIST_H

template <class T>
class CEnumerateByRef;
template <class T>
class CEnumerateByValue;

template <class T>
class CLinkedListByValue
{
protected:
	typedef struct __wraper__
	{
		struct __wraper__	*next, *prev;
		T ptr;
	}WRAPER, *LPWRAPER;

	volatile LPWRAPER	firstElement, lastElement;
	volatile int		count;
public:	
				CLinkedListByValue()
	{
		firstElement = NULL;
		lastElement = NULL;
		count = 0;
	};

				~CLinkedListByValue()
	{
		Clear();
	};

	bool		AddItemLast(T item)
	{
		int size = sizeof(WRAPER);
		LPWRAPER element = (LPWRAPER)malloc(size);
		if(element == NULL)
			return false;

		memset(element, 0, size);

		element->ptr = item;

		if(firstElement == NULL)
		{
			firstElement = element;			
		}else
		{
			lastElement->next = element;
			element->prev = lastElement;			
		}
		lastElement = element;
		
		count++;
		return true;
	}

	bool		AddItemFirst(T item)
	{
		LPWRAPER element = (LPWRAPER)malloc(sizeof(WRAPER));
		if(element == NULL)
			return false;

		memset(element, 0, sizeof(WRAPER));

		element->ptr = item;

		if(lastElement == NULL)
		{
			lastElement = element;
		}else
		{
			firstElement->prev = element;
			element->next = firstElement;			
		}

		firstElement = element;
		count++;
		return true;
	}

	void		DeleteFirstElement()
	{
		if(firstElement == NULL)
			return;

		LPWRAPER element = firstElement;
		firstElement = firstElement->next;
		if(firstElement != NULL)
			firstElement->prev = NULL;
		else
			lastElement = NULL;

		free(element);
		count--;
	}
	void		DeleteLastElement()
	{
		if(lastElement == NULL)
			return;

		LPWRAPER element = lastElement;
		lastElement = lastElement->prev;
		if(lastElement != NULL)
			lastElement->next = NULL;
		else
			firstElement = NULL;

		free(element);
		count--;
	}


	T			GetFirstItem()
	{
		if(firstElement == NULL)
			return NULL;
		return firstElement->ptr;
	}

	T			GetLastItem()
	{
		if(lastElement == NULL)
			return NULL;
		return lastElement->ptr;
	}

	void		Clear()
	{
		while(true)
		{
			LPWRAPER element = firstElement;
			if(element == NULL)
				break;

			firstElement = firstElement->next;
			free(element);
		}

		firstElement = lastElement = NULL;
		count = 0;
	}

	void		DeleteItem(T item)
	{
		LPWRAPER element = firstElement;
		while(element)
		{
			if(element->ptr == item)
			{
				LPWRAPER prev, next;

				prev = element->prev;
				next = element->next;

				if(prev == NULL)
				{
					firstElement = next;
				}else
				{
					prev->next = next;
				}

				if(next == NULL)
				{
					lastElement = prev;
				}else
				{
					next->prev = prev;
				}

				free(element);
				count--;
				break;
			}
			element = element->next;
		}
	}

	void		MoveItemToBegin(T item)
	{
		LPWRAPER element = firstElement;
		while(element)
		{
			if(element->ptr == item)
			{
				if(element == firstElement)
					break;

				if(element->next)
					element->next->prev = element->prev;
				if(element->prev)
					element->prev->next = element->next;
				element->prev = NULL;
				element->next = firstElement;
				firstElement->prev = element;
				firstElement = element;
				break;
			}

			element = element->next;
		}
	}

	int			GetCount() { return count; };

	T			operator[] (int const& index)
	{
		if(index < 0 || index >= count)
			return NULL;
		
		CEnumerateByValue<T> en(this);	
		int counter = 0;	
		for(T ptr=en.GetActValue(); en.IsCurrentValidValue();
			en.SetToNextElement(),ptr=en.GetActValue())
		{
			if(counter == index)
				return ptr;
			counter++;
		}
		return NULL;
	}

	friend class CEnumerateByValue<T>;
};

template <class T>
class CLinkedListByRef
{
protected:
	typedef struct __wraper__
	{
		struct __wraper__	*next, *prev;
		T* ptr;
	}WRAPER, *LPWRAPER;

	volatile LPWRAPER	firstElement, lastElement;
	volatile int		count;
public:	
				CLinkedListByRef()
	{
		firstElement = NULL;
		lastElement = NULL;
		count = 0;
	}

				~CLinkedListByRef()
	{
		Clear();
	}

	void		Clear()
	{
		Clear(true);
	}

	bool		AddItemLast(T* item)
	{
		int size = sizeof(WRAPER);
		LPWRAPER element = (LPWRAPER)malloc(size);
		if(element == NULL)
			return false;

		memset(element, 0, size);

		element->ptr = item;

		if(firstElement == NULL)
		{
			firstElement = element;			
		}else
		{
			lastElement->next = element;
			element->prev = lastElement;			
		}
		lastElement = element;
		
		count++;
		return true;
	}

	bool		AddItemFirst(T* item)
	{
		LPWRAPER element = (LPWRAPER)malloc(sizeof(WRAPER));
		if(element == NULL)
			return false;

		memset(element, 0, sizeof(WRAPER));

		element->ptr = item;

		if(lastElement == NULL)
		{
			lastElement = element;
		}else
		{
			firstElement->prev = element;
			element->next = firstElement;			
		}

		firstElement = element;
		count++;
		return true;
	}

	void		DeleteFirstElement()
	{
		if(firstElement == NULL)
			return;

		LPWRAPER element = firstElement;
		firstElement = firstElement->next;
		if(firstElement != NULL)
			firstElement->prev = NULL;
		else
			lastElement = NULL;

		free(element);
		count--;
	}
	void		DeleteLastElement()
	{
		if(lastElement == NULL)
			return;

		LPWRAPER element = lastElement;
		lastElement = lastElement->prev;
		if(lastElement != NULL)
			lastElement->next = NULL;
		else
			firstElement = NULL;

		free(element);
		count--;
	}


	T*			GetFirstItem()
	{
		if(firstElement == NULL)
			return NULL;
		return firstElement->ptr;
	}

	T*			GetLastItem()
	{
		if(lastElement == NULL)
			return NULL;
		return lastElement->ptr;
	}

	void		Clear(bool deleteElements)
	{
		while(true)
		{
			LPWRAPER element = firstElement;
			if(element == NULL)
				break;

			firstElement = firstElement->next;

			if(deleteElements)
			{
				delete element->ptr;
			}

			free(element);
		}

		firstElement = lastElement = NULL;
		count = 0;
	}

	void		DeleteItem(T* item)
	{
		LPWRAPER element = firstElement;
		while(element)
		{
			if(element->ptr == item)
			{
				LPWRAPER prev, next;

				prev = element->prev;
				next = element->next;

				if(prev == NULL)
				{
					firstElement = next;
				}else
				{
					prev->next = next;
				}

				if(next == NULL)
				{
					lastElement = prev;
				}else
				{
					next->prev = prev;
				}

				free(element);
				count--;
				break;
			}
			element = element->next;
		}
	}

	void		MoveItemToBegin(T* item)
	{
		LPWRAPER element = firstElement;
		while(element)
		{
			if(element->ptr == item)
			{
				if(element == firstElement)
					break;

				if(element->next)
					element->next->prev = element->prev;
				if(element->prev)
					element->prev->next = element->next;
				element->prev = NULL;
				element->next = firstElement;
				firstElement->prev = element;
				firstElement = element;
				break;
			}

			element = element->next;
		}
	}

	int			GetCount() { return count; };

	T*			operator[] (int const& index)
	{
		if(index < 0 || index >= count)
			return NULL;
		
		CEnumerateByRef<T> en(this);	
		int counter = 0;	
		for(T* ptr=en.GetActValue(); ptr && en.IsCurrentValidValue();
			en.SetToNextElement(),ptr=en.GetActValue())
		{
			if(counter == index)
				return ptr;
			counter++;
		}
		return NULL;
	}

	friend class CEnumerateByRef<T>;
};

#endif
