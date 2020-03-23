#ifndef _BIGMATH_H
#define _BIGMATH_H

#ifndef uint32
typedef unsigned int uint32;
#endif

#ifndef uint32
typedef unsigned long long uint64;
#endif

class uint1024
{
	uint32 data[32];

public:
	uint1024()
	{
	}
	uint1024(uint32 val)
	{		
		data[0] = val;
		for(int i=1;i<32;i++)
			data[i] = 0;
	}
	uint1024(uint64 val)
	{		
		data[0] = (uint32)val;
		data[1] = (uint32)(val >> 32);
		for(int i=2;i<32;i++)
			data[i] = 0;
	}

	uint1024 lshift_fast(int bits);
	uint1024 rshift_fast(int bits);

	uint1024 lshift(int bits);
	uint1024 rshift(int bits);

	int bitcount();

	int isZero();

	uint1024 add(uint1024 var);
	uint1024 sub(uint1024 var);

	int cmp(uint1024 var);

	uint1024 div(uint1024 d, uint1024 *rem);
	uint1024 mul(uint1024 d);
	uint32 fastmod(uint32 d);

	uint1024 powm(uint1024 exp, uint1024 mod);
};

#endif
