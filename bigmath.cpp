#include "bigmath.h"

uint1024	uint1024::lshift_fast(int bits)
{
	uint1024 res;
	res.data[0] = data[0] << bits;
	for(int i=1; i < 32; i++)
		res.data[i] = (data[i] << bits) | (data[i-1] >> (32-i));
	return res;
}

uint1024	uint1024::rshift_fast(int bits)
{
	uint1024 res;	
	for(int i=0; i < 31; i++)
		res.data[i] = (data[i] >> bits) | (data[i+1] << (32-i));
	res.data[31] = data[31] >> bits;
	return res;
}

uint1024	uint1024::lshift(int bits)
{
	uint1024 res;
	int index = bits>>5;
	int shift = bits & 0x1F;
	for(int i=0;i<32;i++)
	{
		uint32 src1 = i - index, src2 = i - index - 1;
		uint32 val1 = src1 >= 32 ? 0 : data[src1], val2 = src2 >= 32 ? 0 : data[src2]; // größer weil uint
		res.data[i] = (val1 << shift) | (val2 >> (32-shift));
	}
	return res;
}
uint1024	uint1024::rshift(int bits)
{
	uint1024 res;
	int index = bits>>5;
	int shift = bits & 0x1F;
	for(int i=0;i<32;i++)
	{
		uint32 src1 = i + index, src2 = i + index + 1;
		uint32 val1 = src1 >= 32 ? 0 : data[src1], val2 = src2 >= 32 ? 0 : data[src2];
		res.data[i] = (val1 >> shift) | (val2 << (32-shift));
	}
	return res;
}

int			uint1024::bitcount()
{
	int offset, diff, stop, bitoffset;
	uint32 mask;

	offset = 0;
	for(int i=16;i<32;i++)
	{
		offset = data[i] ? 16 : offset;
	}
	
	diff = 0;
	stop = offset + 16;
	for(int i=offset+8; i < stop; i++)
	{
		diff = data[i] ? 8 : diff;
	}
	offset += diff;

	diff = 0;
	stop = offset + 8;
	for(int i=offset+4; i < stop; i++)
	{
		diff = data[i] ? 4 : diff;
	}
	offset += diff;

	diff = 0;
	stop = offset + 4;
	for(int i=offset+2; i < stop; i++)
	{
		diff = data[i] ? 2 : diff;
	}
	offset += diff;

	offset += (data[offset+1] != 0) ? 1 : 0;

	bitoffset = 0;
	mask = 0xFFFF0000;
	bitoffset = (data[offset] & mask) ? 16 : 0;

	mask = 0xFF00 << bitoffset;
	bitoffset += (data[offset] & mask) ? 8 : 0;

	mask = 0xF0 << bitoffset;
	bitoffset += (data[offset] & mask) ? 4 : 0;

	mask = 0xC << bitoffset;
	bitoffset += (data[offset] & mask) ? 2 : 0;

	mask = 0x2 << bitoffset;
	bitoffset += (data[offset] & mask) ? 1 : 0;
	
	return (offset<<5) + bitoffset;
}

int 		uint1024::isZero()
{
	int res = 1;
	for(int i=0;i<32;i++)
		res = (data[i] == 0) ? res : 0;
	return res;
}

uint1024	uint1024::add(uint1024 var)
{
	uint32 carry=0;
	uint1024 res;

	for(int i=0;i<32;i++)
	{
		res.data[i] = data[i] + var.data[i] + carry;
		carry = res.data[i] < data[i] ? 1 : 0;
	}

	return res;
}

uint1024	uint1024::sub(uint1024 var)
{
	uint32 carry=0;
	uint1024 res;

	for(int i=0;i<32;i++)
	{
		res.data[i] = data[i] - var.data[i] - carry;
		carry = res.data[i] > data[i] ? 1 : 0;
	}

	return res;
}

int			uint1024::cmp(uint1024 var)
{
	int result = 0, tmp;
	for(int i=31;i>=0;i--)
	{
		tmp = ((data[i]>var.data[i]) ? 1 : 0) | ((data[i]<var.data[i]) ? -1 : 0);
		result = result ? result : tmp;		
	}

	return result;
}

uint1024	uint1024::div(uint1024 d, uint1024 *rem)
{
	int bitDiff = bitcount() - d.bitcount();
	bitDiff = bitDiff < 0 ? 0 : bitDiff;

	uint1024 D = d.lshift(bitDiff);
	*rem = *this;
	uint1024 dv((uint32)0);

	for(int i=0;i<1024;i++)
	{
		int actDiff = bitDiff - i;

		int cmp = rem->cmp(D);

		uint1024 tmp = dv.lshift_fast(1);
		tmp.data[0] |= (cmp >= 0) ? 1 : 0;
		dv = (actDiff >= 0) ? tmp : dv;

		tmp = rem->sub(D);
		*rem = (cmp >= 0 && actDiff >= 0) ? tmp : *rem;
		
		tmp = D.rshift_fast(1);
		D = (actDiff >= 0) ? tmp : D;
	}

	return dv;
}

uint1024	uint1024::mul(uint1024 d)
{
	uint1024 addSum((uint32)0);
	uint1024 me = *this;

	for(int i=0;i<1024;i++)
	{
		uint1024 tmp((uint32)0);
		tmp = (me.data[0] & 0x01) ? d : tmp;
		addSum = addSum.add(tmp);
		me = me.rshift_fast(1);
		d = d.lshift_fast(1);
	}
	return addSum;
}

uint1024	uint1024::powm(uint1024 exp, uint1024 mod)
{
	uint1024 x((uint32)1);
	uint1024 y = *this;

	for(int i=0;i<1024;i++)
	{		
		uint1024 a,b;

		a = x.mul(y);
		a.div(mod, &b);

		x = (exp.data[0] & 0x01) ? b : x;

		a = y.mul(y);
		a.div(mod, &y);

		exp = exp.rshift_fast(1);
	}

	x.div(mod, &y);
	return y;
}

uint32 uint1024::fastmod(uint32 d)
{
	uint32 x=0,y=0,z=0;

	for(int i=31;i>=0;i--)
	{
		x = data[i];
		y = (y << 16) | (x >> 16);
		z = y / d;
		y -= z*d;
		x <<= 16;
		y = (y << 16) | (x >> 16);
		z = y / d;
		y -= z*d;
	}

	return y;

}
