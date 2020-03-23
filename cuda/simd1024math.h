
#include <stdint.h>

typedef uint32_t simd1024;

__device__ void simd1024_leftshift(simd1024 &r, const simd1024 &a, const int amount)
{
	int laneId = threadIdx.x & (warpSize-1);
	int words = amount>>5;
	if (words > 0) {
		r = (uint32_t)__shfl_up((int)a, words);
		r = (laneId >= words) ? r : 0;
	} else r = a;
	int bits = amount&31;
	if (bits > 0) {
		uint32_t b = (uint32_t)__shfl_up((int)r, 1);
		b = (laneId == words) ? 0 : b; 
		r = (r << bits) | (b >> (32-bits));
	}
}

__device__ void simd1024_lshift1(simd1024 &r, simd1024 a)
{
	int laneId = threadIdx.x & (warpSize-1); 
	uint32_t b = (uint32_t)__shfl_up((int)a, 1);
	b = (laneId == 0) ? 0 : b; 
	r = (a << 1) | (b >> (31));
}

__device__ void simd1024_rshift1(simd1024 &r, simd1024 a)
{
	int laneId = threadIdx.x & (warpSize-1); 
	uint32_t b = (uint32_t)__shfl_down((int)a, 1);
	b = (laneId == 31) ? 0 : b; 
	r = (a >> 1) | (b << (31));
}

__device__ bool simd1024_add(simd1024 &r, simd1024 a, const simd1024 &b)
{
	int laneId = threadIdx.x & (warpSize-1);
	r = a + b;
	bool carry, carry_hi = false;
	do {
		carry = r < a;
		carry_hi |= __shfl((int)carry, 31);
		carry = (uint32_t)__shfl_up((int)carry, 1) && laneId > 0;
		a = r; r += carry;
	} while(__any(carry));

	return carry_hi;
}

__device__ bool simd1024_add_ui(simd1024 &r, simd1024 &a, uint64_t ui)
{
	int laneId = threadIdx.x & (warpSize-1);

	uint32_t lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(ui));
	lo = (uint32_t)__shfl((int)lo,0); hi = (uint32_t)__shfl((int)hi,0);
	lo = laneId == 0 ? lo : 0;
	lo = laneId == 1 ? hi : lo;
	
	return simd1024_add(r, a, lo);
}

__device__ bool simd1024_addc(simd1024 &r, simd1024 a, const simd1024 &b, bool carry_in)
{
	bool carry, carry_hi = false;
	int laneId = threadIdx.x & (warpSize-1);
	r = a + b;
	do {
		carry = r < a;
		carry_hi |= __shfl((int)carry, 31);
		carry = (uint32_t)__shfl_up((int)carry, 1) && laneId > 0;
		a = r; r += carry;
	} while(__any(carry));

	if (carry_in)
		carry_hi |= simd1024_add_ui(r, r, 1);

	return carry_hi;
}

__device__ void simd1024_sub(simd1024 &r, simd1024 a, const simd1024 &b)
{
	int laneId = threadIdx.x & (warpSize-1);
	r = a - b;
	bool carry;
	do {
		carry = r > a;
		carry = (uint32_t)__shfl_up((int)carry, 1) && laneId > 0;
		a = r; r -= carry;
	} while(__any(carry));
}

__device__ void simd1024_sub_ui(simd1024 &r, simd1024 &a, uint64_t ui)
{
	int laneId = threadIdx.x & (warpSize-1);

	uint32_t lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(ui));
	lo = (uint32_t)__shfl((int)lo,0); hi = (uint32_t)__shfl((int)hi,0);
	lo = laneId == 0 ? lo : 0;
	lo = laneId == 1 ? hi : lo;
	
	simd1024_sub(r, a, lo);
}

__device__ int simd1024_bitcount(const simd1024 &a)
{
	int laneId = threadIdx.x & (warpSize-1);
	int bits = (32 - __clz(a));
	int bitCount = (bits > 0) ? ((laneId<<5) + bits) : 0;

#pragma unroll 5
	for (int mask = 16; mask > 0; mask = mask >> 1) 
		bitCount = max(__shfl_xor((int)bitCount, mask), bitCount);

	bitCount = __shfl((int)bitCount, 31);
	return bitCount;
}

__device__ void simd1024_mul(simd1024 &r, const simd1024 &a, const simd1024 &b)
{
	int laneId = threadIdx.x & (warpSize-1);

	r = 0;
	for (int limb=31; limb >= 0; --limb)
	{
		simd1024 factor = (simd1024)__shfl((int)a,limb);
		simd1024_add(r, r, __umulhi(factor, b));
		r = (simd1024)__shfl_up((int)r, 1);
		r = (laneId==0) ? 0 : r;
		simd1024_add(r, r, factor*b);
	}
}

__device__ void simd1024_mul_full(simd1024 &r_lo, simd1024 &r_hi, const simd1024 &a, const simd1024 &b)
{
	int laneId = threadIdx.x & (warpSize-1);

	r_lo = 0;
	r_hi = 0;
	for (int limb=31; limb >= 0; --limb)
	{
		simd1024 factor = (simd1024)__shfl((int)a,limb);
		if (simd1024_add(r_lo, r_lo, __umulhi(factor, b)))
			simd1024_add_ui(r_hi, r_hi, 1);
		simd1024 carrylimb = (simd1024)__shfl((int)r_lo, 31);
		r_hi = (simd1024)__shfl_up((int)r_hi, 1);
		r_hi = (laneId==0) ? carrylimb : r_hi;
		r_lo = (simd1024)__shfl_up((int)r_lo, 1);
		r_lo = (laneId==0) ? 0 : r_lo;
		if (simd1024_add(r_lo, r_lo, factor*b))
			simd1024_add_ui(r_hi, r_hi, 1);
	}
}

__device__ int simd1024_compare(const simd1024 &a, const simd1024 &b)
{
	int laneId = (threadIdx.x & (warpSize-1));
	
	int bitVal = 1<<(laneId & 0x0F);
	int bits = (a > b) ? bitVal : ((a < b) ? -bitVal : 0);

#pragma unroll 4
	for (int mask = 8; mask > 0; mask = mask >> 1) 
		bits += __shfl_xor((int)bits, mask, 16);

	int p31 = __shfl((int)bits, 31);

	return (p31 == 0) ? __shfl((int)bits, 15) : p31;
}

__device__ int simd1024_compare_ui(const simd1024 &a, uint64_t ui)
{
	int laneId = threadIdx.x & (warpSize-1);

	uint32_t lo, hi;
	asm volatile("mov.b64 {%0,%1}, %2;":"=r"(lo),"=r"(hi):"l"(ui));
	lo = (uint32_t)__shfl((int)lo,0); hi = (uint32_t)__shfl((int)hi,0);
	lo = laneId == 0 ? lo : 0;
	lo = laneId == 1 ? hi : lo;
	
	return simd1024_compare(a, lo);
}

__device__ void simd1024_div(simd1024 &r, simd1024 a, simd1024 d, simd1024 &rem)
{
	int laneId = threadIdx.x & (warpSize-1);

	if(__all(d == 0))
	{
		r = 0;
		rem = 0;
		return;
	}

	if(simd1024_compare(a, d) < 0)
	{
		rem = a;
		r = 0;
		return;
	}

	int nBitsA, nBitsD, bitDiff;
	nBitsA = simd1024_bitcount(a);
	nBitsD = simd1024_bitcount(d);
	bitDiff = nBitsA - nBitsD;

	rem = a;
	r = 0;

	simd1024_leftshift(d, d, bitDiff);

	for(int i=bitDiff; i>=0;i--)
	{
		simd1024_lshift1(r, r);

		int cmp = simd1024_compare(rem, d);
		if(cmp >= 0)
		{
			if(laneId == 0) r |= 1;
			simd1024_sub(rem, rem, d);
		}
		simd1024_rshift1(d, d);
	}
}

__device__ void simd1024_gcd(const simd1024 &mod, simd1024 &u, simd1024 &v)
{
    int laneId = threadIdx.x & (warpSize-1);
    
    simd1024 alpha, beta;
    
    u = (laneId > 0) ? 0 : 1;
    v = 0;
    beta = mod;
    alpha = (laneId == 31) ? 0x80000000 : 0;
    
    for(int i=0;i<1024;i++)
    {
        bool isEven = __shfl((int)u & 1, 0) == 0;
        if(isEven)
        {
            simd1024_rshift1(u,u); simd1024_rshift1(v,v);
        }else
        {
            simd1024 tmp;
            simd1024_rshift1(tmp, u ^ beta);
            simd1024_add(u, u & beta, tmp);
            
            simd1024_rshift1(tmp, v);
            simd1024_add(v, alpha, tmp);
        }
    }
}

__device__ simd1024 simd1024_calcBarExp2(const uint32_t &exp, const simd1024 &n)
{
    int count = simd1024_bitcount(n);
    simd1024 nTmp;
    simd1024_leftshift(nTmp, n, 1024-count);
    simd1024 a;
    simd1024_sub(a, 0, nTmp);
    
    while(simd1024_compare(a, n) >= 0)
    {
        simd1024_rshift1(nTmp, nTmp);
        if(simd1024_compare(a, nTmp) >= 0)
            simd1024_sub(a, a, nTmp);
    }
    
    for(int i=0;i<exp;i++)
    {
        simd1024_lshift1(a, a);
        if(simd1024_compare(a, n) >= 0)
            simd1024_sub(a, a, n);
    }
    
    return a;
}

__device__ simd1024 simd1024_montmul1(const simd1024 &abar, const simd1024 &m, const simd1024 &mInverse)
{
    simd1024 tm, u;
    simd1024 tmm_lo, tmm_hi;
    
    simd1024_mul(tm, abar, mInverse);
    simd1024_mul_full(tmm_lo, tmm_hi, tm, m);
    
    bool carry = simd1024_add(u, abar, tmm_lo);
    bool carry_hi = false;
    if(carry)
        carry_hi = simd1024_add_ui(u, tmm_hi, 1);
    else
        u = tmm_hi;
    if(carry_hi || simd1024_compare(u, m) >= 0) simd1024_sub(u, u, m);
    return u;
}

__device__ simd1024 simd1024_montmul(const simd1024 &abar, const simd1024 &bbar, const simd1024 &m, const simd1024 &mInverse)
{
    simd1024 tm, u;
    simd1024 t_lo, t_hi;
    simd1024 tmm_lo, tmm_hi;
    
    simd1024_mul_full(t_lo, t_hi, abar, bbar);
    simd1024_mul(tm, t_lo, mInverse);
    simd1024_mul_full(tmm_lo, tmm_hi, tm, m);
    
    bool carry = simd1024_add(u, t_lo, tmm_lo);
    bool carry_hi = simd1024_addc(u, t_hi, tmm_hi, carry);
    if(carry_hi || simd1024_compare(u, m) >= 0) simd1024_sub(u, u, m);
    return u;
}

__device__ void simd1024_powm2(simd1024 &r, simd1024 exp, const simd1024 &mod)
{
    simd1024 pInverse, modInverse;
    
    simd1024_gcd(mod, pInverse, modInverse);
    
    simd1024 M, x;
    M = simd1024_calcBarExp2(1, mod);
    x = simd1024_calcBarExp2(0, mod);

    const int bits = 5;
    const int ncombos = (1U<<bits);
    simd1024 factor[1U<<bits];
    factor[0] = x;

    for (int bit = bits-1, nbranches=1; bit >= 0; bit--, nbranches*=2) {
        for (unsigned int branch = 0; branch < nbranches; branch++) {
            uint32_t combo = branch << (bit+1);
            if (combo != 0)
                factor[combo] = simd1024_montmul(factor[combo], factor[combo], mod, modInverse);
            factor[combo | (1U<<bit)] = simd1024_montmul(M, factor[combo], mod, modInverse);
        }
    }
    
    bool flag = false;
    for(int i=1024; i>0; i-=bits)
    {
        if (flag)
            for (int j=0; j<bits; j++)
                x = simd1024_montmul(x, x, mod, modInverse);
        unsigned int combo = ((uint32_t)__shfl((int)(exp >> (32-bits)), 31));
        if (combo > 0) {
            x = simd1024_montmul(factor[combo], x, mod, modInverse);
            flag = true;
        }

        simd1024_leftshift(exp, exp, min(bits,i));
    }

    r = simd1024_montmul1(x, mod, modInverse);
}
