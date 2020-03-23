#include <ctype.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <cuda.h>
#include "cuda_runtime.h"

#include <stdint.h>
#include <map>
#include <sys/time.h>

#include "cuda.h"

#if _WIN32
#include <Winsock2.h> // for struct timeval
#endif

static __device__ unsigned long long MAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
	#if __CUDA_ARCH__ >= 130
		return __double_as_longlong(__hiloint2double(HI, LO));
	#else
		return (unsigned long long)LO | (((unsigned long long)HI) << 32);
	#endif
}

template<unsigned int sharedSizeKB, unsigned int nThreadsPerBlock, unsigned int nOffsetsA>
__global__ void primesieve_kernelA(unsigned char *g_bit_array_sieve, unsigned int nBitArray_Size, unsigned int nOffsets, uint4 *primes, unsigned int *base_remainders, uint64_t base_offset, uint32_t base_index, unsigned int nPrimorialEndPrime, unsigned int nPrimeLimitA);

__device__ __forceinline__
unsigned mod_p_small(uint64_t a, unsigned p, uint64_t recip) {
	uint64_t q = __umul64hi((uint64_t)a, recip);
	int64_t r = a - p*q;
	if (r >= p) { r -= p; }
	return (unsigned int)r;
}

#define checkCudaErrors(x) \
{ \
    cudaGetLastError(); \
    x; \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) \
    { \
        fprintf(stderr, "GPU #%d: cudaError %d (%s) calling '%s' (%s line %d)\n", device_map[thr_id], err, cudaGetErrorString(err), #x, __FILE__, __LINE__); \
    } \
}

int device_map[8] = {0,1,2,3,4,5,6,7};

extern "C" void cuda_reset_device()
{
	cudaDeviceReset();
}

extern "C" int cuda_num_devices()
{
    int version;
    cudaError_t err = cudaDriverGetVersion(&version);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Unable to query CUDA driver version! Is an nVidia driver installed?");
        exit(1);
    }

    int maj = version / 1000, min = version % 100; // same as in deviceQuery sample
    if (maj < 5 || (maj == 5 && min < 5))
    {
        fprintf(stderr, "Driver does not support CUDA %d.%d API! Update your nVidia driver!", 5, 5);
        exit(1);
    }

    int GPU_N;
    err = cudaGetDeviceCount(&GPU_N);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Unable to query number of CUDA devices! Is an nVidia driver installed?");
        exit(1);
    }
    return GPU_N;
}

static bool substringsearch(const char *haystack, const char *needle, int &match)
{
    int hlen = strlen(haystack);
    int nlen = strlen(needle);
    for (int i=0; i < hlen; ++i)
    {
        if (haystack[i] == ' ') continue;
        int j=0, x = 0;
        while(j < nlen)
        {
            if (haystack[i+x] == ' ') {++x; continue;}
            if (needle[j] == ' ') {++j; continue;}
            if (needle[j] == '#') return ++match == needle[j+1]-'0';
            if (tolower(haystack[i+x]) != tolower(needle[j])) break;
            ++j; ++x;
        }
        if (j == nlen) return true;
    }
    return false;
}

extern "C" int cuda_finddevice(char *name)
{
    int num = cuda_num_devices();
    int match = 0;
    for (int i=0; i < num; ++i)
    {
        cudaDeviceProp props;
        if (cudaGetDeviceProperties(&props, i) == cudaSuccess)
            if (substringsearch(props.name, name, match)) return i;
    }
    return -1;
}

extern "C" const char* cuda_devicename(int index)
{
	const char *result = NULL;
	cudaDeviceProp props;
	if (cudaGetDeviceProperties(&props, index) == cudaSuccess)
		result = strdup(props.name);

	return result;
}

typedef struct { double value[8]; } tsumarray;
cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id)
{
    cudaError_t result = cudaSuccess;
    if (situation >= 0)
    {   
        static std::map<int, tsumarray> tsum;

        double a = 0.95, b = 0.05;
        if (tsum.find(situation) == tsum.end()) { a = 0.5; b = 0.5; }

        double tsync = 0.0;
        double tsleep = 0.95 * tsum[situation].value[thr_id];
        if (cudaStreamQuery(stream) == cudaErrorNotReady)
        {
            usleep((useconds_t)(1e6*tsleep));
            struct timeval tv_start, tv_end;
            gettimeofday(&tv_start, NULL);
            result = cudaStreamSynchronize(stream);
            gettimeofday(&tv_end, NULL);
            tsync = 1e-6 * (tv_end.tv_usec-tv_start.tv_usec) + (tv_end.tv_sec-tv_start.tv_sec);
        }
        if (tsync >= 0) tsum[situation].value[thr_id] = a * tsum[situation].value[thr_id] + b * (tsleep+tsync);
    }
    else
        result = cudaStreamSynchronize(stream);
    return result;
}

extern "C" bool cuda_init(int thr_id)
{
    static bool init[8] = {0,0,0,0,0,0,0,0};
    bool result = init[thr_id];

    if (!init[thr_id])
    {
        fprintf(stderr, "thread %d maps to CUDA device #%d\n", thr_id, device_map[thr_id]);

        CUcontext ctx;
        cuCtxCreate( &ctx, CU_CTX_SCHED_AUTO, device_map[thr_id] );
        cuCtxSetCurrent(ctx);

        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//        cudaFuncSetCacheConfig(primesieve_kernelA, cudaFuncCachePreferShared);

        init[thr_id] = true;
    }

    return result;
}

#include "simd1024math.h"

unsigned int *h_primes;
__constant__ uint64_t c_zTempVar[17];
__constant__ simd1024 c_zFirstSieveElement[32];
__constant__ int c_offsetsB[32];
__constant__ uint16_t c_primes[512];
__constant__ uint16_t c_blockoffset_mod_p[62][512];
uint4 *d_primesInverseInvk[8];
//uint32_t *d_inverses[8];
uint32_t *d_base_remainders[8];
uint16_t *d_prime_remainders[8];
uint64_t *d_origins[8];
uint32_t *d_nonce_offsets[8];
unsigned char *d_bit_array_sieve[8] = {0,0,0,0,0,0,0,0};

extern "C" void cuda_set_zTempVar(unsigned int thr_id, const uint64_t *limbs)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_zTempVar, limbs, 17*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

extern "C" void cuda_set_zFirstSieveElement(unsigned int thr_id, const uint64_t *limbs)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_zFirstSieveElement, limbs, 32*sizeof(simd1024), 0, cudaMemcpyHostToDevice));
}

extern "C" void cuda_set_primes(unsigned int thr_id, unsigned int *primes, unsigned int *inverses, uint64_t *invk, unsigned int nPrimeLimit, unsigned int nBitArray_Size, unsigned int nOrigins)
{
	h_primes = primes;

	uint32_t *lst = (uint32_t*)malloc(sizeof(uint32_t) * 4 * nPrimeLimit);
	checkCudaErrors(cudaMalloc(&d_primesInverseInvk[thr_id],  nPrimeLimit*sizeof(uint32_t)*4));
	for (int i = 0;i < nPrimeLimit;i++)
	{
		memcpy(&lst[i * 4 + 0], &primes[i], sizeof(uint32_t));
		memcpy(&lst[i * 4 + 1], &inverses[i], sizeof(uint32_t));
		memcpy(&lst[i * 4 + 2], &invk[i], sizeof(uint64_t));
	}
	checkCudaErrors(cudaMemcpy(d_primesInverseInvk[thr_id], lst, sizeof(uint32_t) * 4 * nPrimeLimit, cudaMemcpyHostToDevice));
	free(lst);
	
    checkCudaErrors(cudaMalloc(&d_base_remainders[thr_id],  nPrimeLimit * sizeof(uint32_t)));
	checkCudaErrors(cudaMalloc(&d_prime_remainders[thr_id], 512 * nOrigins * sizeof(uint16_t) * 16));
	checkCudaErrors(cudaMalloc(&d_origins[thr_id], nOrigins * sizeof(uint64_t)));

    checkCudaErrors(cudaMalloc(&d_nonce_offsets[thr_id], (nBitArray_Size>>3) * sizeof(uint32_t)));
}

__device__ unsigned int mpi_mod_int(uint64_t *A, unsigned int B, uint64_t recip)
{
	if (B == 1)
		return 0;
	else if (B == 2)
		return A[0]&1;

	#define biH (sizeof(uint64_t)<<2)
	int i;
	uint64_t x,y;

#pragma unroll 16
	for( i = 17 - 1, y = 0; i > 0; i-- )
	{
		x  = A[i-1];
		y = (y << 32) | (x >> 32);
		y = mod_p_small(y, B, recip);
		
		x <<= 32;		
		y = (y << 32) | (x >> 32);
		y = mod_p_small(y, B, recip);
	}

	return (unsigned int)y;
}

__global__ void base_remainders_kernel(uint4 *g_primes, uint32_t *g_base_remainders, unsigned int nPrimorialEndPrime, unsigned int nPrimeLimit)
{
	unsigned int i = nPrimorialEndPrime + blockDim.x * blockIdx.x + threadIdx.x;
	if (i < nPrimeLimit)
	{
		uint4 tmp = g_primes[i];
		uint64_t rec = MAKE_ULONGLONG(tmp.z, tmp.w);
		g_base_remainders[i] = mpi_mod_int(c_zTempVar, tmp.x, rec);
	}
}

extern "C" void cuda_compute_base_remainders(unsigned int thr_id, unsigned int *base_remainders, unsigned int nPrimorialEndPrime, unsigned int nPrimeLimit)
{
    int nThreads = nPrimeLimit - nPrimorialEndPrime;
    int nThreadsPerBlock = 256;
    int nBlocks = (nThreads + nThreadsPerBlock-1) / nThreadsPerBlock;

    dim3 block(nThreadsPerBlock);
    dim3 grid(nBlocks);

    base_remainders_kernel<<<grid, block, 0>>>(d_primesInverseInvk[thr_id], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimit);

    MyStreamSynchronize(NULL, 0, thr_id);

    cudaError_t err;
    err = cudaGetLastError(); if (err != 0)
        fprintf(stderr, "Thread %d (GPU #%d) base_remainders_kernel CUDA error %d: %s\n", thr_id, device_map[thr_id], err,cudaGetErrorString(err));

    checkCudaErrors(cudaMemcpy(base_remainders, d_base_remainders[thr_id], nPrimeLimit*sizeof(uint32_t), cudaMemcpyDeviceToHost));

    for(unsigned int i=0; i<nPrimorialEndPrime; i++) base_remainders[i] = 0;
}

template<int Begin, int End, int Step = 1>
struct Unroller {
    template<typename Action>
    __device__ __forceinline__ static void step(Action& action) {
        action(Begin);
        Unroller<Begin+Step, End, Step>::step(action);
    }
};

template<int End, int Step>
struct Unroller<End, End, Step> {
    template<typename Action>
    __device__ __forceinline__ static void step(Action& action) {
    }
};


__global__ void primesieve_kernelA0(int thr_id, uint64_t *origins, unsigned int sharedSizeKB, uint4 *primes, unsigned int *base_remainders, uint16_t *prime_remainders, unsigned int nPrimorialEndPrime, unsigned int nPrimeLimitA)
{	
	unsigned int position = (blockIdx.x << 9) + threadIdx.x;
	uint32_t memory_position = position << 4;
	uint4 tmp = primes[threadIdx.x];
	uint64_t rec = MAKE_ULONGLONG(tmp.z, tmp.w);
	uint32_t a, c;

    // note that this kernel currently hardcodes 9 specific offsets
	a = mod_p_small(origins[blockIdx.x] +
		base_remainders[threadIdx.x], tmp.x, rec);
	c = mod_p_small((uint64_t)(tmp.x - a)*tmp.y, tmp.x, rec);
	prime_remainders[memory_position] = c;

	a += 6; if(a >= tmp.x) a -= tmp.x;	
	c = mod_p_small((uint64_t)(tmp.x - a)*tmp.y, tmp.x, rec);
	prime_remainders[memory_position + 1] = c;

	a += 2; if(a >= tmp.x) a -= tmp.x;	
	c = mod_p_small((uint64_t)(tmp.x - a)*tmp.y, tmp.x, rec);
	prime_remainders[memory_position + 2] = c;

	a += 4; if(a >= tmp.x) a -= tmp.x;	
	c = mod_p_small((uint64_t)(tmp.x - a)*tmp.y, tmp.x, rec);
	prime_remainders[memory_position + 3] = c;

	a += 6; if(a >= tmp.x) a -= tmp.x;	
	c = mod_p_small((uint64_t)(tmp.x - a)*tmp.y, tmp.x, rec);
	prime_remainders[memory_position + 4] = c;

	a += 2; if(a >= tmp.x) a -= tmp.x;	
	c = mod_p_small((uint64_t)(tmp.x - a)*tmp.y, tmp.x, rec);
	prime_remainders[memory_position + 5] = c;

	a += 6; if(a >= tmp.x) a -= tmp.x;	
	c = mod_p_small((uint64_t)(tmp.x - a)*tmp.y, tmp.x, rec);
	prime_remainders[memory_position + 6] = c;

	a += 4; if(a >= tmp.x) a -= tmp.x;	
	c = mod_p_small((uint64_t)(tmp.x - a)*tmp.y, tmp.x, rec);
	prime_remainders[memory_position + 7] = c;

	a += 2; if(a >= tmp.x) a -= tmp.x;	
	c = mod_p_small((uint64_t)(tmp.x - a)*tmp.y, tmp.x, rec);
	prime_remainders[memory_position + 8] = c;
}

extern "C" void cuda_set_origins(unsigned int thr_id, uint64_t *origins, unsigned int nOrigins, unsigned int sharedSizeKB, unsigned int nPrimorialEndPrime, unsigned int nPrimeLimitA)
{
	checkCudaErrors(cudaMemcpy(d_origins[thr_id], origins, nOrigins*sizeof(uint64_t), cudaMemcpyHostToDevice));

	dim3 block(nPrimeLimitA);
	dim3 grid(nOrigins);
	primesieve_kernelA0<<<grid, block, 0>>>(thr_id, d_origins[thr_id], sharedSizeKB, d_primesInverseInvk[thr_id], d_base_remainders[thr_id], d_prime_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);

	MyStreamSynchronize(NULL, 10, thr_id);
	cudaError_t err = cudaGetLastError(); if (err != 0)
		fprintf(stderr, "Thread %d (GPU #%d) primesieve_kernelA0 CUDA error %d: %s\n", thr_id, device_map[thr_id], err, cudaGetErrorString(err));
}

template<unsigned int sharedSizeKB, unsigned int nThreadsPerBlock, unsigned int nOffsetsA>
__global__ void primesieve_kernelA(unsigned char *g_bit_array_sieve, unsigned int nBitArray_Size, uint4 *primes, uint16_t *prime_remainders, unsigned int *base_remainders, uint64_t base_offset, uint32_t base_index, unsigned int nPrimorialEndPrime, unsigned int nPrimeLimitA)
{
	extern __shared__ uint32_t shared_array_sieve[];

	unsigned int sizeSieve = sharedSizeKB*1024*8;

	if (sharedSizeKB == 32 && nThreadsPerBlock == 1024) {
#pragma unroll 8
		for (int i=0; i <  8; i++) shared_array_sieve[threadIdx.x+i*1024] = 0;
	}
	else if (sharedSizeKB == 48 && nThreadsPerBlock == 768) {
#pragma unroll 16
		for (int i=0; i < 16; i++) shared_array_sieve[threadIdx.x+i*768] = 0;
	}
	else if (sharedSizeKB == 32 && nThreadsPerBlock == 512) {
#pragma unroll 16
		for (int i=0; i < 16; i++) shared_array_sieve[threadIdx.x+i*512] = 0;
	}
	__syncthreads();
		
	for (int i = nPrimorialEndPrime; i < nPrimeLimitA; i++)
	{
		uint32_t pr = c_primes[i];
		uint32_t pre2 = c_blockoffset_mod_p[blockIdx.x][i];

		// precompute pIdx, nAdd
		unsigned int pIdx = threadIdx.x * pr;
		unsigned int nAdd;
		if (nThreadsPerBlock == 1024)     nAdd = pr << 10;
		else if (nThreadsPerBlock == 768) nAdd = (pr << 9) + (pr << 8);
		else if (nThreadsPerBlock == 512) nAdd = pr << 9;
		else                              nAdd = pr * nThreadsPerBlock;

		uint32_t pre1[nOffsetsA];
		auto pre = [&pre1, &prime_remainders, &base_index, &i](unsigned int o){
			pre1[o] = prime_remainders[(((base_index << 9) + i) << 4) + o]; // << 4 because we have space for 16 offsets
		};
		Unroller<0, nOffsetsA>::step(pre);

		auto loop = [&sizeSieve, &pIdx, &nAdd, &prime_remainders, &base_index, &i, &pre1, &pre2, &pr](unsigned int o){
			uint32_t tmp = pre1[o] + pre2;
			tmp = (tmp >= pr) ? (tmp - pr) : tmp;
			for(unsigned int index = tmp + pIdx;index < sizeSieve;index+= nAdd)				
				atomicOr(&shared_array_sieve[index >> 5], 1 << (index & 31));
		};
		Unroller<0, nOffsetsA>::step(loop);
	}

	__syncthreads();
	g_bit_array_sieve += sharedSizeKB*1024 * blockIdx.x;

	if (sharedSizeKB == 32 && nThreadsPerBlock == 1024) {
#pragma unroll 8
		for (int i = 0; i < 8192; i+=1024) // fixed value
			((uint32_t*)g_bit_array_sieve)[threadIdx.x+i] = shared_array_sieve[threadIdx.x+i];
	}
	else if (sharedSizeKB == 48 && nThreadsPerBlock == 768) {
#pragma unroll 16
		for (int i = 0; i < 12288; i+=768) // fixed value
			((uint32_t*)g_bit_array_sieve)[threadIdx.x+i] = shared_array_sieve[threadIdx.x+i];
	}
	else if (sharedSizeKB == 32 && nThreadsPerBlock == 512) {
#pragma unroll 16
		for (int i = 0; i < 8192; i += 512) // fixed value
			((uint32_t*)g_bit_array_sieve)[threadIdx.x+i] = shared_array_sieve[threadIdx.x+i];
	}
}

__global__ void primesieve_kernelB(unsigned char *g_bit_array_sieve, unsigned int nBitArray_Size, unsigned int nOffsets, uint4 *primes, unsigned int *base_remainders, uint64_t base_offset, unsigned int nPrimorialEndPrime, unsigned int nPrimeLimit)
{
	unsigned int position = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int i = nPrimorialEndPrime + (position >> 3);
	unsigned int o = position & 0x7;	
	if ( i < nPrimeLimit && o < nOffsets)
	{
		uint32_t *g_word_array_sieve = (uint32_t*)g_bit_array_sieve;
		uint4 tmp = primes[i];

		unsigned int p = tmp.x;
		unsigned int inv = tmp.y;
		uint64_t recip = MAKE_ULONGLONG(tmp.z, tmp.w);
		unsigned int remainder = mod_p_small(base_offset + base_remainders[i] + c_offsetsB[o], p, recip);
		unsigned int index = mod_p_small((uint64_t)(p - remainder)*inv, p, recip);

		for (; index < nBitArray_Size; index += p)
		{
			atomicOr(&g_word_array_sieve[index >> 5], 1 << (index & 31));
		}
	}
}

const int nOffsetsA = 9;
const int nOffsetsB = 6;

void kernelA_launcher(unsigned int thr_id, unsigned int sharedSizeKB, unsigned int nThreadsPerBlock, unsigned int nBitArray_Size, uint64_t base_offset, uint32_t base_index, unsigned int nPrimorialEndPrime, int shared_primes)
{
    const int sharedSizeBits = sharedSizeKB * 1024 * 8;

    if (d_bit_array_sieve[thr_id] == NULL)
    {        
        size_t allocSize = ((nBitArray_Size + sharedSizeBits - 1) / sharedSizeBits) * sharedSizeBits;
        checkCudaErrors(cudaMalloc(&d_bit_array_sieve[thr_id], (allocSize+31)/32 * sizeof(uint32_t)));
    }

	int nBlocks = (nBitArray_Size + sharedSizeBits-1) / sharedSizeBits;

	static bool init[8];
	if (!init[thr_id])
	{
		unsigned int primeLimitA = nPrimorialEndPrime + shared_primes;

		uint16_t primes[512];
		for (int i = 0; i < primeLimitA; i++)
			primes[i] = h_primes[i];
		cudaMemcpyToSymbol(c_primes, primes, primeLimitA * sizeof(uint16_t), 0, cudaMemcpyHostToDevice);
		
		for (int block = 0; block < nBlocks; block++)
		{
			unsigned int blockOffset = sharedSizeBits * block;
			uint16_t offsets[512];
			for (int i = 0; i < primeLimitA; i++)
				offsets[i] = h_primes[i] - (blockOffset % h_primes[i]);
			cudaMemcpyToSymbol(c_blockoffset_mod_p, offsets, primeLimitA * sizeof(uint16_t), block*512*sizeof(uint16_t), cudaMemcpyHostToDevice);
		}

		init[thr_id] = true;
	}

	dim3 block(nThreadsPerBlock);
	dim3 grid(nBlocks);

	if (sharedSizeKB == 32 && nThreadsPerBlock == 1024)
	{
		switch (nOffsetsA)
		{
			case 5: primesieve_kernelA<32, 1024, 5><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 6: primesieve_kernelA<32, 1024, 6><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 7: primesieve_kernelA<32, 1024, 7><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 8: primesieve_kernelA<32, 1024, 8><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 9: primesieve_kernelA<32, 1024, 9><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 10: primesieve_kernelA<32, 1024, 10><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 11: primesieve_kernelA<32, 1024, 11><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 12: primesieve_kernelA<32, 1024, 12><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			default: fprintf(stderr, "KernelA supports 5-12 sieving offsets at the moment!\n"); exit(1);
		}
	}
	else if (sharedSizeKB == 48 && nThreadsPerBlock == 768)
	{
		switch (nOffsetsA)
		{
			case 5: primesieve_kernelA<48, 768, 5><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 6: primesieve_kernelA<48, 768, 6><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 7: primesieve_kernelA<48, 768, 7><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 8: primesieve_kernelA<48, 768, 8><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 9: primesieve_kernelA<48, 768, 9><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 10: primesieve_kernelA<48, 768, 10><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 11: primesieve_kernelA<48, 768, 11><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			case 12: primesieve_kernelA<48, 768, 12><<<grid, block, sharedSizeBits/8>>>(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
			default: fprintf(stderr, "KernelA supports 5-12 sieving offsets at the moment!\n"); exit(1);
		}
	}
	else if (sharedSizeKB == 32 && nThreadsPerBlock == 512)
	{
		switch (nOffsetsA)
		{
		case 5: primesieve_kernelA<32, 512, 5> << <grid, block, sharedSizeBits / 8 >> >(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
		case 6: primesieve_kernelA<32, 512, 6> << <grid, block, sharedSizeBits / 8 >> >(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
		case 7: primesieve_kernelA<32, 512, 7> << <grid, block, sharedSizeBits / 8 >> >(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
		case 8: primesieve_kernelA<32, 512, 8> << <grid, block, sharedSizeBits / 8 >> >(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
		case 9: primesieve_kernelA<32, 512, 9> << <grid, block, sharedSizeBits / 8 >> >(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
		case 10: primesieve_kernelA<32, 512, 10> << <grid, block, sharedSizeBits / 8 >> >(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
		case 11: primesieve_kernelA<32, 512, 11> << <grid, block, sharedSizeBits / 8 >> >(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
		case 12: primesieve_kernelA<32, 512, 12> << <grid, block, sharedSizeBits / 8 >> >(d_bit_array_sieve[thr_id], nBitArray_Size, d_primesInverseInvk[thr_id], d_prime_remainders[thr_id], d_base_remainders[thr_id], base_offset, base_index, nPrimorialEndPrime, nPrimorialEndPrime + shared_primes); break;
		default: fprintf(stderr, "KernelA supports 5-12 sieving offsets at the moment!\n"); exit(1);
		}
	}
	else
	{
		fprintf(stderr, "Unsupported Shared Mem / Block size configuration for KernelA! Use 32kb, 1024 threads or 48kb, 768 threads!\n");
		exit(1);
	}
	cudaError_t err = cudaGetLastError(); if (err != 0)
		fprintf(stderr, "Thread %d (GPU #%d) primesieve_kernelA CUDA error %d: %s\n", thr_id, device_map[thr_id], err,cudaGetErrorString(err));
}


extern "C" void cuda_compute_primesieve(unsigned int thr_id, unsigned int nSharedSizeKB, unsigned int nThreadsKernelA, unsigned char *bit_array_sieve, unsigned int *base_remainders, uint64_t base_offset, uint32_t base_index, unsigned int nPrimorialEndPrime, unsigned int nPrimeLimitA, unsigned int nPrimeLimitB, unsigned int nBitArray_Size, unsigned int nDifficulty, int validationEnabled)
{
    cudaError_t err;
    
    // hardcoded sieving offsets for B kernel
	int no[nOffsetsB];
	no[0] = 0;
	no[1] = no[0] + 8;
	no[2] = no[1] + 4;
	no[3] = no[2] + 6;
	no[4] = no[3] + 2;
	no[5] = no[4] + 10;
    checkCudaErrors(cudaMemcpyToSymbol(c_offsetsB, no, nOffsetsB*sizeof(int), 0, cudaMemcpyHostToDevice));

    if (nPrimeLimitA > nPrimorialEndPrime)
		kernelA_launcher(thr_id, nSharedSizeKB, nThreadsKernelA, nBitArray_Size, base_offset, base_index, nPrimorialEndPrime, nPrimeLimitA - nPrimorialEndPrime);

    if (d_bit_array_sieve[thr_id] == NULL)
        checkCudaErrors(cudaMalloc(&d_bit_array_sieve[thr_id], (nBitArray_Size+31)/32 * sizeof(uint32_t)));
    {
		if (nOffsetsB > 8)
            exit(1);

        int nThreads = nPrimeLimitB - nPrimeLimitA;
        int nThreadsPerBlock = 32 * 8;        
        int nBlocks = (nThreads + nThreadsPerBlock - 1) / (nThreadsPerBlock / 8);

        if (nBlocks > 0)
        {
            dim3 block(nThreadsPerBlock);
            dim3 grid(nBlocks);

            primesieve_kernelB<<<grid, block, 0>>>(d_bit_array_sieve[thr_id], nBitArray_Size, nOffsetsB, d_primesInverseInvk[thr_id], d_base_remainders[thr_id], base_offset, nPrimeLimitA, nPrimeLimitB);
            err = cudaGetLastError(); if (err != 0) {
				fprintf(stderr, "Thread %d (GPU #%d) primesieve_kernelB CUDA error %d: %s\n", thr_id, device_map[thr_id], err,cudaGetErrorString(err));
				exit(1);
			}
                
        }
    }

	if(validationEnabled > 0)
		checkCudaErrors(cudaMemcpy(bit_array_sieve, d_bit_array_sieve[thr_id], nBitArray_Size/8, cudaMemcpyDeviceToHost));
}


__global__ void
	fermat_kernel(uint32_t *g_nonce_offsets, size_t numberOfCandidates, int constellation_offset, uint64_t nPrimorial)
{
	int laneId = threadIdx.x & (warpSize-1);
	int prime_index = blockIdx.x * (blockDim.x / warpSize) + (threadIdx.x / warpSize);
	if (prime_index < numberOfCandidates)
	{
		simd1024 number;
		simd1024_add_ui(number, c_zFirstSieveElement[laneId], nPrimorial * g_nonce_offsets[prime_index] + constellation_offset);

		simd1024 exponent;
		simd1024_sub_ui(exponent, number, 1);
	
		simd1024 result;
		simd1024_powm2(result, exponent, number);

		if (simd1024_compare_ui(result, 1))
		{
			g_nonce_offsets[prime_index] = 0xFFFFFFFF;
		}
	}
}

extern "C" void cuda_compute_fermat(unsigned int thr_id, uint32_t *nonce_offsets, size_t numberOfCandidates, int constellation_offset, uint64_t nPrimorial)
{
	int threads_per_block = 512;
	int primes_per_block = threads_per_block / 32;
	int num_blocks = (numberOfCandidates + primes_per_block-1) / primes_per_block;

	if (num_blocks > 0)
	{
		dim3 block(threads_per_block);
		dim3 grid(num_blocks);

		checkCudaErrors(cudaMemcpy(d_nonce_offsets[thr_id], nonce_offsets, numberOfCandidates * sizeof(uint32_t), cudaMemcpyHostToDevice));

		fermat_kernel << <grid, block, 0 >> > (d_nonce_offsets[thr_id], numberOfCandidates, constellation_offset, nPrimorial);
		MyStreamSynchronize(NULL, 3, thr_id);
		cudaError_t err = cudaGetLastError(); if (err != 0)
			fprintf(stderr, "Thread %d (GPU #%d) fermat_kernel CUDA error %d: %s\n", thr_id, device_map[thr_id], err, cudaGetErrorString(err));

		checkCudaErrors(cudaMemcpy(nonce_offsets, d_nonce_offsets[thr_id], numberOfCandidates * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	}
}
