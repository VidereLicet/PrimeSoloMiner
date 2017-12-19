#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_30_intrinsics.h"

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

#if _WIN32
#include <Winsock2.h> // for struct timeval
#endif

extern int device_map[8];


extern unsigned int nBitArray_Size[8];


static cudaDeviceProp props[16];

static uint32_t *d_tempBranch1Nonces[16];
static uint32_t *d_numValid[16];
static uint32_t *h_numValid[16];

extern unsigned char *d_bit_array_sieve[16];
uint32_t *d_nonces[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static uint32_t *d_partSum[2][16];

#define THREADS_PER_BLOCK 512

// This is based on the example provided within the nVidia SDK
extern cudaError_t MyStreamSynchronize(cudaStream_t stream, int situation, int thr_id);

typedef uint32_t(*cuda_compactionFunction_t)(uint32_t *inpHash, int position);

extern "C" __host__ void compaction_gpu_init(int thr_id, int threads)
{
	cudaGetDeviceProperties(&props[thr_id], device_map[thr_id]);

	cudaMalloc(&d_tempBranch1Nonces[thr_id], sizeof(uint32_t) * threads * 2);	
	cudaMalloc(&d_numValid[thr_id], 2*sizeof(uint32_t));
	cudaMallocHost(&h_numValid[thr_id], 2*sizeof(uint32_t));

	uint32_t s1;
	s1 = (threads / THREADS_PER_BLOCK) * 2;

	cudaMalloc(&d_partSum[0][thr_id], sizeof(uint32_t) * s1);
	cudaMalloc(&d_partSum[1][thr_id], sizeof(uint32_t) * s1);
	cudaMalloc(&d_nonces[thr_id], (nBitArray_Size[thr_id]>>3) * sizeof(uint32_t));
}

__global__ void compaction_gpu_SCANFirst(uint32_t *data, int width, uint32_t *partial_sums, int threads, uint32_t *inpHashes)
{
	extern __shared__ uint32_t sums[];
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int lane_id = id % width;
	int warp_id = threadIdx.x / width;

	sums[lane_id] = 0;

	uint32_t bits = id & 31, bytes = id >> 5;
	uint32_t value = (id < threads) ? ((inpHashes[bytes] & (1 << bits)) == 0) : 0;

	__syncthreads();

#pragma unroll

	for (int i=1; i<=width; i*=2)
	{
		uint32_t n = __shfl_up((int)value, i, width);

		if (lane_id >= i) value += n;
	}

	if (threadIdx.x % width == width-1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();
	
	if (warp_id == 0)
	{
		uint32_t warp_sum = sums[lane_id];

		for (int i=1; i<=width; i*=2)
		{
			uint32_t n = __shfl_up((int)warp_sum, i, width);

		if (lane_id >= i) warp_sum += n;
		}

		sums[lane_id] = warp_sum;
	}

	__syncthreads();

	uint32_t blockSum = 0;

	if (warp_id > 0)
	{
		blockSum = sums[warp_id-1];
	}

	value += blockSum;

	data[id] = value;

	if (partial_sums != NULL && threadIdx.x == blockDim.x-1)
	{
		partial_sums[blockIdx.x] = value;
	}
}

__global__ void compaction_gpu_SCAN(uint32_t *data, int width, uint32_t *partial_sums = NULL)
{
	extern __shared__ uint32_t sums[];
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int lane_id = id % width;
	int warp_id = threadIdx.x / width;

	sums[lane_id] = 0;
	uint32_t value = data[id];	

	__syncthreads();
#pragma unroll

	for (int i = 1; i <= width; i *= 2)
	{
		uint32_t n = __shfl_up((int)value, i, width);

		if (lane_id >= i) value += n;
	}
	if (threadIdx.x % width == width - 1)
	{
		sums[warp_id] = value;
	}

	__syncthreads();
	if (warp_id == 0)
	{
		uint32_t warp_sum = sums[lane_id];

		for (int i = 1; i <= width; i *= 2)
		{
			uint32_t n = __shfl_up((int)warp_sum, i, width);

			if (lane_id >= i) warp_sum += n;
		}

		sums[lane_id] = warp_sum;
	}

	__syncthreads();
	uint32_t blockSum = 0;

	if (warp_id > 0)
	{
		blockSum = sums[warp_id - 1];
	}

	value += blockSum;

	data[id] = value;

	if (partial_sums != NULL && threadIdx.x == blockDim.x - 1)
	{
		partial_sums[blockIdx.x] = value;
	}
}

__global__ void compaction_gpu_ADD(uint32_t *data, uint32_t *partial_sums, int len)
{
	__shared__ uint32_t buf;
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);

	if (id > len) return;

	if (threadIdx.x == 0)
	{
		buf = partial_sums[blockIdx.x];
	}

	__syncthreads();
	data[id] += buf;
}

__global__ void compaction_gpu_SCATTER(uint32_t *sum, uint32_t *outp, int threads, uint32_t *inpHashes)
{
	int id = ((blockIdx.x * blockDim.x) + threadIdx.x);
	uint32_t actNounce = id;
	uint32_t bits = id & 31, bytes = id >> 5;
	int32_t value = (id < threads) ? ((inpHashes[bytes] & (1 << bits)) == 0) : 0;	
	if( value )
	{
		int idx = sum[id];
		if(idx > 0)
			outp[idx-1] = id;
	}
}

extern "C" __host__ static uint32_t compaction_roundUpExp(uint32_t val)
{
	if(val == 0)
		return 0;

	uint32_t mask = 0x80000000;
	while( (val & mask) == 0 ) mask = mask >> 1;

	if( (val & (~mask)) != 0 )
		return mask << 1;

	return mask;
}

extern "C" __host__ void compaction_gpu_singleCompaction(int thr_id, int threads, uint32_t *nrm,
														uint32_t *d_nonces1,
														uint32_t *inpHashes, uint32_t *d_validNonceTable)
{
	int orgThreads = threads;
	threads = (int)compaction_roundUpExp((uint32_t)threads);	
	int blockSize = THREADS_PER_BLOCK;
	int nSums = threads / blockSize;

	int thr1 = (threads+blockSize-1) / blockSize;
	int thr2 = threads / (blockSize*blockSize);
	int blockSize2 = (nSums < blockSize) ? nSums : blockSize;
	int thr3 = (nSums + blockSize2-1) / blockSize2;

	bool callThrid = (thr2 > 0) ? true : false;
	
	compaction_gpu_SCANFirst<<<thr1,blockSize, 32*sizeof(uint32_t)>>>(
		d_tempBranch1Nonces[thr_id], 32, d_partSum[0][thr_id], orgThreads, inpHashes);	

	if(callThrid)
	{		
		compaction_gpu_SCAN<<<thr2,blockSize, 32*sizeof(uint32_t)>>>(d_partSum[0][thr_id], 32, d_partSum[1][thr_id]);
		compaction_gpu_SCAN<<<1, thr2, 32*sizeof(uint32_t)>>>(d_partSum[1][thr_id], (thr2>32) ? 32 : thr2);
	}else
	{
		compaction_gpu_SCAN<<<thr3,blockSize2, 32*sizeof(uint32_t)>>>(d_partSum[0][thr_id], (blockSize2>32) ? 32 : blockSize2);
	}
	
	if(callThrid)
	{
		compaction_gpu_ADD<<<thr2-1, blockSize>>>(d_partSum[0][thr_id]+blockSize, d_partSum[1][thr_id], blockSize*thr2);
	}
	compaction_gpu_ADD<<<thr1-1, blockSize>>>(d_tempBranch1Nonces[thr_id]+blockSize, d_partSum[0][thr_id], threads);
		
	compaction_gpu_SCATTER<<<thr1,blockSize,0>>>(d_tempBranch1Nonces[thr_id], d_nonces1, 
		orgThreads, inpHashes);
	
	MyStreamSynchronize(NULL, 5, thr_id);
	if (callThrid)
		cudaMemcpy(nrm, &(d_partSum[1][thr_id])[thr2 - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
	else
		cudaMemcpy(nrm, &(d_partSum[0][thr_id])[nSums - 1], sizeof(uint32_t), cudaMemcpyDeviceToHost);
}

extern "C" __host__ void compaction_gpu(int thr_id, int threads, uint32_t *h_nonces1, size_t *nrm1)
{
	compaction_gpu_singleCompaction(thr_id, threads, h_numValid[thr_id], d_nonces[thr_id], (uint32_t*)d_bit_array_sieve[thr_id], NULL);
	*nrm1 = (size_t)h_numValid[thr_id][0];
	cudaMemcpy(h_nonces1, d_nonces[thr_id], *nrm1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
}
