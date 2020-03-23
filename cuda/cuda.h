
extern int device_map[8];

extern "C" int cuda_num_devices();
extern "C" int cuda_finddevice(char *name);
extern "C" const char* cuda_devicename(int index);
extern "C" bool cuda_init(int thr_id);
extern "C" void cuda_set_zTempVar(unsigned int thr_id, const uint64_t *limbs);
extern "C" void cuda_set_zFirstSieveElement(unsigned int thr_id, const uint64_t *limbs);
extern "C" void cuda_set_primes(unsigned int thr_id, unsigned int *primes, unsigned int *inverses, uint64_t *invk, unsigned int nPrimeLimit, unsigned int nBitArray_Size, unsigned int nOrigins);
extern "C" void cuda_set_origins(unsigned int thr_id, uint64_t *origins, unsigned int nOrigins, unsigned int sharedSizeKB, unsigned int nPrimorialEndPrime, unsigned int nPrimeLimitA);
extern "C" void cuda_compute_base_remainders(unsigned int thr_id, unsigned int *base_remainders, unsigned int nPrimorialEndPrime, unsigned int nPrimeLimit);
extern "C" void cuda_compute_primesieve(unsigned int thr_id, unsigned int nSharedSizeKB, unsigned int nThreadsKernelA, unsigned char *bit_array_sieve, unsigned int *base_remainders, uint64_t base_offset, uint32_t base_index, unsigned int nPrimorialEndPrime, unsigned int nPrimeLimitA, unsigned int nPrimeLimitB, unsigned int nBitArray_Size, unsigned int nDifficulty, int validationEnabled);
extern "C" void cuda_compute_fermat(unsigned int thr_id, uint32_t *nonce_offsets, size_t numberOfCandidates, int constellation_offset, uint64_t nPrimorial);
extern "C" void compaction_gpu_init(int thr_id, int threads);
extern "C" void compaction_gpu(int thr_id, int threads, uint32_t *h_nonces1, size_t *nrm1);
extern "C" void cuda_reset_device();
