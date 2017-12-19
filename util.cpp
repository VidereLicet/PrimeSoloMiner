#include "util.h"
#include "cuda/cuda.h"

/** Find out how many cores the CPU has. **/
int GetTotalCores()
{
#if 0
	int nProcessors = boost::thread::hardware_concurrency();
	if (nProcessors < 1)
		nProcessors = 1;
			
	return nProcessors;
#else
    return cuda_num_devices();
#endif
}

