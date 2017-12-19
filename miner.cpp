#include "core.h"
#include "cuda/cuda.h"
#include "iniParser.h"

#include <inttypes.h>
#include <queue>
#include <fstream>
#include <thread>

// fix  unresolved external symbol __iob_func error when linking OpenSSL 1.0.2j to this miner
// see http://stackoverflow.com/questions/30412951/unresolved-external-symbol-imp-fprintf-and-imp-iob-func-sdl2
#ifdef _WIN32
FILE _iob[] = { *stdin, *stdout, *stderr };
extern "C" FILE * __cdecl __iob_func(void) { return _iob; }
#endif

volatile unsigned int nBlocksFoundCounter = 0;
volatile unsigned int nBlocksAccepted = 0;
volatile unsigned int nBlocksRejected = 0;
volatile unsigned int nDifficulty = 0;
volatile unsigned int nBestHeight = 0;
extern volatile unsigned int nInternalPrimeLimit;
volatile bool isBlockSubmission = false;
unsigned int nStartTimer = 0;
extern unsigned int nPrimeLimit;
volatile unsigned int gpuThreadWaitTime = 0;
unsigned int nThreadsCPU;
std::atomic<uint64_t> SievedBits;
std::atomic<uint64_t> CheckedCandidates;
std::atomic<uint64_t> PrimesFound;
std::atomic<uint64_t> PrimesChecked;

extern unsigned int nBitArray_Size[8];
extern unsigned int nPrimeLimitA[8];
extern unsigned int nPrimeLimitB[8];
extern unsigned int nPrimeLimit;           // will be set to the maximum of any in nPrimeLimitB[]
extern unsigned int nSharedSizeKB[8];
extern unsigned int nThreadsKernelA[8];

extern unsigned int nFourChainsFoundCounter;
extern unsigned int nFiveChainsFoundCounter;
extern unsigned int nSixChainsFoundCounter;
extern unsigned int nSevenChainsFoundCounter;
extern unsigned int nEightChainsFoundCounter;

#if WIN32
static inline void affine_to_cpu(int id, int cpu)
{
    DWORD mask = 1 << cpu;
    SetThreadAffinityMask(GetCurrentThread(), mask);
}
#else
static inline void affine_to_cpu(int id, int cpu)
{
	cpu_set_t set;

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	sched_setaffinity(0, sizeof(&set), &set);
}
#endif

void signal_handler(const boost::system::error_code& error,
	int signal_number)
{
	if (!error)
	{
		time_t now = time(0);
		unsigned int SecondsElapsed = (unsigned int)now - nStartTimer;
	}
}

namespace Core
{

	/** Class to hold the basic data a Miner will use to build a Block.
		Used to allow one Connection for any amount of threads. **/

	extern std::queue<work_info> result_queue;
	extern std::deque<work_info> work_queue;
	extern boost::mutex work_mutex;

	class MinerThreadGPU
	{
	public:
		int threadIndex;
		int threadAffinity;
		CBlock* BLOCK;
		bool fBlockFound, fNewBlock;
		LLP::Thread_t THREAD;
		boost::mutex MUTEX;

		unsigned int nSearches;
		
		MinerThreadGPU(int tid, int affinity) : threadIndex(tid), threadAffinity(affinity), BLOCK(NULL), fBlockFound(false), fNewBlock(true), nSearches(0), THREAD(boost::bind(&MinerThreadGPU::PrimeMiner, this)) { }
		
		
		/** Main Miner Thread. Bound to the class with boost. Might take some rearranging to get working with OpenCL. **/
		void PrimeMiner()
		{
			affine_to_cpu(threadIndex, threadAffinity); // all CUDA threads run on CPU core 0 + threadIndex
			
			loop
			{
				try
				{
					/* Keep thread at idle CPU usage if waiting to submit or recieve block. **/
					Sleep(1);

					if(!(fNewBlock || fBlockFound || !BLOCK))
					{
						if(gpuThreadWaitTime > 0)
							Sleep(gpuThreadWaitTime);

						nDifficulty = BLOCK->nBits;
						BLOCK->nNonce = 0;
						PrimeSieve(threadIndex, BLOCK->GetPrime(), BLOCK->nBits, BLOCK->nHeight, BLOCK->hashMerkleRoot);
						fNewBlock = true;
					}
				}
				catch(std::exception& e){ printf("ERROR: %s\n", e.what()); }
			}
		}
	};

	class MinerThreadCPU
	{
	public:
		int threadIndex;
		int threadAffinity;
		CBlock* BLOCK;
		bool fBlockFound, fNewBlock;
		LLP::Thread_t THREAD;
		boost::mutex MUTEX;

		unsigned int nSearches;
		
		MinerThreadCPU(int tid, int affinity) : threadIndex(tid), threadAffinity(affinity), BLOCK(NULL), fBlockFound(false), fNewBlock(true), nSearches(0), THREAD(boost::bind(&MinerThreadCPU::PrimeMiner, this)) { }
		
		
		/** Main Miner Thread. Bound to the class with boost. Might take some rearranging to get working with OpenCL. **/
		void PrimeMiner()
		{
			affine_to_cpu(threadIndex, threadAffinity);
			
			loop
			{
				try
				{
					if (!PrimeQuery())
					{
						Sleep(100);
					}
				}
				catch(std::exception& e){ printf("ERROR: %s\n", e.what()); }
			}
		}
	};
	
	
	/** Class to handle all the Connections via Mining LLP.
		Independent of Mining Threads for Higher Efficiency. **/
	class ServerConnection
	{
	public:
		LLP::Miner* CLIENT;
		int nThreadsGPU, nThreadsCPU, nTimeout;
		std::vector<MinerThreadGPU*> THREADS_GPU;
		std::vector<MinerThreadCPU*> THREADS_CPU;
		LLP::Thread_t THREAD;
		LLP::Timer    TIMER;
		std::string   IP, PORT;
		
		ServerConnection(std::string ip, std::string port, int nMaxThreadsGPU, int nMaxThreadsCPU, int nMaxTimeout) : IP(ip), PORT(port), TIMER(), nThreadsGPU(nMaxThreadsGPU), nThreadsCPU(nMaxThreadsCPU), nTimeout(nMaxTimeout), THREAD(boost::bind(&ServerConnection::ServerThread, this))
		{

			int affinity = 0;
			int nthr = std::thread::hardware_concurrency();

			for(int nIndex = 0; nIndex < nThreadsGPU; nIndex++)
				THREADS_GPU.push_back(new MinerThreadGPU(nIndex, (affinity++)%nthr));

			for(int nIndex = 0; nIndex < nThreadsCPU; nIndex++)
				THREADS_CPU.push_back(new MinerThreadCPU(nIndex, (affinity++)%nthr));
		}
		
		/** Reset the block on each of the Threads. **/
		void ResetThreads()
		{
		
			/** Reset each individual flag to tell threads to stop mining. **/
			for(int nIndex = 0; nIndex < THREADS_GPU.size(); nIndex++)
			{
				THREADS_GPU[nIndex]->fNewBlock   = true;
			}
				
		}
		
		/** Main Connection Thread. Handles all the networking to allow
			Mining threads the most performance. **/
		void ServerThread()
		{
			/** Don't begin until all mining threads are Created. **/
			while(THREADS_GPU.size() != nThreadsGPU)
				Sleep(1);
			while(THREADS_CPU.size() != nThreadsCPU)
				Sleep(1);
				
				
			/** Initialize the Server Connection. **/
			CLIENT = new LLP::Miner(IP, PORT);
				
				
			/** Initialize a Timer for the Hash Meter. **/
			TIMER.Start();
			
			bool first = true;
			loop
			{
				try
				{
					/** Run this thread at 1 Cycle per Second. **/
					Sleep(1000);
					
					
					/** Attempt with best efforts to keep the Connection Alive. **/
					if(!CLIENT->Connected() || CLIENT->Errors())
					{
						ResetThreads();
						
						if(!CLIENT->Connect())
							continue;
						else
							CLIENT->SetChannel(1);
					}
					
					
					/** Check the Block Height. **/
					unsigned int nHeight = CLIENT->GetHeight(nTimeout);
					if(nHeight == 0)
					{
						printf("Failed to Update Height...\n");
						CLIENT->Disconnect();
						continue;
					}
					
					/** If there is a new block, Flag the Threads to Stop Mining. **/
					if(nHeight != nBestHeight)
					{
						isBlockSubmission = false;
						nBestHeight = nHeight;
						printf("\n[MASTER] Nexus Network: New Block %u\n", nHeight);

						ResetThreads();
					}

					/** Rudimentary Meter **/
					if(first || TIMER.Elapsed() >= 30)
					{
						first = false;
						time_t now = time(0);
						unsigned int SecondsElapsed = (unsigned int)now - nStartTimer;
						unsigned int nElapsed = TIMER.Elapsed();
							
						uint64_t bps = SievedBits.load() / nElapsed;
						SievedBits = 0;
						uint64_t cand = CheckedCandidates.load();
						CheckedCandidates = 0;
						uint64_t cps = cand / nElapsed;
						double   pratio = (double)PrimesFound.load() / PrimesChecked.load();
						PrimesFound = 0;
						PrimesChecked = 0;
						printf("\n[METERS] %u Block(s) ACC=%u REJ=%u| Height = %u | Diff = %f | %02d:%02d:%02d\n", nBlocksFoundCounter, nBlocksAccepted, nBlocksRejected, nBestHeight, (double)nDifficulty/10000000.0, (SecondsElapsed/3600)%60, (SecondsElapsed/60)%60, (SecondsElapsed)%60);
						printf("[METERS] Sieved %.1fM Bits/s, Checked %u Candidates/s, Prime ratio: %.3f %%\n", (double)bps / 1e6, (unsigned int)cps, (double)100 * pratio);
						printf("[METERS] Prime Clusters Found: Four=%u | Five=%u | Six=%u | Seven=%u | Eight=%u\n", nFourChainsFoundCounter, nFiveChainsFoundCounter, nSixChainsFoundCounter, nSevenChainsFoundCounter, nEightChainsFoundCounter);						
						TIMER.Reset();

					}

					for(int nIndex = 0; nIndex < THREADS_GPU.size(); nIndex++)
					{
						/** Attempt to get a new block from the Server if Thread needs One. **/
						if(THREADS_GPU[nIndex]->fNewBlock)
						{
							/** Retrieve new block from Server. **/
							CBlock* BLOCK = CLIENT->GetBlock(nTimeout);
							
							
							/** If the block is good, tell the Mining Thread its okay to Mine. **/
							if(BLOCK)
							{
								THREADS_GPU[nIndex]->BLOCK = BLOCK;
								
								THREADS_GPU[nIndex]->fBlockFound = false;
								THREADS_GPU[nIndex]->fNewBlock   = false;
							}
							
							/** If the Block didn't come in properly, Reconnect to the Server. **/
							else
							{
								CLIENT->Disconnect();
								
								break;
							}
								
						}
					}

					{
						boost::mutex::scoped_lock lock(work_mutex);
						while(result_queue.empty() == false)
						{
							nBlocksFoundCounter++;

							printf("\nSubmitting Block...\n");
							work_info work = result_queue.front();
							result_queue.pop();
							
							double difficulty = (double)work.nNonceDifficulty / 10000000.0;

							printf("\n[MASTER] Prime Cluster of Difficulty %f Found\n", difficulty);
								
							/** Attempt to Submit the Block to Network. **/
							unsigned char RESPONSE = CLIENT->SubmitBlock(work.merkleRoot, work.nNonce, nTimeout);
							
							/** Check the Response from the Server.**/
							if(RESPONSE == 200)
							{
								printf("\n[MASTER] Block Accepted By Nexus Network.\n");
								
								ResetThreads();
								nBlocksAccepted++;
							}
							else if(RESPONSE == 201)
							{
								printf("\n[MASTER] Block Rejected by Nexus Network.\n");
								isBlockSubmission = false;
								nBlocksRejected++;
							}
								
							/** If the Response was Bad, Reconnect to Server. **/
							else 
							{
								printf("\n[MASTER] Failure to Submit Block. Reconnecting...\n");
								CLIENT->Disconnect();
								
								break;
							}
						}
					}
				}
				catch(std::exception& e)
				{
					printf("%s\n", e.what()); CLIENT = new LLP::Miner(IP, PORT); 
				}
			}
		}
	};
}

#include <primesieve.hpp>


int main(int argc, char *argv[])
{
	if(argc < 3)
	{
		printf("Too Few Arguments. The Required Arguments are Ip and Port\n");
		printf("Default Arguments are Total Threads = nVidia GPUs and Connection Timeout = 10 Seconds\n");
		printf("Format for Arguments is 'IP PORT DEVICELIST CPUTHREADS TIMEOUT'\n");
		
		Sleep(10000);
		
		return 0;
	}

	// the io_service.run() replaces the nonterminating sleep loop
	boost::asio::io_service io_service;

	// construct a signal set registered for process termination.
	boost::asio::signal_set signals(io_service, SIGINT, SIGTERM);

	// start an asynchronous wait for one of the signals to occur.
	signals.async_wait(signal_handler);

	std::string IP = argv[1];
	std::string PORT = argv[2];
	unsigned int nThreadsGPU = GetTotalCores();
	unsigned int nTimeout = 10;
	nThreadsCPU = GetTotalCores();
	
	if(argc > 3) {
		int num_processors = nThreadsGPU;
		char * pch = strtok (argv[3],",");
		nThreadsGPU = 0;
		while (pch != NULL) {
			if (pch[0] >= '0' && pch[0] <= '9' && pch[1] == '\0')
			{
				if (atoi(pch) < num_processors)
					device_map[nThreadsGPU++] = atoi(pch);
				else {
					fprintf(stderr, "Non-existant CUDA device #%d specified\n", atoi(pch));
					exit(1);
				}
			} else {
				int device = cuda_finddevice(pch);
				if (device >= 0 && device < num_processors)
					device_map[nThreadsGPU++] = device;
				else {
					fprintf(stderr, "Non-existant CUDA device '%s' specified\n", pch);
					exit(1);
				}
			}
			pch = strtok (NULL, ",");
		}
	}
	if(argc > 4)
		nThreadsCPU = boost::lexical_cast<int>(argv[4]);
	
	if(argc > 5)
		nTimeout = boost::lexical_cast<int>(argv[5]);
	
	printf("\nLoading configuration...\n");
	
    std::ifstream t("config.ini");
    std::stringstream buffer;
    buffer << t.rdbuf();
	std::string config = buffer.str();

	CIniParser parser;
	if (parser.Parse(config.c_str()) == false)
	{
		fprintf(stderr, "Unable to parse config.ini");
	}

	for (int i=0 ; i < nThreadsGPU; i++)
	{
		const char *devicename = cuda_devicename(device_map[i]);

		if (!parser.GetValueAsInteger(devicename, "nPrimeLimitA", (int*)&nPrimeLimitA[i]))
			parser.GetValueAsInteger("GENERAL", "nPrimeLimitA", (int*)&nPrimeLimitA[i]);
		if (!parser.GetValueAsInteger(devicename, "nPrimeLimitB", (int*)&nPrimeLimitB[i]))
			parser.GetValueAsInteger("GENERAL", "nPrimeLimitB", (int*)&nPrimeLimitB[i]);
		if (!parser.GetValueAsInteger(devicename, "nBitArray_Size", (int*)&nBitArray_Size[i]))
			parser.GetValueAsInteger("GENERAL", "nBitArray_Size", (int*)&nBitArray_Size[i]);
		if (!parser.GetValueAsInteger(devicename, "nSharedSizeKB", (int*)&nSharedSizeKB[i]))
			parser.GetValueAsInteger("GENERAL", "nSharedSizeKB", (int*)&nSharedSizeKB[i]);
		if (!parser.GetValueAsInteger(devicename, "nThreadsKernelA", (int*)&nThreadsKernelA[i]))
			parser.GetValueAsInteger("GENERAL", "nThreadsKernelA", (int*)&nThreadsKernelA[i]);

		printf("\nGPU thread %d, device %d [%s]\n", i, device_map[i], devicename);
		printf("nPrimeLimitA = %d\n", nPrimeLimitA[i]);
		printf(" nPrimeLimitB = %d\n", nPrimeLimitB[i]);
		printf(" nBitArray_Size = %d\n", nBitArray_Size[i]);
		printf(" nSharedSizeKB = %d\n", nSharedSizeKB[i]);
		printf(" nThreadsKernelA = %d\n", nThreadsKernelA[i]);

		if (nPrimeLimitB[i] > nPrimeLimit) nPrimeLimit = nPrimeLimitB[i];
	}

	Core::InitializePrimes();
	nStartTimer = (unsigned int)time(0);
	printf("Initializing Miner %s:%s Threads = %i (GPU), %i (CPU) Timeout = %i\n", IP.c_str(), PORT.c_str(), nThreadsGPU, nThreadsCPU, nTimeout);
	Core::ServerConnection MINERS(IP, PORT, nThreadsGPU, nThreadsCPU, nTimeout);

	io_service.run();
	
	return 0;
}
