#ifndef NEXUS_LLP_CORE_H
#define NEXUS_LLP_CORE_H


#include "types.h"
#include "bignum.h"
#ifndef _WIN32
#include <gmp.h>
#include <gmpxx.h>
#else
#include <mpir.h>
#include <mpirxx.h>
#endif

namespace Core
{
	class CBlock
	{
	public:

		/** Begin of Header.   BEGIN(nVersion) **/
		unsigned int  nVersion;
		uint1024 hashPrevBlock;
		uint512 hashMerkleRoot;
		unsigned int  nChannel;
		unsigned int   nHeight;
		unsigned int     nBits;
		uint64          nNonce;
		/** End of Header.     END(nNonce). 
			All the components to build an SK1024 Block Hash. **/
			
		CBlock()
		{
			nVersion       = 0;
			hashPrevBlock  = 0;
			hashMerkleRoot = 0;
			nChannel       = 0;
			nHeight        = 0;
			nBits          = 0;
			nNonce         = 0;
		}
			
		uint1024 GetHash() const
		{
			return SK1024(BEGIN(nVersion), END(nBits));
		}
		
		CBigNum GetPrime() const
		{
			return CBigNum(GetHash() + nNonce);
		}
	};

	class work_info
	{
public:
		work_info() {}
		work_info(CBigNum basehash, unsigned int difficulty, unsigned int height, unsigned int thr_idx, uint32_t *nonce_begin, uint32_t *nonce_end, mpz_t firstsieveelement, uint512 merkle, unsigned int primelimit) :
			BaseHash(basehash), nDifficulty(difficulty), nHeight(height), zFirstSieveElement(firstsieveelement), nNonce(false), nNonceDifficulty(0), merkleRoot(merkle), nPrimes(primelimit)
			{ gpu_thread = thr_idx; nonce_offsets.assign(nonce_begin, nonce_end); }
		~work_info() {}
		
		// input
		unsigned int nHeight;
		unsigned int nDifficulty;
		CBigNum BaseHash;
		uint512 merkleRoot;
		mpz_class zFirstSieveElement;
		unsigned int nPrimes;

		// gpu intermediate result
		unsigned int gpu_thread;
		std::vector<uint32_t> nonce_offsets;

		// result
		uint64_t nNonce;
		unsigned int nNonceDifficulty;
#if _WIN32
		LARGE_INTEGER StartingTime, EndingTime;
#endif
	};
	
	void InitializePrimes();
	unsigned int SetBits(double nDiff);
	double GetPrimeDifficulty(CBigNum prime, int checks);
	unsigned int GetPrimeBits(CBigNum prime, int checks);
	unsigned int GetFractionalDifficulty(CBigNum composite);
	std::vector<unsigned int> Eratosthenes(int nSieveSize);
	bool DivisorCheck(CBigNum test);
	void PrimeSieve(int threadIndex, CBigNum BaseHash, unsigned int nDifficulty, unsigned int nHeight, uint512 merkeRoot);
	bool PrimeQuery();
	bool PrimeCheck(CBigNum test, int checks);
	CBigNum FermatTest(CBigNum n, CBigNum a);
	bool Miller_Rabin(CBigNum n, int checks);
}

namespace LLP
{
	class Outbound : public Connection
	{
		Service_t IO_SERVICE;
		std::string IP, PORT;
		
	public:
		Packet ReadNextPacket(int nTimeout = 10)
		{
			Packet NULL_PACKET;
			while(!PacketComplete())
			{
				if(Timeout(nTimeout) || Errors())
					return NULL_PACKET;
				
				ReadPacket();
			
				Sleep(1);
			}
			
			return this->INCOMING;
		}
		
	public:
		/** Outgoing Client Connection Constructor **/
		Outbound(std::string ip, std::string port) : IP(ip), PORT(port), Connection() { }
		
		bool Connect()
		{
			try
			{
				using boost::asio::ip::tcp;
				
				tcp::resolver 			  RESOLVER(IO_SERVICE);
				tcp::resolver::query      QUERY   (tcp::v4(), IP.c_str(), PORT.c_str());
				tcp::resolver::iterator   ADDRESS = RESOLVER.resolve(QUERY);
				
				this->SOCKET = Socket_t(new tcp::socket(IO_SERVICE));
				this->SOCKET -> connect(*ADDRESS, this->ERROR_HANDLE);
				
				if(Errors())
				{
					this->Disconnect();
					
					printf("Failed to Connect to Mining LLP Server...\n");
					return false;
				}
				
				this->CONNECTED = true;
				this->TIMER.Start();
				
				printf("Connected to %s:%s...\n", IP.c_str(), PORT.c_str());

				return true;
			}
			catch(...){ }
			
			this->CONNECTED = false;
			return false;
		}
		
	};
	
	
	class Miner : public Outbound
	{
	public:
		Miner(std::string ip, std::string port) : Outbound(ip, port){}
		
		enum
		{
			/** DATA PACKETS **/
			BLOCK_DATA   = 0,
			SUBMIT_BLOCK = 1,
			BLOCK_HEIGHT = 2,
			SET_CHANNEL  = 3,
					
			/** REQUEST PACKETS **/
			GET_BLOCK    = 129,
			GET_HEIGHT   = 130,
			
			/** RESPONSE PACKETS **/
			GOOD     = 200,
			FAIL     = 201,
					
			/** GENERIC **/
			PING     = 253,
			CLOSE    = 254
		};
		
		inline void SetChannel(unsigned int nChannel)
		{
			Packet packet;
			packet.HEADER = SET_CHANNEL;
			packet.LENGTH = 4;
			packet.DATA   = uint2bytes(nChannel);
			
			this -> WritePacket(packet);
		}
		
		inline Core::CBlock* GetBlock(int nTimeout = 30)
		{
			Packet packet;
			packet.HEADER = GET_BLOCK;
			this -> WritePacket(packet);
			
			Packet RESPONSE = ReadNextPacket(nTimeout);
			
			if(RESPONSE.IsNull())
				return NULL;
				
			Core::CBlock* BLOCK = DeserializeBlock(RESPONSE.DATA);
			ResetPacket();
			
			return BLOCK;
		}
		
		inline unsigned int GetHeight(int nTimeout = 30)
		{
			Packet packet;
			packet.HEADER = GET_HEIGHT;
			this -> WritePacket(packet);
			
			Packet RESPONSE = ReadNextPacket(nTimeout);
			
			if(RESPONSE.IsNull())
				return 0;
				
			unsigned int nHeight = bytes2uint(RESPONSE.DATA);
			ResetPacket();
			
			return nHeight;
		}
		
		inline unsigned char SubmitBlock(uint512 hashMerkleRoot, uint64 nNonce, int nTimeout = 30)
		{
			Packet PACKET;
			PACKET.HEADER = SUBMIT_BLOCK;
			
			PACKET.DATA = hashMerkleRoot.GetBytes();
			std::vector<unsigned char> NONCE  = uint2bytes64(nNonce);
			
			PACKET.DATA.insert(PACKET.DATA.end(), NONCE.begin(), NONCE.end());
			PACKET.LENGTH = 72;
			
			this->WritePacket(PACKET);
			Packet RESPONSE = ReadNextPacket(nTimeout);
			if(RESPONSE.IsNull())
				return 0;
			
			ResetPacket();
			
			return RESPONSE.HEADER;
		}
		
	private:
		Core::CBlock* DeserializeBlock(std::vector<unsigned char> DATA)
		{
			Core::CBlock* BLOCK = new Core::CBlock();
			BLOCK->nVersion      = bytes2uint(std::vector<unsigned char>(DATA.begin(), DATA.begin() + 4));
			
			BLOCK->hashPrevBlock.SetBytes (std::vector<unsigned char>(DATA.begin() + 4, DATA.begin() + 132));
			BLOCK->hashMerkleRoot.SetBytes(std::vector<unsigned char>(DATA.begin() + 132, DATA.end() - 20));
			
			BLOCK->nChannel      = bytes2uint(std::vector<unsigned char>(DATA.end() - 20, DATA.end() - 16));
			BLOCK->nHeight       = bytes2uint(std::vector<unsigned char>(DATA.end() - 16, DATA.end() - 12));
			BLOCK->nBits         = bytes2uint(std::vector<unsigned char>(DATA.end() - 12,  DATA.end() - 8));
			BLOCK->nNonce        = bytes2uint64(std::vector<unsigned char>(DATA.end() - 8,  DATA.end()));
			
			return BLOCK;
		}
		
		/** Convert a 32 bit Unsigned Integer to Byte Vector using Bitwise Shifts. **/
		std::vector<unsigned char> uint2bytes(unsigned int UINT)
		{
			std::vector<unsigned char> BYTES(4, 0);
			BYTES[0] = UINT >> 24;
			BYTES[1] = UINT >> 16;
			BYTES[2] = UINT >> 8;
			BYTES[3] = UINT;
			
			return BYTES;
		}
		
		unsigned int bytes2uint(std::vector<unsigned char> BYTES, int nOffset = 0) { return (BYTES[0 + nOffset] << 24) + (BYTES[1 + nOffset] << 16) + (BYTES[2 + nOffset] << 8) + BYTES[3 + nOffset]; }
		
		
		/** Convert a 64 bit Unsigned Integer to Byte Vector using Bitwise Shifts. **/
		std::vector<unsigned char> uint2bytes64(uint64 UINT)
		{
			std::vector<unsigned char> INTS[2];
			INTS[0] = uint2bytes((unsigned int) UINT);
			INTS[1] = uint2bytes((unsigned int) (UINT >> 32));
			
			std::vector<unsigned char> BYTES;
			BYTES.insert(BYTES.end(), INTS[0].begin(), INTS[0].end());
			BYTES.insert(BYTES.end(), INTS[1].begin(), INTS[1].end());
			
			return BYTES;
		}
		
		uint64 bytes2uint64(std::vector<unsigned char> BYTES) { return (bytes2uint(BYTES) | ((uint64)bytes2uint(BYTES, 4) << 32)); }
		
	};
}



#endif
