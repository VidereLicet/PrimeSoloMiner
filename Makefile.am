
ACLOCAL_AMFLAGS		= -I m4

EXTRA_DIST		= autogen.sh README.txt LICENSE.txt \
			  compat/gettimeofday.c compat/getopt/getopt_long.c gpuminer-config.h.in

#SUBDIRS		= compat

bin_PROGRAMS	= gpuminer

gpuminer_SOURCES	=  bigmath.cpp  miner.cpp  prime.cpp  util.cpp \
			   bigmath.h bignum.h core.h types.h util.h \
			   iniParser.cpp \
			   cuda/sieve.cu cuda/cuda.h cuda/simd1024math.h cuda/compaction.cu \
			   hash/brg_endian.h  hash/crypto_hash.h       hash/KeccakDuplex.c  hash/KeccakF-1600-interface.h  hash/KeccakHash.h    hash/KeccakSponge.h   hash/skein.cpp  hash/skein_iv.h    hash/templates.h \
			   hash/brg_types.h   hash/Keccak-compact64.c  hash/KeccakDuplex.h  hash/KeccakHash.c              hash/KeccakSponge.c  hash/skein_block.cpp  hash/skein.h    hash/skein_port.h  hash/uint1024.h


gpuminer_LDFLAGS	= $(PTHREAD_FLAGS) @CUDA_LDFLAGS@ $(BOOST_LDFLAGS) $(BOOST_FILESYSTEM_LDFLAGS) $(BOOST_SYSTEM_LDFLAGS) $(BOOST_THREAD_LDFLAGS) $(BOOST_PROGRAM_OPTIONS_LDFLAGS)
gpuminer_LDADD		= @PTHREAD_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@ -lcuda -lgmp -lprimesieve $(BOOST_FILESYSTEM_LIBS) $(BOOST_SYSTEM_LIBS) $(BOOST_THREAD_LIBS) $(BOOST_PROGRAM_OPTIONS_LIBS)
gpuminer_CPPFLAGS	= -std=c++0x -msse2 @OPENMP_CFLAGS@ $(PTHREAD_FLAGS) $(BOOST_CPPFLAGS) -fno-strict-aliasing

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) -std=c++11 @CFLAGS@ -D_FORCE_INLINES -I . -Xptxas "-v" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_52,code=\"sm_52,compute_52\" --maxrregcount=80 --ptxas-options=-v -o $@ -c $<

cuda/sieve.o:	cuda/sieve.cu cuda/simd1024math.h cuda/cuda.h
	$(NVCC) -std=c++11 @CFLAGS@ -D_FORCE_INLINES -I . -Xptxas "-v" -gencode=arch=compute_30,code=\"sm_30,compute_30\" -gencode=arch=compute_35,code=\"sm_35,compute_35\" -gencode=arch=compute_52,code=\"sm_52,compute_52\" --maxrregcount=24 --ptxas-options=-v -o $@ -c $<
