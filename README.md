
Dense Prime Cluster GPU (CUDA) Miner for Coinshield CPU Channel.

This is a fork of Viz' PrimeSoloMiner project, adding GPU (CUDA) acceleration for sieving.
CUDA additions by cbuchner1, ChrisH

Commandline Arguments are IP PORT CUDADEVICES CPUTHREADS TIMEOUT.

IP and PORT and CUDADEVICES are required, CPUTHREADS default is CPU Cores, TIMEOUT default is 10 seconds.


Example to mine on a local wallet using 3 GPUs (device 0,1,2) and doing
primality testing on 6 CPU threads. Timeout shall be 30 seconds.

./gpuminer 127.0.0.1 9323 0,1,2 6 30


Dots indicate that your CPU cannot keep up with primality testing.
What to try if you are getting dots on the console:

a) use a faster CPU
b) specify more CPU threads to do primality testing
c) increase nPrimeLimitB in the config.ini file (multiples of 1024 recommended)

Try to find an operating point that maxes out most of your CPU and generates few or no dots.
You need to balance out GPUs and CPUs.


Kind regards,

Christian Buchner
