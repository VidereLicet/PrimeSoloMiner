#ifndef COINSHIELD_UTIL_H
#define COINSHIELD_UTIL_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include "hash/templates.h"

#ifdef WIN32
	typedef int pid_t;
#else
	#include <sys/types.h>
	#include <sys/time.h>
	#include <sys/resource.h>
#endif

#include <string>
#include <vector>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>


#define loop                for(;;)
#define BEGIN(a)            ((char*)&(a))
#define END(a)              ((char*)&((&(a))[1]))
#define Sleep(a)            boost::this_thread::sleep(boost::posix_time::milliseconds(a))
#define LOCK(a)             boost::lock_guard<boost::mutex> lock(a)


int GetTotalCores();



#endif
