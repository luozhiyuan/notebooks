#ifndef _TIMED_OPERATION_HPP_
#define _TIMED_OPERATION_HPP_

#include <chrono>

template<typename F, typename... Args>
std::chrono::milliseconds 
TimedOperation(std::chrono::milliseconds ms, const F& f, Args&&... args)
{
	using clock = std::chrono::system_clock;
	auto start = clock::now();
	auto end = clock::now();
	while((end - start) < ms)
	{
		f(args...);
		end = clock::now();
	}
	return std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
}

#endif