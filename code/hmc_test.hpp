#ifndef _HMC_TEST_HPP_
#define _HMC_TEST_HPP_


#include "rng.hpp"
void TestHessianHamiltonMC(int w, int h, float plotRadius, float* data, SimpleURNG& urng);
void TestHMC(int w, int h, float plotRadius, float* data, SimpleURNG& urng);
	
#endif
