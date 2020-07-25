#ifndef _HMC_TEST_HPP_
#define _HMC_TEST_HPP_


#include "rng.hpp"
void TestHessianHamiltonMC(int w, int h, int components, float plotRadius, unsigned char* data, SimpleURNG& urng);
void TestHMC(int w, int h, int components, float plotRadius, unsigned char* data, SimpleURNG& urng);
	
#endif
