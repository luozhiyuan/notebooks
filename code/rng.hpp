#ifndef _RNG_HPP_
#define _RNG_HPP_

#include <random>

class SimpleURNG
{
	std::random_device r;
	std::mt19937 mt;
public:
	SimpleURNG(const SimpleURNG&) = delete;
	SimpleURNG() :mt(r()) {}
	void Get1D(double* zeta)
	{
		*zeta = std::generate_canonical<double, std::numeric_limits<double>::digits>(mt);
	}

	void Get2D(double* zeta0, double* zeta1)
	{
		Get1D(zeta0);
		Get1D(zeta1);
	}

};

template<int Dim>
class SimpleNRNG
{
	std::random_device r;
	std::mt19937 mt;
	std::array<std::normal_distribution<double>, Dim> nds;
public:
	SimpleNRNG(const SimpleNRNG&) = delete;
	SimpleNRNG() :mt(r())
	{
		for (int i = 0; i < Dim; i++)
		{
			nds[i] = std::normal_distribution<double>(0, 1);
		}
	}
	void Get(std::array<double, Dim>* zeta)
	{
		for (size_t i = 0; i < zeta->size(); i++)
		{
			(*zeta)[i] = (nds[i])(mt);
		}
	}
};

#endif
