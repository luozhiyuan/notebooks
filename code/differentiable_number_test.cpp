#include <iostream>
#include <iomanip>
#include <random>
#include <numeric>
#include "differentiable_number.hpp"
#include "partial_differentiable_number.hpp"
#include "rng.hpp"
#include "hmc_test.hpp"

template<int Order, typename Number>
DifferentiableNumber<Order, Number>
test_function(const DifferentiableNumber<Order, Number>& x)
{
	return x*Sin(Pow(x, 2));
}
//int main()
//{
//	{
//		auto x = DifferentiableNumber<3>(3);
//		auto d = test_function(x);
//		std::cout << "f(x) = " << d.Derivative(0) << std::endl;
//		std::cout << "f'(x) = " << d.Derivative(1) << std::endl;
//		std::cout << "f''(x) = " << d.Derivative(2) << std::endl;
//		std::cout << "f'''(x) = " << d.Derivative(3) << std::endl;
//	}
//}
