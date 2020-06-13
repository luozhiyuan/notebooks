#ifndef _DIFFERENTIABLE_NUMBER_HPP_
#define _DIFFERENTIABLE_NUMBER_HPP_

#include <array>
#include <cmath>

//a number in a number field module
template<int Order, typename Number = double>
class DifferentiableNumber
{
	std::array<Number, Order + 1> value;

public:

	static DifferentiableNumber<Order, Number> Constant(Number x)
	{
		DifferentiableNumber<Order, Number> r;
		r.value[0] = x;
		for (int i = 1; i < Order + 1; i++)
		{
			r.value[i] = 0;
		}
		return r;
	}

	DifferentiableNumber() :DifferentiableNumber(Number())
	{
	}

	DifferentiableNumber(Number x)
	{
		value[0] = x;
		value[1] = 1;
		for (int i = 2; i < Order + 1; i++)
		{
			value[i] = 0;
		}
	}

	//TODO operator /
	//
	DifferentiableNumber<Order, Number>& operator-=(const DifferentiableNumber<Order, Number>& other)
	{
		for (int i = 0; i < Order + 1; i++)
		{
			value[i] -= other.value[i];
		}
		return *this;
	}

	DifferentiableNumber<Order, Number>& operator+=(const DifferentiableNumber<Order, Number>& other)
	{
		for (int i = 0; i < Order + 1; i++)
		{
			value[i] += other.value[i];
		}
		return *this;
	}

	DifferentiableNumber<Order, Number>& operator*=(const DifferentiableNumber<Order, Number>& other)
	{
		std::array<Number, Order + 1> new_value;
		for (int i = 0; i < Order + 1; i++)
		{
			new_value[i] = Number();
			for (int j = 0; j <= i; j++)
			{
				int k = i - j;
				new_value[i] += value[j] * other.value[k];
			}
		}
		value = new_value;
		return *this;
	}

	DifferentiableNumber<Order, Number> operator*(Number number)const
	{
		DifferentiableNumber<Order, Number> result;
		for (int i = 0; i < Order + 1; i++)
		{
			result.value[i] = value[i] * number;
		}
		return result;
	}

	DifferentiableNumber<Order, Number> operator/(Number number)const
	{
		DifferentiableNumber<Order, Number> result;
		for (int i = 0; i < Order + 1; i++)
		{
			result.value[i] = value[i] / number;
		}
		return result;
	}

	const Number& operator[](int i)const
	{
		return value[i];
	}
	Number& operator[](int i)
	{
		return value[i];
	}

	Number Derivative(int order)const
	{
		Number result = value[order];
		for(int i = 0; i < order; i++)
		{
			result *= (i + 1);
		}
		return result;
	}
};

template<int Order, typename Number>
DifferentiableNumber<Order, Number> operator+(const DifferentiableNumber<Order, Number>& a, const DifferentiableNumber<Order, Number>& b)
{
	DifferentiableNumber<Order, Number> result(a);
	return result += b;
}

template<int Order, typename Number>
DifferentiableNumber<Order, Number> operator-(const DifferentiableNumber<Order, Number>& a, const DifferentiableNumber<Order, Number>& b)
{
	DifferentiableNumber<Order, Number> result(a);
	return result -= b;
}


template<int Order, typename Number>
DifferentiableNumber<Order, Number> operator*(const DifferentiableNumber<Order, Number>& a, const DifferentiableNumber<Order, Number>& b)
{
	DifferentiableNumber<Order, Number> result(a);
	return result *= b;
}

template<int Order, typename Number>
DifferentiableNumber<Order, Number> operator/(const DifferentiableNumber<Order, Number>& a, Number b)
{
	DifferentiableNumber<Order, Number> result(a);
	return result *= (1 / b);
}

template<int Order, typename Number>
DifferentiableNumber<Order, Number> operator*(const DifferentiableNumber<Order, Number>& a, Number s)
{
	return DifferentiableNumber<Order, Number>(a) *= s;
}

template<int Order, typename Number>
DifferentiableNumber<Order, Number> operator*(Number s, const DifferentiableNumber<Order, Number>& a)
{
	return a * s;
}
inline int Factorial(int x)
{
	int v = 1;
	for(int i = 0; i < x; i++)
	{
		v*= x;
	}
	return v;
}


template<typename Number>
Number DPow(int order, Number x, int n)
{
	Number v = std::pow(x, n - order);
	for(int i = n; i > (n - order); i--)
	{
		v *= i;
	}
	return v;
}

template<int Order, typename Number>
DifferentiableNumber<Order, Number> Pow(DifferentiableNumber<Order, Number> x, int n)
{
	DifferentiableNumber<Order, Number> p = DifferentiableNumber<Order, Number>::Constant(1);
	for(int i = 0; i < n; i++)
	{
		p *= x;
	}
	for(int i = 1; i < Order; i++)
	{
		p[i] = DPow(i, x[0], n);
	}
	return p;
}



template<typename Number>
Number DSin(int order, Number x)
{
	if (order % 4 == 1)
	{
		return std::cos(x);
	}
	else if (order % 4 == 2)
	{
		return -std::sin(x);
	}
	else if (order % 4 == 3)
	{
		return -std::cos(x);
	}
	else 
		return std::sin(x);
	
}
//compute any order direvatives for elemetary function
template<int Order, typename Number>
DifferentiableNumber<Order, Number> Sin(DifferentiableNumber<Order, Number> x)
{
	Number value = x[0];
	DifferentiableNumber<Order, Number> sinx = DifferentiableNumber<Order, Number>::Constant(std::sin(value));
	for(int i = 1;i < Order+1; i++)
	{
		if( i == 1)
		{
			sinx[i] = x[i] * DSin(i, value);
		}
		else if( i == 2)
		{
			sinx[i] = Number(1)/Number(2) * (x[i-1] * x[i-1] * DSin(i, value) + x[i] * DSin(i-1, value));
		}
		else if (i == 3)
		{
			sinx[i] = Number(1)/Number(2 * 3) * (DSin(i, value)* x[i-2] * x[i-2] * x[i-2] + 3 * x[i-2] * x[i-1] * DSin(i-1, value) + x[i] * DSin(i-2, value));
		}
		else
		{
			//TODO: taylor series of f(g(x))
		}
	}
	return sinx;
}

#endif
