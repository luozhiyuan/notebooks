#ifndef _PARTIAL_DIFFERENTIABLE_NUMBER_HPP_
#define _PARTIAL_DIFFERENTIABLE_NUMBER_HPP_

#include <array>

//multivariable automatic differentiation demo 
template<typename Number = double>
class PartialDifferentiableNumber 
{
	int index = 0;
public:
	Number x;
	Number dx_i;
	Number dx_j;
	Number dxx;

	PartialDifferentiableNumber(Number x):x(x)
	{
		dx_i = dx_j = dxx = Number();
	}
	PartialDifferentiableNumber<Number>&
		operator +=(const PartialDifferentiableNumber<Number>& other) 
	{
		x += other.x;
		dx_i += other.dx_i;
		dx_j += other.dx_j;
		dxx += other.dxx;
		return *this;
	}

	PartialDifferentiableNumber<Number>&
		operator -=(const PartialDifferentiableNumber<Number>& other) 
	{
		x -= other.x;
		dx_i -= other.dx_i;
		dx_j -= other.dx_j;
		dxx -= other.dxx;
		return *this;
	}

	
	PartialDifferentiableNumber<Number>&
		operator*=(const PartialDifferentiableNumber<Number>& other)
	{
		dxx = x * other.dxx + dx_i * other.dx_j + dx_j * other.dx_i + dxx * other.x;
		dx_j = dx_j * other.x + x * other.dx_j;
		dx_i = dx_i * other.x + x * other.dx_i;
		x *= other.x;
		return *this;
	}


	PartialDifferentiableNumber<Number>
		Inverse()const
	{
		PartialDifferentiableNumber<Number> inv;
		inv.x = Number(1) / x;
		inv.dx_i = -dx_i / (x*x);
		inv.dx_j = -dx_j / (x*x);
		inv.dxx = Number(2) * dx_i * dx_j / (x*x*x) - dxx / (x*x);
	}
	PartialDifferentiableNumber<Number>&
		operator/=(const PartialDifferentiableNumber<Number>& other) 
	{
		return (*this) *= other.Inverse();
	}
	
};

template<typename Number>
PartialDifferentiableNumber<Number>
operator *(const PartialDifferentiableNumber<Number>& lhs, const PartialDifferentiableNumber<Number>& rhs)
{
	auto result(lhs);
	return result *= rhs;
}

template<typename Number>
PartialDifferentiableNumber<Number>
operator +(const PartialDifferentiableNumber<Number>& lhs, const PartialDifferentiableNumber<Number>& rhs)
{
	auto result(lhs);
	return result += rhs;
}

template<typename Number>
PartialDifferentiableNumber<Number>
operator /(const PartialDifferentiableNumber<Number>& lhs, const PartialDifferentiableNumber<Number>& rhs)
{
	auto result(lhs);
	return result /= rhs;
}

template<typename Number,typename F, typename DF, typename DDF>
PartialDifferentiableNumber<Number>
ScalarF(const PartialDifferentiableNumber<Number>& x, const F& f, const DF& df, const DDF& ddf) 
{
	PartialDifferentiableNumber<Number> r(f(x.x));
	r.dx_i = x.dx_i * df(x.x);
	r.dx_j = x.dx_j * df(x.x);
	r.dxx = x.dxx * df(x.x) + x.dx_i * x.dx_j * ddf(x.x);
	return r;
}



#endif