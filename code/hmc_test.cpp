#include <iostream>
#include <iomanip>
#include <random>
#include <numeric>
#include <cstring>
#include "partial_differentiable_number.hpp"
#include "rng.hpp"
#include "hmc_test.hpp"
#include "timed_operation.hpp"
#include "test_config.hpp"
#include "h2mc/commondef.h"

template<typename Number>
PartialDifferentiableNumber<Number>
xysinxy(const PartialDifferentiableNumber<Number>& x, const PartialDifferentiableNumber<Number>& y)
{
	auto Sin = [](Number x) {return std::sin(x); };
	auto DSin = [](Number x) {return std::cos(x); };
	auto DDSin = [](Number x) {return -std::sin(x); };
	return (PartialDifferentiableNumber<Number>(1)+ScalarF(
		x * x + y * y, 
		Sin, 
		DSin, 
		DDSin))*PartialDifferentiableNumber<Number>(0.5);
}

template<typename Number>
PartialDifferentiableNumber<Number>
negative_log(const PartialDifferentiableNumber<Number>& x)
{
	auto log = [](Number x) {return -std::log(x); };
	auto dlog = [](Number x) {return -1/x; };
	auto ddlog = [](Number x) {return 1/(x*x); };
	return ScalarF(x, log, dlog, ddlog);
}

//懒得写一个vector类, array凑合一下
template<typename Number, int Dim>
Number dot(const std::array<Number, Dim>& lhs, const std::array<Number, Dim>& rhs)
{
	Number r = Number();
	for (size_t i = 0; i < lhs.size(); i++) 
	{
		r += lhs[i] * rhs[i];
	}
	return r;
}

template<typename Number, int Dim>
std::array<Number, Dim> operator -(const std::array<Number, Dim>& rhs)
{
	std::array<Number, Dim> r;
	for (size_t i = 0; i < rhs.size(); i++) 
	{
		r[i] = 0 - rhs[i];
	}
	return r;
}

template<typename Number, int Dim>
std::array<Number, Dim> operator +(const std::array<Number, Dim>& lhs, const std::array<Number, Dim>& rhs)
{
	std::array<Number, Dim> r;
	for (size_t i = 0; i < lhs.size(); i++) 
	{
		r[i] = lhs[i] + rhs[i];
	}
	return r;
}



template<typename Number, int Dim>
std::array<Number, Dim> operator -(const std::array<Number, Dim>& lhs, const std::array<Number, Dim>& rhs)
{
	std::array<Number, Dim> r;
	for (size_t i = 0; i < lhs.size(); i++) 
	{
		r[i] = lhs[i] - rhs[i];
	}
	return r;
}

template<typename Number, int Dim>
std::array<Number, Dim> operator *(Number lhs, const std::array<Number, Dim>& rhs)
{
	std::array<Number, Dim> r;
	for (size_t i = 0; i < rhs.size(); i++) 
	{
		r[i] = lhs * rhs[i];
	}
	return r;
}

template<typename Number, int Dim>
std::array<Number, Dim> operator *(const std::array<Number, Dim>& rhs, Number lhs)
{
	return lhs * rhs;
}

class TargetDist 
{
	static PartialDifferentiableNumber<Float> f2(const PartialDifferentiableNumber<Float>& x, const PartialDifferentiableNumber<Float>& y)
	{
		return negative_log(
			PartialDifferentiableNumber<Float>(1)  // 这里加上一个常数缩放一下原分布，因为我们本身就只能求正比于原函数的函数，而且这里原分布会取到0，取-log，出现类似于黑洞一样的无穷大，导致无法逃离，从而会卡在一个地方很久才能逃离，出现较大的variance。
			+ xysinxy(x, y));
	}
public:
	Float f(Float x, Float y) {
		return (1 + std::sin(x*x + y * y))/2;
	}

	std::array<Float, 2> Grad(const std::array<Float, 2>& q)const 
	{
		std::array<Float, 2> result;
		PartialDifferentiableNumber<Float> x(q[0]), y(q[1]);
		//d/dx
		x.dx_i = 1;
		y.dx_i = 0;
		result[0] = f2(x,y).dx_i;

		x.dx_i = 0;
		y.dx_i = 1;
		result[1] = f2(x, y).dx_i;
		return result;
	}

	//00, 01, 10, 11
	std::array<Float, 4> Hessian(const std::array<Float, 2>& q)const 
	{
		std::array<Float, 4> result;
		PartialDifferentiableNumber<Float> x(q[0]), y(q[1]);
		//d/dxdx
		x.dx_i = 1;
		x.dx_j = 1;
		y.dx_i = 0;
		y.dx_j = 0;
		result[0] = f2(x,y).dxx;

		//d/dxdy
		x.dx_i = 1;
		x.dx_j = 0;
		y.dx_i = 0;
		y.dx_j = 1;
		result[2] = result[1] = f2(x, y).dxx;

		//d/dydy
		x.dx_i = 0;
		x.dx_j = 0;
		y.dx_i = 1;
		y.dx_j = 1;
		result[3] = f2(x, y).dxx;
		return result;
	}
	Float operator()(const std::array<Float, 2>& q)const
	{
		PartialDifferentiableNumber<Float> x(q[0]), y(q[1]);
		return f2(x,y).x;
	}
};



template<typename U, typename Number, int Dim>
bool HMC(const U& u_gradu, Number epsilon, int L, const std::array<Number, Dim>& current_state, SimpleNRNG<Dim>& nrng, SimpleURNG& urng, std::array<Number, Dim>* proposal_state)
{
	auto q = current_state;
	std::array<Float, Dim> p;
	nrng.Get(&p);
	auto current_p = p;

	//integrate
	for (int i = 0; i < L; i++) 
	{
		p = p - (static_cast<Number>(0.5 * epsilon) * u_gradu.Grad(q));
		q = q + (epsilon * p);
		p = p - (static_cast<Number>(epsilon * 0.5) * u_gradu.Grad(q));
	}

	//time reversable
	p = -p;
	auto current_U = u_gradu(current_state);
	auto current_K = dot(current_p, current_p) / 2.0;
	auto proposed_U = u_gradu(q);
	auto proposed_K = dot(p, p) / 2.0;

	Float zeta = 0;
	urng.Get1D(&zeta);
	auto exponent = current_U - proposed_U + current_K - proposed_K;
	if (zeta < std::exp(exponent)) 
	{
		*proposal_state = q;
		return true;
	}
	*proposal_state = current_state;
	return false;
}

void HamiltonianMCMC(int w, int h, float plotRadius, float* data, SimpleURNG& rng) 
{
	TargetDist u;

	Float boostrap_total = 0;
	const int boostrap_count = w * h;
	for (int i = 0; i < boostrap_count; i++) {
		Float zeta1, zeta2;
		rng.Get2D(&zeta1, &zeta2);
		boostrap_total += u.f(static_cast<Float>(plotRadius * (zeta1 - 0.5)), static_cast<Float>(plotRadius * (zeta2 - 0.5)));
	}
	
	Float normalization = boostrap_total / boostrap_count;

	SimpleNRNG<2> nrng;

	Float target_accept_rate = 0.8f;
	Float epsilon =  static_cast<Float>(plotRadius * 1.0 / (std::max(w, h) ));
	std::array<Float, 2> q;
	Float zeta1, zeta2;
	rng.Get2D(&zeta1, &zeta2);
	q[0] = static_cast<Float>(plotRadius * (zeta1 - 0.5));
	q[1] = static_cast<Float>(plotRadius * (zeta2 - 0.5));

	Float* accept_score = new Float[w*h];
	::memset(accept_score, 0, sizeof(Float) * w * h);

	Float total_count = 0;
	Float total_accept = 0;
	int integral_steps = 50;
	auto real_time = TimedOperation(kTestTime, [&]()
	{
		int batch_size = w*h;
		while(batch_size -- )
		{
			auto current_q = q;
			bool accept = HMC<TargetDist, Float, 2>(u, epsilon, integral_steps, current_q, nrng, rng, &q);

			int x;
			int y;

			x = (int)((q[0] / plotRadius + 0.5) * w);
			y = int((q[1] / plotRadius + 0.5)* h);
			if (x >= 0 && y >= 0 && x < w && y < h)
			{
				accept_score[y * w + x] += normalization;
			}
			else
			{
				q = current_q;
			}
			total_count++;
			if (accept)
				total_accept++;
			if (total_accept / total_count < target_accept_rate) {
				epsilon -= epsilon / integral_steps;
				epsilon = std::max(epsilon, static_cast<Float>(1.0 / (std::max(w, h))));
			}
		}
	});
	for (int i = 0; i < w*h; i++) 
	{
		data[i] = (float)(std::min(accept_score[i] * boostrap_count / total_count, static_cast<Float>(1.0)) );
	}
	delete[]accept_score;

	std::cout <<"HamiltonianMCMC cost: " <<real_time.count() << "ms" << " with accpet rate: " << total_accept / total_count << std::endl;
}

//直接用了论文代码，而不是照论文写了一遍，因为论文中实现和公式有错误，与代码不一致，而且更简化
//2d h2mc 
#include "h2mc/gaussian.h"
#include "h2mc/h2mc.h"


struct MarkovState {
	Float score;
	std::array<Float, 2> q;
    Gaussian gaussian;
};

template<typename U, typename Number>
void InitializeState(MarkovState* state, U u_gradu, const H2MCParam& h2mc_parameters, const std::array<Number, 2>& q)
{
	state->q = q;

	AlignedStdVector vGrad;
    AlignedStdVector vHess;
	vGrad.resize(2);
	vHess.resize(2*2);
	auto grad = u_gradu.Grad(q);
	vGrad[0] = grad[0];
	vGrad[1] = grad[1];
	auto hess = u_gradu.Hessian(q);
	vHess[0] = hess[0];
	vHess[1] = hess[1];
	vHess[2] = hess[2];
	vHess[3] = hess[3];
	auto score = u_gradu(q);
	state->score = score;
	ComputeGaussian(h2mc_parameters, score, vGrad, vHess, state->gaussian);
}

template<typename U>
Float H2MC(const U u_gradu, Float plotRadius, const H2MCParam& h2mc_params, const MarkovState& current_state, MarkovState* proposal_state, RNG& rng)
{
	Vector offset(2);
	GenerateSample(current_state.gaussian, offset, rng);
	//

	auto new_pos = decltype(current_state.q){current_state.q[0] + offset[0]*plotRadius , current_state.q[1] + offset[1]*plotRadius};
	InitializeState(proposal_state, u_gradu, h2mc_params, new_pos);
	Float py = GaussianLogPdf(offset, current_state.gaussian);
	Float px = GaussianLogPdf(-offset, proposal_state->gaussian);

	Float score = current_state.score;
	if (std::abs(score) < std::numeric_limits<Float>::epsilon()) 
	{
		score = current_state.score > 0 ? std::numeric_limits<Float>::epsilon() : -std::numeric_limits<Float>::epsilon();
	}
	Float a = std::exp(px - py) * (proposal_state->score) / score;
	a = std::max(Float(0), std::min(Float(1), a));
	return a;
}

void HessianHamiltonianMC(int w, int h, float plotRadius, float* data, SimpleURNG& rng)
{
	TargetDist u;

	Float normalization;
	int boostrap_count = w * h;
	//算法本身是无偏的, 但可惜, MCMC只能得到一个与真实期望一致的结果(正比)
	const bool reality = false;
	if(reality)
	{
		//boostrap may introduce bais
		Float boostrap_total = 0;
		for (int i = 0; i < boostrap_count; i++) {
			Float zeta1, zeta2;
			rng.Get2D(&zeta1, &zeta2);
			boostrap_total += u.f(static_cast<Float>(plotRadius * (zeta1 - 0.5)), static_cast<Float>(plotRadius * (zeta2 - 0.5)));
		}
		normalization = boostrap_total / boostrap_count;
	}
	else
	{
		//cheating here try to get unbaised result
		Float boostrap_total = 0;
		boostrap_count = 0;
		for(int y= 0; y < h; y++)
		{
			for(int x = 0; x < w; x++)
			{
				Float zeta1, zeta2;
				zeta1 = Float(x)/Float(w);
				zeta2 = Float(y)/Float(h);
				boostrap_total += u.f(static_cast<Float>(plotRadius * (zeta1 - 0.5)), static_cast<Float>(plotRadius * (zeta2 - 0.5)));
				boostrap_count ++;
			}
		}
		normalization = boostrap_total / Float(boostrap_count);
	}

	
	Float target_accept_rate = 0.6f;//adaptive
	std::array<Float, 2> q;
	Float zeta1, zeta2;
	rng.Get2D(&zeta1, &zeta2);
	q[0] = static_cast<Float>(plotRadius * (zeta1 - 0.5));
	q[1] = static_cast<Float>(plotRadius * (zeta2 - 0.5));

	Float* accept_accumulation = new Float[w*h];
	::memset(accept_accumulation, 0, sizeof(Float) * w * h);

	Float total_count = 0;
	Float total_accept = 0;

	H2MCParam h2mc_parameters(Float(1));
	RNG std_rng;

	std::uniform_real_distribution<Float> uniform_distribution(Float(0.0), Float(1.0));
	//initialize q
	MarkovState state;
	MarkovState proposal_state;
	InitializeState(&state, u, h2mc_parameters, q);
	auto real_time = TimedOperation(kTestTime, [&]()
	{
		int batch_size = w*h;
		const int adaptive_check = 10000;
		while(batch_size -- > 0 )
		{
			auto current_state = state;
			Float accept_probability = H2MC(u, plotRadius, h2mc_parameters, current_state, &proposal_state, std_rng);
			bool accepted = uniform_distribution(std_rng) < accept_probability;

			int x;
			int y;
			if (accepted) 
			{
				//出界了重来吧
				x = (int)((proposal_state.q[0] / plotRadius + 0.5) * w);
				y = (int)((proposal_state.q[1] / plotRadius + 0.5)* h);
				if (!(x >= 0 && y >= 0 && x < w && y < h)) 
				{
					batch_size++;
					continue;
				}
			}
    		if (accepted) {
				q = proposal_state.q;
			}
			else
			{
				q = current_state.q;
			}
			x = (int)((q[0] / plotRadius + 0.5) * w);
			y = (int)((q[1] / plotRadius + 0.5)* h);

			if (x >= 0 && y >= 0 && x < w && y < h)
			{
				if (accepted)
				{
					accept_accumulation[y * w + x] += normalization * accept_probability;
					state = proposal_state;
				}
				else
				{
					accept_accumulation[y * w + x] += normalization * (1-accept_probability);
					state = current_state;
				}
			}
			else
			{
				state = current_state;
			}
			total_count++;
			if (accepted)
				total_accept++;
			if (( int(total_count) % adaptive_check) == 0 
				&& total_accept / total_count < target_accept_rate)  //try adaptive
			{
				//narrow down the gaussian
				h2mc_parameters.sigma *= 0.9f;
			}
		}
	});

	for (int i = 0; i < w*h; i++) 
	{
		data[i] = float(std::min(accept_accumulation[i] * boostrap_count/total_count, static_cast<Float>(1.0)) );
	}
	delete[]accept_accumulation;

	std::cout << "HessianHamiltonianMC cost: " << real_time.count() << "ms " << "with accpet rate: " << total_accept / total_count << "boostrap_count/total_count: " << boostrap_count/total_count << std::endl;
}


void TestHessianHamiltonMC(int w, int h, float plotRadius, float* data, SimpleURNG& urng) 
{
	::memset(data, 0, sizeof(float) * w * h);
	HessianHamiltonianMC(w, h, plotRadius, data, urng);
}

void TestHMC(int w, int h, float plotRadius, float* data, SimpleURNG& urng) 
{
	::memset(data, 0, sizeof(float) * w * h);
	HamiltonianMCMC(w, h, plotRadius, data, urng);
}
