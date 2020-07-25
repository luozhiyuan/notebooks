#include <iostream>
#include <iomanip>
#include <random>
#include <numeric>
#include "differentiable_number.hpp"
#include "partial_differentiable_number.hpp"
#include "rng.hpp"
#include "hmc_test.hpp"

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
	static PartialDifferentiableNumber<double> f2(const PartialDifferentiableNumber<double>& x, const PartialDifferentiableNumber<double>& y)
	{
		return negative_log(
			PartialDifferentiableNumber<double>(1)  // 这里加上一个常数缩放一下原分布，因为我们本身就只能求正比于原函数的函数，而且这里原分布会取到0，取-log，出现类似于黑洞一样的无穷大，导致无法逃离，从而会卡在一个地方很久才能逃离，出现较大的variance。
			+ xysinxy(x, y));
	}
public:
	double f(double x, double y) {
		return (1 + std::sin(x*x + y * y))/2;
	}

	std::array<double, 2> Grad(const std::array<double, 2>& q)const 
	{
		std::array<double, 2> result;
		PartialDifferentiableNumber<double> x(q[0]), y(q[1]);
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
	std::array<double, 4> Hessian(const std::array<double, 2>& q)const 
	{
		std::array<double, 4> result;
		PartialDifferentiableNumber<double> x(q[0]), y(q[1]);
		//d/dxdx
		x.dx_i = 1;
		x.dx_j = 1;
		y.dx_i = 0;
		y.dx_j = 0;
		result[0] = f2(x,y).dxx;

		x.dx_i = 1;
		x.dx_j = 0;
		y.dx_i = 0;
		y.dx_j = 1;
		result[2] = result[1] = f2(x, y).dxx;

		x.dx_i = 0;
		x.dx_j = 0;
		y.dx_i = 1;
		y.dx_j = 1;
		result[3] = f2(x, y).dxx;
		return result;
	}
	double operator()(const std::array<double, 2>& q)const
	{
		PartialDifferentiableNumber<double> x(q[0]), y(q[1]);
		return f2(x,y).x;
	}
};



template<typename U, typename Number, int Dim>
bool HMC(const U u_gradu, Number epsilon, int L, const std::array<Number, Dim>& current_q, SimpleNRNG<Dim>& nrng, SimpleURNG& urng, std::array<Number, Dim>* new_q)
{
	auto q = current_q;
	std::array<double, Dim> p;
	nrng.Get(&p);
	auto current_p = p;

	//integrate
	for (int i = 0; i < L; i++) 
	{
		p = p - 0.5 * epsilon * u_gradu.Grad(q);
		q = q + epsilon * p;
		p = p - epsilon * 0.5 * u_gradu.Grad(q);
	}

	//time reversable
	p = -p;
	auto current_U = u_gradu(current_q);
	auto current_K = dot(current_p, current_p) / 2.0;
	auto proposed_U = u_gradu(q);
	auto proposed_K = dot(p, p) / 2.0;

	double zeta = 0;
	urng.Get1D(&zeta);
	auto exponent = current_U - proposed_U + current_K - proposed_K;
	if (zeta < std::exp(exponent)) 
	{
		*new_q = q;
		return true;
	}
	*new_q = current_q;
	return false;
}

void hamiltonMCMC(int w, int h, int components, float plotRadius, unsigned char* data, SimpleURNG& rng) 
{
	TargetDist u;

	double boostrap_total = 0;
	const int boostrap_count = w * h;
	for (int i = 0; i < boostrap_count; i++) {
		double zeta1, zeta2;
		rng.Get2D(&zeta1, &zeta2);
		boostrap_total += u.f(plotRadius * (zeta1 - 0.5), plotRadius * (zeta2 - 0.5));
	}
	
	double normalization = boostrap_total / boostrap_count;
	


	SimpleNRNG<2> nrng;

	double target_accept_rate = 0.8f;
	double epsilon =  plotRadius * 1.0 / (std::max(w, h) );
	const int PerPixelSample = 10;
	std::array<double, 2> q;
	double zeta1, zeta2;
	rng.Get2D(&zeta1, &zeta2);
	q[0] = plotRadius * (zeta1 - 0.5);
	q[1] = plotRadius * (zeta2 - 0.5);

	double* accept_score = new double[w*h*components];
	::memset(accept_score, 0, sizeof(double) * w * h);

	double total_count = 0;
	double total_accept = 0;
	int integral_steps = 50;
	for (int i = 0; i < h; i++) 
	{
		for (int j = 0; j < w; j++) 
		{
			for (int sample = 0; sample < PerPixelSample; sample++) 
			{
				auto current_q = q;
				bool accept = HMC(u, epsilon, integral_steps, current_q, nrng, rng, &q);

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
					epsilon = std::max(epsilon, 1.0/(std::max(w,h)));
				}
			}
		}
	}
	std::cout << "accpet rate: " << total_accept / total_count << std::endl;
	for (int i = 0; i < w*h; i++) 
	{
		data[i*components] = unsigned char(255 * std::min(accept_score[i] * boostrap_count / total_count, 1.0) );
	}

	//auto norm = std::accumulate(accept_score, accept_score + w * h*PerPixelSample, 0.0, [](double x, double init) {return std::max(x, init); });
	//for (int i = 0; i < w*h; i++) 
	//{
	//	data[i*components] = unsigned char(255 * std::min(accept_score[i] / norm, 1.0) );
	//}
	delete[]accept_score;
}

//直接用了论文代码，而不是照论文写了一遍，因为论文中实现和公式有错误，与代码不一致，而且更简化
//2d h2mc 
#include "h2mc/gaussian.h"
#include "h2mc/h2mc.h"


struct MarkovState {
	Float score;
    //Float scoreSum;
	std::array<double, 2> current_q;
    Gaussian gaussian;
};

template<typename U, typename Number>
void InitializeState(MarkovState* state, U u_gradu, const H2MCParam& h2mc_parameters, const std::array<Number, 2>& q)
{
	state->current_q = q;

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
Float H2MC(const U u_gradu, Float plotRadius, const H2MCParam& h2mc_params, const MarkovState& current_q, MarkovState* new_q, RNG& rng)
{
	Vector offset(2);
	GenerateSample(current_q.gaussian, offset, rng);
	//

	auto new_pos = current_q.current_q + decltype(current_q.current_q){offset[0]*plotRadius, offset[1]*plotRadius};
	InitializeState(new_q, u_gradu, h2mc_params, new_pos);
	Float py = GaussianLogPdf(offset, current_q.gaussian);
	Float px = GaussianLogPdf(-offset, new_q->gaussian);

	Float score = current_q.score;
	if (std::abs(score) < std::numeric_limits<Float>::epsilon()) 
	{
		score = current_q.score > 0 ? std::numeric_limits<Float>::epsilon() : -std::numeric_limits<Float>::epsilon();
	}
	Float a = std::exp(px - py) * (new_q->score) / score;
	a = std::max(Float(0), std::min(Float(1), a));
	return a;
}

void HessianHamiltonMC(int w, int h, int components, float plotRadius, unsigned char* data, SimpleURNG& rng)
{
	TargetDist u;

	double boostrap_total = 0;
	const int boostrap_count = w * h;
	for (int i = 0; i < boostrap_count; i++) {
		double zeta1, zeta2;
		rng.Get2D(&zeta1, &zeta2);
		boostrap_total += u.f(plotRadius * (zeta1 - 0.5), plotRadius * (zeta2 - 0.5));
	}
	
	double normalization = boostrap_total / boostrap_count;
	


	double target_accept_rate = 0.8f;
	const int PerPixelSample = 512;
	std::array<double, 2> q;
	double zeta1, zeta2;
	rng.Get2D(&zeta1, &zeta2);
	q[0] = plotRadius * (zeta1 - 0.5);
	q[1] = plotRadius * (zeta2 - 0.5);

	double* accept_count = new double[w*h*components];
	::memset(accept_count, 0, sizeof(double) * w * h);

	double total_count = 0;
	double total_accept = 0;

	H2MCParam h2mc_parameters;
	RNG std_rng;

	std::uniform_real_distribution<Float> uniDist(Float(0.0), Float(1.0));
	//initialize q
	MarkovState state;
	MarkovState new_state;
	InitializeState(&state, u, h2mc_parameters, q);
	for (int i = 0; i < h; i++) 
	{
		for (int j = 0; j < w; j++) 
		{
			for (int sample = 0; sample < PerPixelSample; sample++) 
			{
				auto current_q = state;
				Float accept = H2MC(u, plotRadius, h2mc_parameters, current_q, &new_state, std_rng);
				bool accepted = uniDist(std_rng) < accept;
				int x; 
				int y; 
				if (accepted) {
					q = new_state.current_q;
				}
				else 
				{
					q = current_q.current_q;
				}
				x = (int)((q[0] / plotRadius + 0.5) * w);
				y = int((q[1] / plotRadius + 0.5)* h);
	
    			if (x >= 0 && y >= 0 && x < w && y < h)
				{
					if (accepted)
					{
						state = new_state;
					}
					else 
					{
						state = current_q;
					}
					accept_count[y * w + x] += normalization;
     			}
				else
				{
					state = current_q;
				}
				total_count++;
				if (accepted)
					total_accept++;
				if (total_accept / total_count < target_accept_rate) {
				}
			}
		}
	}
	std::cout << "accpet rate: " << total_accept / total_count << std::endl;
	for (int i = 0; i < w*h; i++) 
	{
		data[i*components] = unsigned char(255 * std::min(accept_count[i] * boostrap_count/total_count, 1.0) );
	}
	//auto norm = std::accumulate(accept_count, accept_count + w * h*PerPixelSample, 0.0, [](double x, double init) {return std::max(x, init); });
	//for (int i = 0; i < w*h; i++) 
	//{
	//	data[i*components] = unsigned char(255 * std::min(accept_count[i] / norm, 1.0) );
	//}
	delete[]accept_count;
}


void TestHessianHamiltonMC(int w, int h, int components, float plotRadius, unsigned char* data, SimpleURNG& urng) 
{
	::memset(data, 0, sizeof(unsigned char) * w * h *components);
	HessianHamiltonMC(w, h, components, plotRadius, data, urng);
}

void TestHMC(int w, int h, int components, float plotRadius, unsigned char* data, SimpleURNG& urng) 
{
	::memset(data, 0, sizeof(unsigned char) * w * h *components);
	hamiltonMCMC(w, h, components, plotRadius, data, urng);
}

//#include "stb_image_write.h"
//void testHMC() {
//	int w = 1024;
//	int h = 1024;
//	int components = 1;
//	float plotRadius = 32;
//	unsigned char* data = new unsigned char[w*h*components];
//	::memset(data, 0, sizeof(unsigned char) * w * h *components);
//	SimpleURNG urng;
//	hamiltonMCMC(w, h, components, plotRadius, data, urng);
//	stbi_write_png("hmc.png", w, h, components, data, 0);
//	delete[]data;
//}
//
//void testH2MC() {
//	int w = 1024;
//	int h = 1024;
//	int components = 1;
//	float plotRadius = 32;
//	unsigned char* data = new unsigned char[w*h*components];
//	::memset(data, 0, sizeof(unsigned char) * w * h *components);
//	SimpleURNG urng;
//	HessianHamiltonMC(w, h, components, plotRadius, data, urng);
//	stbi_write_png("h2mc.png", w, h, components, data, 0);
//	delete[]data;
//}
//


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
//
//	{
//		testHMC();
//	}
//}
