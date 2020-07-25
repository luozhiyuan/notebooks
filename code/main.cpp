#include <memory>
#include <cmath>
#include <array>
#include <map>
#include <algorithm>

#include <iostream>
#include "test_config.hpp"
#include "rng.hpp"
#include "render.hpp"
#include "hmc_test.hpp"
#include "timed_operation.hpp"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


void GroundTruth(int w, int h, int components, float plotRadius, unsigned char* data, SimpleURNG& rng)
{
	for(int i = 0; i < h; i++)
	{
		float y = (float(i)/h - 0.5f) * plotRadius;
		for(int j = 0; j < w; j++)
		{
			float x = (float(j)/w - 0.5f) * plotRadius;
			unsigned char value =  render(x, y) * 255;
			data[(i * w + j ) * components] = value;
		}
	}
}


void SaveResult(const char* fileName, int w, int h, float plotRadius, void (*f)(int w, int h,int components, float plotRadius, unsigned char* data, SimpleURNG& rng))
{
	SimpleURNG rng;

	int components = 1;
	unsigned char* data = new unsigned char[w*h*components];
	::memset(data, 0, w*h*components * sizeof(unsigned char));

	f(w, h, components, plotRadius, data, rng);

	stbi_write_png(fileName, w, h, components, data, 0);

	delete[] data;
}

//we just happens to know the distribution
void MonteCarloInverse(int w, int h, int components, float plotRadius, unsigned char* data, SimpleURNG& rng)
{
	int size = w * h;

	//cheating from pdf
	unsigned char* pdf = new unsigned char[size * components];
	GroundTruth(w, h, components, plotRadius, pdf, rng);
	unsigned long *cdf = new unsigned long[size * components];
	cdf[0] = pdf[0];
	for(int i = 1; i < size; i++)
	{
		cdf[i] += cdf[i-1] + pdf[i];
	}


	unsigned long total = cdf[size - 1];
	double zeta = 0;

	double* accumulated_data = new double[size];
	::memset(accumulated_data, 0, sizeof(double) * size);

	int total_samples = 0;
	auto duration = TimedOperation(kTestTime, [&]() {
		int one_iteration_count = size;
		while (one_iteration_count-- > 0) {
			rng.Get1D(&zeta);
			int index = std::lower_bound(cdf, cdf + size, zeta * total) - cdf;
			accumulated_data[index] += pdf[index];
			total_samples++;
		}
	});

	for(int i = 0; i < size; i++)
	{
		data[i] = std::min(255.0, accumulated_data[i] * double(size)/double(total_samples));
	}

	delete [] cdf;
	delete [] pdf;

	std::cout << "MonteCarloInverse real time: " << duration.count() << "ms" << std::endl;
}

void MonteCarloRejection(int w, int h, int components, float plotRadius, unsigned char* data, SimpleURNG& rng)
{
	const int size = w*h;

	double* accumulated_data = new double[size];
	::memset(accumulated_data, 0, sizeof(double) * size);

	int total_sample_count = 0;
	double zeta1, zeta2;
	auto duration = TimedOperation(kTestTime, [&]() {
		int one_iteration_count = size;
		while (one_iteration_count-- > 0) {
			rng.Get2D(&zeta1, &zeta2);
			int i = int(h*zeta1);
			int j = int(w*zeta2);
			float y = (zeta1 - 0.5f) * plotRadius;
			float x = (zeta2 - 0.5f) * plotRadius;
			double zeta3 = 0;
			int accept = 0;
			rng.Get1D(&zeta3);
			accept += (zeta3 < render(x, y)) ? 1 : 0;
			accumulated_data[(i * w + j) * components] += accept;
			total_sample_count++;
		}
	});

	for(int i = 0; i < size; i++)
	{
		data[i] = (255 * std::min(1.0, accumulated_data[i] / (double(total_sample_count)/double(size))));
	}

	delete[] accumulated_data;

	std::cout << "MonteCarloRejection real time: " << duration.count() << "ms" << std::endl;
}

//reference from http://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Metropolis_Sampling.html
//with symmetric transition
void Mutate(SimpleURNG& rng, double zeta1, double zeta2, double *next_zeta1, double *next_zeta2)
{
	double u = 0;
	rng.Get1D(&u);
	if(u < 0.1) //large step
	{
		rng.Get2D(next_zeta1, next_zeta2);
	}
	else //mutate
	{
		double u1,u2;
		rng.Get2D(&u1, &u2);
		*next_zeta1 = std::fmod(1+ zeta1 + 0.001*(u1-0.5), 1.0);
		*next_zeta2 = std::fmod(1+ zeta2 + 0.001*(u2-0.5), 1.0);
	}
}

void Record(double* data, int w, int h , int components, double zeta1, double zeta2, double f)
{
	int i = int(h*zeta1);
	int j = int(w*zeta2);
	data[(i * w + j ) * components] += f;
}

void MetropolisMCMC(int w, int h, int components, float plotRadius, unsigned char* data, SimpleURNG& rng)
{
	//initial state
	double zeta1, zeta2;
	rng.Get2D(&zeta1, &zeta2);
	double y = (zeta1 - 0.5f) * plotRadius;
	double x = (zeta2 - 0.5f) * plotRadius;
	double f = render(x, y);
	double *accumulated = new double[w*h*components];
	::memset(accumulated, 0, sizeof(double) * w*h*components);

	int total_sample_count = 0;
	TimedOperation(kTestTime, [&]()
	{
		int batch_size = w*h;
		while (batch_size-- > 0)
		{
			double next_zeta1, next_zeta2;
			Mutate(rng, zeta1, zeta2, &next_zeta1, &next_zeta2);

			//next state
			double next_y = (next_zeta1 - 0.5f) * plotRadius;
			double next_x = (next_zeta2 - 0.5f) * plotRadius;
			double next_f = render(next_x, next_y);

			//acceptance
			if (std::abs(next_f) < 1e-10f)
			{
				next_f = 1e-10f;
			}
			double a = std::min(1.0, next_f / f);
			//expected value optimization
			Record(accumulated, w, h, components, zeta1, zeta2, (1 - a) * f);
			Record(accumulated, w, h, components, next_zeta1, next_zeta2, a * next_f);
			double u = 0;
			rng.Get1D(&u);
			if (u < a)
			{
				zeta1 = next_zeta1;
				zeta2 = next_zeta2;
				f = next_f;
			}
			total_sample_count++;
		}
	}
	);
	for (int i = 0; i < w*h*components; i++)
	{
		data[i] = 255 * std::min(1.0, accumulated[i] / (double(total_sample_count) / double(w*h)));
	}
	delete[] accumulated;
}

int main()
{
	int w = 1024;
	int h = 1024;
	float plotRadius = 32;
	SaveResult("xx_yy_ground_truth.png", w,h, plotRadius, GroundTruth);
	SaveResult("xx_yy_mc_inverse.png", w, h, plotRadius, MonteCarloInverse);
	SaveResult("xx_yy_mc_rejection.png", w, h, plotRadius, MonteCarloRejection);
	SaveResult("xx_yy_mc_metropolis.png", w, h, plotRadius, MetropolisMCMC);
	//SaveResult("xx_yy_hmc.png", w, h, plotRadius, TestHMC);
	//SaveResult("xx_yy_h2mc.png", w, h, plotRadius, TestHessianHamiltonMC);
}
