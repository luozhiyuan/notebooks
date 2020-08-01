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


void GroundTruth(int w, int h, float plotRadius, float* data, SimpleURNG& rng)
{
	for(int i = 0; i < h; i++)
	{
		float y = (float(i)/h - 0.5f) * plotRadius;
		for(int j = 0; j < w; j++)
		{
			float x = (float(j)/w - 0.5f) * plotRadius;
			float value =  static_cast<float>(render(x, y));
			data[(i * w + j )] = value;
		}
	}
}


void SaveResult(const char* fileName, int w, int h, float plotRadius, void (*f)(int w, int h, float plotRadius, float* data, SimpleURNG& rng))
{
	SimpleURNG rng;

	int size = w * h;
	float *intensity = new float[size];
	f(w, h, plotRadius, intensity, rng);

	int components = 1;
	unsigned char* data = new unsigned char[size*components];
	::memset(data, 0, size*components * sizeof(unsigned char));

	//ramp ?
	for (int i = 0; i < size; i++) 
	{
		float result = intensity[i];
		for (int c = 0; c < components; c++)
		{
			data[i*components + c] = static_cast<unsigned char>(255 * result);
		}
	}

	stbi_write_png(fileName, w, h, components, data, 0);

	delete[] data;
	delete[] intensity;
}

//we just happens to know the distribution
void MonteCarloInverse(int w, int h, float plotRadius, float* data, SimpleURNG& rng)
{
	int size = w * h;

	//cheating from pdf
	float* pdf = new float[size];
	GroundTruth(w, h, plotRadius, pdf, rng);
	float *cdf = new float[size];
	::memset(cdf, 0, sizeof(float) * size);
	cdf[0] = pdf[0];
	for(int i = 1; i < size; i++)
	{
		cdf[i] += cdf[i-1] + pdf[i];
	}


	float total = cdf[size - 1];
	float zeta = 0;

	float* accumulated_data = new float[size];
	::memset(accumulated_data, 0, sizeof(float) * size);

	int total_samples = 0;
	auto duration = TimedOperation(kTestTime, [&]() {
		int one_iteration_count = size;
		while (one_iteration_count-- > 0) {
			rng.Get1D(&zeta);
			auto index = std::lower_bound(cdf, cdf + size, zeta * total) - cdf;
			accumulated_data[index] += pdf[index];
			total_samples++;
		}
	});

	for(int i = 0; i < size; i++)
	{
		data[i] = (std::min(1.0f, accumulated_data[i] * float(size)/float(total_samples)));
	}

	delete [] cdf;
	delete [] pdf;
	delete[] accumulated_data;

	std::cout << "MonteCarloInverse real time: " << duration.count() << "ms" << std::endl;
}

void MonteCarloRejection(int w, int h, float plotRadius, float* data, SimpleURNG& rng)
{
	const int size = w*h;

	float* accumulated_data = new float[size];
	::memset(accumulated_data, 0, sizeof(float) * size);

	int total_sample_count = 0;
	float zeta1, zeta2;
	auto duration = TimedOperation(kTestTime, [&]() {
		int one_iteration_count = size;
		while (one_iteration_count-- > 0) {
			rng.Get2D(&zeta1, &zeta2);
			int i = int(h*zeta1);
			int j = int(w*zeta2);
			float y = (zeta1 - 0.5f) * plotRadius;
			float x = (zeta2 - 0.5f) * plotRadius;
			float zeta3 = 0;
			int accept = 0;
			rng.Get1D(&zeta3);
			accept += (zeta3 < render(x, y)) ? 1 : 0;
			accumulated_data[(i * w + j)] += accept;
			total_sample_count++;
		}
	});

	for(int i = 0; i < size; i++)
	{
		data[i] = static_cast<float>(std::min(1.0f, accumulated_data[i] / (float(total_sample_count)/float(size))));
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

void Record(double* data, int w, int h, double zeta1, double zeta2, double f)
{
	int i = int(h*zeta1);
	int j = int(w*zeta2);
	data[(i * w + j )] += f;
}

void MetropolisMCMC(int w, int h, float plotRadius, float* data, SimpleURNG& rng)
{
	//initial state
	double zeta1, zeta2;
	rng.Get2D(&zeta1, &zeta2);
	double y = (zeta1 - 0.5f) * plotRadius;
	double x = (zeta2 - 0.5f) * plotRadius;
	double f = render(x, y);
	double *accumulated = new double[w*h];
	::memset(accumulated, 0, sizeof(double) * w*h);

	int total_sample_count = 0;
	auto duration = TimedOperation(kTestTime, [&]()
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
			Record(accumulated, w, h, zeta1, zeta2, (1 - a) * f);
			Record(accumulated, w, h, next_zeta1, next_zeta2, a * next_f);
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
	for (int i = 0; i < w*h; i++)
	{
		data[i] = static_cast<float>(std::min(1.0, accumulated[i] / (double(total_sample_count) / double(w*h))));
	}
	delete[] accumulated;
	std::cout << "MetropolisMCMC real time: " << duration.count() << "ms" << std::endl;
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
	SaveResult("xx_yy_hmc.png", w, h, plotRadius, TestHMC);
	SaveResult("xx_yy_h2mc.png", w, h, plotRadius, TestHessianHamiltonMC);

	char pause;
	std::cin>>pause;
	return 0;
}
