#include <memory>
#include <cmath>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
int main()
{
	int w = 1024;
	int h = 1024;
	int components = 1;
	unsigned char* data = new unsigned char[w*h*components];
	float PlotRadius = 20;
	for(int i = 0; i < h; i++)
	{
		float y = (float(i)/h - 0.5f) * PlotRadius;
		for(int j = 0; j < w; j++)
		{
			float x = (float(j)/w - 0.5f) * PlotRadius;
			unsigned char value = (1 + std::sin( x*x + y*y))/2.0f * 255;
			data[(i * w + j ) * components] = value;
		}
	}
	stbi_write_png("xx_yy.png", w, h, components, data, 0);
}
