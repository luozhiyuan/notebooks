#ifndef _RENDER_HPP_
#define _RENDER_HPP_

double render(double x, double y)
{
	double v = std::sin(x*x + y*y);
	return (1 + v)/2.0 + 0.001;
}

#endif
