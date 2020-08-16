#pragma once

#include "erand48.h"
#include <cmath>

const double PI  =3.141592653589793238463;
const float  PI_F=3.14159265358979f;

inline void uniform_sampling_sphere(unsigned short* Xi, double* x, double* y, double* z)
{
	double theta = 2 * PI * erand48(Xi);
	double phi = acos(1 - 2 * erand48(Xi));
	*x = sin(phi) * cos(theta);
	*y = sin(phi) * sin(theta);
	*z = cos(phi);
}

//Point2f ConcentricSampleDisk(const Point2f &u) {
//    // Map uniform random numbers to $[-1,1]^2$
//    Point2f uOffset = 2.f * u - Vector2f(1, 1);
//
//    // Handle degeneracy at the origin
//    if (uOffset.x == 0 && uOffset.y == 0) return Point2f(0, 0);
//
//    // Apply concentric mapping to point
//    Float theta, r;
//    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
//        r = uOffset.x;
//        theta = PiOver4 * (uOffset.y / uOffset.x);
//    } else {
//        r = uOffset.y;
//        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
//    }
//    return r * Point2f(std::cos(theta), std::sin(theta));
//}
//
//inline Vector3f CosineSampleHemisphere(const Point2f &u) {
//    Point2f d = ConcentricSampleDisk(u);
//    Float z = std::sqrt(std::max((Float)0, 1 - d.x * d.x - d.y * d.y));
//    return Vector3f(d.x, d.y, z);
//}
//


