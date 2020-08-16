#pragma once


struct Vec {        // Usage: time ./smallpt 5000 && xv image.ppm 
   double x, y, z;                  // position, also color (r,g,b) 
   Vec(double x_=0, double y_=0, double z_=0){ x=x_; y=y_; z=z_; } 
   Vec operator+(const Vec &b) const { return Vec(x+b.x,y+b.y,z+b.z); } 
   Vec operator-(const Vec &b) const { return Vec(x-b.x,y-b.y,z-b.z); } 
   Vec operator*(double b) const { return Vec(x*b,y*b,z*b); } 
   Vec mult(const Vec &b) const { return Vec(x*b.x,y*b.y,z*b.z); } 
   Vec norm()const { return *this * (1 / length()); }
   double dot(const Vec &b) const { return x*b.x+y*b.y+z*b.z; } // cross: 
   Vec operator%(Vec&b){return Vec(y*b.z-z*b.y,z*b.x-x*b.z,x*b.y-y*b.x);} 
   double length()const { return sqrt(length_sqr()); }
   double length_sqr()const { return (dot(*this)); }

   bool operator == (const Vec& other)const
   {
	   return other.x == x && other.y == y && other.z == z;
   }
   bool operator != (const Vec& other)const
   {
	   return !(*this == other);
   }
 }; 
 struct Ray { Vec o, d; Ray(Vec o_, Vec d_) : o(o_), d(d_) {} }; 
 enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance() 

 const double kRayEpsilon = 1e-4;

 struct Sphere { 
   double rad;       // radius 
   Vec p, e, c;      // position, emission, color 
   Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive) 
   Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_): 
     rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {} 
   double intersect(const Ray &r) const { // returns distance, 0 if nohit 
     Vec op = p-r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
     double t, eps=1e-4, b=op.dot(r.d), det=b*b-op.dot(op)+rad*rad; 
     if (det<0) return 0; else det=sqrt(det); 
     return (t=b-det)>eps ? t : ((t=b+det)>eps ? t : 0); 
   } 
 }; 

inline double clamp(double x){ return x<0 ? 0 : x>1 ? 1 : x; }
inline Vec clamp(const Vec& v){ return Vec(clamp(v.x), clamp(v.y),clamp(v.z)); }
inline int toInt(float x){ return int(pow(clamp(x),1/2.2)*255+.5); }

