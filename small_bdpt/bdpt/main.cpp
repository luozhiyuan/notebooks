#define _USE_MATH_DEFINES
#include <cmath>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <cstdlib> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt
#include <cstdio>  //        Remove "-fopenmp" for g++ version < 4.2
#include "erand48.h"
#include "smallpt.h"
#include "sampling.h"
#include <random>
#include <iostream>
#include "scoped_assignment.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define USE_BDPT

const int kGroundTruthSPP = 4096*8;

const int kMaxDepth = 12;
#ifdef _DEBUG
const int kDefaultSPP = 16;

Vec debug_ray_dir;
bool debug_ray_on = false;
#else
const int kDefaultSPP = 1024;
#endif

const int kLightIndex = 8;
Sphere spheres[] = {//Scene: radius, position, emission, color, material 
   Sphere(1e5, Vec(1e5 + 1,40.8,81.6), Vec(),Vec(.75,.25,.25),DIFF),//Left 
   Sphere(1e5, Vec(-1e5 + 99,40.8,81.6),Vec(),Vec(.25,.25,.75),DIFF),//Rght 
   Sphere(1e5, Vec(50,40.8, 1e5),     Vec(),Vec(.75,.75,.75),DIFF),//Back 
   Sphere(1e5, Vec(50,40.8,-1e5 + 170), Vec(),Vec(),           DIFF),//Frnt 
   Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(),Vec(.75,.75,.75),DIFF),//Botm 
   Sphere(1e5, Vec(50,-1e5 + 81.6,81.6),Vec(),Vec(.75,.75,.75),DIFF),//Top 
   Sphere(16.5,Vec(27,16.5,47),       Vec(),Vec(1,1,1)*.999, SPEC),//Mirr 
   Sphere(16.5,Vec(73,16.5,78),       Vec(),Vec(1,1,1)*.999, REFR),//Glas 
   Sphere(6, Vec(50,6,81.6),Vec(10,10,10),  Vec(), DIFF) //Lite 
};
const Vec kCameraPos = {50, 52, 295.6};

double g_A;
Vec g_camera_dir;
double g_camera_start = 140;

inline double AbsDot(const Vec& w0, const Vec& w1)
{
	return fabs(w0.dot(w1));
}

inline void CameraPDF(const Vec& dir, double *pdf_pos, double* pdf_dir)
{
	*pdf_pos = 1;
	double cos_theta = dir.dot(g_camera_dir);
	*pdf_dir = 1.0 / (g_A * cos_theta * cos_theta * cos_theta);
}
inline void CameraPDF(const Vec& pos, Vec* sample_pos, Vec* wi, double *pdf)
{
	*wi = kCameraPos - pos;
	double dist = wi->length();

	if (dist <= g_camera_start  ) //check face dir
	{
		*pdf = 0;
		return;
	}
	*wi = *wi * (1 / dist);


	Vec near = kCameraPos - *wi * g_camera_start;
	if ((pos - kCameraPos).norm().dot((near - kCameraPos).norm()) < 0)
	{
		*pdf = 0;
		return;
	}


	if (g_camera_dir.dot(*wi) > 0)
		*pdf = 0;
	else
		*pdf = (dist*dist) / AbsDot(g_camera_dir, *wi);
	*sample_pos = near;
}



inline double brdf_pdf(Refl_t brdf, const Vec& wo, const Vec& wi, const Vec& normal)
{
	if(brdf == DIFF)
		return wo.dot(normal) * wi.dot(normal) > 0? 1.0 / M_PI:0;
	return 0;
}

inline bool intersect(const Ray &r, double &t, int &id) {
	double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
	for (int i = int(n); i--;) if ((d = spheres[i].intersect(r)) && d < t) { t = d; id = i; }
	return t < inf;
}

inline bool intersect(const Ray &r, double ray_length, double &t, int &id) {
	t = ray_length + 1;
	double n = sizeof(spheres) / sizeof(Sphere), d;
	for (int i = int(n); i--;) if ((d = spheres[i].intersect(r)) && d < t && d < ray_length)
	{
		t = d; id = i;
		return true; 
	}
	return t < ray_length;
}

bool is_visible(const Vec& p0, const Vec& p1)
{
	double t;
	int id;
	Vec dir = (p1 - p0);
	double ray_length = dir.length();
	dir = dir * (1.0 / ray_length);
	return !intersect(Ray(p0 + dir * kRayEpsilon, dir), ray_length - 2 * kRayEpsilon, t, id);
}
//
struct PathVertex
{
	enum class VertexType { Camera, Light, Surface };

	static PathVertex CreateSurfaceVertex(int obj_id, Refl_t brdf, const Vec& reflectance, const Vec& position, const Vec& normal, double pdf, const PathVertex& prev)
	{
		PathVertex v;
		v.type = VertexType::Surface;
		v.reflectance = reflectance;
		v.position = position;
		v.normal = normal;
		v.pdfFwd = prev.ConvertDensity(pdf, position, normal, v.type);
		v.brdf_type = brdf;
		//v.pdfRev = pdfRev;
		v.object_id = obj_id;
		v.delta = brdf != Refl_t::DIFF;
		return v;
	}

	static PathVertex CreateCameraVertex(const Vec& position,const Vec& reflectance)
	{
		PathVertex v;
		v.type = VertexType::Camera;
		v.reflectance = reflectance;
		v.position = position;

		//v.pdfFwd = pdf;
		return v;
	}

	static PathVertex CreateLightVertex(Refl_t brdf, const Vec& reflectance, const Vec& position, const Vec& normal, double pdf)
	{
		PathVertex v;
		v.type = VertexType::Light;
		v.reflectance = reflectance;
		v.position = position;
		v.normal = normal;
		v.brdf_type = brdf;
		v.pdfFwd = pdf;
		return v;
	}

	VertexType type;
	Refl_t brdf_type;
	Vec reflectance; //in
	bool delta = false;

	//
	Vec position;
	Vec normal;

	double pdfFwd = 0;
	double pdfRev = 0;
	int object_id = -1;

	double ConvertDensity(double pdf, const Vec& p, const Vec& n, VertexType vertex_type)const
	{
		Vec w = p - position;
		if (w.length() == 0) return 0;
		double inv_dist2 = 1 / w.dot(w);
		if (IsOnSurface())
			pdf *= AbsDot(normal,w.norm());
		return pdf * inv_dist2;

	}

	bool IsLight()const
	{
		return (brdf_type == Refl_t::DIFF && (kLightIndex == object_id)) || type == VertexType::Light;
	}
	bool IsDelta()const
	{
		return delta;
	}

	Vec f(const PathVertex& next)const
	{
		Vec wi = next.position - position;
		if(wi.length_sqr() == 0) 
			return Vec();
		if(type == VertexType::Surface)
		{
			return spheres[object_id].c * (1.0/ M_PI);
		}
		return Vec();
	}
	Vec Le(const PathVertex& v)const
	{
		Vec w = v.position - position;
		if (w.dot(normal) <= 0)
			return 0;
		return spheres[object_id].e;
	}
	bool IsConnectible()const
	{
		if(type == VertexType::Camera || type == VertexType::Light)//arealight
			return true;
		return brdf_type == Refl_t::DIFF;
	}

	bool IsDeltaLight()const
	{
		return false;
	}
	bool IsOnSurface()const
	{
		return normal != Vec();
	}

	double Pdf(const PathVertex *prev, const PathVertex &next)const
	{
		if(type == VertexType::Light) return PdfLight(next);
		Vec wn = next.position - position;
		if(wn.length_sqr() == 0) return 0;
		wn = wn.norm();
		Vec wp;
		if(prev) 
		{
			wp = prev->position - position;
			if(wp.length_sqr() == 0) return 0;
			wp = wp.norm();
		}
		double pdf = 0;
		if(type == VertexType::Camera)
		{
			double unused;
			CameraPDF(wn, &unused, &pdf);
		}
		else if(type == VertexType::Surface)
		{
			pdf = brdf_pdf(brdf_type, wp, wn, normal);
			if(pdf == 0)
			{
				int debug_break = 0;
			}
		}
		return ConvertDensity(pdf, next.position, next.normal, next.type);
	}
	double PdfLight(const PathVertex &v)const
	{
		Vec w = v.position - position;
		double invDist2 = 1/w.length_sqr();
		double pdf = 1/M_PI * invDist2;
		if(v.IsOnSurface())
			pdf *= AbsDot(v.normal, w.norm());
		return pdf;
	}
	double PdfLightOrigin(const PathVertex &v)const
	{
		auto light_radius = spheres[kLightIndex].rad;
		return 1/(4*M_PI*(light_radius*light_radius));
	}
};

enum class PathTraceMode
{
	CameraRay,
	LightRay
};
void path_trace(PathTraceMode mode, const Ray &r, int max_depth, int depth, unsigned short *Xi, std::vector<PathVertex>* path, double pdfFwd, const Vec& sp)
{
	double t;                               // distance to intersection 
	int id = 0;                               // id of intersected object 
	if (!intersect(r, t, id) || depth >= max_depth) 
		return; // if miss or exceeds, return black 
	const Sphere &obj = spheres[id];        // the hit object 
	Vec x = r.o + r.d*t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;

	if(f.length_sqr() == 0 && obj.e.length_sqr() == 0)
	{ 
		//end
		if (mode == PathTraceMode::CameraRay) {
			(*path)[depth].pdfFwd = pdfFwd;
			(*path)[depth].pdfRev = 1;
		}

		return; 
	}

	//rr prob
	double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl 
	++depth;
	double pdfRev = 0;
	if (obj.refl == DIFF) {                  // Ideal DIFFUSE reflection 
		double r1 = 2 * M_PI*erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
		Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
		Vec d = (u*cos(r1)*r2s + v * sin(r1)*r2s + w * sqrt(1 - r2)).norm();

		pdfRev = brdf_pdf(obj.refl, r.d * (-1), d, nl);
		//TODO: add a path vertex
		auto new_vertex = PathVertex::CreateSurfaceVertex(id, obj.refl, sp, x, nl, pdfFwd, (*path)[depth - 1]);
		path->push_back(new_vertex);
		if(new_vertex.IsLight()){
			int debug = 0;
		}
		if (new_vertex.object_id == 3)
		{
			int debug = 0;
		}

		double pdf = brdf_pdf(obj.refl, d, r.d*(-1), nl);
		f = f * (1/M_PI); //lambertian brdf
		double abs_cos = AbsDot(nl, d.norm() * (-1));
		Vec passthrough = pdf == 0 ? Vec() : (f.mult(sp)  * ((abs_cos / pdf)));

		if (mode == PathTraceMode::CameraRay)
			passthrough = passthrough + obj.e * (abs_cos);
		path_trace(mode, Ray(x + d * kRayEpsilon, d), max_depth, depth, Xi, path, pdf,
			passthrough
		); // INV_PI/INV_PI
	}
	else if (obj.refl == SPEC)            // Ideal SPECULAR reflection 
	{
		//TODO:
		//add delta path vertex
		auto new_vertex = PathVertex::CreateSurfaceVertex(id, obj.refl, sp, x, nl, pdfFwd, (*path)[depth - 1]);
		path->push_back(new_vertex);
		Vec d = (r.d - n * 2 * n.dot(r.d)).norm();
		path_trace(mode, Ray(x + d * kRayEpsilon, d), max_depth, depth, Xi, path, 0, f.mult(sp));
		//return obj.e + f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi)); 
	}
	else //refr
	{
		Vec reflDir = (r.d - n * 2 * n.dot(r.d)).norm();
		Ray reflRay(x + reflDir * kRayEpsilon, reflDir);     // Ideal dielectric REFRACTION 
		bool into = n.dot(nl) > 0;                // Ray from outside going in? 
		double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
		if ((cos2t = 1 - nnt * nnt*(1 - ddn * ddn)) < 0)    // Total internal reflection 
		{
			//TODO:
			//add delta path vertex
			auto new_vertex = PathVertex::CreateSurfaceVertex(id, obj.refl, sp, x, nl, pdfFwd, (*path)[depth - 1]);
			path->push_back(new_vertex);

			path_trace(mode, reflRay, max_depth, depth, Xi, path, 0, f.mult(sp));
			//return obj.e + f.mult(radiance(reflRay,depth,Xi)); 
		}
		else 
		{
			Vec tdir = (r.d*nnt - n * ((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))).norm();
			double a = nt - nc, b = nt + nc, R0 = a * a / (b*b), c = 1 - (into ? -ddn : tdir.dot(n));
			double Re = R0 + (1 - R0)*c*c*c*c*c, Tr = 1 - Re, P = .25 + .5*Re, RP = Re / P, TP = Tr / (1 - P);

			//TODO: refraction
			//add delta path vertex
			//if(depth > 2) 
			{
				double p = erand48(Xi); //Russian roulette

				if (p < P)
				{
					auto new_vertex = PathVertex::CreateSurfaceVertex(id, obj.refl, sp, x, nl, pdfFwd, (*path)[depth - 1]);
					path->push_back(new_vertex);

					path_trace(mode, reflRay, max_depth, depth, Xi, path, 0, sp.mult(f*RP));
				}
				else
				{
					auto new_vertex = PathVertex::CreateSurfaceVertex(id, obj.refl, sp, x, nl, pdfFwd, (*path)[depth - 1]);
					path->push_back(new_vertex);

					path_trace(mode, Ray(x + tdir * kRayEpsilon, tdir), max_depth, depth, Xi, path, 0, sp.mult(f*TP));
				}
				//TODO:
				//add delta path vertex
				 //radiance(reflRay,depth,Xi)*RP:radiance(Ray(x,tdir),depth,Xi)*TP)
			}
			//else
			{
				//TODO
				//add delta path vertex
				//path_trace(reflRay)
				 //radiance(reflRay,depth,Xi)*Re+radiance(Ray(x,tdir),depth,Xi)*Tr)
			}

			//return obj.e + f.mult(depth>2 ? (erand48(Xi)<P ?   // Russian roulette 
			//  radiance(reflRay,depth,Xi)*RP:radiance(Ray(x,tdir),depth,Xi)*TP) : 
			//  radiance(reflRay,depth,Xi)*Re+radiance(Ray(x,tdir),depth,Xi)*Tr);


		}

	}
	(*path)[depth - 1].pdfRev =
		(*path)[depth].ConvertDensity(pdfRev, (*path)[depth - 1].position, (*path)[depth - 1].normal, (*path)[depth - 1].type);
}

double G(const PathVertex& v0, const PathVertex& v1)
{
	Vec d = v0.position - v1.position;
	double g = 1 / d.dot(d);
	d = d * sqrt(g);
	if (v0.IsOnSurface()) g *= AbsDot(v0.normal,d);
	if (v1.IsOnSurface()) g *= AbsDot(v1.normal,d);
	return g;
}

double MISWeight(
	std::vector<PathVertex>& light_path, std::vector<PathVertex>& camera_path,
	const PathVertex& sampled, size_t s, size_t t)
{
	if(s + t == 2)
		return 1;

	double sumRi = 0;
    // Define helper function _remap0_ that deals with Dirac delta functions
    auto remap0 = [](double f) -> double { return f != 0 ? f : 1; };

    // Temporarily update vertex properties for current strategy

    // Look up connection vertices and their predecessors
    PathVertex *qs = s > 0 ? &light_path[s - 1] : nullptr,
           *pt = t > 0 ? &camera_path[t - 1] : nullptr,
           *qsMinus = s > 1 ? &light_path[s - 2] : nullptr,
           *ptMinus = t > 1 ? &camera_path[t - 2] : nullptr;

    // Update sampled vertex for $s=1$ or $t=1$ strategy
    ScopedAssignment<PathVertex> a1;
    if (s == 1)
        a1 = {qs, sampled};
    else if (t == 1)
        a1 = {pt, sampled};

    // Mark connection vertices as non-degenerate
    ScopedAssignment<bool> a2, a3;
    if (pt) a2 = {&pt->delta, false};
    if (qs) a3 = {&qs->delta, false};

    // Update reverse density of vertex $\pt{}_{t-1}$
    ScopedAssignment<double> a4;
    if (pt)
        a4 = {&pt->pdfRev, s > 0 ? qs->Pdf(qsMinus, *pt)
                                 : pt->PdfLightOrigin(*ptMinus)};

    // Update reverse density of vertex $\pt{}_{t-2}$
    ScopedAssignment<double> a5;
    if (ptMinus)
        a5 = {&ptMinus->pdfRev, s > 0 ? pt->Pdf(qs, *ptMinus)
                                      : pt->PdfLight(*ptMinus)};

    // Update reverse density of vertices $\pq{}_{s-1}$ and $\pq{}_{s-2}$
    ScopedAssignment<double> a6;
    if (qs) a6 = {&qs->pdfRev, pt->Pdf(ptMinus, *qs)};
    ScopedAssignment<double> a7;
    if (qsMinus) a7 = {&qsMinus->pdfRev, qs->Pdf(pt, *qsMinus)};

    // Consider hypothetical connection strategies along the camera subpath
    double ri = 1;
    for (int i = t - 1; i > 0; --i) {
		if(camera_path[i].pdfRev == 1 || camera_path[i].pdfFwd == 1)
		{
			int debug_break = 0;
		}
        ri *=
            remap0(camera_path[i].pdfRev) / remap0(camera_path[i].pdfFwd);
        if (!camera_path[i].delta && !camera_path[i - 1].delta)
            sumRi += ri;
    }

    // Consider hypothetical connection strategies along the light subpath
    ri = 1;
    for (int i = s - 1; i >= 0; --i) {
		if(light_path[i].pdfRev == 1 || light_path[i].pdfFwd == 1)
		{
			int debug_break = 0;
		}
        ri *= remap0(light_path[i].pdfRev) / remap0(light_path[i].pdfFwd);
        bool deltaLightvertex = i > 0 ? light_path[i - 1].delta
                                      : light_path[0].IsDeltaLight();
        if (!light_path[i].delta && !deltaLightvertex) sumRi += ri;
    }
    return 1 / (1 + sumRi);
}

Vec ConnectBDPT(std::vector<PathVertex>& light_path, std::vector<PathVertex>& camera_path,
	size_t s, size_t t, unsigned short *Xi, double *misWeight, Vec* raster_dir)
{
	Vec L;
	PathVertex sampled;
	if (s == 0)
	{
		const PathVertex& pt = camera_path[t - 1];
		if (pt.IsLight())
			L = pt.Le(camera_path[t - 2]).mult(pt.reflectance);
	}
	else if (t == 1) 
	{
		const PathVertex& pt = camera_path[t-1];
		const PathVertex& qs = light_path[s - 1];

		bool debug_break = false;
		if (s > 2 && (light_path[s - 2].brdf_type == Refl_t::REFR))
		{
			debug_break = true;
		}
			
		Vec wi;
		double pdf;
		Vec sample_pos;
		CameraPDF(qs.position, &sample_pos, &wi, &pdf);

#ifdef _DEBUG
		if (debug_ray_dir.dot(wi * (-1)) >= 0.999999 )
		{
			//debug_break = true;
		}
#endif
		Vec sp = Vec(1,1,1);
		sampled = PathVertex::CreateCameraVertex(sample_pos, sp * (1.0/pdf) );
		if(pdf > 0 && qs.IsConnectible() && is_visible(sampled.position, qs.position))
		{
			if (debug_break )
			{
#ifdef _DEBUG
				double t;
				int id;
				if (intersect(Ray(sampled.position, debug_ray_dir), t, id)) 
				{
					bool visible = is_visible(sampled.position, qs.position);
					Vec pos = sampled.position + debug_ray_dir * t;
					int debug = 0;
				}
#endif
			}
			L = qs.reflectance.mult(qs.f(sampled)).mult( sampled.reflectance );
			if(qs.IsOnSurface())
			{
				L = L * AbsDot(wi, qs.normal);
			}
			*raster_dir  = wi * (-1.0);
		}
	}
	else if (s == 1)
	{

		const PathVertex& pt = camera_path[t - 1];
		//createt light vertex
		double light_radius = spheres[kLightIndex].rad;
		double light_x, light_y, light_z;
		uniform_sampling_sphere(Xi, &light_x, &light_y, &light_z);
		Vec light_sample_pos = spheres[kLightIndex].p + Vec(light_x, light_y, light_z);
		double pos_pdf =  1/ (4*M_PI * light_radius * light_radius);
		Vec dir((pt.position - light_sample_pos).norm());
		Vec pos(light_sample_pos);
		//double light_pdf = (4*M_PI*light_radius*light_radius);
		
		double dist_sqr = (pt.position - light_sample_pos).length_sqr();
		//double light_pdf = (4*M_PI * dist_sqr);
		double light_pdf = 1 / (2 * M_PI* (1 - sqrt(1 - ((light_radius*light_radius) / dist_sqr))));
			
		Vec wi = (pos - pt.position);
		if (wi.length_sqr() <= DBL_EPSILON)
			light_pdf = 0;
			
		wi = wi.norm();
		if (light_pdf > 0) {
			Vec reflectance = spheres[kLightIndex].e*(1.0 / (pos_pdf * light_pdf));// *(1.0 / (pt.position - spheres[kLightIndex].p).length_sqr());
#ifdef _DEBUG
			bool debug = false;

			if (t > 2)
			{
				Vec debug_dir = (camera_path[1].position - camera_path[0].position).norm();
				if (debug_ray_on)
				{
					auto dot = debug_ray_dir.dot(debug_dir) >= 0.99;
					debug = true;
				}

				if (debug == true && is_visible(spheres[kLightIndex].p + dir * (light_radius + 1), pt.position))
				{
					bool visible = is_visible(pt.position, pos);
					debug = true;
				}
				if (debug)
				{
					debug = (t == 4
						&& camera_path[1].brdf_type == Refl_t::REFR
						&& camera_path[2].object_id == kLightIndex);
				}

			}

#endif
			sampled =
				PathVertex::CreateLightVertex(Refl_t::DIFF, reflectance, pos, dir, 0);
			if (pt.IsConnectible() && is_visible(pt.position, pos))
			{
#ifdef _DEBUG
				if (debug)
					int debug_break = 0;
#endif
				L = pt.reflectance.mult(pt.f(sampled)).mult(sampled.reflectance);
				if (pt.IsOnSurface())
					L = L * AbsDot(pt.normal, wi);

#ifdef _DEBUG
				else if (debug)
				{
					int debug_break = 0;
				}
				if (std::isnan(L.x)) {
					int debug_break = 0;
				}
#endif
			}
		}
	}
	else
	{
		if(t == 2){
			int debug_break = 0;
		}
		const PathVertex &qs = light_path[s - 1], &pt = camera_path[t - 1];
		if (qs.IsConnectible() && pt.IsConnectible() && is_visible(qs.position, pt.position))
		{
			L = qs.reflectance.mult(qs.f(pt)).mult(pt.f(qs)).mult(pt.reflectance);
			L = L * G(qs, pt);
		}
	}

	if(L.length_sqr() == 0)
		return L;
	*misWeight = MISWeight(light_path, camera_path, sampled, s, t);
	L = L * *misWeight;
	return L;
}

//reference: http://www.pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing.html#

Vec bdpt_radiance(const Ray &r, int depth, unsigned short *Xi, std::vector<std::pair<Vec, Vec> >* raster_dir_radiance)
{
	//camera path
	std::vector<PathVertex> camera_path;
	{
		Vec sp = Vec(1,1,1);
		double pdf;
		double unused;
		CameraPDF(r.d, &unused, &pdf);
		auto camera_start = PathVertex::CreateCameraVertex(r.o, sp * (1.0/pdf));
		camera_path.push_back(camera_start);
		path_trace(PathTraceMode::CameraRay, r, kMaxDepth + 1, 0, Xi, &camera_path, pdf, sp * (1.0/pdf));
	}
	//light path
	std::vector<PathVertex> light_path;
	{
		double light_radius = spheres[kLightIndex].rad;
		double light_pdf_pos = 1 / (4 * M_PI*light_radius*light_radius);
		double light_x, light_y, light_z;
		uniform_sampling_sphere(Xi, &light_x, &light_y, &light_z);
		Vec light_normal(light_x, light_y, light_z);
		Vec pos(spheres[kLightIndex].p + light_normal * light_radius);

		Vec dir;
		double light_pdf_dir = 1 / M_PI;
		{
			Point2f u{erand48(Xi), erand48(Xi)};
			Vec w = CosineSampleHemisphere(u);
		    light_pdf_dir = w.z/(1/M_PI);
			Vec v1, v2, n(light_normal);
			CoordinateSystem(n, &v1, &v2);
			dir = v1 * w.x + v2 * w.y +  n * w.z;
		}
		double light_pdf = light_pdf_dir * light_pdf_pos;
		Vec reflectance = spheres[kLightIndex].e * (1.0 / (light_pdf));
		PathVertex light_start = PathVertex::CreateLightVertex(Refl_t::DIFF, reflectance, pos, light_normal, light_pdf);
		light_path.push_back(light_start);
		path_trace(PathTraceMode::LightRay, Ray(pos, dir), kMaxDepth, 0, Xi, &light_path, light_pdf_dir, reflectance);
	}

	//connect
	Vec L;
	for (size_t t = 1; t <= camera_path.size(); t++)
	{
			//debug
			//if (t != 1)
			//	continue;


		for (size_t s = 0; s <= light_path.size(); s++)
		{
			int depth = t + s - 2;
			if ((s == 1 && t == 1) || depth > kMaxDepth)
				continue;

			//debug
			//if (s > 1)
			//	continue;

			double misWeight = 0.f;

			Vec raster_dir;
			Vec Lpath = ConnectBDPT(light_path, camera_path, s, t, Xi, &misWeight, &raster_dir);

			if (misWeight != 0)
			{
				if(raster_dir != Vec())
				{
					raster_dir_radiance->push_back(std::make_pair(raster_dir, Lpath));
				}
				else
				{
					L = L + Lpath;
				}
			}
		}
	}
	return L;
}

Vec radiance(const Ray &r, int depth, unsigned short *Xi) {
	double t;                               // distance to intersection 
	int id = 0;                               // id of intersected object 
	if (!intersect(r, t, id) || depth > kMaxDepth) return Vec(); // if miss or exceeds, return black 
	const Sphere &obj = spheres[id];        // the hit object 
	Vec x = r.o + r.d*t, n = (x - obj.p).norm(), nl = n.dot(r.d) < 0 ? n : n * -1, f = obj.c;
	double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl 
	if (++depth > 5)
		if (erand48(Xi) < p) f = f * (1 / p);
		else return obj.e; //R.R. 

	if (obj.refl == DIFF) {                  // Ideal DIFFUSE reflection 
		double r1 = 2 * M_PI*erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
		Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
		Vec d = (u*cos(r1)*r2s + v * sin(r1)*r2s + w * sqrt(1 - r2)).norm();
		return obj.e + f.mult(radiance(Ray(x, d), depth, Xi));
	}
	else if (obj.refl == SPEC)            // Ideal SPECULAR reflection 
		return obj.e + f.mult(radiance(Ray(x, r.d - n * 2 * n.dot(r.d)), depth, Xi));

	Ray reflRay(x, r.d - n * 2 * n.dot(r.d));     // Ideal dielectric REFRACTION 
	bool into = n.dot(nl) > 0;                // Ray from outside going in? 
	double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
	if ((cos2t = 1 - nnt * nnt*(1 - ddn * ddn)) < 0)    // Total internal reflection 
		return obj.e + f.mult(radiance(reflRay, depth, Xi));

	Vec tdir = (r.d*nnt - n * ((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))).norm();
	double a = nt - nc, b = nt + nc, R0 = a * a / (b*b), c = 1 - (into ? -ddn : tdir.dot(n));
	double Re = R0 + (1 - R0)*c*c*c*c*c, Tr = 1 - Re, P = .25 + .5*Re, RP = Re / P, TP = Tr / (1 - P);
	return obj.e + f.mult(depth > 2 ? (erand48(Xi) < P ?   // Russian roulette 
		radiance(reflRay, depth, Xi)*RP : radiance(Ray(x, tdir), depth, Xi)*TP) :
		radiance(reflRay, depth, Xi)*Re + radiance(Ray(x, tdir), depth, Xi)*Tr);
}

bool dir_to_raster(double *x, double *y, const Vec& dir, int w, int h, const Ray& cam_center_ray, const Vec& cx, const Vec& cy)
{
	Vec p = cam_center_ray.o + dir * (1.0 / (dir.dot(cam_center_ray.d)));
	Vec film_center = cam_center_ray.o + cam_center_ray.d;
	Vec from_center = p - film_center;

	int raster_x = (int)((from_center.dot(cx)/cx.length_sqr() + 0.5) * w);
	//flip y
	int raster_y = (int)(((0.5 - from_center.dot(cy)/cy.length_sqr())) * h);

	if (raster_x < 0 || raster_x >= w || raster_y < 0 || raster_y >= h)
		return false;

	*x = raster_x;
	*y = raster_y;
	return true;
}

int main(int argc, char *argv[]) {
	int w = 1024, h = 768, samps = argc == 2 ? atoi(argv[1]) / 4 : kDefaultSPP; // # samples
	Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir
	Vec cx = Vec(w*.5135 / h), cy = (cx%cam.d).norm()*.5135, r, *c = new Vec[w*h];

#ifdef _DEBUG
	///
	//debug
		//int x = 697, y = 571; //behind spec ball
		//int x = 841, y = 461; //wall
		//int debug_x = 584, debug_y = 610; //on sphere
		//int debug_x = 258, debug_y = 578; //on sphere
		int debug_x = 461, debug_y = 704; //on floor
		Vec d = cx * (debug_x / w - .5) + cy * (((h- debug_y) / h) - .5) + cam.d;
		//debug
		debug_ray_dir = d.norm();
		double window_w = 20, window_h = 20;
		int x_min = debug_x - window_w, y_min = debug_y - window_h, x_max = debug_x + window_w, y_max = debug_y + window_h;
#endif

	///
	g_A = 2*cx.length() * 2 * cy.length();
	g_camera_dir = cam.d;

	Vec *splat = new Vec[w*h];
	int *splat_count = new int[w*h];
	::memset(c, 0, sizeof(Vec) * w*h);

	for (int s = 0; s < samps; s++)
	{
		::memset(splat, 0, sizeof(Vec)*w*h);
		::memset(splat_count, 0, sizeof(int)*w*h);
//#pragma omp parallel for schedule(dynamic, 1) private(r) 
		for (int y = 0; y < h; y++) {                       // Loop over image rows
			unsigned short Xi[] = { s, samps - s, y*y*y };
			fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100.*(s*h + y) / (samps*h));
			for (unsigned short x = 0; x < w; x++)   // Loop cols
			{

#ifdef _DEBUG
				if(!(x>=x_min && x <= x_max && (h-y) >= y_min && (h-y) <= y_max))
				{
					continue;
				}
				debug_ray_on = debug_x == x && debug_y == (h-y);
#endif
				int i = (h - y - 1)*w + x;
				r = Vec();
				for (int sy = 0; sy < 2; sy++)     // 2x2 subpixel rows
				{
					for (int sx = 0; sx < 2; sx++) {        // 2x2 subpixel cols
						float r1 = 2 * erand48(Xi), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
						float r2 = 2 * erand48(Xi), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
						Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) +
							cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.d;

#ifdef USE_BDPT
						std::vector<std::pair<Vec, Vec> > raster_dir_to_radiance;
						r = r + bdpt_radiance(Ray(cam.o + d * g_camera_start, d.norm()), 0, Xi, &raster_dir_to_radiance) * 0.25;
						//raster splat
						for(size_t k = 0; k < raster_dir_to_radiance.size(); k++)
						{
							const Vec& dir = raster_dir_to_radiance[k].first;
							const Vec& radiance = raster_dir_to_radiance[k].second;

							double raster_x, raster_y;
#ifdef _DEBUG
							if (dir.dot(debug_ray_dir) >= 0.999999) 
							{
								int debug_break = 0;
							}
#endif
							if(dir.dot(cam.d) > 0 && dir_to_raster(&raster_x, &raster_y, dir, w, h, cam, cx, cy))
							{
								int raster_index = ((int)(raster_y) * w + (int)(raster_x));
								splat[raster_index] = splat[raster_index] + radiance;
								splat_count[raster_index] ++;
							}
					}
#else
						r = r + radiance(Ray(cam.o + d * g_camera_start, d.norm()), 0, Xi) * 0.25;
#endif
					}
				}
				c[i] = (c[i] * s + Vec(clamp(r.x), clamp(r.y), clamp(r.z))) * (1.0 / (double)(s + 1));
			}
		}

		for(int i = 0; i < w*h; i++)
		{
			if(splat_count[i] > 0)
				c[i] = clamp( c[i] + splat[i]  *  (1.0 / splat_count[i]  / (s+1))  );
		}
	}

	//int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
	std::vector<unsigned char> pixels;
	pixels.resize(w*h * 3);
	for (int i = 0; i < w*h; i++)
	{
		pixels[i * 3] = toInt(c[i].x);
		pixels[i * 3 + 1] = toInt(c[i].y);
		pixels[i * 3 + 2] = toInt(c[i].z);
	}
#ifdef USE_BDPT
	char file_name[128];
	sprintf_s(file_name,"image_bdpt_%dspp.png", samps*4);
#else
	sprintf_s(file_name,"image_pt_%dspp.png", samps*4);
#endif
	
	stbi_write_png(file_name, w, h, 3, pixels.data(), 0);

	//FILE *f;
	//fopen_s(&f, "image.ppm", "w");         // Write image to PPM file.
	//fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	//for (int i=0; i<w*h; i++)
	//  fprintf(f,"%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
	delete []c;
	delete []splat;
	delete[]splat_count;
}