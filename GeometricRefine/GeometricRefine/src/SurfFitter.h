#pragma once

#include <vector>
#include "Helper.h"
#include "MySurf.h"
#include "Mathematics/BSplineSurfaceFit.h"
#include "Mathematics/BSplineSurface.h"
#include <Mathematics/IntpThinPlateSpline2.h>

class SurfFitter : public MySurf
{
public:
	SurfFitter()
	{
		/*u_split = 30;
		v_split = 30;*/
		u_split = 20;
		v_split = 20;
		//depend on input
		dim_u_input = -1;
		dim_v_input = -1; // only set once, by set_points
		//flag_using_aqd = true;
		flag_using_aqd = false;
		flag_input_normal = false;
	}

	SurfFitter(bool using_aqd)
	{
		/*u_split = 30;
		v_split = 30;*/
		u_split = 20;
		v_split = 20;
		//depend on input
		dim_u_input = -1;
		dim_v_input = -1; // only set once, by set_points
		//flag_using_aqd = true;
		flag_using_aqd = using_aqd;
		flag_input_normal = false;
	}

	void set_uv_split(int usp, int vsp)
	{
		u_split = usp;
		v_split = vsp;
	}

	void set_points(const std::vector<vec3d>& pts)
	{
		input_pts = pts;
		flag_input_normal = false;
		us.clear();
		vs.clear();
		if (dim_u_input < 0)
		{
			if (input_pts.size() == 400)
			{
				dim_u_input = 20;
				dim_v_input = 20;
			}
			else
			{
				dim_v_input = 10;
				dim_u_input = 10;
			}
		}
	}
	
	void set_bspline_nongrid_points(const std::vector<vec3d>& pts, const std::vector<double>& weights)
	{
		//only work for BSpline
		nongrid_pts = pts;
		nongrid_weights = weights;
	}

	void set_points(const std::vector<vec3d>& pts, const std::vector<double> pts_weight)
	{
		input_pts = pts;
		flag_input_normal = false;
		us.clear();
		vs.clear();
		assert(pts.size() == pts_weight.size());
		input_pts_weight = pts_weight;
		//update weight
		if (dim_u_input < 0)
		{
			if (input_pts.size() == 400)
			{
				dim_u_input = 20;
				dim_v_input = 20;
			}
			else
			{
				dim_v_input = 10;
				dim_u_input = 10;
			}
		}
		
	}

	void expand_input_points(double dist = 0.05, double expand_weight = 0.01);
	

	void subdivide_grid(int split = 10);

	void set_normals(const std::vector<vec3d>& pts)
	{
		assert(pts.size() == input_pts.size());
		input_normals = pts;
		flag_input_normal = true;
	}
	

	virtual void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv = false) = 0;
	virtual void projection_with_normal(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, std::vector<vec3d>& tgt_normal, bool save_uv = false)
	{
		//only impl for cylinder, cone, nurbs
	}

	virtual void print_params()
	{
		std::cout << "no implemented" << std::endl;
	}

	virtual void set_axis(const vec3d& axis)
	{
		//impl for cylinder and cone
	}

	virtual const vec3d& get_axis()
	{
		//impl for cylinder and cone
		return vec3d(0.0, 0.0, 0.0);
	}

	virtual void set_radius_offset(double r)
	{
		//only impl for cylidner
	}

	void get_self_projection(bool update_uv = false) {
		projection(input_pts, projection_pts, update_uv);
	}
	void get_nn_proj_pt(const vec3d& src, vec3d& tgt);
	void get_nn_proj_pts(const std::vector<vec3d>& src, std::vector<vec3d>& tgt);
	void get_grid_pts(bool with_normal = false);
	void get_nn_grid_pts(const std::vector<vec3d>& src, std::vector<vec3d>& tgt);
	void get_grid_tri(std::vector<vec3d>& pts, std::vector<std::vector<size_t>>& faces);
	void save_input_patch_obj(const std::string& fn);

	//virtual void write_obj_surf(std::ostream& out, size_t& vcounter, int u_div, int v_div);
	//get_uv()
	virtual void fitting() = 0;

	/*virtual vec3d GetNormal(double u, double v)
	{
		return vec3d(0.0, 0.0, 0.0);
	}*/

	void write_data_surf(std::ostream& out, int u_div, int v_div, bool flag_normal = false);

	void get_grid_pts_normal(int u_div, int v_div, std::vector<vec3d>& pts, std::vector<vec3d>& normals, bool flag_diff = false);


	//virtual void fitting();
	bool flag_using_aqd;
	bool flag_input_normal;
	std::vector<vec3d> input_normals;
	std::vector<vec3d> input_pts;
	std::vector<vec3d> projection_pts;
	std::vector<double> us;
	std::vector<double> vs;
	std::vector<double> input_pts_weight;

	int u_split;
	int v_split;
	std::vector<vec3d> grid_pts;
	std::vector<vec3d> grid_normals;
	
	//for NURBS fitting
	int dim_u_input; //default number is 10
	int dim_v_input;
	std::vector<vec3d> nongrid_pts;
	std::vector<double> nongrid_weights;
};



class CylinderFitter: public SurfFitter
{
public:
	CylinderFitter()
		: radius(0.0)
	{
		flag_normalize = false;
		//flag_normalize = true;
		flag_first_time = true;
		flag_input_axis = false; //once axis is set, aqd not used, first time set to false
	}

	CylinderFitter(bool using_aqd)
		: radius(0.0)
	{
		flag_normalize = false;
		//flag_normalize = true;
		flag_first_time = true;
		flag_input_axis = false; //once axis is set, aqd not used, first time set to false
		flag_using_aqd = using_aqd;
	}

	CylinderFitter(const vec3d& _loc, const vec3d& _xdir, const vec3d& _ydir, const vec3d& _zdir, double _radius,
		bool _u_closed, double _u_min, double _u_max, double _v_min, double _v_max)
		: loc(_loc), xdir(_xdir), ydir(_ydir), zdir(_zdir), radius(_radius)
	{
		this->u_closed = _u_closed, this->v_closed = false;
		this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max, this->v_min = _v_min, this->v_max = _v_max;
		flag_normalize = false;
		//flag_normalize = true;
		flag_first_time = true;
		flag_input_axis = false; //once axis is set, aqd not used, first time set to false
	}
	vec3d GetPosition(double u, double v)
	{
		if (flag_normalize)
		{
			vec3d tmp = loc + radius * (cos(u) * xdir + sin(u) * ydir) + v * zdir;
			return tmp * scale + translation;
		}
		else
		{
			return loc + radius * (cos(u) * xdir + sin(u) * ydir) + v * zdir;
		}
		//std::cout << "cylinder pos" << std::endl;
	}


	vec3d GetNormal(double u, double v)
	{
		//no need to consider normalization
		vec3d uvec = radius * (-sin(u) * xdir + cos(u) * ydir);
		vec3d vvec = zdir;
		vec3d normal = uvec.Cross(vvec);
		normal.Normalize();
		return normal;
	}

	void set_axis(const vec3d& axis)
	{
		zdir = axis;
		flag_using_aqd = false;
		flag_first_time = false;
	}

	const vec3d& get_axis()
	{
		return zdir;
	}

	void set_radius_offset(double r)
	{
		radius += r;
	}
//protected:
	vec3d loc, xdir, ydir, zdir;
	double radius;
	bool flag_normalize;
	vec3d translation;
	double scale;
	bool flag_first_time; 
	bool flag_input_axis;

//private:
	void estimate_normal();
	void aqd_fitting();
	void fitting();
	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv = false);
	void projection_with_normal(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, std::vector<vec3d>& tgt_normal,bool save_uv = false);
};

class PlaneFitter : public SurfFitter
{
public:
	PlaneFitter()
	{
	}
	PlaneFitter(const vec3d& _loc, const vec3d& _xdir, const vec3d& _ydir,
		double _u_min, double _u_max, double _v_min, double _v_max)
		: loc(_loc), xdir(_xdir), ydir(_ydir)
	{
		this->u_closed = this->v_closed = false;
		this->u_min = _u_min, this->u_max = _u_max, this->v_min = _v_min, this->v_max = _v_max;
		//set zdir
		zdir = xdir.Cross(ydir);
		zdir.Normalize();
	}
	vec3d GetPosition(double u, double v)
	{
		return loc + u * xdir + v * ydir;
	}
	vec3d GetNormal(double u, double v)
	{
		return zdir;
	}

	void print_params()
	{
		std::cout << "zdir:" << zdir << std::endl;
		std::cout << "d: " << zdir.Dot(loc) << std::endl;
	}

	vec3d loc, xdir, ydir;
public:
	void fitting();
	//void aqd_fitting(); #not usable
	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv = false);
	vec3d zdir;
};

class ConeFitter : public SurfFitter
{
public:
	ConeFitter()
		:radius(0.0), angle(0.0)
	{
		flag_first_time = true;
		//flag_sfpn = true;
		flag_sfpn = false;
		flag_set_axis = false;
		flag_single_side_proj = false; //only consider single side, update 0221
		
	}

	ConeFitter(bool using_aqd)
		:radius(0.0), angle(0.0)
	{
		flag_first_time = true;
		//flag_sfpn = true;
		flag_sfpn = false;
		flag_using_aqd = using_aqd;
		flag_set_axis = false;
		flag_single_side_proj = false; //only consider single side, update 0221
	}
	
	ConeFitter(const vec3d& _loc, const vec3d& _xdir, const vec3d& _ydir, const vec3d& _zdir,
		double _radius, double _angle,
		bool _u_closed, double _u_min, double _u_max, double _v_min, double _v_max)
		: loc(_loc), xdir(_xdir), ydir(_ydir), zdir(_zdir), radius(_radius), angle(_angle)
	{
		this->u_closed = _u_closed, this->v_closed = false;
		this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max, this->v_min = _v_min, this->v_max = _v_max;
		flag_first_time = true;
		flag_single_side_proj = false; //only consider single side, update 0221
	}
	vec3d GetPosition(double u, double v)
	{
		return loc + (radius + v * sin(angle)) * (cos(u) * xdir + sin(u) * ydir) + v * cos(angle) * zdir;
	}

	vec3d GetNormal(double u, double v)
	{
		//no need to consider normalization
		vec3d uvec = (radius + v * sin(angle)) * (-sin(u) * xdir + cos(u) * ydir);
		vec3d vvec = sin(angle) * (cos(u) * xdir + sin(u) * ydir) + cos(angle) * zdir;
		vec3d normal = uvec.Cross(vvec);
		normal.Normalize();
		return normal;
	}

	void set_axis(const vec3d& axis)
	{
		zdir = axis;
		flag_using_aqd = false;
		flag_sfpn = true;
		flag_set_axis = true;
	}

	const vec3d& get_axis()
	{
		return zdir;
	}

	void print_params()
	{
		std::cout << "zdir: " << zdir << std::endl;
		std::cout << "loc: " << loc << std::endl;
		std::cout << "angle: " << angle << std::endl;
		std::cout << "radius: " << radius << std::endl;
	}

	vec3d loc, xdir, ydir, zdir;
	double radius, angle;
	bool flag_first_time;
	bool flag_sfpn;
	bool flag_set_axis;
	bool flag_single_side_proj;

public:
	void estimate_normal();
	void aqd_fitting();
	void sfpn_fitting();
	void fitting();
	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv = false);
	void projection_with_normal(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, std::vector<vec3d>& tgt_normal, bool save_uv = false);
};

class SphereFitter : public SurfFitter
{
public:
	SphereFitter()
		: radius(0.0)
	{
		//canonical frame
		xdir[0] = 1.0;
		xdir[1] = 0.0;
		xdir[2] = 0.0;
		ydir[0] = 0.0;
		ydir[1] = 1.0;
		ydir[2] = 0.0;
		zdir[0] = 0.0;
		zdir[1] = 0.0;
		zdir[2] = 1.0;
	}
	SphereFitter(const vec3d& _loc, const vec3d& _xdir, const vec3d& _ydir, const vec3d& _zdir,
		double _radius, bool _u_closed, bool _v_closed,
		double _u_min, double _u_max, double _v_min, double _v_max)
		: loc(_loc), xdir(_xdir), ydir(_ydir), zdir(_zdir), radius(_radius)
	{
		this->u_closed = _u_closed, this->v_closed = _v_closed;
		this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
		this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? M_PI : _v_max;
	}
	vec3d GetPosition(double u, double v)
	{
		return loc + radius * (cos(v) * (cos(u) * xdir + sin(u) * ydir) + sin(v) * zdir);
	}

	vec3d GetNormal(double u, double v)
	{
		vec3d normal = GetPosition(u, v) - loc;
		normal.Normalize();
		return normal;
	}

	vec3d loc, xdir, ydir, zdir;
	double radius;
public:
	void fitting();
	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv = false);
};

class TorusFitter : public SurfFitter
{
public:
	TorusFitter()
		: min_radius(0.0), max_radius(1.0)
	{}
	TorusFitter(const vec3d& _loc, const vec3d& _xdir, const vec3d& _ydir, const vec3d& _zdir,
		double _max_radius, double _min_radius, bool _u_closed, bool _v_closed,
		double _u_min, double _u_max, double _v_min, double _v_max)
		: loc(_loc), xdir(_xdir), ydir(_ydir), zdir(_zdir),
		max_radius(_max_radius), min_radius(_min_radius)
	{
		this->u_closed = _u_closed, this->v_closed = _v_closed;
		this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
		this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
	}
	vec3d GetPosition(double u, double v)
	{
		return loc + (max_radius + min_radius * cos(v)) * (cos(u) * xdir + sin(u) * ydir) + min_radius * sin(v) * zdir;
	}

	vec3d GetNormal(double u, double v)
	{
		//no need to consider normalization
		vec3d uvec = (max_radius + min_radius * cos(v)) * (-sin(u) * xdir + cos(u) * ydir);
		vec3d vvec = (0.0 - min_radius * sin(v)) * (cos(u) * xdir + sin(u) * ydir) + min_radius * cos(v) * zdir;
		vec3d normal = uvec.Cross(vvec);
		normal.Normalize();
		return normal;
	}

	vec3d loc, xdir, ydir, zdir;
	double max_radius, min_radius;

public:
	void fitting();
	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv = false);
};


class SplineFitter : public SurfFitter
{
public:
	SplineFitter()
	{
		spline = NULL; //update during fitting
		numControls = 5;
		degree = 3;
		numSamples[0] = 10;
		numSamples[1] = 10;
		nurbs_surf = NULL;
	}
	~SplineFitter()
	{
		if (spline)
			delete spline;
	}

	SplineFitter(int u_degree, int v_degree,
		const std::vector<gte::Vector<3, double>>& controls,
		const std::vector<double>& my_uknots, const std::vector<double>& my_vknots,
		const std::vector<double>& myweights,
		bool _u_closed, bool _v_closed,
		double _u_min, double _u_max, double _v_min, double _v_max);

	//input must be grid point cloud with size numSamples[0] * numSamples[1]
	vec3d GetPosition(double u, double v);

	vec3d GetNormal(double u, double v);
	
	void fitting();
	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv = false);

	//fitter can only get position
	gte::BSplineSurface<3, double> *spline;
	std::vector<vec3d> spline_pts;
	int numControls;
	int degree;
	int numSamples[2];
	gte::NURBSSurface<3, double>* nurbs_surf;

};


class ThinPlateSpline : public SurfFitter
{
public:
	std::vector<gte::IntpThinPlateSpline2<double>*> vec_thinplate;
	ThinPlateSpline()
	{
		smoothness = 10.0;
		for (size_t i = 0; i < 3; i++)
		{
			vec_thinplate.push_back(NULL);
		}
	}
	~ThinPlateSpline()
	{
		//if (vec_thinplate)
		//	delete vec_thinplate;
		if (!vec_thinplate.empty())
		{
			for (size_t i = 0; i < vec_thinplate.size(); i++)
			{
				delete vec_thinplate[i];
			}
		}
	}

	vec3d GetPosition(double u, double v);
	void fitting();
	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv = false);

	

	std::vector<vec3d> spline_pts;

	double smoothness;

};

//////////////////////////////////////////////
class RevolutionFitter : public SurfFitter
{
	//no fitting module
public:
	RevolutionFitter(
		int u_degree,
		const vec3d& _loc, const vec3d& _zdir,
		const std::vector<gte::Vector<3, double>>& controls,
		const std::vector<double>& my_uknots,
		const std::vector<double>& myweights,
		bool _u_closed, bool _v_closed,
		double _u_min, double _u_max, double _v_min, double _v_max)
		: loc(_loc), zdir(_zdir)
	{
		this->u_closed = _u_closed, this->v_closed = _v_closed;
		this->u_min = _u_min, this->u_max = _u_max;
		this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
		mycurve = new MySplineCurve(u_degree, controls, my_uknots, myweights, _u_min, _u_max, _u_closed);
	}

	RevolutionFitter(const vec3d& _start, const vec3d& _end,
		const vec3d& _loc, const vec3d& _zdir, bool _v_closed, double _u_min, double _u_max, double _v_min, double _v_max)
		: loc(_loc), zdir(_zdir)
	{
		this->u_closed = false;
		this->v_closed = _v_closed;
		this->u_min = _u_min, this->u_max = _u_max;
		this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
		mycurve = new MyLine(_start, _end);
	}
	RevolutionFitter(const vec3d& c_loc, const vec3d& _dirx, const vec3d& _diry, double _radius, bool _u_closed, bool _v_closed,
		const vec3d& _loc, const vec3d& _zdir, double _u_min, double _u_max, double _v_min, double _v_max)
		: loc(_loc), zdir(_zdir)
	{
		this->u_closed = _u_closed, this->v_closed = _v_closed;
		this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
		this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
		mycurve = new MyCircle(c_loc, _dirx, _diry, _radius, _u_min, _u_max, _u_closed);
	}
	RevolutionFitter(const vec3d& c_loc, const vec3d& _dirx, const vec3d& _diry, double _x_radius, double _y_radius, bool _u_closed, bool _v_closed,
		const vec3d& _loc, const vec3d& _zdir, double _u_min, double _u_max, double _v_min, double _v_max)
		: loc(_loc), zdir(_zdir)
	{
		this->u_closed = _u_closed, this->v_closed = _v_closed;
		this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
		this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
		mycurve = new MyEllipse(c_loc, _dirx, _diry, _x_radius, _y_radius, _u_min, _u_max, _u_closed);
	}

	~RevolutionFitter()
	{
		if (mycurve)
			delete mycurve;
	}

	vec3d GetPosition(double u, double v)
	{
		const double cos_angle = cos(v);
		const double sin_angle = sin(v);
		vec3d V = mycurve->GetPosition(u) - loc;
		return loc + cos_angle * V + (1 - cos_angle) * V.Dot(zdir) * zdir + sin_angle * zdir.Cross(V);
	}

	vec3d GetNormal(double u, double v)
	{
		//normalized version
		const double cos_angle = cos(v);
		const double sin_angle = sin(v);
		vec3d X = mycurve->GetPosition(u) - loc;
		vec3d tang_X = mycurve->GetTangent(u);
		vec3d utang = cos_angle * tang_X + (1 - cos_angle) * tang_X.Dot(zdir) * zdir + sin_angle * zdir.Cross(tang_X);
		vec3d vtang = -sin_angle * X + sin_angle * X.Dot(zdir) * zdir + cos_angle * zdir.Cross(X);
		vec3d normal = utang.Cross(vtang);
		normal.Normalize();
		return normal;
	}

	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv = false)
	{
		//no impl
	}
	
	void fitting()
	{
		//no impl
	}


protected:
	MyCurve* mycurve;
	vec3d loc, zdir;
};


class ExtrusionFitter : public SurfFitter
{
public:
	ExtrusionFitter(
		int u_degree,
		const vec3d& _zdir,
		const std::vector<gte::Vector<3, double>>& controls,
		const std::vector<double>& my_uknots,
		const std::vector<double>& myweights,
		bool _u_closed,
		double _u_min, double _u_max, double _v_min, double _v_max)
		: zdir(_zdir)
	{
		this->u_closed = _u_closed, this->v_closed = false;
		this->u_min = _u_min, this->u_max = _u_max;
		this->v_min = _v_min, this->v_max = _v_max;
		mycurve = new MySplineCurve(u_degree, controls, my_uknots, myweights, u_min, u_max, u_closed);
	}
	ExtrusionFitter(const vec3d& _start, const vec3d& _end,
		const vec3d& _zdir, double _u_min, double _u_max, double _v_min, double _v_max)
		: zdir(_zdir)
	{
		this->u_closed = this->v_closed = false;
		this->u_min = _u_min, this->u_max = _u_max;
		this->v_min = _v_min, this->v_max = _v_max;
		mycurve = new MyLine(_start, _end);
	}
	ExtrusionFitter(const vec3d& _loc, const vec3d& _dirx, const vec3d& _diry, double _radius, bool _u_closed,
		const vec3d& _zdir, double _u_min, double _u_max, double _v_min, double _v_max)
		: zdir(_zdir)
	{
		this->u_closed = _u_closed, this->v_closed = false;
		this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
		this->v_min = _v_min, this->v_max = _v_max;
		mycurve = new MyCircle(_loc, _dirx, _diry, _radius, _u_min, _u_max, u_closed);
	}
	ExtrusionFitter(const vec3d& _loc, const vec3d& _dirx, const vec3d& _diry, double _x_radius, double _y_radius, bool _u_closed,
		const vec3d& _zdir, double _u_min, double _u_max, double _v_min, double _v_max)
		: zdir(_zdir)
	{
		this->u_closed = _u_closed, this->v_closed = false;
		this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
		this->v_min = _v_min, this->v_max = _v_max;
		mycurve = new MyEllipse(_loc, _dirx, _diry, _x_radius, _y_radius, _u_min, _u_max, u_closed);
	}

	~ExtrusionFitter()
	{
		if (mycurve)
			delete mycurve;
	}

	vec3d GetPosition(double u, double v)
	{
		return mycurve->GetPosition(u) + v * zdir;
	}

	vec3d GetNormal(double u, double v)
	{
		//normalized
		vec3d utang = mycurve->GetTangent(u);
		vec3d normal = utang.Cross(zdir);
		normal.Normalize();
		return normal;
	}

	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv = false)
	{
		//no impl
	}

	void fitting()
	{
		//no impl
	}

protected:
	MyCurve* mycurve;
	vec3d zdir;
};