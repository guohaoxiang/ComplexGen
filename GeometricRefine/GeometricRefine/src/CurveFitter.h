#pragma once

#include <cmath>
#include "TinyVector.h"
#include "MyCurve.h"
#include "TinyMatrix.h"
#include "Mathematics/BSplineCurve.h"
#include "Mathematics/NURBSCurve.h"


#define MAX_RADIUS 1e6

typedef TinyVector<double, 3> vec3d;

class CurveFitter : public MyCurve
{
public:
	CurveFitter()
	{
		t_split = 34;
		dirz = vec3d(0.0, 0.0, 0.0);
	}
	void set_points(const std::vector<vec3d>& pts)
	{
		input_pts = pts;
		ts.clear();
	}

	void set_bspline_nongrid_points(const std::vector<vec3d>& pts, const std::vector<double>& weights)
	{
		//only work for BSpline
		nongrid_pts = pts;
		nongrid_weights = weights;
	}
	
	virtual void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_t) = 0;
	void get_self_projection() {
		projection(input_pts, projection_pts, true);
	}
	//virtual void write_obj_surf(std::ostream& out, size_t& vcounter, int u_div, int v_div);
	//get_uv()

	virtual void set_axis(const vec3d& axis)
	{
		//impl only for circle
	}

	virtual bool check_validness()
	{
		//only impl for ellipse
		return true;
	}

	void get_nn_proj_pt(const vec3d& src, vec3d& tgt)
	{
		assert(projection_pts.size() != 0);
		//find nn pts of src from projection_pts
		//tgt.clear();
		//for (size_t i = 0; i < src.size(); i++)
		{
			double min_dist = -1.0;
			size_t min_id = -1;
			for (size_t j = 0; j < projection_pts.size(); j++)
			{
				double tmp_dist = (src - projection_pts[j]).Length();
				if (min_dist < 0.0 || min_dist > tmp_dist)
				{
					min_dist = tmp_dist;
					min_id = j;
				}
			}
			assert(min_dist > -0.5);
			//tgt.push_back(projection_pts[min_id]);
			tgt = projection_pts[min_id];
		}
	}

	virtual void get_seg_pts() //similar to grid, rewrite for NURBS
	{
		seg_pts.clear();
		double t_inter = 1.0 / (t_split - 1);
		for (size_t i = 0; i < t_split; i++)
		{
			double dt = t_min + i * (t_max - t_min) * t_inter;
			seg_pts.push_back(GetPosition(dt));
		}
	}

	void get_nn_proj_pts(const std::vector<vec3d>& src, std::vector<vec3d>& tgt)
	{
		assert(projection_pts.size() != 0);
		//find nn pts of src from projection_pts
		tgt.clear();
		for (size_t i = 0; i < src.size(); i++)
		{
			double min_dist = -1.0;
			size_t min_id = -1;
			for (size_t j = 0; j < projection_pts.size(); j++)
			{
				double tmp_dist = (src[i] - projection_pts[j]).Length();
				if (min_dist < 0.0 || min_dist > tmp_dist)
				{
					min_dist = tmp_dist;
					min_id = j;
				}
			}
			assert(min_dist > -0.5);
			tgt.push_back(projection_pts[min_id]);
		}
	}
	virtual void fitting() = 0;
	std::vector<vec3d> input_pts;
	std::vector<vec3d> projection_pts;
	std::vector<double> ts;
	
	int t_split;
	std::vector<vec3d> seg_pts;
	std::vector<vec3d> nongrid_pts;
	std::vector<double> nongrid_weights;
	vec3d dirz;
};

class LineFitter : public CurveFitter
{
public:
	LineFitter()
	{
		this->closed = false;
		this->t_min = 0;
		this->t_max = 0;
		dir = vec3d(0, 0, 0);
		start = vec3d(0, 0, 0);
	}

	LineFitter(const vec3d& _start, const vec3d& _end)
		: start(_start)
	{
		this->closed = false;
		this->t_min = 0;
		//this->t_max = 1;
		dir = _end - _start;
		this->t_max = dir.Length();
		dir.Normalize();
	}
	vec3d GetPosition(double t)
	{
		return start + t * dir;
	}

	vec3d start;
	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_t);
	void fitting();
	vec3d dir;
};

class CircleFitter : public CurveFitter
{
public:
	CircleFitter()
		: radius(1.0)
	{
		flag_set_axis = false;
	}
	CircleFitter(const vec3d& _loc, const vec3d& _dirx, const vec3d& _diry,
		double _radius, double _t_min, double _t_max, bool _closed)
		: loc(_loc), dirx(_dirx), diry(_diry), radius(_radius)
	{
		this->t_min = _t_min;
		this->t_max = _t_max;
		this->closed = _closed;
		flag_set_axis = false;
		dirz = dirx.Cross(diry);
		dirz.Normalize();
	}
	vec3d GetPosition(double t)
	{
		return loc + radius * (cos(t) * dirx + sin(t) * diry);
	}

	void set_axis(const vec3d& axis)
	{
		dirz = axis;
		flag_set_axis = true;
	}

	vec3d loc, dirx, diry;
	double radius;
	bool flag_set_axis = false;

	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_t = false);
	void fitting();
	//vec3d dirz;
	ColumnMatrix3d mat_v2s, mat_s2v; //volume to surface
};

class EllipseFitter : public CurveFitter
{
public:
	EllipseFitter() {
		x_radius = 1.0;
		y_radius = 1.0;
	}

	EllipseFitter(const vec3d& _loc, const vec3d& _dirx, const vec3d& _diry,
		double _x_radius, double _y_radius, double _t_min, double _t_max, bool _closed)
		: loc(_loc), dirx(_dirx), diry(_diry), x_radius(_x_radius), y_radius(_y_radius)
	{
		this->t_min = _t_min;
		this->t_max = _t_max;
		this->closed = _closed;
	}
	vec3d GetPosition(double t)
	{
		return loc + x_radius * cos(t) * dirx + y_radius * sin(t) * diry;
	}

	bool check_validness()
	{
		if (x_radius < 1.0 && y_radius < 1.0)
			return true;
		return false;
	}

	vec3d loc, dirx, diry;
	double x_radius, y_radius;


	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_t);
	void fitting();
	//vec3d dirz;
	ColumnMatrix3d mat_v2s, mat_s2v; //volume to surface

};

class SplineCurveFitter : public CurveFitter
{
public:
	SplineCurveFitter()
	{
		numControls = 34;
		degree = 3;
		//to be changed
		numSamples = 34;
		spline_curve = NULL;
		nurbs_curve = NULL;
	}
	~SplineCurveFitter()
	{
		if (spline_curve)
			delete spline_curve;
	}
	
	SplineCurveFitter(int degree,
		const std::vector<gte::Vector<3, double>>& controls,
		const std::vector<double>& myknots,
		const std::vector<double>& myweights,
		double _t_min, double _t_max, bool _closed)
	{
		//not for fitting, only to get NURBS position
		this->t_min = _t_min;
		this->t_max = _t_max;
		this->closed = _closed;

		gte::BasisFunctionInput<double> my_input;
		my_input.degree = degree;
		my_input.numControls = (int)controls.size();
		my_input.periodic = _closed;
		my_input.uniform = false;

		std::vector<std::pair<double, int>> knots_stataus;
		knots_stataus.push_back(std::make_pair(myknots[0], 1));
		for (size_t i = 1; i < myknots.size(); i++)
		{
			if (myknots[i] == knots_stataus.back().first)
				knots_stataus.back().second++;
			else
				knots_stataus.push_back(std::make_pair(myknots[i], 1));
		}

		my_input.numUniqueKnots = (int)knots_stataus.size();
		my_input.uniqueKnots.resize(my_input.numUniqueKnots);
		for (size_t i = 0; i < knots_stataus.size(); i++)
		{
			my_input.uniqueKnots[i].t = knots_stataus[i].first;
			my_input.uniqueKnots[i].multiplicity = knots_stataus[i].second;
		}
		nurbs_curve = new gte::NURBSCurve<3, double>(my_input, controls.data(), myweights.data());
		//store number of control points
		numControls = (int)controls.size();
	}
	
	vec3d GetPosition(double t);
	void projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_t);
	void fitting();

	gte::BSplineCurve<3, double>* spline_curve;
	std::vector<vec3d> spline_pts;
	int numControls;
	int degree;
	int numSamples;

	//set a nurbs curve for generating grid
	gte::NURBSCurve<3, double>* nurbs_curve;
};