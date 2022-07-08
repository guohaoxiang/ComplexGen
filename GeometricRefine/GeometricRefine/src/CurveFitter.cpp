#include <iostream>
#include "CurveFitter.h"
#include "Mathematics/ApprOrthogonalLine3.h"
#include "Mathematics/ApprCircle2.h"
#include "Mathematics/ApprEllipse2.h"
#include "Mathematics/BSplineCurveFit.h"
#include "SurfFitter.h"
#include "Helper.h"


using namespace gte;

void LineFitter::fitting()
{
	assert(input_pts.size() != 0);
	//ApprOrthogonalPlane3<double>  fitter;
	std::vector<Vector3<double>> positions;
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		Vector3<double> data;
		data[0] = input_pts[i][0];
		data[1] = input_pts[i][1];
		data[2] = input_pts[i][2];
		positions.push_back(data);
	}
	
	ApprOrthogonalLine3<double> fitter;
	fitter.Fit(positions.size(), positions.data());

	for (size_t i = 0; i < 3; i++)
	{
		start[i] = fitter.GetParameters().origin[i];
		dir[i] = fitter.GetParameters().direction[i];
	}
	
	dir.Normalize();
}

void LineFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_t)
{
	tgt.clear();
	if (save_t)
	{
		this->ts.clear();
	}
	for (size_t i = 0; i < src.size(); i++)
	{
		vec3d tmp = src[i] - start;
		double t = tmp * dir;
		if (save_t)
		{
			ts.push_back(t);
		}

		tgt.push_back(start + t * dir);
	}

	if (save_t)
	{
		//update u_max, umin
		//end point defined as the point with maximum t
		t_max = *std::max_element(ts.begin(), ts.end());
		t_min = *std::min_element(ts.begin(), ts.end());
	}
}

void CircleFitter::fitting()
{
	PlaneFitter pf;
	pf.set_points(input_pts);
	pf.fitting();
	//get normal and loc is needed
	vec3d plane_loc = pf.loc;
	if (!flag_set_axis)
	{
		dirz = pf.zdir;
	}
	
	get_rotmat_from_normal(dirz, mat_v2s);
	mat_s2v = mat_v2s.Inverse();

	assert(input_pts.size() != 0);
	std::vector<Vector2<double>> positions;
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		Vector2<double> data;
		vec3d proj_pt = mat_v2s * (input_pts[i] - plane_loc);
		data[0] = proj_pt[0];
		data[1] = proj_pt[1];
		positions.push_back(data);
	}
	
	ApprCircle2<double> fitter;
	Circle2<double> circle;
	fitter.FitUsingSquaredLengths((int)positions.size(), positions.data(), circle);
	radius = circle.radius;
	vec3d circle_center(0, 0, 0);
	circle_center[0] = circle.center[0];
	circle_center[1] = circle.center[1];
	
	loc = plane_loc + mat_s2v * circle_center;
	get_vertical_vectors(dirz, dirx, diry);
}

void CircleFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_t)
{
	tgt.clear();
	if (save_t)
	{
		this->ts.clear();
	}

	//std::vector<double> ts_ref;

	for (size_t i = 0; i < src.size(); i++)
	{
		vec3d tmp = src[i] - loc;
		tmp = tmp - tmp * dirz * dirz;
		tmp.Normalize();
		double projx = tmp * dirx;
		double projy = tmp * diry;
		double t = safe_acos(projx);
		if (projy < 0.0)
		{
			t = 2 * M_PI - t;
		}

		if (save_t)
		{
			ts.push_back(t);
			/*double t_ref = t;
			if (t_ref > M_PI) t_ref = t - 2 * M_PI;
			ts_ref.push_back(t_ref);*/
		}

		tgt.push_back(GetPosition(t));
	}

	if (save_t)
	{
		//update u_max, umin
		//end point defined as the point with maximum t
		t_max = *std::max_element(ts.begin(), ts.end());
		t_min = *std::min_element(ts.begin(), ts.end());
		//std::cout << "t min:  " << t_min << " t max: " << t_max << "t diff: " << t_max - t_min << std::endl;

		//if ((t_max - t_min) > 2 * M_PI - TH_CIRCLE)
		if ((t_max - t_min) > M_PI) //update 0127 
		{
			//double t_max_ref = *std::max_element(ts_ref.begin(), ts_ref.end());
			//double t_min_ref = *std::min_element(ts_ref.begin(), ts_ref.end());
			//if ((t_max_ref - t_min_ref) < 2 * M_PI - TH_CIRCLE)
			//{
			//	//ref not span a circle
			//	t_max = t_max_ref;
			//	t_min = t_min_ref;
			//}
			
			update_minmax(ts, t_min, t_max);
			
			//for reverse case
			/*int tmp_split = 8;
			update_minmax(ts, t_min, t_max, tmp_split);*/


			//update 1029, not change closeness anymore

			/*if ((t_max - t_min) > 2 * M_PI - TH_CIRCLE)
			{
				closed = true;
			}*/
			//ignore reverse direction

			/*else
			{
				closed = false;
			}*/
		}
		/*else
		{
			closed = false;
		}*/
		
	}
}

void EllipseFitter::fitting()
{
	PlaneFitter pf;
	pf.set_points(input_pts);
	pf.fitting();
	//get normal and loc is needed
	vec3d plane_loc = pf.loc;
	dirz = pf.zdir;
	get_rotmat_from_normal(dirz, mat_v2s);
	mat_s2v = mat_v2s.Inverse();

	assert(input_pts.size() != 0);
	std::vector<Vector2<double>> positions;
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		Vector2<double> data;
		vec3d proj_pt = mat_v2s * (input_pts[i] - plane_loc);
		data[0] = proj_pt[0];
		data[1] = proj_pt[1];
		positions.push_back(data);
	}

	size_t max_iter = 32;
	bool use_initial_guess = false;

	ApprEllipse2<double> fitter;
	Ellipse2<double> ellipse;
	fitter(positions, max_iter, use_initial_guess, ellipse);
	x_radius = ellipse.extent[0];
	y_radius = ellipse.extent[1];
	
	if (isnan(x_radius) || isinf(x_radius))
	{
		x_radius = MAX_RADIUS;
	}
	
	if (isnan(y_radius) || isinf(y_radius))
	{
		y_radius = MAX_RADIUS;
	}

	//ApprCircle2<double> fitter;
	//Circle2<double> circle;
	//fitter.FitUsingSquaredLengths((int)positions.size(), positions.data(), circle);
	//radius = circle.radius;
	vec3d center(0, 0, 0);
	center[0] = ellipse.center[0];
	center[1] = ellipse.center[1];

	loc = plane_loc + mat_s2v * center;
	//get_vertical_vectors(dirz, dirx, diry);
	vec3d axisx(0, 0, 0), axisy(0, 0, 0);
	for (size_t i = 0; i < 2; i++)
	{
		axisx[i] = ellipse.axis[0][i];
		axisy[i] = ellipse.axis[1][i];
	}
	
	dirx = mat_s2v * axisx;
	diry = mat_s2v * axisy;
}

void EllipseFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_t)
{
	tgt.clear();
	if (save_t)
	{
		this->ts.clear();
	}
	//std::vector<double> ts_ref;
	for (size_t i = 0; i < src.size(); i++)
	{
		vec3d tmp = src[i] - loc;
		tmp = tmp - tmp * dirz * dirz;
		//tmp.Normalize();
		double projx = tmp * dirx / x_radius;
		double projy = tmp * diry / y_radius;
		double t = safe_acos(projx / sqrt(projx * projx + projy * projy));
		if (projy < 0.0)
		{
			t = 2 * M_PI - t;
		}

		if (save_t)
		{
			ts.push_back(t);
			/*double t_ref = t;
			if (t_ref > M_PI) t_ref = t - 2 * M_PI;
			ts_ref.push_back(t_ref);*/
		}

		tgt.push_back(GetPosition(t));
	}

	if (save_t)
	{
		//update u_max, umin
		//end point defined as the point with maximum t
		t_max = *std::max_element(ts.begin(), ts.end());
		t_min = *std::min_element(ts.begin(), ts.end());

		if ((t_max - t_min) > 2 * M_PI - TH_CIRCLE)
		{
			//double t_max_ref = *std::max_element(ts_ref.begin(), ts_ref.end());
			//double t_min_ref = *std::min_element(ts_ref.begin(), ts_ref.end());
			//if ((t_max_ref - t_min_ref) < 2 * M_PI - TH_CIRCLE)
			//{
			//	//ref not span a circle
			//	t_max = t_max_ref;
			//	t_min = t_min_ref;
			//}
			update_minmax(ts, t_min, t_max);
			/*if ((t_max - t_min) > 2 * M_PI - TH_CIRCLE)
			{
				closed = true;
			}
			else
			{
				closed = false;
			}*/
		}
		/*else
		{
			closed = false;
		}*/
	}
}


vec3d SplineCurveFitter::GetPosition(double t)
{
	//if nurbs set, return nurbs value:
	if (nurbs_curve)
	{
		gte::Vector<3, double> V = nurbs_curve->GetPosition(t);
		return vec3d(V[0], V[1], V[2]);
	}

	assert(spline_curve != NULL);
	vec3d value;
	Vector3<double> jet;
	spline_curve->Evaluate(t, 0, &jet);
	for (size_t i = 0; i < 3; i++)
	{
		value[i] = jet[i];
	}
	return value;
}


void SplineCurveFitter::fitting()
{
	assert(input_pts.size() != 0);
	//ApprOrthogonalPlane3<double>  fitter;
	//std::vector<Vector3<double>> positions;
	std::vector<double> positions;
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			positions.push_back(input_pts[i][j]);
		}
	}

	BSplineCurveFit<double> fitter(3, numSamples, positions.data(), degree, numControls);

	//set spline
	BasisFunctionInput<double> basis_input;
	//set basis
	basis_input.degree = fitter.GetBasis().GetDegree();
	basis_input.numControls = fitter.GetBasis().GetNumControls();
	basis_input.periodic = fitter.GetBasis().IsPeriodic();
	basis_input.uniform = fitter.GetBasis().IsOpen();
	basis_input.numUniqueKnots = fitter.GetBasis().GetNumUniqueKnots();
	basis_input.uniqueKnots.resize(basis_input.numUniqueKnots);
	auto nots_begin = fitter.GetBasis().GetUniqueKnots();
	std::copy(nots_begin, nots_begin + basis_input.numUniqueKnots, basis_input.uniqueKnots.begin());

	auto control_pts = fitter.GetControlData();
	std::vector<Vector3<double>> control_pts_new;
	control_pts_new.resize(basis_input.numControls);
	for (size_t i = 0; i < basis_input.numControls; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			control_pts_new[i][j] = control_pts[3 * i + j];
		}
	}

	if (spline_curve)
		delete spline_curve;

	spline_curve = new BSplineCurve<3, double>(basis_input, control_pts_new.data());

	spline_pts.clear();
	double split = 1.0 / (numControls - 1);
	for (size_t i = 0; i < numControls; i++)
	{
		spline_pts.push_back(GetPosition(split * i));
	}
}


void SplineCurveFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_t)
{
	//tgt = src;
	//return
	//nn
	
	//if nurbs is set, get spline_pts firstly
	if (nurbs_curve && spline_pts.empty())
	{
		double split = 1.0 / (numControls - 1);
		for (size_t i = 0; i < numControls; i++)
		{
			spline_pts.push_back(GetPosition(split * i));
		}
	}

	tgt.clear();
	assert(spline_pts.size() > 0);
	for (size_t i = 0; i < src.size(); i++)
	{
		double min_dist = -1.0;
		size_t min_id = 0;
		for (size_t j = 0; j < spline_pts.size(); j++)
		{
			double tmp_dist = (spline_pts[j] - src[i]).Length();
			if (min_dist < 0.0 || min_dist > tmp_dist)
			{
				min_dist = tmp_dist;
				min_id = j;
			}
		}
		assert(min_dist > -0.5);
		tgt.push_back(spline_pts[min_id]);
	}
	
}