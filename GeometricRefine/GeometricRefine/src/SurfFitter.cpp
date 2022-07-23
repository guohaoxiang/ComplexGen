#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <math.h>
#include"SurfFitter.h"

#include "Mathematics/ApprCylinder3.h"
#include "Mathematics/ApprOrthogonalPlane3.h"
#include "Mathematics/ApprCone3.h"
#include "Mathematics/ApprSphere3.h"
#include "Mathematics/ApprTorus3.h"
#include "Mathematics/BSplineSurfaceFit.h"

#define TH_POS 1e-5

//using namespace Eigen;

using namespace gte;
//using gte::Vector3;
//aqd part
#include "quadricfitting.h"

using allquadrics::data_pnw;

FILE _iob[] = { *stdin, *stdout, *stderr };

extern"C" FILE * __cdecl __iob_func(void)
{
	return _iob;
}

const char* quadricTypeNames[] = { "general", "rotationally symmetric", "plane", "sphere",
	"general cylinder", "circular cylinder", "general cone", "circular cone",
	"ellipsoid (BIASED METHOD)", "hyperboloid (BIASED METHOD)",
	"ellipsoid (IMPROVED)", "hyperboloid (IMPROVED)",
	"hyperboloid (1 sheet)", "hyperboloid (2 sheet)", "paraboloid",
	"paraboloid (elliptical)", "paraboloid (hyperbolic)",
	"elliptical cylinder", "hyperbolic cylinder", "parabolic cylinder" };
void outputQuadric(allquadrics::Quadric& q) {
	cout << q.q[0];
	for (int ii = 1; ii < 10; ii++) {
		cout << ", " << q.q[ii];
	}
}

void SurfFitter::expand_input_points(double dist, double expand_weight)
{
	//modify dim_u_input and dim_v_input here
	assert(!v_closed);

	std::vector<vec3d> input_pts_new;
	expand_grid_points(input_pts, dim_u_input, dim_v_input, dist, u_closed, input_pts_new);
	input_pts = input_pts_new;

	/*std::vector<vec3d> u_neg(dim_u_input), u_pos(dim_u_input);
	for (size_t i = 0; i < dim_u_input; i++)
	{
		vec3d dir_neg = input_pts[i * dim_v_input] - input_pts[i * dim_v_input + 1];
		dir_neg.Normalize();
		u_neg[i] = input_pts[i * dim_v_input] + dist * dir_neg;
		vec3d dir_pos = input_pts[i * dim_v_input + dim_v_input - 1] - input_pts[i * dim_v_input + dim_v_input - 2];
		dir_pos.Normalize();
		u_pos[i] = input_pts[i * dim_v_input + dim_v_input - 1] + dir_pos * dist;
	}*/

	if (!u_closed)
	{
		//std::vector<vec3d> v_neg(dim_v_input), v_pos(dim_v_input);
		//for (size_t i = 0; i < dim_v_input; i++)
		//{
		//	vec3d dir_neg = input_pts[i] - input_pts[i + dim_v_input];
		//	dir_neg.Normalize();
		//	v_neg[i] = input_pts[i] + dist * dir_neg;
		//	vec3d dir_pos = input_pts[(dim_u_input - 1) * dim_v_input + i] - input_pts[(dim_u_input - 2) * dim_v_input + i];
		//	dir_pos.Normalize();
		//	v_pos[i] = input_pts[(dim_u_input - 1) * dim_v_input + i] + dist * dir_pos;
		//}
		//
		////four corners
		//vec3d v_neg_neg, v_neg_pos, v_pos_neg, v_pos_pos;
		//vec3d dir_neg = v_neg[0] - v_neg[1];
		//dir_neg.Normalize();
		//v_neg_neg = v_neg[0] + dist * dir_neg;
		//vec3d dir_pos = v_neg[dim_v_input - 1] - v_neg[dim_v_input - 2];
		//dir_pos.Normalize();
		//v_neg_pos = v_neg[dim_v_input - 1] + dist * dir_pos;
		//
		//dir_neg = v_pos[0] - v_pos[1];
		//dir_neg.Normalize();
		//v_pos_neg = v_pos[0] + dist * dir_neg;
		//dir_pos = v_pos[dim_v_input - 1] - v_pos[dim_v_input - 2];
		//dir_pos.Normalize();
		//v_pos_pos = v_pos[dim_v_input - 1] + dist * dir_pos;

		//std::vector<vec3d> input_pts_new;
		//input_pts_new.push_back(v_neg_neg);
		//input_pts_new.insert(input_pts_new.end(), v_neg.begin(), v_neg.end());
		//input_pts_new.push_back(v_neg_pos);
		//for (size_t i = 0; i < dim_u_input; i++)
		//{
		//	input_pts_new.push_back(u_neg[i]);
		//	for (size_t j = 0; j < dim_v_input; j++)
		//	{
		//		input_pts_new.push_back(input_pts[i * dim_v_input + j]);
		//	}
		//	input_pts_new.push_back(u_pos[i]);
		//}
		//input_pts_new.push_back(v_pos_neg);
		//input_pts_new.insert(input_pts_new.end(), v_pos.begin(), v_pos.end());
		//input_pts_new.push_back(v_pos_pos);
		//
		//input_pts = input_pts_new;
		if (!input_pts_weight.empty())
		{
			std::vector<double> input_pts_weight_new;
			for (size_t i = 0; i < dim_v_input + 2; i++)
			{
				input_pts_weight_new.push_back(expand_weight);
			}

			for (size_t i = 0; i < dim_u_input; i++)
			{
				input_pts_weight_new.push_back(expand_weight);
				for (size_t j = 0; j < dim_v_input; j++)
				{
					input_pts_weight_new.push_back(input_pts_weight[i * dim_v_input + j]);
				}
				input_pts_weight_new.push_back(expand_weight);
			}

			for (size_t i = 0; i < dim_v_input + 2; i++)
			{
				input_pts_weight_new.push_back(expand_weight);
			}

			assert(input_pts_weight_new.size() == input_pts_new.size());
			input_pts_weight = input_pts_weight_new;
		}


		dim_u_input = dim_u_input + 2;
	}
	else
	{
		/*input_pts.insert(input_pts.begin(), u_neg.begin(), u_neg.end());
		input_pts.insert(input_pts.end(), u_pos.begin(), u_pos.end());*/
		/*std::vector<vec3d> input_pts_new;
		for (size_t i = 0; i < dim_u_input; i++)
		{
			input_pts_new.push_back(u_neg[i]);
			for (size_t j = 0; j < dim_v_input; j++)
			{
				input_pts_new.push_back(input_pts[i * dim_v_input + j]);
			}
			input_pts_new.push_back(u_pos[i]);
		}
		input_pts = input_pts_new;*/

		if (!input_pts_weight.empty())
		{
			std::vector<double> input_pts_weight_new;
			for (size_t i = 0; i < dim_u_input; i++)
			{
				input_pts_weight_new.push_back(expand_weight);
				for (size_t j = 0; j < dim_v_input; j++)
				{
					input_pts_weight_new.push_back(input_pts_weight[i * dim_v_input + j]);
				}
				input_pts_weight_new.push_back(expand_weight);
			}
			assert(input_pts_weight_new.size() == input_pts_new.size());
			input_pts_weight = input_pts_weight_new;
		}
	}

	dim_v_input = dim_v_input + 2;

}

void SurfFitter::subdivide_grid(int split)
{
	std::vector<vec3d> input_pts_usplit, input_pts_vsplit;
	//update dim_u_input, dim_v_input
	//int dim_u_input_ori = dim_u_input, dim_v_input_ori = dim_v_input;
	std::vector<double> weights_usplit, weights_vsplit;
	for (size_t i = 0; i < dim_u_input - 1; i++)
	{
		for (size_t j = 0; j < split; j++)
		{
			for (size_t k = 0; k < dim_v_input; k++)
			{
				vec3d ustart = input_pts[i * dim_v_input + k], uend = input_pts[(i + 1) * dim_v_input + k], udiff = (uend - ustart) / split;
				input_pts_usplit.push_back(ustart + j * 1.0 * udiff);
			}
		}
	}

	if (!input_pts_weight.empty())
	{
		for (size_t i = 0; i < dim_u_input - 1; i++)
		{
			for (size_t j = 0; j < split; j++)
			{
				for (size_t k = 0; k < dim_v_input; k++)
				{
					/*vec3d ustart = input_pts[i * dim_v_input + k], uend = input_pts[(i + 1) * dim_v_input + k], udiff = (uend - ustart) / split;
					input_pts_usplit.push_back(ustart + j * 1.0 * udiff);*/
					weights_usplit.push_back(input_pts_weight[i * dim_v_input + k]);
				}
			}
		}
	}

	if (u_closed)
	{
		for (size_t j = 0; j < split; j++)
		{
			for (size_t k = 0; k < dim_v_input; k++)
			{
				vec3d ustart = input_pts[(dim_u_input - 1) * dim_v_input + k], uend = input_pts[k], udiff = (uend - ustart) / split;
				input_pts_usplit.push_back(ustart + j * 1.0 * udiff);
			}
		}

		if (!input_pts_weight.empty())
		{
			for (size_t j = 0; j < split; j++)
			{
				for (size_t k = 0; k < dim_v_input; k++)
				{
					weights_usplit.push_back(input_pts_weight[(dim_u_input - 1) * dim_v_input + k]);
				}
			}
		}

		dim_u_input = dim_u_input * split;

	}
	else
	{
		for (size_t k = 0; k < dim_v_input; k++)
		{
			input_pts_usplit.push_back(input_pts[(dim_u_input - 1) * dim_v_input + k]);
		}

		if (!input_pts_weight.empty())
		{
			for (size_t k = 0; k < dim_v_input; k++)
			{
				//input_pts_usplit.push_back(input_pts[(dim_u_input - 1) * dim_v_input + k]);
				weights_usplit.push_back(input_pts_weight[(dim_u_input - 1) * dim_v_input + k]);
			}
		}

		dim_u_input = (dim_u_input - 1) * split + 1;
	}
	//input_pts = input_pts_usplit;

	//v_part
	for (size_t i = 0; i < dim_u_input; i++)
	{
		for (size_t j = 0; j < dim_v_input - 1; j++)
		{
			vec3d vstart = input_pts_usplit[i * dim_v_input + j], vend = input_pts_usplit[i * dim_v_input + j + 1], vdiff = (vend - vstart) / split;
			for (size_t k = 0; k < split; k++)
			{
				input_pts_vsplit.push_back(vstart + k * 1.0 * vdiff);
			}
		}
		input_pts_vsplit.push_back(input_pts_usplit[i * dim_v_input + dim_v_input - 1]);
	}

	if (!input_pts_weight.empty())
	{
		for (size_t i = 0; i < dim_u_input; i++)
		{
			for (size_t j = 0; j < dim_v_input - 1; j++)
			{
				for (size_t k = 0; k < split; k++)
				{
					weights_vsplit.push_back(weights_usplit[i * dim_v_input + j]);
				}
			}
			weights_vsplit.push_back(weights_usplit[i * dim_v_input + dim_v_input - 1]);
		}
		input_pts_weight = weights_vsplit;
		assert(input_pts_weight.size() == input_pts_vsplit.size());
	}

	dim_v_input = (dim_v_input - 1) * split + 1;
	input_pts = input_pts_vsplit;

	//weight
}

void SurfFitter::get_nn_proj_pt(const vec3d& src, vec3d& tgt)
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

void SurfFitter::get_nn_proj_pts(const std::vector<vec3d>& src, std::vector<vec3d>& tgt)
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

void SurfFitter::get_grid_pts(bool with_normal)
{
	//better call this function after getting the right uv
	grid_pts.clear();
	grid_normals.clear();
	int denom_u = u_split - 1, denom_v = v_split - 1;
	if (u_closed)
	{
		denom_u = u_split;
	}
	if (v_closed)
	{
		denom_v = v_split;
	}

	double u_inter = 1.0 / denom_u;
	double v_inter = 1.0 / denom_v;
	for (size_t i = 0; i < u_split; i++)
	{
		double du = u_min + i * (u_max - u_min) * u_inter;
		for (size_t j = 0; j < v_split; j++)
		{
			double dv = v_min + j * (v_max - v_min) * v_inter;
			grid_pts.push_back(GetPosition(du, dv));
			grid_normals.push_back(GetNormal(du, dv));
		}
	}
}

void SurfFitter::get_nn_grid_pts(const std::vector<vec3d>& src, std::vector<vec3d>& tgt)
{
	assert(grid_pts.size() != 0);
	//find nn pts of src from projection_pts
	tgt.clear();
	for (size_t i = 0; i < src.size(); i++)
	{
		double min_dist = -1.0;
		size_t min_id = -1;
		for (size_t j = 0; j < grid_pts.size(); j++)
		{
			double tmp_dist = (src[i] - grid_pts[j]).Length();
			if (min_dist < 0.0 || min_dist > tmp_dist)
			{
				min_dist = tmp_dist;
				min_id = j;
			}
		}
		assert(min_dist > -0.5);
		tgt.push_back(grid_pts[min_id]);
	}
}

void SurfFitter::get_grid_tri(std::vector<vec3d>& pts, std::vector<std::vector<size_t>>& faces)
{
	get_grid_pts();
	pts = grid_pts;
	faces.clear();
	size_t counter = 0;
	for (size_t j = 0; j < u_split - 1; j++)
	{
		for (size_t k = 0; k < v_split - 1; k++)
		{
			//faces.push_back(std::vector<size_t>({ counter + j * v_split + k, counter + (j + 1) * v_split + k, counter + (j + 1) * v_split + k + 1, counter + j * v_split + k + 1 }));
			faces.push_back(std::vector<size_t>({ counter + j * v_split + k, counter + (j + 1) * v_split + k, counter + (j + 1) * v_split + k + 1 }));
			faces.push_back(std::vector<size_t>({ counter + j * v_split + k, counter + (j + 1) * v_split + k + 1, counter + j * v_split + k + 1 }));

		}
	}

	if (u_closed)
	{
		for (size_t j = 0; j < v_split - 1; j++)
		{
			//faces.push_back(std::vector<size_t>({ counter + (u_split - 1) * v_split + j, counter + j, counter + j + 1, counter + (u_split - 1) * v_split + j + 1 }));
			faces.push_back(std::vector<size_t>({ counter + (u_split - 1) * v_split + j, counter + j, counter + j + 1 }));
			faces.push_back(std::vector<size_t>({ counter + (u_split - 1) * v_split + j,  counter + j + 1, counter + (u_split - 1) * v_split + j + 1 }));
		}
	}

	if (v_closed)
	{
		for (size_t j = 0; j < u_split - 1; j++)
		{
			//faces.push_back(std::vector<size_t>({ counter + j * v_split + v_split - 1, counter + (j + 1) * v_split + v_split - 1, counter + (j + 1) * v_split, counter + j * v_split }));
			faces.push_back(std::vector<size_t>({ counter + j * v_split + v_split - 1, counter + (j + 1) * v_split + v_split - 1, counter + (j + 1) * v_split }));
			faces.push_back(std::vector<size_t>({ counter + j * v_split + v_split - 1, counter + (j + 1) * v_split, counter + j * v_split }));
		}
	}

	if (u_closed && v_closed)
	{
		//faces.push_back(std::vector<size_t>({ counter, counter + (u_split - 1) * v_split, counter + u_split * v_split - 1, counter + v_split - 1 }));
		faces.push_back(std::vector<size_t>({ counter, counter + (u_split - 1) * v_split, counter + u_split * v_split - 1 }));
		faces.push_back(std::vector<size_t>({ counter, counter + u_split * v_split - 1, counter + v_split - 1 }));
	}
}

void SurfFitter::save_input_patch_obj(const std::string& fn)
{
	//save input grid obj according to dim_u, dim_v, u_close, v_close
	std::ofstream out(fn);
	assert(input_pts.size() == dim_u_input * dim_v_input);
	int denom_u = dim_u_input - 1, denom_v = dim_v_input - 1;
	int u_div = dim_u_input, v_div = dim_v_input;
	if (u_closed)
	{
		denom_u = dim_u_input;
	}
	if (v_closed)
	{
		denom_v = dim_v_input;
	}

	for (int i = 0; i < dim_u_input; i++)
	{
		//double du = u_min + i * (u_max - u_min) / denom_u;
		for (int j = 0; j < dim_v_input; j++)
		{
			//double dv = v_min + j * (v_max - v_min) / denom_v;
			out << "v " << input_pts[i * dim_v_input + j] << std::endl;
		}
	}

	int vcounter = 1;

	for (int i = 0; i < u_div - 1; i++)
	{
		for (int j = 0; j < v_div - 1; j++)
		{
			out << "f "
				<< vcounter + i * v_div + j << ' '
				<< vcounter + (i + 1) * v_div + j << ' '
				<< vcounter + (i + 1) * v_div + j + 1 << ' '
				<< vcounter + i * v_div + j + 1 << std::endl;
		}
	}

	if (u_closed)
	{
		for (int j = 0; j < v_div - 1; j++)
		{
			out << "f "
				<< vcounter + (u_div - 1) * v_div + j << ' '
				<< vcounter + j << ' '
				<< vcounter + j + 1 << ' '
				<< vcounter + (u_div - 1) * v_div + j + 1 << std::endl;
		}
	}

	if (v_closed)
	{
		for (int i = 0; i < u_div - 1; i++)
		{
			out << "f "
				<< vcounter + i * v_div + v_div - 1 << ' '
				<< vcounter + (i + 1) * v_div + v_div - 1 << ' '
				<< vcounter + (i + 1) * v_div << ' '
				<< vcounter + i * v_div << std::endl;
		}
	}

	if (u_closed && v_closed)
	{
		out << "f "
			<< vcounter << ' '
			<< vcounter + (u_div - 1) * v_div << ' '
			<< vcounter + u_div * v_div - 1 << ' '
			<< vcounter + v_div - 1 << std::endl;
	}

	out.close();
}

void SurfFitter::write_data_surf(std::ostream& out, int u_div, int v_div, bool flag_normal)
{
	out << "surf " << (u_closed ? "uclosed " : "uopen ") << (v_closed ? "vclosed " : "vopen ") << u_div << ' ' << v_div << std::endl;
	int denom_u = u_div - 1, denom_v = v_div - 1;
	if (u_closed)
		denom_u = u_div;
	if (v_closed)
		denom_v = v_div;

	for (int i = 0; i < u_div; i++)
	{
		double du = u_min + i * (u_max - u_min) / denom_u;
		for (int j = 0; j < v_div; j++)
		{
			double dv = v_min + j * (v_max - v_min) / denom_v;
			if (!flag_normal)
				out << GetPosition(du, dv) << std::endl;
			else
			{
				out << GetPosition(du, dv) << " " << GetNormal(du, dv) << std::endl;
			}
		}
	}
}

void SurfFitter::get_grid_pts_normal(int u_div, int v_div, std::vector<vec3d>& pts, std::vector<vec3d>& normals, bool flag_diff)
{
	//use difference or not
	pts.clear();
	normals.clear();
	int denom_u = u_div - 1, denom_v = v_div - 1;
	if (u_closed)
		denom_u = u_div;
	if (v_closed)
		denom_v = v_div;

	if (!flag_diff)
	{
		for (int i = 0; i < u_div; i++)
		{
			double du = u_min + i * (u_max - u_min) / denom_u;
			for (int j = 0; j < v_div; j++)
			{
				double dv = v_min + j * (v_max - v_min) / denom_v;
				pts.push_back(GetPosition(du, dv));
				normals.push_back(GetNormal(du, dv));
			}
		}
	}
	else
	{
		for (int i = 0; i < u_div; i++)
		{
			double du = u_min + i * (u_max - u_min) / denom_u;
			for (int j = 0; j < v_div; j++)
			{
				double dv = v_min + j * (v_max - v_min) / denom_v;
				pts.push_back(GetPosition(du, dv));
			}
		}

		for (int i = 0; i < u_div; i++)
		{
			for (int j = 0; j < v_div; j++)
			{
				int curx = i, nextx = i + 1, cury = j, nexty = j + 1;
				if (nextx == u_div)
				{
					curx = i - 1;
					nextx = i;
				}
				if (nexty == v_div)
				{
					cury = j - 1;
					nexty = j;
				}
				vec3d udiff = pts[v_div * nextx + j] - pts[v_div * curx + j];
				vec3d vdiff = pts[v_div * i + nexty] - pts[v_div * i + cury];
				vec3d normal = udiff.Cross(vdiff);
				normal.Normalize();
				normals.push_back(normal);
			}
		}

	}

}


void CylinderFitter::estimate_normal()
{
	//esimate normal from points
	estimate_normal_from_grid(input_pts, input_normals, dim_u_input, dim_v_input, u_closed);
	//check output
}

void CylinderFitter::aqd_fitting()
{
	//normals are not used in all fitting, but used for estimating normals
	//fitting by aqd methods
	if (!flag_input_normal)
	{
		estimate_normal();
		//return;
	}
	
	std::vector<data_pnw> datapts;
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		vec3 pos(input_pts[i][0], input_pts[i][1], input_pts[i][2]);
		vec3 normal(input_normals[i][0], input_normals[i][1], input_normals[i][2]);
		datapts.push_back(data_pnw(pos, normal, 1.0));
	}

	//not scaled and centered yet
	allquadrics::Quadric circularCylinder;
	fitCircularCylinder(datapts, circularCylinder);
	/*cout << "cylinder para: " << endl;
	outputQuadric(circularCylinder);
	cout << endl;*/

	Eigen::Matrix3d A;
	A(0, 0) = circularCylinder.q[4];
	A(1, 0) = A(0, 1) = circularCylinder.q[5] * 0.5;
	A(2, 0) = A(0, 2) = circularCylinder.q[6] * 0.5;
	A(1, 1) = circularCylinder.q[7];
	A(1, 2) = A(2, 1) = circularCylinder.q[8] * 0.5;
	A(2, 2) = circularCylinder.q[9];
	
	//std::cout << A << std::endl;
	//EigenSolver<Matrix3d> es(A);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(A);
	/*std::cout << "The eigenvalues of A are:" << endl << es.eigenvalues() << endl;
	std::cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;*/


	//move zero vector to z
	std::vector<int> id_vec({ 0,1,2 });
	double min_error = abs(es.eigenvalues()[0]);
	int min_id = 0;
	
	for (int i = 1; i < 3; i++)
	{
		if (abs(es.eigenvalues()[i]) < min_error)
		{
			min_error = abs(es.eigenvalues()[i]);
			min_id = i;
		}
	}

	if (min_id != 2)
	{
		id_vec[min_id] = 2;
		id_vec[2] = min_id;
	}
	
	Eigen::Matrix3d R, D, RT;
	//std::cout << "R: " << R << std::endl;
	//std::cout << "D: " << D << std::endl;
	//std::cout << "RT: " << RT << std::endl;

	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			D(i, j) = 0.0;
		}
	}

	for (size_t i = 0; i < 3; i++)
	{
		D(i, i) = es.eigenvalues()[id_vec[i]];
	}

	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			R(i, j) = es.eigenvectors()(i, id_vec[j]);
		}
	}

	if (R.determinant() < 0.0)
	{
		R = -R;
	}

	RT = R.transpose();
	//std::cout << "D value: " << std::endl << D << std::endl;
	//std::cout << "R: " << std::endl << R << std::endl;
	//std::cout << "RT: " << std::endl << RT << std::endl;
	//std::cout << "R * RT: " << std::endl << R * RT << std::endl;
	//std::cout << "det R: " << R.determinant() << std::endl;

	//R served as a transformation
	//x' = RT x - trans, a = 0.5 DP RT b
	
	//compute the pseudo inverse of D
	Eigen::Matrix3d DP;
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			DP(i, j) = 0.0;
		}
	}
	
	for (size_t i = 0; i < 3; i++)
	{
		if (abs(D(i, i)) > TH_POS)
		{
			DP(i, i) = 1.0 / D(i, i);
		}
		else
		{
			DP(i, i) = 0.0;
		}
	}

	Eigen::Vector3d b(circularCylinder.q[1], circularCylinder.q[2], circularCylinder.q[3]);
	Eigen::Vector3d rotb = RT * b;
	Eigen::Vector3d bnew = rotb - DP * D * rotb;
	Eigen::Vector3d trans = -0.5 * DP * RT * b;
	
	//std::cout << "bnew: " << bnew << std::endl;
	//double cnew = (trans.transpose() * (D * trans)) + rotb.transpose() * trans + circularCone.q[0];
	double cnew = (trans.transpose() * (D * trans)) + circularCylinder.q[0] + rotb.transpose() * trans;
	//std::cout << "cnew: " << cnew << std::endl;
	
	double r2 = ( -2.0 * cnew / (D(0, 0) + D(1, 1)));
	//std::cout << "radius: " << r << std::endl;
	assert(r2 > 0);
	this->radius = sqrt(r2);
	Eigen::Vector3d tmpx(1, 0, 0), tmpy(0, 1, 0), tmpz(0, 0, 1), tmploc = R * trans;

	tmpx = R * tmpx;
	tmpy = R * tmpy;
	tmpz = R * tmpz;
	//double cnew = circularCone.q[0];
	
	for (size_t i = 0; i < 3; i++)
	{
		loc[i] = tmploc[i];
		xdir[i] = tmpx[i];
		ydir[i] = tmpy[i];
		zdir[i] = tmpz[i];
	}
	

	//convert cylinder back

	//allquadrics::TriangleMesh inputMesh;
	//char* defaultInput = "cylinder.obj";
	//inputMesh.loadObj(defaultInput);
	//inputMesh.centerAndScale(1);
	//allquadrics::Quadric circularCone;
	//fitCircularCylinder(inputMesh, circularCone);
	//cout << "direct fit for circular cylinder: " << endl;
	//outputQuadric(circularCone);
	//cout << endl << endl;
}

void CylinderFitter::fitting()
{
	if (flag_normalize)
	{
		vec3d minv = input_pts[0], maxv = input_pts[0];
		for (size_t i = 1; i < input_pts.size(); i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				if (minv[j] > input_pts[i][j])
					minv[j] = input_pts[i][j];
				if (maxv[j] < input_pts[i][j])
					maxv[j] = input_pts[i][j];
			}
		}

		vec3d scalev = maxv - minv;
		scale = scalev[0];

		for (size_t j = 0; j < 3; j++)
		{
			if (scale < scalev[j])
				scale = scalev[j];
		}
		translation = (minv + maxv) / 2.0;

		for (size_t i = 0; i < input_pts.size(); i++)
		{
			input_pts[i] = (input_pts[i] - translation) / scale;
		}

	}

	//commented on 0812
	if (flag_using_aqd)
	{
		aqd_fitting();
		flag_first_time = false;
		return;
	}
	assert(input_pts.size() != 0);
	unsigned int numThreads = std::thread::hardware_concurrency();
	
	ApprCylinder3<double>* fitter = NULL;
	if (flag_first_time)
	{
		//ApprCylinder3<double> fitter(numThreads, 1024, 512);
		fitter = new ApprCylinder3<double>(numThreads, 1024, 512);
		//flag_first_time = false;
	}
	else
	{
		Vector3<double> axis;
		for (size_t i = 0; i < 3; i++)
		{
			axis[i] = zdir[i];
		}
		//std::cout << "axis: " << axis[0] << " " << axis[1] << " " << axis[2] << std::endl;
		//specify axis, GT never change it.
		fitter = new ApprCylinder3<double>(axis);
	}

	//ApprCylinder3<double> fitter(numThreads, 2048, 1024);
	//ApprCylinder3<double> fitter(1);
	std::vector<Vector3<double>> positions;

	for (size_t i = 0; i < input_pts.size(); i++)
	{
		Vector3<double> data;
		data[0] = input_pts[i][0];
		data[1] = input_pts[i][1];
		data[2] = input_pts[i][2];
		positions.push_back(data);
	}
	unsigned int numVertices = static_cast<unsigned int>(positions.size());
	Cylinder3<double> cylinder;
	double fitting_error = (*fitter)(numVertices, positions.data(), cylinder);
	//std::cout << "min error for fitting cylinder = " << fitting_error << std::endl;
	for (size_t i = 0; i < 3; i++)
	{
		this->loc[i] = cylinder.axis.origin[i];
		this->zdir[i] = cylinder.axis.direction[i];
	}
	this->radius = cylinder.radius;
	//this->height = cylinder.height;
	get_vertical_vectors(this->zdir, this->xdir, this->ydir);
	//std::cout << "x dir" << this->xdir << " y dir : " << this->ydir << std::endl;

	//other parameters to be changed
}

void CylinderFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv)
{
	tgt.clear();
	if (save_uv)
	{
		this->us.clear();
		this->vs.clear();
	}
	//std::vector<double> us_ref;
	for (size_t i = 0; i < src.size(); i++)
	{
		vec3d tmp = src[i] - this->loc;
		double v = tmp * this->zdir;
		if (save_uv)
		{
			this->vs.push_back(v);
		}
		vec3d zpar = v * this->zdir + this->loc;
		vec3d unit_radius = (src[i] - zpar);
		unit_radius.Normalize();
		tgt.push_back(zpar + unit_radius * this->radius);
		if (save_uv)
		{
			double u = safe_acos(unit_radius * this->xdir);
			if (unit_radius * this->ydir < 0.0)
				u = 2 * M_PI - u;
			this->us.push_back(u);

			//update us_ref
			/*double u_ref = u;
			if (u_ref > M_PI)
				u_ref = u - 2 * M_PI;
			us_ref.push_back(u_ref);*/
		}
	}

	if (save_uv)
	{
		//update u_max, umin
		this->u_max = *std::max_element(this->us.begin(), this->us.end());
		this->u_min = *std::min_element(this->us.begin(), this->us.end());
		this->v_max = *std::max_element(this->vs.begin(), this->vs.end());
		this->v_min = *std::min_element(this->vs.begin(), this->vs.end());


		if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
		{
			/*double u_max_ref = *std::max_element(us_ref.begin(), us_ref.end());
			double u_min_ref = *std::min_element(us_ref.begin(), us_ref.end());
			if ((u_max_ref - u_min_ref) < 2 * M_PI - TH_CIRCLE)
			{
				u_max = u_max_ref;
				u_min = u_min_ref;
			}*/
			update_minmax(us, u_min, u_max);
			if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
			{
				u_closed = true;
			}
			/*else
			{
				u_closed = false;
			}*/

			if (u_closed)
			{
				u_max = 2 * M_PI;
				u_min = 0.0;
			}
		}
		/*else
		{
			u_closed = false;
		}*/
		
	}
}

void CylinderFitter::projection_with_normal(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, std::vector<vec3d>& tgt_normal, bool save_uv)
{
	tgt.clear();
	tgt_normal.clear();
	if (save_uv)
	{
		this->us.clear();
		this->vs.clear();
	}
	std::vector<double> us_ref;
	for (size_t i = 0; i < src.size(); i++)
	{
		vec3d tmp = src[i] - this->loc;
		double v = tmp * this->zdir;
		if (save_uv)
		{
			this->vs.push_back(v);
		}
		vec3d zpar = v * this->zdir + this->loc;
		vec3d unit_radius = (src[i] - zpar);
		unit_radius.Normalize();
		tgt.push_back(zpar + unit_radius * this->radius);
		double u = safe_acos(unit_radius * this->xdir);
		if (unit_radius * this->ydir < 0.0)
			u = 2 * M_PI - u;
		tgt_normal.push_back(GetNormal(u, v));

		if (save_uv)
		{
			this->us.push_back(u);
			//update us_ref
			double u_ref = u;
			if (u_ref > M_PI)
				u_ref = u - 2 * M_PI;
			us_ref.push_back(u_ref);
		}
	}

	if (save_uv)
	{
		//update u_max, umin
		this->u_max = *std::max_element(this->us.begin(), this->us.end());
		this->u_min = *std::min_element(this->us.begin(), this->us.end());
		this->v_max = *std::max_element(this->vs.begin(), this->vs.end());
		this->v_min = *std::min_element(this->vs.begin(), this->vs.end());


		if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
		{
			update_minmax(us, u_min, u_max);
			if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
			{
				u_closed = true;
			}
			/*else
			{
				u_closed = false;
			}*/

			if (u_closed)
			{
				u_max = 2 * M_PI;
				u_min = 0.0;
			}
		}

	}
}


void PlaneFitter::fitting()
{
	//aqd_fitting();
	assert(input_pts.size() != 0);
	ApprOrthogonalPlane3<double>  fitter;
	std::vector<Vector3<double>> positions;
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		Vector3<double> data;
		data[0] = input_pts[i][0];
		data[1] = input_pts[i][1];
		data[2] = input_pts[i][2];
		positions.push_back(data);
	}
	int numVertices = static_cast<unsigned int>(positions.size());
	//Cylinder3<double> cylinder;
	fitter.Fit(numVertices, positions.data());
	for (size_t i = 0; i < 3; i++)
	{
		zdir[i] = fitter.GetParameters().second[i];
		loc[i] = fitter.GetParameters().first[i];
	}
	get_vertical_vectors(this->zdir, this->xdir, this->ydir);
}

void PlaneFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv)
{
	tgt.clear();
	if (save_uv)
	{
		this->us.clear();
		this->vs.clear();
	}
	for (size_t i = 0; i < src.size(); i++)
	{
		vec3d tmp = src[i] - this->loc;
		double dist = tmp * zdir;
		tgt.push_back(src[i] - dist * zdir);
		if (save_uv)
		{
			double u = (tmp - dist * zdir) * xdir;
			double v = (tmp - dist * zdir) * ydir;
			this->us.push_back(u);
			this->vs.push_back(v);
		}
	}

	if (save_uv)
	{
		//update u_max, umin
		this->u_max = *std::max_element(this->us.begin(), this->us.end());
		this->u_min = *std::min_element(this->us.begin(), this->us.end());
		this->v_max = *std::max_element(this->vs.begin(), this->vs.end());
		this->v_min = *std::min_element(this->vs.begin(), this->vs.end());
	}
}

void CreateLMCone(std::vector<Vector3<double>> const& X,
	Vector3<double>& coneVertex, Vector3<double>& coneAxis, double& coneAngle)
{
	ApprCone3<double> fitter;
	size_t const maxIterations = 32;
	//size_t const maxIterations = 128;
	double const updateLengthTolerance = 1e-04;
	double const errorDifferenceTolerance = 1e-08;
	////init value
	double const lambdaFactor = 0.001;
	double const lambdaAdjust = 10.0;

	//modified value
	/*double const lambdaFactor = 0.001;
	double const lambdaAdjust = 100.0;*/
	size_t const maxAdjustments = 8;
	//size_t const maxAdjustments = 32;
	bool useConeInputAsInitialGuess = false;

	fitter(static_cast<int>(X.size()), X.data(),
		maxIterations, updateLengthTolerance, errorDifferenceTolerance,
		lambdaFactor, lambdaAdjust, maxAdjustments, useConeInputAsInitialGuess,
		coneVertex, coneAxis, coneAngle);
}

void CreateGNCone(std::vector<Vector3<double>> const& X,
	Vector3<double>& coneVertex, Vector3<double>& coneAxis, double& coneAngle)
{
	ApprCone3<double> fitter;
	size_t const maxIterations = 32;
	double const updateLengthTolerance = 1e-04;
	double const errorDifferenceTolerance = 1e-08;
	bool useConeInputAsInitialGuess = false;

	fitter(static_cast<int>(X.size()), X.data(),
		maxIterations, updateLengthTolerance, errorDifferenceTolerance,
		useConeInputAsInitialGuess, coneVertex, coneAxis, coneAngle);
}

void ConeFitter::estimate_normal()
{
	estimate_normal_from_grid(input_pts, input_normals, dim_u_input, dim_v_input, u_closed);
}

void ConeFitter::aqd_fitting()
{
	//normals are not used in all fitting, but used for estimating normals
	//fitting by aqd methods
	if (!flag_input_normal)
	{
		estimate_normal();
		//return;
	}

	std::vector<data_pnw> datapts;
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		vec3 pos(input_pts[i][0], input_pts[i][1], input_pts[i][2]);
		vec3 normal(input_normals[i][0], input_normals[i][1], input_normals[i][2]);
		datapts.push_back(data_pnw(pos, normal, 1.0));
	}

	//not scaled and centered yet
	allquadrics::Quadric circularCone;
	//fitCircularCylinder(datapts, circularCone);
	fitCircularCone(datapts, circularCone);

	/*cout << "Cone para: " << endl;
	outputQuadric(circularCone);
	cout << endl;*/

	Eigen::Matrix3d A;
	A(0, 0) = circularCone.q[4];
	A(1, 0) = A(0, 1) = circularCone.q[5] * 0.5;
	A(2, 0) = A(0, 2) = circularCone.q[6] * 0.5;
	A(1, 1) = circularCone.q[7];
	A(1, 2) = A(2, 1) = circularCone.q[8] * 0.5;
	A(2, 2) = circularCone.q[9];

	//std::cout << A << std::endl;
	//EigenSolver<Matrix3d> es(A);
	if (A.determinant() > 0)
	{
		A = -A;
	}

	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(A);
	/*std::cout << "The eigenvalues of A are:" << endl << es.eigenvalues() << endl;
	std::cout << "The matrix of eigenvectors, V, is:" << endl << es.eigenvectors() << endl << endl;*/


	//move zero vector to z
	std::vector<int> id_vec({ 0,1,2 });
	double min_error = es.eigenvalues()[0];
	int min_id = 0;

	for (int i = 1; i < 3; i++)
	{
		if (es.eigenvalues()[i] < min_error)
		{
			min_error = es.eigenvalues()[i];
			min_id = i;
		}
	}

	if (min_id != 2)
	{
		id_vec[min_id] = 2;
		id_vec[2] = min_id;
	}

	Eigen::Matrix3d R, D, RT;
	//std::cout << "R: " << R << std::endl;
	//std::cout << "D: " << D << std::endl;
	//std::cout << "RT: " << RT << std::endl;

	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			D(i, j) = 0.0;
		}
	}

	for (size_t i = 0; i < 3; i++)
	{
		D(i, i) = es.eigenvalues()[id_vec[i]];
	}

	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			R(i, j) = es.eigenvectors()(i, id_vec[j]);
		}
	}

	if (R.determinant() < 0.0)
	{
		R = -R;
	}

	RT = R.transpose();
	//std::cout << "D value: " << std::endl << D << std::endl;
	//std::cout << "R: " << std::endl << R << std::endl;
	//std::cout << "RT: " << std::endl << RT << std::endl;
	//std::cout << "R * RT: " << std::endl << R * RT << std::endl;
	//std::cout << "det R: " << R.determinant() << std::endl;

	//R served as a transformation
	//x' = RT x - trans, trans = -0.5 * A-1 b

	//compute the pseudo inverse of D
	Eigen::Matrix3d DP;
	for (size_t i = 0; i < 3; i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			DP(i, j) = 0.0;
		}
	}

	for (size_t i = 0; i < 3; i++)
	{
		if (abs(D(i, i)) > TH_POS)
		{
			DP(i, i) = 1.0 / D(i, i);
		}
		else
		{
			DP(i, i) = 0.0;
		}
	}

	//std::cout << "D inverse: " << std::endl << DP << std::endl;

	Eigen::Vector3d b(circularCone.q[1], circularCone.q[2], circularCone.q[3]);
	//Vector3d rotb = RT * b;
	//Vector3d bnew = rotb - DP * D * rotb;
	//Vector3d trans = -0.5 * DP * RT * b;

	////std::cout << "bnew: " << bnew << std::endl;
	////double cnew = (trans.transpose() * (D * trans)) + rotb.transpose() * trans + circularCone.q[0];
	//double cnew = (trans.transpose() * (D * trans)) + circularCone.q[0] + rotb.transpose() * trans;
	
	Eigen::Vector3d trans = -0.5 * DP * RT * b;
	Eigen::Vector3d bnew = (RT * b + 2.0 * D * trans);

	/*std::cout << "R: " << std::endl << R << std::endl;
	std::cout << "trans: " << trans << std::endl;
	std::cout << "bnew: " << bnew << std::endl;*/


	double cnew = (trans.transpose() * (D * trans)) +  circularCone.q[0]  + b.transpose() * R * trans;
	
	//std::cout << "cnew: " << cnew << std::endl;

	//double r2 = (-2.0 * cnew / (D(0, 0) + D(1, 1)));
	////std::cout << "radius: " << r << std::endl;
	//assert(r2 > 0);
	//this->radius = sqrt(r2);
	
	double cota = sqrt(-(D(0, 0) + D(1, 1)) / (2.0 * D(2, 2)));
	//std::cout << "tana: " << cota << std::endl;
	//angle = asin(std::max(std::min(sina, 1.0), -1.0));
	angle = atan(1.0 / cota); //positive
	//std::cout << "sin(angle) : " << sin(angle) << " cos(angle): " << cos(angle) << std::endl;
	radius = 0.0;

	Eigen::Vector3d tmpx(1, 0, 0), tmpy(0, 1, 0), tmpz(0, 0, 1), tmploc = R * trans;

	tmpx = R * tmpx;
	tmpy = R * tmpy;
	tmpz = R * tmpz;
	//double cnew = circularCone.q[0];

	for (size_t i = 0; i < 3; i++)
	{
		loc[i] = tmploc[i];
		xdir[i] = tmpx[i];
		ydir[i] = tmpy[i];
		zdir[i] = tmpz[i];
	}
	u_max = 2 * M_PI;

	//std::cout << "zdir: " << zdir << std::endl;
	//std::cout << "loc: " << tmploc << std::endl;

	//convert cylinder back

	//allquadrics::TriangleMesh inputMesh;
	//char* defaultInput = "cylinder.obj";
	//inputMesh.loadObj(defaultInput);
	//inputMesh.centerAndScale(1);
	//allquadrics::Quadric circularCone;
	//fitCircularCylinder(inputMesh, circularCone);
	//cout << "direct fit for circular cylinder: " << endl;
	//outputQuadric(circularCone);
	//cout << endl << endl;
}

void ConeFitter::sfpn_fitting()
{
	if (!flag_input_normal)
	{
		estimate_normal();
		//return;
	}
	
	double lambda_ls = 0.001;
	double theta_min = 0.001;
	if (!flag_set_axis)
	{
		PlaneFitter pf;
		pf.set_points(input_normals);
		pf.fitting();
		zdir = pf.zdir;
	}

	int num_points = input_pts.size();
	Eigen::MatrixXd A(num_points, 3);
	Eigen::VectorXd b(num_points, 1);
	for (size_t i = 0; i < num_points; i++)
	{
		double sum = 0.0;
		for (size_t j = 0; j < 3; j++)
		{
			A(i, j) = input_normals[i][j];
			sum += input_normals[i][j] * input_pts[i][j];
		}
		b(i) = sum;
	}

	//least square fitting
	Eigen::MatrixXd ATA = A.transpose() * A;
	Eigen::MatrixXd ATb = A.transpose() * b;
	Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(ATA);
	//std::cout << "The rank of ATA is " << lu_decomp.rank() << std::endl;
	if (lu_decomp.rank() != 3)
	{
		ATA += lambda_ls * Eigen::MatrixXd::Identity(3,3);
	}

	//get loc
	Eigen::VectorXd sol = ATA.ldlt().solve(ATb);
	for (size_t i = 0; i < 3; i++)
	{
		loc[i] = sol(i);
	}
	
	//fix zdir
	vec3d diff_all(0.0, 0.0, 0.0);
	vec3d tmp;
	for (size_t i = 0; i < num_points; i++)
	{
		tmp = input_pts[i] - loc;
		//tmp.Normalize();
		diff_all += tmp;
	}
	diff_all /= num_points;
	diff_all.Normalize();
	if (diff_all.Dot(zdir) < 0.0)
	{
		zdir = -zdir;
	}

	//get theta
	double sum = 0.0;
	for (size_t i = 0; i < num_points; i++)
	{
		tmp = input_pts[i] - loc;
		tmp.Normalize();
		double cosv = std::abs(tmp.Dot(zdir));
		cosv = std::min(cosv, 0.999999);
		sum += std::acos(cosv);		//always positive
	}
	angle = sum / num_points;
	//get xdir and ydir
	get_vertical_vectors(this->zdir, this->xdir, this->ydir);

}

void ConeFitter::fitting()
{
	if (flag_sfpn)
	{
		sfpn_fitting();
		flag_first_time = false;
		return;
	}
	

	if (flag_using_aqd)
	{

		aqd_fitting();
		flag_first_time = false;
		return;
	}

	assert(input_pts.size() != 0);
	std::vector<Vector3<double>> positions;
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		Vector3<double> data;
		data[0] = input_pts[i][0];
		data[1] = input_pts[i][1];
		data[2] = input_pts[i][2];
		positions.push_back(data);
	}
	Vector3<double> coneVertex;
	Vector3<double> coneAxis;
	if (flag_first_time) //parameter not passing to fitter, should set useConeInputAsInitialGuess as true
		CreateLMCone(positions, coneVertex, coneAxis, angle);
	else
	{
		//init cone vertex and cone axis
		for (size_t i = 0; i < 3; i++)
		{
			coneVertex[i] = loc[i];
			coneAxis[i] = zdir[i];
		}

		CreateLMCone(positions, coneVertex, coneAxis, angle); //axis and angle should be given simultaneously
		//CreateGNCone(positions, coneVertex, coneAxis, angle);

	}
	//CreateGNCone(positions, coneVertex, coneAxis, angle);

	for (size_t i = 0; i < 3; i++)
	{
		loc[i] = coneVertex[i];
		zdir[i] = coneAxis[i];
	}
	get_vertical_vectors(this->zdir, this->xdir, this->ydir);
}

void ConeFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv)
{
	//modified on 20210602, not used tmporaly
	/*tgt = src;
	return;*/

	tgt.clear();
	if (save_uv)
	{
		this->us.clear();
		this->vs.clear();
	}
	//std::vector<double> us_ref;
	for (size_t i = 0; i < src.size(); i++)
	{
		vec3d tmp = src[i] - this->loc;
		double v = tmp * zdir / cos(angle);
		vec3d rad_vec = tmp - (tmp * zdir) * zdir;
		rad_vec.Normalize();
		double cosvalue = rad_vec * xdir;
		cosvalue = std::min(std::max(-0.999999, cosvalue), 0.999999);
		double u = acos(cosvalue);
		//fit u accordig to v
		if (std::isnan(u))
		{
			std::cout << "cos value: " << rad_vec * xdir << std::endl;
			std::cout << "!!!!NAN found !!!!" << std::endl;
		}
		if (rad_vec * ydir < 0.0)
		{
			u = 2 * M_PI - u;
		}

		if (flag_single_side_proj && v < 0.0)
		{
			tgt.push_back(loc);
			continue;
		}

		if (v < 0.0)
		{
			u = u - M_PI; //opposite
			if (u < 0.0)
			{
				u = u + 2 * M_PI;
			}
		}
		
		if (save_uv)
		{
			us.push_back(u);
			vs.push_back(v);
			/*double u_ref = u;
			if (u_ref > M_PI)
				u_ref = u - 2 * M_PI;
			us_ref.push_back(u_ref);*/
		}
		//tgt.push_back(loc + (tmp * zdir) * zdir + (radius + v * sin(angle)) * rad_vec);
		tgt.push_back(loc + (tmp * zdir) * zdir + (radius + v * sin(angle)) * (xdir * cos(u) + ydir * sin(u)));
	}

	if (save_uv)
	{
		//update u_max, umin
		this->u_max = *std::max_element(this->us.begin(), this->us.end());
		this->u_min = *std::min_element(this->us.begin(), this->us.end());
		this->v_max = *std::max_element(this->vs.begin(), this->vs.end());
		this->v_min = *std::min_element(this->vs.begin(), this->vs.end());

		if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
		{
			/*double u_max_ref = *std::max_element(us_ref.begin(), us_ref.end());
			double u_min_ref = *std::min_element(us_ref.begin(), us_ref.end());
			if ((u_max_ref - u_min_ref) < 2 * M_PI - TH_CIRCLE)
			{
				u_max = u_max_ref;
				u_min = u_min_ref;
			}*/
			update_minmax(us, u_min, u_max);
			if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
			{
				u_closed = true;
			}
			/*else
			{
				u_closed = false;
			}*/
		}
		/*else
		{
			u_closed = false;
		}*/
	}
}

void ConeFitter::projection_with_normal(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, std::vector<vec3d>& tgt_normal, bool save_uv)
{
	//modified on 20210602, not used tmporaly
	/*tgt = src;
	return;*/

	tgt.clear();
	tgt_normal.clear();
	if (save_uv)
	{
		this->us.clear();
		this->vs.clear();
	}
	std::vector<double> us_ref;
	for (size_t i = 0; i < src.size(); i++)
	{
		vec3d tmp = src[i] - this->loc;
		double v = tmp * zdir / cos(angle);
		vec3d rad_vec = tmp - (tmp * zdir) * zdir;
		rad_vec.Normalize();
		double u = safe_acos(rad_vec * xdir);
		//fit u accordig to v
		if (rad_vec * ydir < 0.0)
		{
			u = 2 * M_PI - u;
		}

		if (v < 0.0)
		{
			u = u - M_PI;
			if (u < 0.0)
			{
				u = u + 2 * M_PI;
			}
		}

		tgt_normal.push_back(GetNormal(u, v));

		if (save_uv)
		{
			us.push_back(u);
			vs.push_back(v);
			double u_ref = u;
			if (u_ref > M_PI)
				u_ref = u - 2 * M_PI;
			us_ref.push_back(u_ref);
		}
		//tgt.push_back(loc + (tmp * zdir) * zdir + (radius + v * sin(angle)) * rad_vec);
		tgt.push_back(loc + (tmp * zdir) * zdir + (radius + v * sin(angle)) * (xdir * cos(u) + ydir * sin(u)));
	}

	if (save_uv)
	{
		//update u_max, umin
		this->u_max = *std::max_element(this->us.begin(), this->us.end());
		this->u_min = *std::min_element(this->us.begin(), this->us.end());
		this->v_max = *std::max_element(this->vs.begin(), this->vs.end());
		this->v_min = *std::min_element(this->vs.begin(), this->vs.end());

		if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
		{
			update_minmax(us, u_min, u_max);
			if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
			{
				u_closed = true;
			}
			/*else
			{
				u_closed = false;
			}*/

			if (u_closed)
			{
				u_max = 2 * M_PI;
				u_min = 0.0;
			}

			/*double u_max_ref = *std::max_element(us_ref.begin(), us_ref.end());
			double u_min_ref = *std::min_element(us_ref.begin(), us_ref.end());
			if ((u_max_ref - u_min_ref) < 2 * M_PI - TH_CIRCLE)
			{
				u_max = u_max_ref;
				u_min = u_min_ref;
			}*/
		}
	}
}



void SphereFitter::fitting()
{
	assert(input_pts.size() != 0);
	std::vector<Vector3<double>> positions;
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		Vector3<double> data;
		data[0] = input_pts[i][0];
		data[1] = input_pts[i][1];
		data[2] = input_pts[i][2];
		positions.push_back(data);
	}
	ApprSphere3<double> fitter;
	Sphere3<double> sphere;
	fitter.FitUsingSquaredLengths((int)positions.size(), positions.data(), sphere);
	for (size_t i = 0; i < 3; i++)
	{
		loc[i] = sphere.center[i];
	}
	radius = sphere.radius;
	
	//canonical frame
	
}


void SphereFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv)
{
	tgt.clear();
	if (save_uv)
	{
		this->us.clear();
		this->vs.clear();
	}
	//std::vector<double> us_ref, vs_ref;
	for (size_t i = 0; i < src.size(); i++)
	{
		vec3d tmp = src[i] - this->loc;
		tmp.Normalize();
		double v = safe_asin(tmp * zdir);
		vec3d rad_vec = tmp - (tmp * zdir) * zdir;
		rad_vec.Normalize();
		double u = safe_acos(rad_vec * xdir);
		//fit u accordig to v
		if (rad_vec * ydir < 0.0)
		{
			u = 2 * M_PI - u;
		}
		if (save_uv)
		{
			us.push_back(u);
			vs.push_back(v);
			double u_ref = u;
			/*if (u_ref > M_PI)
				u_ref = u - 2 * M_PI;
			us_ref.push_back(u_ref);

			double v_ref = v;
			if (v_ref > M_PI)
				v_ref = v - 2 * M_PI;
			vs_ref.push_back(v_ref);*/
		}
		tgt.push_back(loc + radius * tmp);
	}

	if (save_uv)
	{
		//update u_max, umin
		this->u_max = *std::max_element(this->us.begin(), this->us.end());
		this->u_min = *std::min_element(this->us.begin(), this->us.end());
		this->v_max = *std::max_element(this->vs.begin(), this->vs.end());
		this->v_min = *std::min_element(this->vs.begin(), this->vs.end());

		if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
		{
			/*double u_max_ref = *std::max_element(us_ref.begin(), us_ref.end());
			double u_min_ref = *std::min_element(us_ref.begin(), us_ref.end());
			if ((u_max_ref - u_min_ref) < 2 * M_PI - TH_CIRCLE)
			{
				u_max = u_max_ref;
				u_min = u_min_ref;
			}*/
			update_minmax(us, u_min, u_max);
			if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
			{
				u_closed = true;
			}
			/*else
			{
				u_closed = false;
			}*/

			if (u_closed)
			{
				u_max = 2 * M_PI;
				u_min = 0.0;
			}

		}
		/*else
		{
			u_closed = false;
		}*/

		if ((v_max - v_min) > 2 * M_PI - TH_CIRCLE)
		{
			/*double v_max_ref = *std::max_element(vs_ref.begin(), vs_ref.end());
			double v_min_ref = *std::min_element(vs_ref.begin(), vs_ref.end());
			if ((v_max_ref - v_min_ref) < 2 * M_PI - TH_CIRCLE)
			{
				v_max = v_max_ref;
				v_min = v_min_ref;
			}*/
			update_minmax(vs, v_min, v_max);
			if ((v_max - v_min) > 2 * M_PI - TH_CIRCLE)
			{
				v_closed = true;
			}
			else
			{
				v_closed = false;
			}
		}
		else
		{
			v_closed = false;
		}
	}
}


//from TorusFitting windows
//GN torus
void CreateGNTorus(std::vector<Vector3<double>> const& X,
	Vector3<double>& C, Vector3<double>& N, double& r0, double& r1)
{
	ApprTorus3<double> fitter;
	size_t const maxIterations = 128;
	double const updateLengthTolerance = 1e-04;
	double const errorDifferenceTolerance = 1e-08;
	bool useTorusInputAsInitialGuess = false;

	auto result = fitter(static_cast<int>(X.size()), X.data(),
		maxIterations, updateLengthTolerance, errorDifferenceTolerance,
		useTorusInputAsInitialGuess, C, N, r0, r1);
	(void)result;
}

void CreateLMTorus(std::vector<Vector3<double>> const& X,
	Vector3<double>& C, Vector3<double>& N, double& r0, double& r1)
{
	ApprTorus3<double> fitter;
	size_t const maxIterations = 128;
	double const updateLengthTolerance = 1e-04;
	double const errorDifferenceTolerance = 1e-08;
	double const lambdaFactor = 0.001;
	double const lambdaAdjust = 10.0;
	size_t const maxAdjustments = 8;
	bool useTorusInputAsInitialGuess = false;

	auto result = fitter(static_cast<int>(X.size()), X.data(),
		maxIterations, updateLengthTolerance, errorDifferenceTolerance,
		lambdaFactor, lambdaAdjust, maxAdjustments, useTorusInputAsInitialGuess,
		C, N, r0, r1);
	(void)result;
}


void TorusFitter::fitting()
{
	assert(input_pts.size() != 0);
	std::vector<Vector3<double>> positions;
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		Vector3<double> data;
		data[0] = input_pts[i][0];
		data[1] = input_pts[i][1];
		data[2] = input_pts[i][2];
		positions.push_back(data);
	}
	/*ApprSphere3<double> fitter;
	Sphere3<double> sphere;
	fitter.FitUsingSquaredLengths((int)positions.size(), positions.data(), sphere);
	for (size_t i = 0; i < 3; i++)
	{
		loc[i] = sphere.center[i];
	}*/

	//canonical frame
	Vector3<double> center, normal;
	CreateLMTorus(positions, center, normal, max_radius, min_radius);
	//CreateGNTorus(positions, center, normal, max_radius, min_radius);
	
	//check nan
	if (isnan(min_radius))
	{
		min_radius = 0.0;
	}

	for (size_t i = 0; i < 3; i++)
	{
		zdir[i] = normal[i];
		loc[i] = center[i];
	}
	get_vertical_vectors(this->zdir, this->xdir, this->ydir);

}

void TorusFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv)
{
	tgt.clear();
	if (save_uv)
	{
		this->us.clear();
		this->vs.clear();
	}

	//std::vector<double> us_ref, vs_ref;
	for (size_t i = 0; i < src.size(); i++)
	{
		vec3d tmp = src[i] - this->loc;
		double sinv = tmp * zdir / min_radius;
		if (sinv > 1.0)
			sinv = 1.0 - 1e-6;
		if (sinv < -1.0)
			sinv = -1.0 + 1e-6;
		double v = safe_asin(sinv);
		
		vec3d rad_vec = tmp - (tmp * zdir) * zdir;
		double cosv = (rad_vec.Length() - max_radius);
		if (cosv < 0.0)
		{
			v = M_PI - v;
		}
		if (v < 0.0)
		{
			v = 2 * M_PI + v;
		}

		rad_vec.Normalize();
		double u = safe_acos(rad_vec * xdir);
		//fit u accordig to v
		if (rad_vec * ydir < 0.0)
		{
			u = 2 * M_PI - u;
		}
		if (save_uv)
		{
			us.push_back(u);
			vs.push_back(v);

			/*double u_ref = u;
			if (u_ref > M_PI)
				u_ref = u - 2 * M_PI;
			us_ref.push_back(u_ref);

			double v_ref = v;
			if (v_ref > M_PI)
				v_ref = v - 2 * M_PI;
			vs_ref.push_back(v_ref);*/
		}
		tgt.push_back(GetPosition(u,v));
	}

	if (save_uv)
	{
		//update u_max, umin
		this->u_max = *std::max_element(this->us.begin(), this->us.end());
		this->u_min = *std::min_element(this->us.begin(), this->us.end());
		this->v_max = *std::max_element(this->vs.begin(), this->vs.end());
		this->v_min = *std::min_element(this->vs.begin(), this->vs.end());
		//std::cout << "v min " << v_min << " v max: " << v_max << std::endl;
		//std::cout << "u min " << u_min << " u max: " << u_max << std::endl;
		//std::cout << "min radius " << min_radius << " max radiu " << max_radius << std::endl;
		if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
		{
			/*double u_max_ref = *std::max_element(us_ref.begin(), us_ref.end());
			double u_min_ref = *std::min_element(us_ref.begin(), us_ref.end());
			if ((u_max_ref - u_min_ref) < 2 * M_PI - TH_CIRCLE)
			{
				u_max = u_max_ref;
				u_min = u_min_ref;
			}*/
			update_minmax(us, u_min, u_max);
			if ((u_max - u_min) > 2 * M_PI - TH_CIRCLE)
			{
				u_closed = true;
			}

			if (u_closed)
			{
				u_max = 2 * M_PI;
				u_min = 0.0;
			}
			
			/*else
			{
				u_closed = false;
			}*/
		}
		/*else
		{
			u_closed = false;
		}*/

		if ((v_max - v_min) > 2 * M_PI - TH_CIRCLE)
		{
			/*double v_max_ref = *std::max_element(vs_ref.begin(), vs_ref.end());
			double v_min_ref = *std::min_element(vs_ref.begin(), vs_ref.end());
			if ((v_max_ref - v_min_ref) < 2 * M_PI - TH_CIRCLE)
			{
				v_max = v_max_ref;
				v_min = v_min_ref;
			}*/
			update_minmax(vs, v_min, v_max);
			if ((v_max - v_min) > 2 * M_PI - TH_CIRCLE)
			{
				v_closed = true;
			}
			else
			{
				v_closed = false;
			}
		}
		else
		{
			v_closed = false;
		}
	}
}



vec3d SplineFitter::GetPosition(double u, double v)
{
	//if nurbs set, return nurbs value
	if (nurbs_surf)
	{
		gte::Vector<3, double> V = nurbs_surf->GetPosition(u, v);
		return vec3d(V[0], V[1], V[2]);
	}

	assert(spline != NULL);
	vec3d value;
	Vector3<double> jet;
	spline->Evaluate(u, v, 0, &jet);
	for (size_t i = 0; i < 3; i++)
	{
		value[i] = jet[i];
	}
	return value;
}

vec3d SplineFitter::GetNormal(double u, double v)
{
	//if nurbs set, return nurbs value
	if (nurbs_surf)
	{
		gte::Vector<3, double> V = nurbs_surf->GetVTangent(u, v);
		vec3d vtangent(V[0], V[1], V[2]);
		gte::Vector<3, double> U = nurbs_surf->GetUTangent(u, v);
		vec3d utangent(U[0], U[1], U[2]);
		vec3d normal = utangent.Cross(vtangent);
		normal.Normalize();
		return normal;
	}

	assert(spline != NULL);
	gte::Vector<3, double> V = spline->GetVTangent(u, v);
	vec3d vtangent(V[0], V[1], V[2]);
	gte::Vector<3, double> U = spline->GetUTangent(u, v);
	vec3d utangent(U[0], U[1], U[2]);
	vec3d normal = utangent.Cross(vtangent);
	normal.Normalize();
	return normal;
}

void SplineFitter::fitting()
{
	assert(input_pts.size() != 0);
	std::vector<Vector3<double>> positions;
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		Vector3<double> data;
		data[0] = input_pts[i][0];
		data[1] = input_pts[i][1];
		data[2] = input_pts[i][2];
		positions.push_back(data);
	}


	BSplineSurfaceFit<double> fitter(degree, numControls, numSamples[0], degree,
		numControls, numSamples[1], positions.data());

	//set spline
	BasisFunctionInput<double> basis_input[2];
	//set basis
	for (int i = 0; i < 2; i++)
	{
		basis_input[i].degree = fitter.GetBasis(i).GetDegree();
		basis_input[i].numControls = fitter.GetBasis(i).GetNumControls();
		basis_input[i].periodic = fitter.GetBasis(i).IsPeriodic();
		basis_input[i].uniform = fitter.GetBasis(i).IsOpen();
		basis_input[i].numUniqueKnots = fitter.GetBasis(i).GetNumUniqueKnots();
		basis_input[i].uniqueKnots.resize(basis_input[i].numUniqueKnots);
		auto nots_begin = fitter.GetBasis(i).GetUniqueKnots();
		std::copy(nots_begin, nots_begin + basis_input[i].numUniqueKnots, basis_input[i].uniqueKnots.begin());
	}

	auto control_pts = fitter.GetControlData();

	if (spline)
		delete spline;

	spline = new BSplineSurface<3, double>(basis_input, control_pts);

	//bspline fitter
	spline_pts.clear();
	double split = 1.0 / ((double)numControls - 1.0);
	for (size_t i = 0; i < numControls; i++)
	{
		for (size_t j = 0; j < numControls; j++)
		{
			spline_pts.push_back(GetPosition(i * split, j * split));
		}
	}
}

void SplineFitter::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv)
{
	//value assign
	//tgt = src;
	//return;
	//nearest neighbors of current points
	//not udpating uv

	//if nurbs set, get spline_pts firstly
	if (nurbs_surf && spline_pts.empty())
	{
		double split = 1.0 / ((double)numControls - 1.0);
		for (size_t i = 0; i < numControls; i++)
		{
			for (size_t j = 0; j < numControls; j++)
			{
				spline_pts.push_back(GetPosition(i * split, j * split));
			}
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

vec3d ThinPlateSpline::GetPosition(double u, double v)
{
	//assert(spline != NULL);
	assert(vec_thinplate.size() == 3);
	vec3d value;
	//Vector3<double> jet;
	//spline->Evaluate(u, v, 0, &jet);
	for (size_t i = 0; i < 3; i++)
	{
		//value[i] = jet[i];
		value[i] = (*vec_thinplate[i])(u, v);
	}
	return value;
}

void ThinPlateSpline::fitting()
{
	int const numPoints = 100;
	assert(input_pts.size() == numPoints);
	//std::vector<Vector3<double>> positions;
	std::vector<std::array<double, numPoints>> fs(3);
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		//Vector3<double> data;
		fs[0][i] = input_pts[i][0];
		fs[1][i] = input_pts[i][1];
		fs[2][i] = input_pts[i][2];
	}

	//int const numPoints = 9;
	std::array<double, numPoints> x;
	std::array<double, numPoints> y;
	//std::array<double, numPoints> f = { 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0 };

	int dim = (int)std::sqrt(numPoints);
	double invp = 1.0 / (dim - 1);
	for (size_t i = 0; i < dim; i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			x[i * dim + j] = i * invp;
			y[i * dim + j] = j * invp;
		}
	}

	for (size_t i = 0; i < 3; i++)
	{
		if (vec_thinplate[i])
		{
			delete vec_thinplate[i];
		}
		vec_thinplate[i] = new IntpThinPlateSpline2<double>(numPoints, x.data(), y.data(), fs[i].data(), smoothness, false);
		//not transform to unit sphere
	}

	//// Resample on a 7x7 regular grid.
	//int const numResample = 6;
	//double const invResample = 1.0 / static_cast<double>(numResample);
	//double smooth, interp, functional;

	//// No smoothing, exact interpolation at grid points.
	//smooth = 0.0;
	//IntpThinPlateSpline2<double> noSmooth(
	//	numPoints, x.data(), y.data(), f.data(), smooth, false);
	////output << "no smoothing (smooth parameter is 0.0)" << std::endl;
	//for (int j = 0; j <= numResample; ++j)
	//{
	//	for (int i = 0; i <= numResample; ++i)
	//	{
	//		interp = noSmooth(invResample * i, invResample * j);
	//		//output << interp << " ";
	//	}
	//	//output << std::endl;
	//}
	//functional = noSmooth.ComputeFunctional();
	////output << "functional = " << functional << std::endl << std::endl;

	//// Increasing amounts of smoothing.
	//smooth = 0.1;
	//for (int k = 1; k <= 6; ++k, smooth *= 10.0)
	//{
	//	IntpThinPlateSpline2<double> spline(
	//		numPoints, x.data(), y.data(), f.data(), smooth, false);
	//	//output << "smoothing (parameter is " << smooth << ")" << std::endl;
	//	for (int j = 0; j <= numResample; ++j)
	//	{
	//		for (int i = 0; i <= numResample; ++i)
	//		{
	//			interp = spline(invResample * i, invResample * j);
	//			interp = noSmooth(invResample * i, invResample * j);
	//			//output << interp << " ";
	//		}
	//		//output << std::endl;
	//	}
	//	functional = noSmooth.ComputeFunctional();
	//	//output << "functional = " << functional << std::endl << std::endl;
	//}

	spline_pts.clear();
	double split = 1.0 / ((double)dim - 1.0);
	for (size_t i = 0; i < dim; i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			spline_pts.push_back(GetPosition(i * split, j * split));
		}
	}
}

void ThinPlateSpline::projection(const std::vector<vec3d>& src, std::vector<vec3d>& tgt, bool save_uv)
{
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

//spline part
SplineFitter::SplineFitter(int u_degree, int v_degree,
	const std::vector<gte::Vector<3, double>>& controls,
	const std::vector<double>& my_uknots, const std::vector<double>& my_vknots,
	const std::vector<double>& myweights,
	bool _u_closed, bool _v_closed,
	double _u_min, double _u_max, double _v_min, double _v_max)
{
	this->u_closed = _u_closed, this->v_closed = _v_closed;
	// this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
	// this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;
	this->u_min = _u_min, this->u_max = _u_max;
	this->v_min = _v_min, this->v_max = _v_max;

	gte::BasisFunctionInput<double> my_u_input;
	my_u_input.degree = u_degree;
	my_u_input.numControls = (int)my_uknots.size() - u_degree - 1;
	//my_u_input.periodic = _u_closed;
	my_u_input.periodic = false;
	my_u_input.uniform = false;
	std::vector<std::pair<double, int>> knots_stataus;
	knots_stataus.push_back(std::make_pair(my_uknots[0], 1));
	for (size_t i = 1; i < my_uknots.size(); i++)
	{
		if (my_uknots[i] == knots_stataus.back().first)
			knots_stataus.back().second++;
		else
			knots_stataus.push_back(std::make_pair(my_uknots[i], 1));
	}

	my_u_input.numUniqueKnots = (int)knots_stataus.size();
	my_u_input.uniqueKnots.resize(my_u_input.numUniqueKnots);
	for (size_t i = 0; i < knots_stataus.size(); i++)
	{
		my_u_input.uniqueKnots[i].t = knots_stataus[i].first;
		my_u_input.uniqueKnots[i].multiplicity = knots_stataus[i].second;
	}

	gte::BasisFunctionInput<double> my_v_input;
	my_v_input.degree = v_degree;
	my_v_input.numControls = (int)my_vknots.size() - v_degree - 1;
	//my_v_input.periodic = _v_closed;
	my_v_input.periodic = false;
	my_v_input.uniform = false;

	knots_stataus.resize(0);
	knots_stataus.push_back(std::make_pair(my_vknots[0], 1));
	for (size_t i = 1; i < my_vknots.size(); i++)
	{
		if (my_vknots[i] == knots_stataus.back().first)
			knots_stataus.back().second++;
		else
			knots_stataus.push_back(std::make_pair(my_vknots[i], 1));
	}

	my_v_input.numUniqueKnots = (int)knots_stataus.size();
	my_v_input.uniqueKnots.resize(my_v_input.numUniqueKnots);
	for (size_t i = 0; i < knots_stataus.size(); i++)
	{
		my_v_input.uniqueKnots[i].t = knots_stataus[i].first;
		my_v_input.uniqueKnots[i].multiplicity = knots_stataus[i].second;
	}
	nurbs_surf = new gte::NURBSSurface<3, double>(my_u_input, my_v_input, controls.data(), myweights.data());
	numControls = my_u_input.numControls;

	/*this->u_closed = _u_closed, this->v_closed = _v_closed;
	this->u_min = _u_closed ? 0 : _u_min, this->u_max = _u_closed ? 2 * M_PI : _u_max;
	this->v_min = _v_closed ? 0 : _v_min, this->v_max = _v_closed ? 2 * M_PI : _v_max;

	gte::BasisFunctionInput<double> my_u_input;
	my_u_input.degree = u_degree;
	my_u_input.numControls = (int)my_uknots.size() - u_degree - 1;
	my_u_input.periodic = _u_closed;
	my_u_input.uniform = false;
	std::vector<std::pair<double, int>> knots_stataus;
	knots_stataus.push_back(std::make_pair(my_uknots[0], 1));
	for (size_t i = 1; i < my_uknots.size(); i++)
	{
		if (my_uknots[i] == knots_stataus.back().first)
			knots_stataus.back().second++;
		else
			knots_stataus.push_back(std::make_pair(my_uknots[i], 1));
	}

	my_u_input.numUniqueKnots = (int)knots_stataus.size();
	my_u_input.uniqueKnots.resize(my_u_input.numUniqueKnots);
	for (size_t i = 0; i < knots_stataus.size(); i++)
	{
		my_u_input.uniqueKnots[i].t = knots_stataus[i].first;
		my_u_input.uniqueKnots[i].multiplicity = knots_stataus[i].second;
	}

	gte::BasisFunctionInput<double> my_v_input;
	my_v_input.degree = v_degree;
	my_v_input.numControls = (int)my_vknots.size() - v_degree - 1;
	my_v_input.periodic = _v_closed;
	my_v_input.uniform = false;

	knots_stataus.resize(0);
	knots_stataus.push_back(std::make_pair(my_vknots[0], 1));
	for (size_t i = 1; i < my_vknots.size(); i++)
	{
		if (my_vknots[i] == knots_stataus.back().first)
			knots_stataus.back().second++;
		else
			knots_stataus.push_back(std::make_pair(my_vknots[i], 1));
	}

	my_v_input.numUniqueKnots = (int)knots_stataus.size();
	my_v_input.uniqueKnots.resize(my_v_input.numUniqueKnots);
	for (size_t i = 0; i < knots_stataus.size(); i++)
	{
		my_v_input.uniqueKnots[i].t = knots_stataus[i].first;
		my_v_input.uniqueKnots[i].multiplicity = knots_stataus[i].second;
	}
	nurbs_surf = new gte::NURBSSurface<3, double>(my_u_input, my_v_input, controls.data(), myweights.data());
	numControls = my_u_input.numControls;*/
}
