#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <set>
#include <queue>
#include "cxxopts.hpp"
//#include "yaml.h"
#include "MyObjLoader.h"
#include "MyCurve.h"
#include "MySurf.h"
#include "happly.h"
#include "Helper.h"
#include "SurfFitter.h"
#include "CurveFitter.h"
#include "MyPointCloud.h"
#include "Mesh3D.h"
#include "NURBSFittingWrapper.h"
#include "yaml.h"
#include "json.hpp"

#include <cmath>

#define COR_VAL_TH 0.5
#define PATCH_DIST_TH 0.015
#define EPS_SEG 1e-6
#define MIN_TRI_AREA 1e-6

using namespace std;

typedef TinyVector<double, 3> vec3d;

std::string GetFileExtension(const std::string &FileName)
{
    if (FileName.find_last_of(".") != std::string::npos)
        return FileName.substr(FileName.find_last_of(".") + 1);
    return "";
}

void save_all_curves(const std::vector<CurveFitter*>& curve_fitters, const std::vector<string>& curve_type, int split,  const char* fn)
{
	size_t counter = 1;
	std::ofstream file(fn);
	for (size_t i = 0; i < curve_type.size(); i++)
	{
		curve_fitters[i]->write_obj_curve(file, counter, split);
	}
	file.close();
}

void save_all_curves_proj_ply(const std::vector<CurveFitter*>& curve_fitters, const std::vector<string>& curve_type, const char* fn)
{
	std::vector<vec3d> pts;
	for (size_t i = 0; i < curve_fitters.size(); i++)
	{
		pts.insert(pts.end(), curve_fitters[i]->projection_pts.begin(), curve_fitters[i]->projection_pts.end());
	}
	
	save_pts_ply(fn, pts);

}

void save_all_curves_seg_ply(const std::vector<CurveFitter*>& curve_fitters, const std::vector<string>& curve_type, const char* fn)
{
	std::vector<vec3d> pts;
	for (size_t i = 0; i < curve_fitters.size(); i++)
	{
		curve_fitters[i]->get_seg_pts();
		pts.insert(pts.end(), curve_fitters[i]->seg_pts.begin(), curve_fitters[i]->seg_pts.end());
	}

	save_pts_ply(fn, pts);

}

void save_all_input_patch_normals(const char* fn, const std::vector<std::vector<vec3d>> &pts_all, const std::vector<std::vector<vec3d>> &normals_all)
{
	std::vector<vec3d> pts, normals;
	for (size_t i = 0; i < pts_all.size(); i++)
	{
		pts.insert(pts.end(), pts_all[i].begin(), pts_all[i].end());
		normals.insert(normals.end(), normals_all[i].begin(), normals_all[i].end());
	}

	save_pts_normal_xyz(fn, pts, normals);
}

void save_all_patches(const std::vector<SurfFitter*>& surf_fitters, const std::vector<string>& surf_type, int usplit, int vsplit, const char* fn)
{
	size_t counter = 1;
	
	std::ofstream file(fn);
	for (size_t i = 0; i < surf_type.size(); i++)
	{
		surf_fitters[i]->write_obj_surf(file, counter, usplit, vsplit);
	}
	file.close();
}

void gather_all_patches_pts_faces(const std::vector<SurfFitter*>& surf_fitters, std::vector<vec3d>& pts, std::vector<std::vector<size_t>>& faces, std::vector<size_t>& face2patch)
{
	pts.clear();
	faces.clear();
	face2patch.clear();
	size_t counter = 0;
	for (size_t i = 0; i < surf_fitters.size(); i++)
	{
		surf_fitters[i]->get_grid_pts();
		pts.insert(pts.end(), surf_fitters[i]->grid_pts.begin(), surf_fitters[i]->grid_pts.end());
		int u_split = surf_fitters[i]->u_split, v_split = surf_fitters[i]->v_split;
		for (size_t j = 0; j < u_split - 1; j++)
		{
			for (size_t k = 0; k < v_split - 1; k++)
			{
				faces.push_back(std::vector<size_t>({ counter + j * v_split + k, counter + (j + 1) * v_split + k, counter + (j + 1) * v_split + k + 1, counter + j * v_split + k + 1 }));
				face2patch.push_back(i);
			}
		}
		
		if (surf_fitters[i]->u_closed)
		{
			for (size_t j = 0; j < v_split - 1; j++)
			{
				faces.push_back(std::vector<size_t>({ counter + (u_split - 1) * v_split + j, counter + j, counter + j + 1, counter + (u_split - 1) * v_split + j + 1 }));
				face2patch.push_back(i);
			}
		}

		if (surf_fitters[i]->v_closed)
		{
			for (size_t j = 0; j < u_split - 1; j++)
			{
				faces.push_back(std::vector<size_t>({ counter + j * v_split + v_split - 1, counter + (j + 1) * v_split + v_split - 1, counter + (j + 1) * v_split, counter + j * v_split}));
				face2patch.push_back(i);
			}
		}

		if (surf_fitters[i]->u_closed && surf_fitters[i]->v_closed)
		{
			faces.push_back(std::vector<size_t>({ counter, counter + (u_split - 1) * v_split, counter + u_split * v_split - 1, counter + v_split - 1 }));
		}


		counter += surf_fitters[i]->grid_pts.size();
	}
}

bool get_cylindercone_axis_by_neighbors(const std::string& cur_patch_type, const std::vector<int>&nn_curve, const std::vector<CurveFitter*> &curve_fitters, const std::vector<string>& curve_type, const std::vector<std::vector<vec3d>> &curves, double th_same_dir_cos, vec3d& tmp_axis)
{
	bool axis_valid = false;
	//axis stored in tmp_axis
	if (cur_patch_type == "Cylinder")
	{
		if (nn_curve.size() == 2)
		{
			if (curve_type[nn_curve[0]] == "Circle" && curve_type[nn_curve[0]] == "Circle")
			{
				vec3d dir0 = curve_fitters[nn_curve[0]]->dirz, dir1 = curve_fitters[nn_curve[1]]->dirz;
				if (dir0.Dot(dir1) < 0.0)
				{
					dir1 = -dir1;
				}
				if (dir0.Dot(dir1) > th_same_dir_cos)
				{
					tmp_axis = (dir0 + dir1) / 2.0;
					tmp_axis.Normalize();
					axis_valid = true;
				}
			}
		}
		else if (nn_curve.size() == 1)
		{
			//udpate 1209, even with only one circle neighbor, set it as true
			if (curve_type[nn_curve[0]] == "Circle")
			{
				tmp_axis = curve_fitters[nn_curve[0]]->dirz;
				axis_valid = true;
			}

		}
		else
		{
			//line-line case
			std::vector<vec3d> line_dirs;
			for (auto c : nn_curve)
			{
				if (curve_type[c] == "Line")
				{
					vec3d tmp = curves[c].back() - curves[c][0];
					tmp.Normalize();
					line_dirs.push_back(tmp);
				}
			}
			if (line_dirs.size() == 2)
			{
				if (line_dirs[0].Dot(line_dirs[1]) < 0.0)
				{
					line_dirs[0] = -line_dirs[0];
				}

				if (line_dirs[0].Dot(line_dirs[1]) > th_same_dir_cos)
				{
					tmp_axis = (line_dirs[0] + line_dirs[1]) / 2.0;
					tmp_axis.Normalize();
					axis_valid = true;
				}
			}
		}
	}

	if (cur_patch_type == "Cone")
	{
		if (nn_curve.size() == 2)
		{
			if (curve_type[nn_curve[0]] == "Circle" && curve_type[nn_curve[0]] == "Circle")
			{
				vec3d dir0 = curve_fitters[nn_curve[0]]->dirz, dir1 = curve_fitters[nn_curve[1]]->dirz;
				if (dir0.Dot(dir1) < 0.0)
				{
					dir1 = -dir1;
				}
				if (dir0.Dot(dir1) > th_same_dir_cos)
				{
					tmp_axis = (dir0 + dir1) / 2.0;
					tmp_axis.Normalize();
					axis_valid = true;
				}
			}
		}
		else if (nn_curve.size() == 1)
		{
			if (curve_type[nn_curve[0]] == "Circle")
			{
				tmp_axis = curve_fitters[nn_curve[0]]->dirz;
				axis_valid = true;
			}

		}
	}


	return axis_valid;
}

using json = nlohmann::json;

int main(int argc, char** argv)
{
	bool flag_direct_fit_cylinder_cone = false; // not using nurbs, direct fit cylinder and cone
	bool flag_input_pc_normal = true;
	bool save_curve_corner = false;
	int u_split = 20, v_split = 20;
	int t_split = 34;
	bool flag_patch_self_proj = true;
	bool flag_curve_self_proj = false;
	bool flag_debug = false;
	//flag_debug = true; //set as true
	bool flag_debug_curve = false;
	bool flag_debug_patch = false;
	double dist_expand = 0.1;
	double belt_offset = 0.03;
	double min_dist_to_boundary = 0.0001;
	double update_dist_to_boundary = 0.01;
	double cylinder_radius_offset = 0.005;

	double max_patch_nn_dist_ratio = 0.3; //only consider pc and nb corners and curves within radio * bb_length
	double max_patch_grid_diff_ratio = 2.0;
	double smoothness = 1.0;  //double smoothness = 0.1;
	//double smoothness = 100.0;  //double smoothness = 0.1;

	/*double max_cylinder_cone_error = 0.05;
	max_cylinder_cone_error = 0.02;*/
	bool flag_seg_pc = true;
	int dim_u_input = 20, dim_v_input = 20, dim_u_output = 20, dim_v_output = 20;
	bool flag_abandon_oob_spline = true; //out of bound spline
	double range_obb = 1.0;
	//here for testing
	//dim_u_output = 40, dim_v_output = 40;

	int dim_u_ctrl = 20, dim_v_ctrl = 20;
	dim_u_ctrl = 5, dim_v_ctrl = 5;

	//for patch-curve, patch-corner relation
	double max_patch_pc_dist = 0.05; //only consider input pts close to patches
	max_patch_pc_dist = 0.02; //only consider input pts close to patches

	double max_patch_curve_dist = 0.1, max_patch_corner_dist = 0.1, max_curve_corner_dist = 0.1; //shortest distance
	bool flag_fit_patch_with_curve = true, flag_fit_patch_with_corner = true;

	bool flag_only_use_near_nbs = true;
	bool flag_expand_para_range_last = true;

	flag_fit_patch_with_curve = true;
	flag_fit_patch_with_corner = true;
	bool flag_using_aqd = true;
	//bool flag_using_aqd = false;
	bool flag_set_axis_by_neighbor = true;
	double th_same_dir_cos = std::cos(M_PI / 6.0);

	try
	{
		cxxopts::Options options("ParametricFitting", "parametric fitting of point cloud (author: Haoxiang Guo, Email: guohaoxiangxiang@gmail.com)");
		options
			.positional_help("[optional args]")
			.show_positional_help()
			.allow_unrecognised_options()
			.add_options()
			("i,input", "input complex (complex format)", cxxopts::value<std::string>())
			("p,pc", "point cloud (ply format)", cxxopts::value<std::string>())
			//("o,output_prefix", "output prefix (ptangle format)", cxxopts::value<std::string>())
			("e,extend", "extending factor for uv grid", cxxopts::value<double>()) //default: 2.5
			("d,dist", "distance for extend uv-grid", cxxopts::value<double>()) //default: 0.06
			("k,keepratio", "keep ratio for cutting", cxxopts::value<double>()) //default: 0.3
			("curveonpatch", "curve weight on patch", cxxopts::value<int>()) //default: 50
			("pointonpatch", "point weight on patch", cxxopts::value<int>()) //default: 100
			("corneronpatch", "point weight on patch", cxxopts::value<int>()) //default: 50

			//("c,curvecorner", "only save ")
			("s,skip", "skip iterative step")
			("debug", "debug patch and curve")
			("outerone", "fit outer only once")
			("outerzero", "not fit outer")
			("debugfinal", "debug final output")
			("nosegpc", "not segment point cloud")
			("cc", "direct fit cylinder and cone")
			//("c", "cut final mesh")
			("a, axisxyz", "set axis as x/y/z if it is close to normal setting")
			("max_angle_diff_degree", "when rounding, max angle difference in degree", cxxopts::value<double>()) //default: 10
			("set_curve_axis", "set curve axis according to patch axis in final round")
			//("s,segmentation", "strictly segment point cloud")
			("patchpcdist", "patch pc dist", cxxopts::value<double>()) //default: 0.05
			("last_iter", "last iter", cxxopts::value<int>()) //default: 1
			("fit_other_last", "fit other patches last")
			("use_all_nb", "when fitting, use all neighors (not include pc)")
			("nb_th", "when not using all neighors, here to set nb threshold, default: 0.1", cxxopts::value<double>())
			("patch_bb_ratio", "when not using all neighors, here to set patch_bb_ratio, default: 0.3", cxxopts::value<double>())
			//("mp", "maximum number of patches in each colored cluster, only work for csg, default -1(no upper bound)", cxxopts::value<int>())
			("h,help", "print help");

		auto result = options.parse(argc, argv);
		if (result.count("help"))
		{
			std::cout << options.help({ "", "Group" }) << std::endl;
			exit(0);
		}
		if (result.count("cc"))
			flag_direct_fit_cylinder_cone = true;
		
		if (result.count("debug"))
		{
			flag_debug_patch = true;
			flag_debug_curve = true;
		}

		if (result.count("debugfinal"))
		{
			flag_debug = true;
		}

		if (result.count("use_all_nb"))
		{
			flag_only_use_near_nbs = false;
		}

		if (result.count("nb_th"))
		{
			max_patch_curve_dist = result["nb_th"].as<double>();
			max_patch_corner_dist = result["nb_th"].as<double>();
			max_curve_corner_dist = result["nb_th"].as<double>();
		}

		if (result.count("patch_bb_ratio"))
		{
			max_patch_nn_dist_ratio = result["patch_bb_ratio"].as<double>();
		}

		//bool flag_set_cylinder_axis_by_nc = true; //set cylinder axis by neighboring curves.
		bool flag_enlarge_tangent_cylinder_radius = false; //updata 1210, no longer used
		bool flag_skip_control_fitting = false;
		double min_cut_keep_ratio = 0.01; //if area of kept faces/area of total faces is less than this value, then switch network input 
		bool flag_fit_other_last = false;
		int last_fit_iter = 1;
		
		if (result.count("fit_other_last"))
		{
			flag_fit_other_last = true;
		}

		if (result.count("last_iter"))
		{
			last_fit_iter = result["last_iter"].as<int>();
		}
		
		if (result.count("nosegpc"))
		{
			flag_seg_pc = false;
		}
		

		if (result.count("s"))
		{
			flag_skip_control_fitting = true;
		}
		
		if (result.count("k"))
		{
			min_cut_keep_ratio = result["k"].as<double>();
		}

		bool flag_only_save_curve_corner = false;
		if (result.count("c"))
		{
			flag_only_save_curve_corner = true;
		}

		bool flag_axisxyz = false, flag_set_curve_axis = false;
		double max_angle_diff_degree = 10.0;

		if (result.count("a"))
		{
			flag_axisxyz = true;
		}

		if (result.count("set_curve_axis"))
		{
			flag_set_curve_axis = true;
		}

		if (result.count("max_angle_diff_degree"))
		{
			max_angle_diff_degree = result["max_angle_diff_degree"].as<double>();
		}
		
		double enlarge_factor = 2.5; //only used for direct enlarging uv
		if (result.count("extend"))
		{
			enlarge_factor = result["e"].as<double>();
		}

		if (result.count("patchpcdist"))
		{
			max_patch_pc_dist = result["patchpcdist"].as<double>();
		}

		int flag_split = 10, period_split = 30;
		
		if (flag_debug)
			period_split = 10;

		int w_corneroncurve = 5, w_corneronpatch = 5, w_curveoncorner = 1, w_curveonpatch = 5, w_patchoncorner = 1, w_patchoncurve = 1;
		//int w_pc = 1;
		//modify 0116
		//w_corneroncurve = 10.0;
		
		int w_pc_corner = 0, w_pc_curve = 0, w_pc_patch = 10;

		

		int w_self_curve = 1, w_self_patch = 1, w_self_corner = 5;
		double w_expand = 0.1, dist_expand = 0.06;

		if (result.count("d"))
		{
			dist_expand = result["d"].as<double>();
		}
		
		w_pc_patch = 100, w_curveonpatch = 50, w_corneronpatch = 50;


		if (result.count("curveonpatch"))
		{
			w_curveonpatch = result["curveonpatch"].as<int>();
		}

		if (result.count("corneronpatch"))
		{
			w_corneronpatch = result["corneronpatch"].as<int>();
		}

		if (result.count("pointonpatch"))
		{
			w_pc_patch = result["pointonpatch"].as<int>();
		}

		auto& inputfile = result["i"].as<std::string>();
		auto& inputprefix = inputfile.substr(0, inputfile.find_last_of("."));

		std::vector<vec3d> corners;
		std::vector<std::vector<vec3d>> curves, patches, patch_normals;
		std::vector<string> curve_type, patch_type;
		std::vector<bool> patch_close;
		Eigen::VectorXd is_curved_closed;
		Eigen::MatrixXd curve_corner_corres, patch_curve_corres, patch_corner_corres;

		//complex_loader(inputfile.c_str(), corners, curves, patches, curve_type, patch_type, is_curved_closed, curve_corner_corres, patch_curve_corres, patch_normals, patch_close);
		complex_loader(inputfile.c_str(), corners, curves, patches, curve_type, patch_type, is_curved_closed, curve_corner_corres, patch_curve_corres, patch_corner_corres, patch_normals, patch_close);
		//patch_corner_corres = patch_curve_corres * curve_corner_corres / 2.0;
		std::vector<bool> flag_curve_close(curves.size(), false);
		for (size_t i = 0; i < curves.size(); i++)
		{
			if (is_curved_closed[i] > 0.5)
				flag_curve_close[i] = true;
		}
		std::vector<CurveFitter*> curve_fitters;
		std::vector<SurfFitter*> surf_fitters;	
		std::vector<std::vector<int>> corner2curve(corners.size()), corner2patch(corners.size()), curve2corner(curve_type.size()), curve2patch(curve_type.size()), patch2curve(patch_type.size()), patch2corner(patch_type.size());
		std::vector<std::set<int>> patch2patch(patches.size());
		std::vector<bool> patch_flip_uv(patches.size(), false); //for cylinder and cone, flip uv so that v is the line direction

		std::vector<std::vector<vec3d>> patches_ori(patches.size());
		std::vector<string> patch_type_ori = patch_type;

		for (size_t i = 0; i < patches.size(); i++)
		{
			patches_ori[i] = patches[i];
		}


		bool flag_input_normal = false;
		if (!patch_normals.empty())
		{
			flag_input_normal = true;
		}
		else
		{
			patch_normals.resize(patches.size());
		}

		for (size_t i = 0; i < curves.size(); i++)
		{
			for (size_t j = 0; j < corners.size(); j++)
			{
				if (curve_corner_corres(i, j) > COR_VAL_TH)
				{
					curve2corner[i].push_back(j);
					corner2curve[j].push_back(i);
				}
			}
		}

		for (size_t i = 0; i < patches.size(); i++)
		{
			for (size_t j = 0; j < curves.size(); j++)
			{
				if (patch_curve_corres(i, j) > COR_VAL_TH)
				{
					patch2curve[i].push_back(j);
					curve2patch[j].push_back(i);
				}
			}
		}

		//patch2patch
		for (int i = 0; i < patches.size(); i++)
		{
			for (auto eid : patch2curve[i])
			{
				for (auto fid : curve2patch[eid])
				{
					if (fid != i)
					{
						patch2patch[i].insert(fid);
					}
				}
			}
		}

		for (size_t i = 0; i < patches.size(); i++)
		{
			for (size_t j = 0; j < corners.size(); j++)
			{
				if (patch_corner_corres(i,j) > COR_VAL_TH)
				{
					patch2corner[i].push_back(j);
					corner2patch[j].push_back(i);
				}
			}
		}

		//json output
		
		json oj;
		std::vector<json> patchj;
		if (patches.front().size() == 100)
		{
			dim_u_input = 10, dim_v_input = 10;
		}

		for (size_t i = 0; i < patches.size(); i++)
		{
			json j;
			j["type"] = patch_type_ori[i];
			j["u_dim"] = dim_u_input;
			j["v_dim"] = dim_v_input;
			j["u_closed"] = (bool)patch_close[i];
			j["v_closed"] = false;
			j["with_param"] = false;
			std::vector<double> onepatch_pts;
			
			for (size_t j = 0; j < patches[i].size(); j++)
			{
				for (size_t k = 0; k < 3; k++)
				{
					onepatch_pts.push_back(patches[i][j][k]);
				}
			}
			j["grid"] = onepatch_pts;
			patchj.push_back(j);
		}

		oj["patches"] = patchj;
		if (save_curve_corner)
		{
			//std::vector<YAML::Node> curve_nodes;
			json curvej;
			for (size_t i = 0; i < curves.size(); i++)
			{
				json onenode;
				onenode["type"] = curve_type[i];
				//YAML::Node onecurve_pts = YAML::Load("[]");
				std::vector<double> onecurve_pts;
				for (size_t j = 0; j < curves[i].size(); j++)
				{
					for (size_t k = 0; k < 3; k++)
					{
						onecurve_pts.push_back(curves[i][j][k]);
					}
				}
				//onenode["closed"] = (bool)curve_fitters[i]->closed;
				onenode["closed"] = (bool)is_curved_closed[i]; //change to curve fitter later
				onenode["pts"] = onecurve_pts;
				curvej.push_back(onenode);
			}
			oj["curves"] = curvej;
			json cornerj;
			for (size_t i = 0; i < corners.size(); i++)
			{
				json onenode;
				//YAML::Node onecorner_pts = YAML::Load("[]");
				std::vector<double> onecorner_pts;
				for (size_t k = 0; k < 3; k++)
				{
					onecorner_pts.push_back(corners[i][k]);
				}
				onenode["pts"] = onecorner_pts;
				cornerj.push_back(onenode);
			}
			oj["corners"] = cornerj;
		}

		json patch2curvej;
		json curve2cornerj;
		json patch2cornerj;
		for (size_t i = 0; i < patches.size(); i++)
		{
			std::vector<int> onecurve(curves.size(), 0);
			for (size_t j = 0; j < patch2curve[i].size(); j++)
			{
				//patch2curve_node.push_back(patch2curve[i][j]);
				onecurve[patch2curve[i][j]] = 1;
			}
			//YAML::Node onenode = YAML::Load("[]");
			/*std::vector<int> onenode;
			for (size_t j = 0; j < curves.size(); j++)
			{
				onenode.push_back(onecurve[j]);
			}*/
			patch2curvej.push_back(onecurve);
		}


		for (size_t i = 0; i < curves.size(); i++)
		{
			if (corners.size() > 0)
			{
				std::vector<int> onecurve(corners.size(), 0);
				for (size_t j = 0; j < curve2corner[i].size(); j++)
				{
					onecurve[curve2corner[i][j]] = 1;
				}
				/*YAML::Node onenode = YAML::Load("[]");
				for (size_t j = 0; j < corners.size(); j++)
				{
					onenode.push_back(onecurve[j]);
				}*/
				curve2cornerj.push_back(onecurve);
			}

		}

		for (size_t i = 0; i < patches.size(); i++)
		{
			if (corners.size() > 0)
			{
				std::vector<int> onepatch(corners.size(), 0);
				for (size_t j = 0; j < patch2corner[i].size(); j++)
				{
					onepatch[patch2corner[i][j]] = 1;
				}
				/*YAML::Node onenode = YAML::Load("[]");
				for (size_t j = 0; j < corners.size(); j++)
				{
					onenode.push_back(onecurve[j]);
				}*/
				patch2cornerj.push_back(onepatch);
			}

		}
		

		oj["patch2curve"] = patch2curvej;
		oj["curve2corner"] = curve2cornerj;
		oj["patch2corner"] = patch2cornerj;

		std::ofstream ofs;
		ofs.open(inputprefix + "_input.json");
		//ofs << std::setw(2) << oj;
		ofs << oj;
		ofs.close();

		//return 1;

		//set curve_fitters and surf_fitters
		assert(curve_type.size() == curves.size());
		assert(patch_type.size() == patches.size());
		for (size_t i = 0; i < curve_type.size(); i++)
		{
			CurveFitter* cf = NULL;
			if (curve_type[i] == "Line")
				cf = new LineFitter();
			else if (curve_type[i] == "Circle")
				cf = new CircleFitter();
			else if (curve_type[i] == "Ellipse")
				cf = new EllipseFitter();
			else if (curve_type[i] == "BSpline")
				//cf = new SplineCurveFitter();
				cf = new NURBSCurveFitter();

			//cf->closed = is_curved_closed[i];
			cf->closed = true;
			if (is_curved_closed[i] < 0.5)
				cf->closed = false;
			assert(cf != NULL);
			curve_fitters.push_back(cf);

			////debug version
			//curve_type[i] = "Line";
			//CurveFitter* cf = new LineFitter();
			//curve_fitters.push_back(cf);
		}

		for (size_t i = 0; i < patch_type.size(); i++)
		{
			SurfFitter* sf = NULL;
			//if (patch_type[i] == "Plane")
			//	sf = new PlaneFitter();
			//else if (patch_type[i] == "Cylinder")
			//	sf = new CylinderFitter();
			//else if (patch_type[i] == "Torus")
			//{
			//	//sf = new TorusFitter();
			//	sf = new NURBSSurfFitter(); //using nurbs to fit torus
			//	patch_type[i] = "BSpline";
			//}
			//else if (patch_type[i] == "Cone")
			//	sf = new ConeFitter();
			//	//sf = new PlaneFitter();
			//	//sf = new NURBSSurfFitter();
			//else if (patch_type[i] == "Sphere")
			//	sf = new SphereFitter();
			//else if (patch_type[i] == "BSpline")
			//	//sf = new SplineFitter();
			//	//sf = new ThinPlateSpline();
			//	sf = new NURBSSurfFitter();
			
			//original version
			std::cout << "patch close: " << patch_close[i] << std::endl;
			if (!patch_close[i] && (patch_type[i] == "Cylinder" || patch_type[i] == "Cone"))
			//if ((patch_type[i] == "Cylinder" || patch_type[i] == "Cone"))
			{
				std::vector<vec3d> patch_update;
				patch_flip_uv[i] = !check_patch_uv_order(patches[i], dim_u_input, dim_v_input, patch_update);
				std::cout << "id: " << i << " flip uv: " << patch_flip_uv[i] << std::endl;
				if (patch_flip_uv[i])
				{
					patches[i] = patch_update;
				}
			}

			if (!patch_close[i])
			{
				if (patch_type[i] == "Plane")
					sf = new PlaneFitter();
				else if (patch_type[i] == "Cylinder")
				{
					if (!flag_direct_fit_cylinder_cone)
					{
						sf = new NURBSSurfFitter(2, 1, 4, 2);
						patch_type[i] = "BSpline";
					}
					else
					{
						sf = new CylinderFitter(flag_using_aqd);
					}
				}
				else if (patch_type[i] == "Torus")
				{
					sf = new NURBSSurfFitter(2, 2, 4, 4); //using nurbs to fit torus
					patch_type[i] = "BSpline";
				}
				else if (patch_type[i] == "Cone")
				{
					if (!flag_direct_fit_cylinder_cone)
					{
						sf = new NURBSSurfFitter(2, 1, 4, 2);
						patch_type[i] = "BSpline";
					}
					else
					{
						sf = new ConeFitter(flag_using_aqd);
					}
				}
				else if (patch_type[i] == "Sphere")
					//sf = new NURBSSurfFitter(2, 2);
					sf = new SphereFitter();
				else if (patch_type[i] == "BSpline")
					//sf = new NURBSSurfFitter(3, 3, 20, 20);
					//sf = new NURBSSurfFitter(3, 3, 15, 15);
					//sf = new NURBSSurfFitter(3, 3, 15, 15, 0.01);
					sf = new NURBSSurfFitter(3, 3, dim_u_ctrl, dim_v_ctrl, smoothness);
					//sf = new NURBSSurfFitter();
			}
			else
			{
				if (patch_type[i] == "Plane")
					sf = new PlaneFitter();
				else if (patch_type[i] == "Cylinder")
				{
					if (!flag_direct_fit_cylinder_cone)
					{
						sf = new NURBSSurfFitter(3, 1, 7, 2);
						//sf = new NURBSSurfFitter(3, 3, 5, 5);
						patch_type[i] = "BSpline";
					}
					else
					{
						sf = new CylinderFitter(flag_using_aqd);
					}
				}
				else if (patch_type[i] == "Torus")
				{
					sf = new NURBSSurfFitter(3, 2, 7, 4); //using nurbs to fit torus
					patch_type[i] = "BSpline";
				}
				else if (patch_type[i] == "Cone")
				{
					if (!flag_direct_fit_cylinder_cone)
					{
						sf = new NURBSSurfFitter(3, 1, 7, 2);
						patch_type[i] = "BSpline";
					}
					else
					{
						sf = new ConeFitter(flag_using_aqd);
					}
				}
				else if (patch_type[i] == "Sphere")
					//sf = new NURBSSurfFitter(2, 2);
					sf = new SphereFitter();
				else if (patch_type[i] == "BSpline")
					//sf = new NURBSSurfFitter(3, 3, 20, 20);
					//sf = new NURBSSurfFitter();
					//sf = new NURBSSurfFitter(3, 3, 15, 15);
					//sf = new NURBSSurfFitter(3, 3, 15, 15, 0.01);
					sf = new NURBSSurfFitter(3, 3, dim_u_ctrl, dim_v_ctrl, smoothness);

			}
			//sf = new NURBSSurfFitter();
			if (!patch_flip_uv[i])
				sf->u_closed = patch_close[i];
			else
			{
				sf->v_closed = patch_close[i];
			}
			
			//set closeness
			//u v closeness is decided by flipping or not.
			assert(sf != NULL);
			surf_fitters.push_back(sf);


			//debug version
			/*patch_type[i] = "BSpline";
			sf = new NURBSSurfFitter(3, 3, dim_u_ctrl, dim_v_ctrl, smoothness);
			surf_fitters.push_back(sf);*/
		}

		//for patch extension
		std::vector<std::vector<vec3d>> patches_expand(patches.size());
		for (size_t i = 0; i < patches.size(); i++)
		{
			int dim_u = 10, dim_v = 10;
			if (patches[i].size() == 400)
			{
				dim_u = 20, dim_v = 20;
			}
			expand_grid_points(patches[i], dim_u, dim_v, dist_expand, patch_close[i], patches_expand[i]);
		}

		//if (result.count("p"))
		assert(result.count("p"));
		auto& inputpc = result["p"].as<std::string>();
		MyPointCloud pc;
		pc.load_ply(inputpc.c_str(), true); //second parameter: with normals

		std::vector<vec3d> curve_pcnn;
		for (size_t i = 0; i < curve_type.size(); i++)
		{
			std::cout << "curve No. " << i << " type: " << curve_type[i] << std::endl;
			curve_fitters[i]->set_points(curves[i]);
			curve_fitters[i]->fitting();
			curve_fitters[i]->get_self_projection();

			if (!flag_curve_self_proj)
			{
				curve_fitters[i]->get_seg_pts();
				if (flag_debug_curve)
				{
					save_pts_ply(std::to_string(i) + "_curve_seg.ply", curve_fitters[i]->seg_pts);
					save_pts_ply(std::to_string(i) + "_curve_input.ply", curves[i]);
				}
			}
		}
		//save_pts_ply((inputprefix + "_curves_init.ply"), curves);
		std::vector<vec3d> patch_pcnn;
		for (size_t i = 0; i < patch_type.size(); i++)
		{
			std::cout << "patch No. " << i << " type: " << patch_type_ori[i] << std::endl;
			surf_fitters[i]->set_points(patches[i]);
			//surf_fitters[i]->subdivide_grid();
			
			/*if (patch_type_ori[i] == "BSpline")
				surf_fitters[i]->subdivide_grid(2);*/

			if (flag_debug_patch)
			{
				surf_fitters[i]->save_input_patch_obj(std::to_string(i) + "_input_patch.obj");
			}

			surf_fitters[i]->fitting();
			if (flag_patch_self_proj)
			{
				surf_fitters[i]->get_self_projection(true);
				patches[i] = surf_fitters[i]->projection_pts;
			}
			else
			{
				//surf_fitters[i]->projection_pts(patches[i], surf_fitters)
				surf_fitters[i]->get_self_projection(true);
				surf_fitters[i]->get_grid_pts(true);
				//save_pts_ply(std::to_string(i) + "_patch_proj.ply", surf_fitters[i]->grid_pts);
				//save_pts_normal_xyz(std::to_string(i) + "_patch_grid.xyz", surf_fitters[i]->grid_pts, surf_fitters[i]->grid_normals);
			}

			
			if (flag_debug_patch)
			{
				std::ofstream file(std::to_string(i) + "_init_fit_patch.obj");
				size_t counter = 1;
				surf_fitters[i]->write_obj_surf(file, counter, u_split, v_split);
				file.close();
			}

			////for debugging
			//double avg_error = 0.0;
			//for (size_t j = 0; j < surf_fitters[i]->input_pts.size(); j++)
			//{
			//	avg_error += (surf_fitters[i]->input_pts[j] - surf_fitters[i]->projection_pts[j]).Length();
			//}
			//avg_error /= surf_fitters[i]->input_pts.size();
			//std::cout << "patch : " << i << " avg error: " << avg_error << std::endl;
		}

		std::vector<int> control_dims({ 5, 5 });

		if (flag_skip_control_fitting)
			control_dims.clear();

		if (result.count("outerone"))
		{
			control_dims.clear();
			control_dims.push_back(5);
		}
		
		std::vector<std::vector<vec3d>> patch_pcs(patches.size());
		std::vector<std::vector<vec3d>> patch_pc_normals(patches.size());
 		std::vector<std::vector<vec3d>> patch_grids(patches.size());
		if (result.count("outerzero"))
		{
			control_dims.clear();
			//update u v control of splines
			for (size_t i = 0; i < patch_type_ori.size(); i++)
			{
				//set patch grids
				surf_fitters[i]->set_uv_split(dim_u_output, dim_v_output);
				surf_fitters[i]->get_grid_pts();
				patch_grids[i] = surf_fitters[i]->grid_pts;
				//if (patch_type_ori[i] == "BSpline")
				//{
				//	//surf_fitters[i]->get_grid_pts();
				//	std::vector<vec3d> tmp = surf_fitters[i]->grid_pts;
				//	//delete surf_fitters[i];
				//	//surf_fitters[i] = NULL;
				//	////sf = new NURBSSurfFitter(3, 3, dim_u_ctrl, dim_v_ctrl, smoothness);
				//	//surf_fitters[i] = new NURBSSurfFitter(3, 3, control_dims[control_iter], control_dims[control_iter], smoothness);
				//	//surf_fitters[i]->u_closed = patch_close[i];

				//	////fit once for projection
				//	//surf_fitters[i]->set_points(tmp);
				//	//surf_fitters[i]->save_input_patch_obj(std::to_string(i) + "_input_patch_20.obj");
				//	//surf_fitters[i]->fitting();
				//}
				
			}

			for (size_t i = 0; i < patches.size(); i++)
			{
				vec3d bb_min, bb_max, bb_diff;
				get_pcs_bb(patch_grids[i], bb_min, bb_max);
				bb_diff = bb_max - bb_min;
				double prev_range = std::max(bb_diff[0], std::max(bb_diff[1], bb_diff[2]));

				surf_fitters[i]->set_uv_split(dim_u_output, dim_v_output);
				surf_fitters[i]->get_grid_pts();
				get_pcs_bb(surf_fitters[i]->grid_pts, bb_min, bb_max);
				bb_diff = bb_max - bb_min;
				double cur_range = std::max(bb_diff[0], std::max(bb_diff[1], bb_diff[2]));
				if (cur_range < max_patch_grid_diff_ratio * prev_range)
				{
					patch_grids[i] = surf_fitters[i]->grid_pts;
				}
				else
				{
					std::cout << "patch id: " << i << " out of range in iter 1" << std::endl;
				}
			}


		}

		for (size_t control_iter = 0; control_iter < control_dims.size(); control_iter++)
		{
			//patch fitting based on curves and corners
			std::cout << "control pts update iter: " << control_iter << std::endl;
			if (control_iter == control_dims.size() - 1)
			{
				//update u v control of splines
				for (size_t i = 0; i < patch_type_ori.size(); i++)
				{
					//set patch grids
					surf_fitters[i]->set_uv_split(dim_u_output, dim_v_output);
					surf_fitters[i]->get_grid_pts();
					patch_grids[i] = surf_fitters[i]->grid_pts;
					if (patch_type_ori[i] == "BSpline")
					{
						//surf_fitters[i]->get_grid_pts();
						std::vector<vec3d> tmp = surf_fitters[i]->grid_pts;
						delete surf_fitters[i];
						surf_fitters[i] = NULL;
						//sf = new NURBSSurfFitter(3, 3, dim_u_ctrl, dim_v_ctrl, smoothness);
						surf_fitters[i] = new NURBSSurfFitter(3, 3, control_dims[control_iter], control_dims[control_iter], smoothness);
						surf_fitters[i]->u_closed = patch_close[i];
						
						//fit once for projection
						surf_fitters[i]->set_points(tmp);
						surf_fitters[i]->save_input_patch_obj(std::to_string(i) + "_input_patch_20.obj");
						surf_fitters[i]->fitting();
					}
				}
			}
			int max_outer_iter = 5;
			std::vector<std::vector<double>> dist_pc2patch(pc.pts.size(), std::vector<double>(patches.size(), 0.0));
			for (size_t outiter = 0; outiter < max_outer_iter; outiter++)
			{
				std::cout << "outer iter: " << outiter << std::endl;
				for (size_t i = 0; i < patches.size(); i++)
				{
					std::vector<vec3d> nns;
					surf_fitters[i]->projection(pc.pts, nns);
					for (size_t j = 0; j < nns.size(); j++)
					{
						dist_pc2patch[j][i] = (pc.pts[j] - nns[j]).Length();
					}
				}

				//std::vector<std::vector<vec3d>> patch_pcs(patches.size());
				std::vector<double> patch_ranges(patches.size()), th_patch_pc(patches.size());
				for (size_t i = 0; i < patches.size(); i++)
				{
					vec3d bb_min, bb_max;
					get_pcs_bb(patches[i], bb_min, bb_max);
					vec3d bb_diff = bb_max - bb_min;
					patch_ranges[i] = std::max(bb_diff[0], std::max(bb_diff[1], bb_diff[2]));
					th_patch_pc[i] = std::min(max_patch_pc_dist, max_patch_nn_dist_ratio * patch_ranges[i]);
				}

				//get patch pcs, range
				patch_pcs.clear();
				patch_pcs.resize(patches.size());

				if (flag_input_pc_normal)
				{
					patch_pc_normals.clear();
					patch_pc_normals.resize(patches.size());
				}
				
				if (flag_seg_pc) //set as true
				{
					for (size_t i = 0; i < pc.pts.size(); i++)
					{
						auto smallest = std::min_element(dist_pc2patch[i].begin(), dist_pc2patch[i].end());
						if (*smallest < max_patch_pc_dist)
						{
							int curid = std::distance(dist_pc2patch[i].begin(), smallest);
							if (*smallest < th_patch_pc[curid])
							{
								patch_pcs[curid].push_back(pc.pts[i]);
								if (flag_input_pc_normal)
									patch_pc_normals[curid].push_back(pc.normals[i]);
							}
						}
					}
				}
				else
				{
					for (size_t i = 0; i < pc.pts.size(); i++)
					{
						for (size_t j = 0; j < patches.size(); j++)
						{
							if (dist_pc2patch[i][j] < max_patch_pc_dist)
							{
								patch_pcs[j].push_back(pc.pts[i]);
								if (flag_input_pc_normal) patch_pc_normals[j].push_back(pc.normals[i]);
							}
						}
					}
				}

				//check nn pc and refit
				for (size_t i = 0; i < patches.size(); i++)
				{
					//get patch range;
					double patch_range = patch_ranges[i];
					//double patch_range = patch_max - patch_min;
					//std::cout << "patch " << i << " range: " << patch_range << std::endl;
					//double cur_th_patch_pc = std::min(max_patch_pc_dist, max_patch_nn_dist_ratio * patch_range);
					double cur_th_patch_curve = std::min(max_patch_curve_dist, max_patch_nn_dist_ratio * patch_range);
					double cur_th_patch_corner = std::min(max_patch_corner_dist, max_patch_nn_dist_ratio * patch_range);

					std::vector<vec3d> cur_valid_curve, cur_valid_corner, cur_valid_pc;
					if (flag_fit_patch_with_curve)
					{
						for (size_t j = 0; j < patch2curve[i].size(); j++)
						{
							std::vector<vec3d> nns;
							surf_fitters[i]->projection(curves[patch2curve[i][j]], nns);
							double min_dist = -1.0;
							for (size_t k = 0; k < nns.size(); k++)
							{
								double tmp = (curves[patch2curve[i][j]][k] - nns[k]).Length();
								if (min_dist < 0.0 || min_dist > tmp)
								{
									min_dist = tmp;
								}
							}
							//std::cout << "patch id: " << i << " curve id : " << patch2curve[i][j] << " min dist: " << min_dist << std::endl;

							if (flag_only_use_near_nbs)
							{
								if (min_dist < cur_th_patch_curve)
								{
									cur_valid_curve.insert(cur_valid_curve.end(), curves[patch2curve[i][j]].begin(), curves[patch2curve[i][j]].end());
								}
							}
							else
							{
								cur_valid_curve.insert(cur_valid_curve.end(), curves[patch2curve[i][j]].begin(), curves[patch2curve[i][j]].end());

							}
							
						}
					}

					if (flag_fit_patch_with_corner)
					{
						std::vector<vec3d> all_nn_corners;
						for (size_t j = 0; j < patch2corner[i].size(); j++)
						{
							all_nn_corners.push_back(corners[patch2corner[i][j]]);
						}
						std::vector<vec3d> nns;
						surf_fitters[i]->projection(all_nn_corners, nns);
						for (size_t j = 0; j < nns.size(); j++)
						{
							double tmp = (all_nn_corners[j] - nns[j]).Length();
							if (flag_only_use_near_nbs)
							{
								if (tmp < cur_th_patch_corner)
								{
									cur_valid_corner.push_back(all_nn_corners[j]);
								}
							}
							else
							{
								cur_valid_corner.push_back(all_nn_corners[j]);
							}
							
						}
					}


					if (patch_type[i] == "BSpline")
					{
						surf_fitters[i]->set_points(patches[i], std::vector<double>(patches[i].size(), 1.0));
						std::vector<vec3d> nongrid(patch_pcs[i].begin(), patch_pcs[i].end());
						std::vector<double> nongrid_weight(patch_pcs[i].size(), w_pc_patch * 1.0);
						if (flag_fit_patch_with_curve)
						{
							nongrid.insert(nongrid.end(), cur_valid_curve.begin(), cur_valid_curve.end());
							std::vector<double> tmp(cur_valid_curve.size(), w_curveonpatch * 1.0);
							nongrid_weight.insert(nongrid_weight.end(), tmp.begin(), tmp.end());
						}

						if (flag_fit_patch_with_corner)
						{
							nongrid.insert(nongrid.end(), cur_valid_corner.begin(), cur_valid_corner.end());
							std::vector<double> tmp(cur_valid_corner.size(), w_corneronpatch * 1.0);
							nongrid_weight.insert(nongrid_weight.end(), tmp.begin(), tmp.end());
						}
						surf_fitters[i]->set_bspline_nongrid_points(nongrid, nongrid_weight);
					}
					else
					{
						std::vector<vec3d> fitted_pts(patches[i].begin(), patches[i].end());
						for (size_t j = 0; j < w_pc_patch; j++)
						{
							fitted_pts.insert(fitted_pts.end(), patch_pcs[i].begin(), patch_pcs[i].end());
						}

						if (flag_fit_patch_with_curve)
						{
							for (size_t j = 0; j < w_curveonpatch; j++)
							{
								fitted_pts.insert(fitted_pts.end(), cur_valid_curve.begin(), cur_valid_curve.end());
							}
						}

						if (flag_fit_patch_with_corner)
						{
							for (size_t j = 0; j < w_corneronpatch; j++)
							{
								fitted_pts.insert(fitted_pts.end(), cur_valid_corner.begin(), cur_valid_corner.end());
							}
						}
						surf_fitters[i]->set_points(fitted_pts);
						if (patch_type[i] == "Cylinder" || patch_type[i] == "Cone")
						{
							//directly fitting cylinder and cone
							std::vector<vec3d> fitted_normals, tmp, tmp_normals;
							surf_fitters[i]->projection_with_normal(patches[i], tmp, fitted_normals);
							if (flag_input_pc_normal)
							{
								for (size_t j = 0; j < w_pc_patch; j++)
									fitted_normals.insert(fitted_normals.end(), patch_pc_normals[i].begin(), patch_pc_normals[i].end());
							}
							else
							{
								//get projection normals
								surf_fitters[i]->projection_with_normal(patch_pcs[i], tmp, tmp_normals);
								for (size_t j = 0; j < w_pc_patch; j++)
									fitted_normals.insert(fitted_normals.end(), tmp_normals.begin(), tmp_normals.end());
							}

							if (flag_fit_patch_with_curve)
							{
								surf_fitters[i]->projection_with_normal(cur_valid_curve, tmp, tmp_normals);
								for (size_t j = 0; j < w_curveonpatch; j++)
								{
									fitted_normals.insert(fitted_normals.end(), tmp_normals.begin(), tmp_normals.end());
								}
							}

							if (flag_fit_patch_with_corner)
							{
								surf_fitters[i]->projection_with_normal(cur_valid_corner, tmp, tmp_normals);
								for (size_t j = 0; j < w_corneronpatch; j++)
								{
									fitted_normals.insert(fitted_normals.end(), tmp_normals.begin(), tmp_normals.end());
								}
							}

							surf_fitters[i]->set_normals(fitted_normals);

							if (flag_set_axis_by_neighbor)
							{
								bool axis_valid = false;
								vec3d tmp_axis(0.0, 0.0, 0.0);
								axis_valid = get_cylindercone_axis_by_neighbors(patch_type_ori[i], patch2curve[i], curve_fitters, curve_type, curves, th_same_dir_cos, tmp_axis);
								std::cout << "axis validness for patch " << i << " : " << axis_valid << std::endl;
								if (axis_valid)
								{
									surf_fitters[i]->set_axis(tmp_axis);
								}
								else
								{
									//use aqd fitting
									surf_fitters[i]->flag_using_aqd = true;
								}
							}
						}
					}


					surf_fitters[i]->fitting();
					if (flag_patch_self_proj)
					{
						if (patch_type[i] == "BSpline")
						{
							surf_fitters[i]->get_self_projection();
							
							/*double avg_dist = 0.0;
							for (size_t j = 0; j < surf_fitters[i]->input_pts.size(); j++)
							{
								avg_dist += (surf_fitters[i]->input_pts[j] - surf_fitters[i]->projection_pts[j]).Length();
							}
							avg_dist /= 1.0 * surf_fitters[i]->input_pts.size();
							std::cout << "patch id : " << i << " dist: " << avg_dist << std::endl;*/
							
							patches[i] = surf_fitters[i]->projection_pts;
						}
						else
						{
							surf_fitters[i]->projection(patches[i], surf_fitters[i]->projection_pts, true);

							/*double avg_dist = 0.0;
							for (size_t j = 0; j < patches[i].size(); j++)
							{
								avg_dist += (patches[i][j] - surf_fitters[i]->projection_pts[j]).Length();
							}
							avg_dist /= 1.0 * patches[i].size();
							std::cout << "patch id : " << i << " dist: " << avg_dist << std::endl;*/

							patches[i] = surf_fitters[i]->projection_pts;
						}
					}
					if (flag_debug_patch)
					{
						//surf_fitters[i]->save_input_patch_obj(std::to_string(i) + "_patch_input.obj");
						save_pts_ply(std::to_string(i) + "_patch_pc.ply", patch_pcs[i]);
						std::ofstream file(std::to_string(i) + "_refit_patch.obj");
						size_t counter = 1;
						surf_fitters[i]->write_obj_surf(file, counter, u_split, v_split);
						file.close();
						if (flag_fit_patch_with_curve)
						{
							save_pts_ply(std::to_string(i) + "_patch_curve.ply", cur_valid_curve);
						}
						if (flag_fit_patch_with_corner)
						{
							save_pts_ply(std::to_string(i) + "_patch_corner.ply", cur_valid_corner);
						}
					}
				}


				//segmentation
				std::vector<vec3d> valid_pcs;
				std::vector<int> valid_mask;
				for (size_t i = 0; i < patches.size(); i++)
				{
					valid_pcs.insert(valid_pcs.end(), patch_pcs[i].begin(), patch_pcs[i].end());
					std::vector<int> tmp_mask(patch_pcs[i].size(), i);
					valid_mask.insert(valid_mask.end(), tmp_mask.begin(), tmp_mask.end());
				}

				if (flag_debug_patch)
				{
					save_all_patches(surf_fitters, patch_type, u_split, v_split, (inputprefix + "_iter_" + std::to_string(control_iter) + "_" + std::to_string(outiter) + "_all_patches.obj").c_str());
					save_pts_color_ply((inputprefix + "_iter_" + std::to_string(control_iter) + "_" + std::to_string(outiter) + "_pc_seg.ply"), valid_pcs, valid_mask);
				}
			}
			//save_all_patches(surf_fitters, patch_type, u_split, v_split, (inputprefix + "_patches_ctrliter_" + std::to_string(control_iter) + ".obj").c_str());
			
			//if control_iter == 1, update patch_grids
			if (control_iter == control_dims.size() - 1)
			{
				for (size_t i = 0; i < patches.size(); i++)
				{
					vec3d bb_min, bb_max, bb_diff;
					get_pcs_bb(patch_grids[i], bb_min, bb_max);
					bb_diff = bb_max - bb_min;
					double prev_range = std::max(bb_diff[0], std::max(bb_diff[1], bb_diff[2]));
					
					surf_fitters[i]->set_uv_split(dim_u_output, dim_v_output);
					surf_fitters[i]->get_grid_pts();
					get_pcs_bb(surf_fitters[i]->grid_pts, bb_min, bb_max);
					bb_diff = bb_max - bb_min;
					double cur_range = std::max(bb_diff[0], std::max(bb_diff[1], bb_diff[2]));
					if (cur_range < max_patch_grid_diff_ratio * prev_range)
					{
						patch_grids[i] = surf_fitters[i]->grid_pts;
					}
					else
					{
						std::cout << "patch id: " << i << " out of range in iter 1" << std::endl;
					}
				}
			}
			
			//curve fitting based on patches and corners
			for (size_t i = 0; i < curves.size(); i++)
			{
				std::vector<vec3d> fitted_pts, bspline_nongrid_pts;
				std::vector<double> bspline_nongrid_weight;
				for (size_t j = 0; j < w_self_curve; j++)
				{
					fitted_pts.insert(fitted_pts.end(), curves[i].begin(), curves[i].end());
				}
				if (flag_debug_curve)
					save_pts_ply(std::to_string(i) + "curve_init.ply", fitted_pts);
				if (curve_type[i] != "BSpline")
				{
					// curve corners
					std::vector<vec3d> curvecorners_init;
					for (size_t j = 0; j < curve2corner[i].size(); j++)
					{
						curvecorners_init.push_back(corners[curve2corner[i][j]]);
					}

					std::vector<vec3d> curvecorners_proj;
					curve_fitters[i]->projection(curvecorners_init, curvecorners_proj, false);
					std::vector<vec3d> curvecorners;
					for (size_t j = 0; j < curve2corner[i].size(); j++)
					{
						if (flag_only_use_near_nbs)
						{
							if ((curvecorners_init[j] - curvecorners_proj[j]).Length() < max_curve_corner_dist)
							{
								curvecorners.push_back(curvecorners_init[j]);
							}
						}
						else
						{
							curvecorners.push_back(curvecorners_init[j]);
						}
						
					}

					for (size_t j = 0; j < w_corneroncurve; j++)
					{
						fitted_pts.insert(fitted_pts.end(), curvecorners.begin(), curvecorners.end());
					}

					if (flag_debug_curve)
						save_pts_ply(std::to_string(i) + "curve_corner.ply", curvecorners);


					std::vector<vec3d> curvepatches;
					for (size_t j = 0; j < curve2patch[i].size(); j++)
					{
						std::vector<vec3d> tmp;
						//surf_fitters[curve2patch[i][j]]->get_nn_grid_pts(curves[i], tmp);
						surf_fitters[curve2patch[i][j]]->projection(curves[i], tmp);
						double min_dist = (curves[i][0] - tmp[0]).Length();
						for (size_t k = 1; k < curves[i].size(); k++)
						{
							double cur_dist = (curves[i][k] - tmp[k]).Length();
							if (min_dist > cur_dist) min_dist = cur_dist;
						}
						
						if (flag_only_use_near_nbs)
						{
							if (min_dist < max_patch_curve_dist) //not consiser bb yet							
								curvepatches.insert(curvepatches.end(), tmp.begin(), tmp.end());
						}
						else
						{
							curvepatches.insert(curvepatches.end(), tmp.begin(), tmp.end());
						}
						
					}

					for (size_t j = 0; j < w_patchoncurve; j++)
					{
						fitted_pts.insert(fitted_pts.end(), curvepatches.begin(), curvepatches.end());
					}

					if (flag_debug_curve)
						save_pts_ply(std::to_string(i) + "curve_patches.ply", curvepatches);

				}
				else
				{
					//set bspline non grid, only consider corner
					std::vector<vec3d> curvecorners;
					for (size_t j = 0; j < curve2corner[i].size(); j++)
					{
						//no thresholding tmp
						curvecorners.push_back(corners[curve2corner[i][j]]);
					}
					bspline_nongrid_pts.insert(bspline_nongrid_pts.end(), curvecorners.begin(), curvecorners.end());
					std::vector<double> curvecorners_weight(curvecorners.size(), w_corneroncurve * 1.0);
					bspline_nongrid_weight.insert(bspline_nongrid_weight.end(), curvecorners_weight.begin(), curvecorners_weight.end());
				}

				//save_pts_ply("tmp_pts.ply", fitted_pts);
				if (curve_type[i] != "BSpline")
					curve_fitters[i]->set_points(fitted_pts);
				else
				{
					curve_fitters[i]->set_points(curves[i]);
					curve_fitters[i]->set_bspline_nongrid_points(bspline_nongrid_pts, bspline_nongrid_weight);
				}
				curve_fitters[i]->fitting();
				curve_fitters[i]->projection(curves[i], curve_fitters[i]->projection_pts, true);


				if (flag_debug_curve)
					save_pts_ply(std::to_string(i) + "curve_proj.ply", curve_fitters[i]->projection_pts);
				//update curves
				curves[i] = curve_fitters[i]->projection_pts;
			}
			//save all curves
			//save_pts_ply((inputprefix + "_curves_ctrliter_" + std::to_string(control_iter) + ".ply"), curves);
			//corner fitting based on curves and curves
			for (size_t i = 0; i < corners.size(); i++)
			{
				std::vector<vec3d> corner_in;

				for (size_t j = 0; j < w_self_corner; j++)
				{
					corner_in.push_back(corners[i]);

				}
				std::vector<vec3d> onecorner;
				onecorner.push_back(corners[i]);
				//curve to corner
				for (size_t j = 0; j < corner2curve[i].size(); j++)
				{
					std::vector<vec3d> tmp_vec;
					curve_fitters[corner2curve[i][j]]->projection(onecorner, tmp_vec, false);
					
					if (flag_only_use_near_nbs)
					{
						if ((onecorner[0] - tmp_vec[0]).Length() < max_curve_corner_dist)
						{
							for (size_t k = 0; k < w_curveoncorner; k++)
							{
								corner_in.push_back(tmp_vec[0]);
							}
						}
					}
					else
					{
						for (size_t k = 0; k < w_curveoncorner; k++)
						{
							corner_in.push_back(tmp_vec[0]);
						}
					}
					
				}

				//patch to corner
				for (size_t j = 0; j < corner2patch[i].size(); j++)
				{
					std::vector<vec3d> tmp_vec;
					//surf_fitters[corner2patch[i][j]]->get_nn_proj_pt(corners[i], tmp);
					surf_fitters[corner2patch[i][j]]->projection(onecorner, tmp_vec);
					if (flag_only_use_near_nbs)
					{
						if ((onecorner[0] - tmp_vec[0]).Length() < max_patch_corner_dist)
						{
							for (size_t k = 0; k < w_patchoncorner; k++)
							{
								corner_in.push_back(tmp_vec[0]);
							}
						}
					}
					else
					{
						for (size_t k = 0; k < w_patchoncorner; k++)
						{
							corner_in.push_back(tmp_vec[0]);
						}
					}
					
				}

				//average
				vec3d avg_pt(0.0, 0.0, 0.0);
				for (size_t j = 0; j < corner_in.size(); j++)
				{
					avg_pt = avg_pt + corner_in[j];
				}
				corners[i] = avg_pt / corner_in.size();
			}
			//save_pts_ply((inputprefix + "_corners_ctrliter_" + std::to_string(control_iter) + ".ply"), corners);
		}

		//save NURBS yaml file

		if (!flag_direct_fit_cylinder_cone)
		{

			patchj.clear();
			for (size_t i = 0; i < patches.size(); i++)
			{
				json j;
				j["type"] = patch_type_ori[i];
				j["u_dim"] = dim_u_input;
				j["v_dim"] = dim_v_input;
				j["u_closed"] = (bool)patch_close[i];
				j["v_closed"] = false;
				j["with_param"] = false;
				std::vector<double> onepatch_pts;

				for (size_t j = 0; j < patches[i].size(); j++)
				{
					for (size_t k = 0; k < 3; k++)
					{
						onepatch_pts.push_back(patches[i][j][k]);
					}
				}
				j["grid"] = onepatch_pts;
				patchj.push_back(j);
			}

			oj["patches"] = patchj;
			if (save_curve_corner)
			{
				//std::vector<YAML::Node> curve_nodes;
				json curvej;
				for (size_t i = 0; i < curves.size(); i++)
				{
					json onenode;
					onenode["type"] = curve_type[i];
					//YAML::Node onecurve_pts = YAML::Load("[]");
					std::vector<double> onecurve_pts;
					for (size_t j = 0; j < curves[i].size(); j++)
					{
						for (size_t k = 0; k < 3; k++)
						{
							onecurve_pts.push_back(curves[i][j][k]);
						}
					}
					//onenode["closed"] = (bool)curve_fitters[i]->closed;
					onenode["closed"] = (bool)curve_fitters[i]->closed; //change to curve fitter later
					onenode["pts"] = onecurve_pts;
					curvej.push_back(onenode);
				}
				oj["curves"] = curvej;
				json cornerj;
				for (size_t i = 0; i < corners.size(); i++)
				{
					json onenode;
					//YAML::Node onecorner_pts = YAML::Load("[]");
					std::vector<double> onecorner_pts;
					for (size_t k = 0; k < 3; k++)
					{
						onecorner_pts.push_back(corners[i][k]);
					}
					onenode["pts"] = onecorner_pts;
					cornerj.push_back(onenode);
				}
				oj["corners"] = cornerj;
			}
			//save_curves_obj(inputprefix + "_curves_nurbs.obj", curves, flag_curve_close);

			//for (size_t i = 0; i < patches.size(); i++)
			//{
			//	YAML::Node onenode;
			//	onenode["type"] = patch_type_ori[i];
			//	onenode["u_dim"] = dim_u_output;
			//	onenode["v_dim"] = dim_v_output;
			//	onenode["u_closed"] = surf_fitters[i]->u_closed;
			//	onenode["v_closed"] = surf_fitters[i]->v_closed;
			//	onenode["with_param"] = false;
			//	//std::vector<double> onepatch_pts;
			//	//surf_fitters[i]->set_uv_split(dim_u_output, dim_v_output);
			//	//surf_fitters[i]->get_grid_pts();
			//	YAML::Node onepatch_pts = YAML::Load("[]");
			//	for (size_t j = 0; j < patch_grids[i].size(); j++)
			//	{
			//		for (size_t k = 0; k < 3; k++)
			//		{
			//			onepatch_pts.push_back(patch_grids[i][j][k]);
			//		}
			//	}
			//	onenode["grid"] = onepatch_pts;

			//	patch_nodes.push_back(onenode);
			//}




			//if (save_curve_corner)
			//{
			//	std::vector<YAML::Node> curve_nodes;
			//	for (size_t i = 0; i < curves.size(); i++)
			//	{
			//		YAML::Node onenode;
			//		onenode["type"] = curve_type[i];
			//		YAML::Node onecurve_pts = YAML::Load("[]");
			//		for (size_t j = 0; j < curves[i].size(); j++)
			//		{
			//			for (size_t k = 0; k < 3; k++)
			//			{
			//				onecurve_pts.push_back(curves[i][j][k]);
			//			}
			//		}
			//		//onenode["closed"] = (bool)curve_fitters[i]->closed;
			//		onenode["closed"] = (bool)curve_fitters[i]->closed; //change to curve fitter later
			//		onenode["pts"] = onecurve_pts;
			//		curve_nodes.push_back(onenode);
			//	}
			//	output_node["curves"] = curve_nodes;
			//	std::vector<YAML::Node> corner_nodes;
			//	for (size_t i = 0; i < corners.size(); i++)
			//	{
			//		YAML::Node onenode;
			//		YAML::Node onecorner_pts = YAML::Load("[]");
			//		for (size_t k = 0; k < 3; k++)
			//		{
			//			onecorner_pts.push_back(corners[i][k]);
			//		}
			//		onenode["pts"] = onecorner_pts;
			//		corner_nodes.push_back(onenode);
			//	}
			//	output_node["corners"] = corner_nodes;
			//}

			/*output_node["patches"] = patch_nodes;
			output_node["patch2curve"] = patch2curve_node;
			output_node["curve2corner"] = curve2corner_node;

			YAML::Emitter output_inter_emitter;
			output_inter_emitter << output_node;

			ofs_output.open(inputprefix + "_nurbs.yml");
			ofs_output << output_inter_emitter.c_str();
			ofs_output.close();*/

			//fit cylinder and cone again, abandon those with huge distance with NURBS surface
			std::vector<bool> flag_using_nurbs(patches.size(), true);


			for (size_t last_iter = 0; last_iter < last_fit_iter; last_iter++)
			{
				//update cylinder and cone
				std::cout << "last round fitting " << last_iter << std::endl;
				std::vector<double> patch_ranges(patches.size()), th_patch_pc(patches.size());
				for (size_t i = 0; i < patches.size(); i++)
				{
					vec3d bb_min, bb_max;
					get_pcs_bb(patches[i], bb_min, bb_max);
					vec3d bb_diff = bb_max - bb_min;
					patch_ranges[i] = std::max(bb_diff[0], std::max(bb_diff[1], bb_diff[2]));
					th_patch_pc[i] = std::min(max_patch_pc_dist, max_patch_nn_dist_ratio * patch_ranges[i]);
				}
				
				for (size_t i = 0; i < patches.size(); i++)
				{
					if (patch_type[i] == "Plane" || patch_type[i] == "Sphere")
						flag_using_nurbs[i] = false;

					bool flag_fit_current = false;
					if (last_iter == 0 && (patch_type_ori[i] == "Cylinder" || patch_type_ori[i] == "Cone"))
					{
						//surf_fitters[i]->get_grid_pts(true);






































						flag_fit_current = true;
						//using projection as input
						std::vector<vec3d> cur_pts, cur_normals;
						surf_fitters[i]->projection_with_normal(patches[i], cur_pts, cur_normals);

						std::vector<vec3d> pc_pts, pc_normals;
						surf_fitters[i]->projection_with_normal(patch_pcs[i], pc_pts, pc_normals);
						for (size_t j = 0; j < w_pc_patch; j++)
						{
							cur_pts.insert(cur_pts.end(), pc_pts.begin(), pc_pts.end());
							cur_normals.insert(cur_normals.end(), pc_normals.begin(), pc_normals.end());
						}

						SurfFitter* sf = NULL, * sf_fix_axis = NULL;
						if (patch_type_ori[i] == "Cylinder")
						{
							sf = new CylinderFitter(flag_using_aqd);
						}
						else if (patch_type_ori[i] == "Cone")
						{
							sf = new ConeFitter(flag_using_aqd);
						}
						sf->u_closed = surf_fitters[i]->u_closed;
						sf->v_closed = surf_fitters[i]->v_closed;
						/*sf->set_points(surf_fitters[i]->grid_pts);
						sf->set_normals(surf_fitters[i]->grid_normals);*/

						sf->set_points(cur_pts);
						sf->set_normals(cur_normals);

						//projection
						//sf->projection_with_normal()

						sf->fitting();
						sf->get_self_projection(true);
						double avg_error = 0.0;
						for (size_t j = 0; j < sf->input_pts.size(); j++)
						{
							avg_error += (sf->input_pts[j] - sf->projection_pts[j]).Length();
						}
						avg_error /= sf->input_pts.size();

						if (flag_set_axis_by_neighbor)
						{
							bool axis_valid = false;
							vec3d tmp_axis(0.0, 0.0, 0.0);

							axis_valid = get_cylindercone_axis_by_neighbors(patch_type_ori[i], patch2curve[i], curve_fitters, curve_type, curves, th_same_dir_cos, tmp_axis);


							std::cout << "patch id: " << i << " axis valid: " << axis_valid << std::endl;
							if (axis_valid)
							{
								if (patch_type_ori[i] == "Cylinder")
								{
									sf_fix_axis = new CylinderFitter(flag_using_aqd);
								}
								else if (patch_type_ori[i] == "Cone")
								{
									sf_fix_axis = new ConeFitter(flag_using_aqd);
								}
								sf_fix_axis->u_closed = surf_fitters[i]->u_closed;
								sf_fix_axis->v_closed = surf_fitters[i]->v_closed;
								/*sf_fix_axis->set_points(surf_fitters[i]->grid_pts);
								sf_fix_axis->set_normals(surf_fitters[i]->grid_normals);*/
								sf_fix_axis->set_points(cur_pts);
								sf_fix_axis->set_normals(cur_normals);
								if (flag_axisxyz)
								{
									std::cout << "set xyz axis patch id: " << i << std::endl;
									tmp_axis = get_nn_xyz(tmp_axis, max_angle_diff_degree);
								}

								sf_fix_axis->set_axis(tmp_axis);
								//refit
								sf_fix_axis->fitting();
								sf_fix_axis->get_self_projection(true);
								if (flag_debug_patch)
								{
									save_pts_ply(std::to_string(i) + "_cc_proj_axis.ply", sf_fix_axis->projection_pts);
								}
								double avg_error_fix_axis = 0.0;
								for (size_t j = 0; j < sf_fix_axis->input_pts.size(); j++)
								{
									avg_error_fix_axis += (sf_fix_axis->input_pts[j] - sf_fix_axis->projection_pts[j]).Length();
								}
								avg_error_fix_axis /= sf_fix_axis->input_pts.size();
								std::cout << "ori error: " << avg_error << " fix axis error: " << avg_error_fix_axis << std::endl;
								if (flag_axisxyz || avg_error_fix_axis < avg_error)
								{
									avg_error = avg_error_fix_axis;
									std::swap(sf, sf_fix_axis);
									delete sf_fix_axis;
									sf_fix_axis = NULL;
								}
							}
						}

						if (flag_debug_patch)
						{
							//save_pts_normal_xyz(std::to_string(i) + "_cc_input.xyz", surf_fitters[i]->grid_pts, surf_fitters[i]->grid_normals);
							//save_pts_ply();
							save_pts_normal_xyz(std::to_string(i) + "_cc_input.xyz", cur_pts, cur_normals);
							std::ofstream file(std::to_string(i) + "_cc_surf.obj");
							size_t counter = 1;
							//skip cone
							//for (size_t i = 0; i < surf_type.size(); i++)
							{
								//std::cout << "patch No. " << i << " type: " << surf_type[i] << std::endl;
								//save_pts_ply(std::to_string(i) + "_input_patch.ply", patches[i]);
								//save_pts_ply(std::to_string(i) + "_patch_proj.ply", surf_fitters[i]->projection_pts);
								//if (surf_type[i] == "Cone") continue;
								sf->write_obj_surf(file, counter, 20, 20);
								//counter += usplit * vsplit;
							}
							file.close();
							save_pts_ply(std::to_string(i) + "_cc_proj.ply", sf->projection_pts);
						}

						//update 1207, if grid out of range, check back to geometric tools fitting
						bool flag_grid_valid = true;
						sf->set_uv_split(dim_u_output, dim_v_output);
						sf->get_grid_pts();
						patch_grids[i] = sf->grid_pts;
						for (size_t j = 0; j < patch_grids[i].size(); j++)
						{
							for (size_t k = 0; k < 3; k++)
							{
								if (patch_grids[i][j][k] < -range_obb || patch_grids[i][j][k] > range_obb)
								{
									flag_grid_valid = false;
									break;
								}
							}
							if (!flag_grid_valid)
								break;
						}
						if (!flag_grid_valid)
						{
							std::cout << "out of range patch when fitting cone/cylinder: " << i << std::endl;
							SurfFitter* sf_gt = NULL;
							if (patch_type_ori[i] == "Cylinder")
							{
								sf_gt = new CylinderFitter(false);
							}
							else if (patch_type_ori[i] == "Cone")
							{
								sf_gt = new ConeFitter(false);
							}
							sf_gt->u_closed = surf_fitters[i]->u_closed;
							sf_gt->v_closed = surf_fitters[i]->v_closed;
							/*sf_fix_axis->set_points(surf_fitters[i]->grid_pts);
							sf_fix_axis->set_normals(surf_fitters[i]->grid_normals);*/
							sf_gt->set_points(cur_pts);
							//refit
							sf_gt->fitting();
							sf_gt->get_self_projection(true);
							double avg_error_fix_axis = 0.0;
							for (size_t j = 0; j < sf_gt->input_pts.size(); j++)
							{
								avg_error_fix_axis += (sf_gt->input_pts[j] - sf_gt->projection_pts[j]).Length();
							}
							avg_error_fix_axis /= sf_gt->input_pts.size();
							std::cout << "fitting error aqd: " << avg_error << " gt: " << avg_error_fix_axis << std::endl;
							avg_error = avg_error_fix_axis;

							std::swap(sf, sf_gt);
							delete sf_gt;
							sf_gt = NULL;

							sf->set_uv_split(dim_u_output, dim_v_output);
							sf->get_grid_pts();
							patch_grids[i] = sf->grid_pts;
						}

						std::cout << "cylinder cone error: " << avg_error << std::endl;
						//if (avg_error < max_cylinder_cone_error) //error check
						if (true)
						{
							//update surffitter
							delete surf_fitters[i];
							surf_fitters[i] = NULL;
							surf_fitters[i] = sf;
							flag_using_nurbs[i] = false;
						}

						////udpate patch_grid
						//surf_fitters[i]->set_uv_split(dim_u_output, dim_v_output);
						//surf_fitters[i]->get_grid_pts();
						//patch_grids[i] = surf_fitters[i]->grid_pts;

						if (flag_patch_self_proj)
						{
							if (patch_type_ori[i] == "BSpline" || patch_type_ori[i] == "Torus")
							{
								surf_fitters[i]->get_self_projection();

								patches[i] = surf_fitters[i]->projection_pts;
							}
							else
							{
								surf_fitters[i]->projection(patches[i], surf_fitters[i]->projection_pts, true);

								patches[i] = surf_fitters[i]->projection_pts;
							}
						}
					}
					
					//fit other patches
					if (flag_fit_other_last && !flag_fit_current)
					{
						double patch_range = patch_ranges[i];
						//double patch_range = patch_max - patch_min;
						//std::cout << "patch " << i << " range: " << patch_range << std::endl;
						//double cur_th_patch_pc = std::min(max_patch_pc_dist, max_patch_nn_dist_ratio * patch_range);
						double cur_th_patch_curve = std::min(max_patch_curve_dist, max_patch_nn_dist_ratio * patch_range);
						double cur_th_patch_corner = std::min(max_patch_corner_dist, max_patch_nn_dist_ratio * patch_range);

						std::vector<vec3d> cur_valid_curve, cur_valid_corner, cur_valid_pc;
						if (flag_fit_patch_with_curve)
						{
							for (size_t j = 0; j < patch2curve[i].size(); j++)
							{
								std::vector<vec3d> nns;
								surf_fitters[i]->projection(curves[patch2curve[i][j]], nns);
								double min_dist = -1.0;
								for (size_t k = 0; k < nns.size(); k++)
								{
									double tmp = (curves[patch2curve[i][j]][k] - nns[k]).Length();
									if (min_dist < 0.0 || min_dist > tmp)
									{
										min_dist = tmp;
									}
								}
								//std::cout << "patch id: " << i << " curve id : " << patch2curve[i][j] << " min dist: " << min_dist << std::endl;

								if (flag_only_use_near_nbs)
								{
									if (min_dist < cur_th_patch_curve)
									{
										cur_valid_curve.insert(cur_valid_curve.end(), curves[patch2curve[i][j]].begin(), curves[patch2curve[i][j]].end());
									}
								}
								else
								{
									cur_valid_curve.insert(cur_valid_curve.end(), curves[patch2curve[i][j]].begin(), curves[patch2curve[i][j]].end());

								}
								
							}
						}

						if (flag_fit_patch_with_corner)
						{
							std::vector<vec3d> all_nn_corners;
							for (size_t j = 0; j < patch2corner[i].size(); j++)
							{
								all_nn_corners.push_back(corners[patch2corner[i][j]]);
							}
							std::vector<vec3d> nns;
							surf_fitters[i]->projection(all_nn_corners, nns);
							for (size_t j = 0; j < nns.size(); j++)
							{
								double tmp = (all_nn_corners[j] - nns[j]).Length();
								if (flag_only_use_near_nbs)
								{
									if (tmp < cur_th_patch_corner)
									{
										cur_valid_corner.push_back(all_nn_corners[j]);
									}
								}
								else
								{
									cur_valid_corner.push_back(all_nn_corners[j]);

								}
							}
						}


						if (patch_type_ori[i] == "BSpline" || patch_type_ori[i] == "Torus")
						{
							surf_fitters[i]->set_points(patches[i], std::vector<double>(patches[i].size(), 1.0));
							std::vector<vec3d> nongrid(patch_pcs[i].begin(), patch_pcs[i].end());
							std::vector<double> nongrid_weight(patch_pcs[i].size(), w_pc_patch * 1.0);
							if (flag_fit_patch_with_curve)
							{
								nongrid.insert(nongrid.end(), cur_valid_curve.begin(), cur_valid_curve.end());
								std::vector<double> tmp(cur_valid_curve.size(), w_curveonpatch * 1.0);
								nongrid_weight.insert(nongrid_weight.end(), tmp.begin(), tmp.end());
							}

							if (flag_fit_patch_with_corner)
							{
								nongrid.insert(nongrid.end(), cur_valid_corner.begin(), cur_valid_corner.end());
								std::vector<double> tmp(cur_valid_corner.size(), w_corneronpatch * 1.0);
								nongrid_weight.insert(nongrid_weight.end(), tmp.begin(), tmp.end());
							}
							surf_fitters[i]->set_bspline_nongrid_points(nongrid, nongrid_weight);
						}
						else
						{
							std::vector<vec3d> fitted_pts(patches[i].begin(), patches[i].end());
							for (size_t j = 0; j < w_pc_patch; j++)
							{
								fitted_pts.insert(fitted_pts.end(), patch_pcs[i].begin(), patch_pcs[i].end());
							}

							if (flag_fit_patch_with_curve)
							{
								for (size_t j = 0; j < w_curveonpatch; j++)
								{
									fitted_pts.insert(fitted_pts.end(), cur_valid_curve.begin(), cur_valid_curve.end());
								}
							}

							if (flag_fit_patch_with_corner)
							{
								for (size_t j = 0; j < w_corneronpatch; j++)
								{
									fitted_pts.insert(fitted_pts.end(), cur_valid_corner.begin(), cur_valid_corner.end());
								}
							}
							surf_fitters[i]->set_points(fitted_pts);
							if (patch_type_ori[i] == "Cylinder" || patch_type_ori[i] == "Cone")
							{
								std::vector<vec3d> fitted_normals, tmp, tmp_normals;
								surf_fitters[i]->projection_with_normal(patches[i], tmp, fitted_normals);
								if (flag_input_pc_normal)
								{
									for (size_t j = 0; j < w_pc_patch; j++)
										fitted_normals.insert(fitted_normals.end(), patch_pc_normals[i].begin(), patch_pc_normals[i].end());
								}
								else
								{
									//get projection normals
									surf_fitters[i]->projection_with_normal(patch_pcs[i], tmp, tmp_normals);
									for (size_t j = 0; j < w_pc_patch; j++)
										fitted_normals.insert(fitted_normals.end(), tmp_normals.begin(), tmp_normals.end());
								}

								if (flag_fit_patch_with_curve)
								{
									surf_fitters[i]->projection_with_normal(cur_valid_curve, tmp, tmp_normals);
									for (size_t j = 0; j < w_curveonpatch; j++)
									{
										fitted_normals.insert(fitted_normals.end(), tmp_normals.begin(), tmp_normals.end());
									}
								}

								if (flag_fit_patch_with_corner)
								{
									surf_fitters[i]->projection_with_normal(cur_valid_corner, tmp, tmp_normals);
									for (size_t j = 0; j < w_corneronpatch; j++)
									{
										fitted_normals.insert(fitted_normals.end(), tmp_normals.begin(), tmp_normals.end());
									}
								}

								surf_fitters[i]->set_normals(fitted_normals);

								if (flag_set_axis_by_neighbor)
								{
									bool axis_valid = false;
									vec3d tmp_axis(0.0, 0.0, 0.0);
									axis_valid = get_cylindercone_axis_by_neighbors(patch_type_ori[i], patch2curve[i], curve_fitters, curve_type, curves, th_same_dir_cos, tmp_axis);
									std::cout << "axis validness for patch " << i << " : " << axis_valid << std::endl;
									if (axis_valid)
									{
										surf_fitters[i]->set_axis(tmp_axis);
									}
									else
									{
										//use aqd fitting
										surf_fitters[i]->flag_using_aqd = true;
									}
								}
							}
						}


						surf_fitters[i]->fitting();
						if (flag_patch_self_proj)
						{
							if (patch_type_ori[i] == "BSpline" || patch_type_ori[i] == "Torus")
							{
								surf_fitters[i]->get_self_projection();

								patches[i] = surf_fitters[i]->projection_pts;
							}
							else
							{
								surf_fitters[i]->projection(patches[i], surf_fitters[i]->projection_pts, true);

								patches[i] = surf_fitters[i]->projection_pts;
							}
						}
						//grid patch last

						surf_fitters[i]->set_uv_split(dim_u_output, dim_v_output);
						surf_fitters[i]->get_grid_pts();
						patch_grids[i] = surf_fitters[i]->grid_pts;
					}
					

					
					if (flag_expand_para_range_last && last_iter == last_fit_iter - 1)
					{
						//update uv range
						if (!(patch_type_ori[i] == "BSpline" || patch_type_ori[i] == "Torus"))
						{
							//not work for Spline
							std::vector<vec3d> input_pts_nbs, tmp; //consider all nbs
							input_pts_nbs.insert(input_pts_nbs.end(), patches[i].begin(), patches[i].end());
							
							for (size_t j = 0; j < patch2curve[i].size(); j++)
							{
								input_pts_nbs.insert(input_pts_nbs.end(), curves[patch2curve[i][j]].begin(), curves[patch2curve[i][j]].end());
							}

							for (size_t j = 0; j < patch2corner[i].size(); j++)
							{
								input_pts_nbs.push_back(corners[patch2corner[i][j]]);
							}
							
							surf_fitters[i]->projection(input_pts_nbs, tmp, true);

							surf_fitters[i]->set_uv_split(dim_u_output, dim_v_output);
							surf_fitters[i]->get_grid_pts();
							patch_grids[i] = surf_fitters[i]->grid_pts;
						}

						
					}

				}


				//udpate curves and corners again
				//curve fitting based on patches and corners

				//update 1230, increase patch weight on curve and corner
				w_patchoncurve = 3;
				w_patchoncorner = 3;

				for (size_t i = 0; i < curves.size(); i++)
				{
					std::vector<vec3d> fitted_pts, bspline_nongrid_pts;
					std::vector<double> bspline_nongrid_weight;
					for (size_t j = 0; j < w_self_curve; j++)
					{
						fitted_pts.insert(fitted_pts.end(), curves[i].begin(), curves[i].end());
					}
					if (flag_debug_curve)
						save_pts_ply(std::to_string(i) + "curve_init.ply", fitted_pts);

					//CurveFitter* cf_backup = NULL;
					/*if (curve_type[i] == "Ellipse")
					{
						cf_backup = curve_fitters[i];
						bool validness = curve_fitters[i]->check_validness();
						std::cout << "first round validness " << validness << std::endl;
						curve_fitters[i] = new EllipseFitter();
					}*/
					
					if (curve_type[i] != "BSpline")
					{
						// curve corners
						std::vector<vec3d> curvecorners_init;
						for (size_t j = 0; j < curve2corner[i].size(); j++)
						{
							curvecorners_init.push_back(corners[curve2corner[i][j]]);
						}

						std::vector<vec3d> curvecorners_proj;
						curve_fitters[i]->projection(curvecorners_init, curvecorners_proj, false);
						std::vector<vec3d> curvecorners;
						for (size_t j = 0; j < curve2corner[i].size(); j++)
						{
							if (flag_only_use_near_nbs)
							{
								if ((curvecorners_init[j] - curvecorners_proj[j]).Length() < max_curve_corner_dist)
								{
									curvecorners.push_back(curvecorners_init[j]);
								}
							}
							else
							{
								curvecorners.push_back(curvecorners_init[j]);
							}
							
						}

						for (size_t j = 0; j < w_corneroncurve; j++)
						{
							fitted_pts.insert(fitted_pts.end(), curvecorners.begin(), curvecorners.end());
						}

						if (flag_debug_curve)
							save_pts_ply(std::to_string(i) + "curve_corner.ply", curvecorners);


						std::vector<vec3d> curvepatches;
						for (size_t j = 0; j < curve2patch[i].size(); j++)
						{
							std::vector<vec3d> tmp;
							//surf_fitters[curve2patch[i][j]]->get_nn_grid_pts(curves[i], tmp);
							surf_fitters[curve2patch[i][j]]->projection(curves[i], tmp);
							double min_dist = (curves[i][0] - tmp[0]).Length();
							for (size_t k = 1; k < curves[i].size(); k++)
							{
								double cur_dist = (curves[i][k] - tmp[k]).Length();
								if (min_dist > cur_dist) min_dist = cur_dist;
							}
							
							if (flag_only_use_near_nbs)
							{
								if (min_dist < max_patch_curve_dist) //not consiser bb yet							
									curvepatches.insert(curvepatches.end(), tmp.begin(), tmp.end());
							}
							else
							{
								curvepatches.insert(curvepatches.end(), tmp.begin(), tmp.end());
							}
						}

						for (size_t j = 0; j < w_patchoncurve; j++)
						{
							fitted_pts.insert(fitted_pts.end(), curvepatches.begin(), curvepatches.end());
						}

						if (flag_debug_curve)
							save_pts_ply(std::to_string(i) + "curve_patches.ply", curvepatches);

					}
					else
					{
						//set bspline non grid, only consider corner
						std::vector<vec3d> curvecorners;
						for (size_t j = 0; j < curve2corner[i].size(); j++)
						{
							//thresholding
							curvecorners.push_back(corners[curve2corner[i][j]]);
						}
						bspline_nongrid_pts.insert(bspline_nongrid_pts.end(), curvecorners.begin(), curvecorners.end());
						std::vector<double> curvecorners_weight(curvecorners.size(), w_corneroncurve * 1.0);
						bspline_nongrid_weight.insert(bspline_nongrid_weight.end(), curvecorners_weight.begin(), curvecorners_weight.end());
					}

					//save_pts_ply("tmp_pts.ply", fitted_pts);
					if (curve_type[i] != "BSpline")
						curve_fitters[i]->set_points(fitted_pts);
					else
					{
						//not set points yet
						//surf_fitters[i]->set_points(patches[i], std::vector<double>(patches[i].size(), 1.0));
						curve_fitters[i]->set_points(curves[i]);
						curve_fitters[i]->set_bspline_nongrid_points(bspline_nongrid_pts, bspline_nongrid_weight);
					}

					if (flag_set_curve_axis && curve_type[i] == "Circle")
					{
						//only for circle, if it is neighboring cylinders or cones, set axis as their average
						//no angle check yet
						std::vector<vec3d> axis_cands;

						for (size_t j = 0; j < curve2patch[i].size(); j++)
						{
							int cur_patch_id = curve2patch[i][j];
							if (patch_type_ori[cur_patch_id] == "Cylinder" || patch_type_ori[cur_patch_id] == "Cone")
							{
								axis_cands.push_back(surf_fitters[cur_patch_id]->get_axis());
							}
						}

						if (axis_cands.size() == 1 || axis_cands.size() == 2)
						{
							std::cout << "set curve axis" << std::endl;
							vec3d tmp_axis(0, 0, 0);
							if (axis_cands.size() == 1)
							{
								tmp_axis = axis_cands[0];
							}
							else
							{
								double cosvalue = axis_cands[0].Dot(axis_cands[1]);
								if (cosvalue < 0.0)
									axis_cands[1] = -axis_cands[1];

								tmp_axis = (axis_cands[0] + axis_cands[1]) / 2.0;
							}

							if (flag_axisxyz)
							{
								std::cout << "set xyz axis curve id: " << i << std::endl;
								tmp_axis = get_nn_xyz(tmp_axis, max_angle_diff_degree);
							}

							curve_fitters[i]->set_axis(tmp_axis);
						}
					}

					curve_fitters[i]->fitting();
					
					bool cur_para_valid = true;
					if (curve_type[i] == "Ellipse")
					{
						cur_para_valid = curve_fitters[i]->check_validness();
						if (!cur_para_valid)
						{
							std::cout << "fitting ellipse out of range" << std::endl;
							/*std::swap(curve_fitters[i], cf_backup);*/
							curve_fitters[i]->set_points(curves[i]);
							curve_fitters[i]->fitting();
							cur_para_valid = curve_fitters[i]->check_validness();
							std::cout << "second round validness " << cur_para_valid << std::endl;
							//save_pts_ply(std::to_string(i) + "curve_error.ply", curves[i]);
						}
					}
					//save_pts_ply("57curve_error00.ply", curves[57]);

					//std::cout << "validness: " << cur_para_valid << std::endl;
					if (cur_para_valid)
					{
						curve_fitters[i]->projection(curves[i], curve_fitters[i]->projection_pts, true);

						//for seg
						//curve_fitters[i]->get_seg_pts();

						if (flag_debug_curve)
							save_pts_ply(std::to_string(i) + "curve_proj.ply", curve_fitters[i]->projection_pts);
						//update curves
						curves[i] = curve_fitters[i]->projection_pts;
					}
					
					//save_pts_ply("57curve_error01.ply", curves[57]);

					if (cur_para_valid && flag_expand_para_range_last && last_iter == last_fit_iter - 1)
					{
						if (curve_type[i] != "BSpline")
						{
							//re-param
							std::vector<vec3d> tmp;
							std::cout << "update curve range last: " << i << std::endl;
							curve_fitters[i]->projection(fitted_pts, tmp, true);
							curve_fitters[i]->get_seg_pts();
							curves[i] = curve_fitters[i]->seg_pts;
						}
						else
						{
							curve_fitters[i]->get_seg_pts();
							curves[i] = curve_fitters[i]->seg_pts;
						}
					}
				
					//save_pts_ply("57curve_error02.ply", curves[57]);
				}

				//save_pts_ply("57curve_error1.ply", curves[57]);
				//save all curves
				//save_pts_ply((inputprefix + "_final_curves.ply"), curves);
				//corner fitting based on curves and curves
				for (size_t i = 0; i < corners.size(); i++)
				{
					std::vector<vec3d> corner_in;

					for (size_t j = 0; j < w_self_corner; j++)
					{
						corner_in.push_back(corners[i]);

					}
					std::vector<vec3d> onecorner;
					onecorner.push_back(corners[i]);
					//curve to corner
					for (size_t j = 0; j < corner2curve[i].size(); j++)
					{
						std::vector<vec3d> tmp_vec;
						curve_fitters[corner2curve[i][j]]->projection(onecorner, tmp_vec, false);

						if (flag_only_use_near_nbs)
						{
							if ((onecorner[0] - tmp_vec[0]).Length() < max_curve_corner_dist)
							{
								for (size_t k = 0; k < w_curveoncorner; k++)
								{
									corner_in.push_back(tmp_vec[0]);
								}
							}
						}
						else
						{
							for (size_t k = 0; k < w_curveoncorner; k++)
							{
								corner_in.push_back(tmp_vec[0]);
							}
						}
						
					}

					//patch to corner
					for (size_t j = 0; j < corner2patch[i].size(); j++)
					{
						std::vector<vec3d> tmp_vec;
						//surf_fitters[corner2patch[i][j]]->get_nn_proj_pt(corners[i], tmp);
						surf_fitters[corner2patch[i][j]]->projection(onecorner, tmp_vec);
						if (flag_only_use_near_nbs)
						{
							if ((onecorner[0] - tmp_vec[0]).Length() < max_patch_corner_dist)
							{
								for (size_t k = 0; k < w_patchoncorner; k++)
								{
									corner_in.push_back(tmp_vec[0]);
								}
							}
						}
						else
						{
							for (size_t k = 0; k < w_patchoncorner; k++)
							{
								corner_in.push_back(tmp_vec[0]);
							}
						}
						
					}

					//average
					vec3d avg_pt(0.0, 0.0, 0.0);
					for (size_t j = 0; j < corner_in.size(); j++)
					{
						avg_pt = avg_pt + corner_in[j];
					}
					corners[i] = avg_pt / corner_in.size();
				}
			}
		}

		//save_pts_ply("57curve_error2.ply", curves[57]);
		//save_pts_ply((inputprefix + "_corners_final.ply"), corners);

		//0406, add curve validness
		std::vector<bool> curve_validness(curves.size(), true);
		/*curve_validness[3] = false;
		curve_validness[16] = false;*/
		//curve_validness[15] = false;
		//curve_validness[11] = false;


		//save_pts_xyz((inputprefix + "_corners_final.xyz"), corners);
		//save_curves_obj(inputprefix + "_curves_final.obj", curves, flag_curve_close);
		//save_valid_curves_obj(inputprefix + "_curves_final.obj", curves, curve_validness, flag_curve_close);

		//save_all_patches(surf_fitters, patch_type, u_split, v_split, (inputprefix + "_patches_final.obj").c_str());
		//save grid instead
		std::vector<vec3d> all_grid_pts;
		for (size_t i = 0; i < patches.size(); i++)
		{
			all_grid_pts.insert(all_grid_pts.end(), patch_grids[i].begin(), patch_grids[i].end());
		}
		//save_pts_ply(inputprefix + "_patches_final.ply", all_grid_pts);

		//output yml
		//YAML::Node output_node;
		//std::vector<YAML::Node> patch_nodes;
		//output json
		patchj.clear();
		for (size_t i = 0; i < patches.size(); i++)
		{
			json j;
			j["type"] = patch_type_ori[i];
			j["u_dim"] = dim_u_input;
			j["v_dim"] = dim_v_input;
			j["u_closed"] = (bool)patch_close[i];
			j["v_closed"] = false;
			j["with_param"] = false;

			bool flag_grid_valid = false;

			if (patch_type_ori[i] != "Sphere")
			{
				//surf_fitters[i]->get_grid_pts();
				//YAML::Node onepatch_pts = YAML::Load("[]");
				std::vector<double> onepatch_pts;
				if (flag_abandon_oob_spline)
				{
					//check grid_pts
					flag_grid_valid = true;
					for (size_t j = 0; j < patch_grids[i].size(); j++)
					{
						for (size_t k = 0; k < 3; k++)
						{
							if (patch_grids[i][j][k] < -range_obb || patch_grids[i][j][k] > range_obb)
							{
								flag_grid_valid = false;
								break;
							}
						}
						if (!flag_grid_valid)
							break;
					}
					if (!flag_grid_valid)
					{
						std::cout << "out of range patch: " << i << std::endl;
					}
				}
				else
				{
					flag_grid_valid = true;
				}

				if (flag_grid_valid)
				{
					for (size_t j = 0; j < patch_grids[i].size(); j++)
					{
						for (size_t k = 0; k < 3; k++)
						{
							onepatch_pts.push_back(patch_grids[i][j][k]);
						}
					}
					j["grid"] = onepatch_pts;
				}
			}

			if (!flag_grid_valid)
			{
				//for sphere, simply use projection
				if (patches[i].size() != dim_u_output * dim_v_output)
				{
					//input 20, output 40
					surf_fitters[i]->set_points(patches[i]);
					surf_fitters[i]->subdivide_grid(2);
				}
				else
				{
					surf_fitters[i]->set_points(patches[i]);
				}

				surf_fitters[i]->get_self_projection();
				j["u_dim"] = surf_fitters[i]->dim_u_input;
				j["v_dim"] = surf_fitters[i]->dim_v_input;
				//save_pts_ply("sphere.ply", surf_fitters[i]->projection_pts);
				//YAML::Node onepatch_pts = YAML::Load("[]");
				std::vector<double> onepatch_pts;
				for (size_t j = 0; j < surf_fitters[i]->projection_pts.size(); j++)
				{
					for (size_t k = 0; k < 3; k++)
					{
						onepatch_pts.push_back(surf_fitters[i]->projection_pts[j][k]);
					}
				}
				j["grid"] = onepatch_pts;
			}
			
			//parametric part
			if (last_fit_iter != 0 && (patch_type_ori[i] != "BSpline" && patch_type_ori[i] != "Torus"))
			{
				j["with_param"] = true;
				std::vector<double> param(7, 0.0);
				if (patch_type_ori[i] == "Plane")
				{
					PlaneFitter* pf = dynamic_cast<PlaneFitter*>(surf_fitters[i]);
					for (size_t j = 0; j < 3; j++)
					{
						param[j] = pf->zdir[j];
					}
					param[3] = pf->zdir.Dot(pf->loc);
					/*surf_fitters[i]->print_params();
					save_pts_ply("cppplane.ply", patch_grids[i]);*/
					
					//distance evaluation
				}
				else if (patch_type_ori[i] == "Sphere")
				{
					SphereFitter* sf = dynamic_cast<SphereFitter*>(surf_fitters[i]);
					for (size_t j = 0; j < 3; j++)
					{
						param[j] = sf->loc[j];
					}
					param[3] = sf->radius;
				}
				else if (patch_type_ori[i] == "Cylinder")
				{
					CylinderFitter* cf = dynamic_cast<CylinderFitter*>(surf_fitters[i]);
					for (size_t j = 0; j < 3; j++)
					{
						param[j] = cf->zdir[j];
					}
					
					for (size_t j = 0; j < 3; j++)
					{
						param[j + 3] = cf->loc[j];
					}
					param[6] = cf->radius;
				}
				else if (patch_type_ori[i] == "Cone")
				{
					ConeFitter* cf = dynamic_cast<ConeFitter*>(surf_fitters[i]);
					for (size_t j = 0; j < 3; j++)
					{
						param[j] = cf->zdir[j];
					}

					//no matter sfpn or aqd is used, radius should be zero, loc equals to apex
					std::cout << "cone radius: " << cf->radius << std::endl;
					assert(std::abs(cf->radius) < 0.00001);
					for (size_t j = 0; j < 3; j++)
					{
						param[j + 3] = cf->loc[j];
					}
					param[6] = cf->angle;

					//parameter evaluation
					/*std::cout << "cone patch id: " << i << std::endl;
					surf_fitters[i]->print_params();

					save_pts_ply("cppcone.ply", patch_grids[i]);
					std::vector<vec3d> tmp_grids;
					surf_fitters[i]->projection(patch_grids[i], tmp_grids);
					double tmp_dist = 0.0;
					for (size_t j = 0; j < patch_grids[i].size(); j++)
					{
						tmp_dist += (patch_grids[i][j] - tmp_grids[j]).Length();
					}
					tmp_dist /= patch_grids[i].size();
					std::cout << "cone error: " << tmp_dist << std::endl;*/
					
				}

				j["param"] = param;
			}
			
			patchj.push_back(j);
		}

		oj["patches"] = patchj;
		if (save_curve_corner)
		{
			//std::vector<YAML::Node> curve_nodes;
			json curvej;
			for (size_t i = 0; i < curves.size(); i++)
			{
				json onenode;
				onenode["type"] = curve_type[i];
				//YAML::Node onecurve_pts = YAML::Load("[]");
				std::vector<double> onecurve_pts;
				for (size_t j = 0; j < curves[i].size(); j++)
				{
					for (size_t k = 0; k < 3; k++)
					{
						onecurve_pts.push_back(curves[i][j][k]);
					}
				}
				//onenode["closed"] = (bool)curve_fitters[i]->closed;
				onenode["closed"] = (bool)curve_fitters[i]->closed; //change to curve fitter later
				onenode["pts"] = onecurve_pts;
				curvej.push_back(onenode);
			}
			oj["curves"] = curvej;
			json cornerj;
			for (size_t i = 0; i < corners.size(); i++)
			{
				json onenode;
				//YAML::Node onecorner_pts = YAML::Load("[]");
				std::vector<double> onecorner_pts;
				for (size_t k = 0; k < 3; k++)
				{
					onecorner_pts.push_back(corners[i][k]);
				}
				onenode["pts"] = onecorner_pts;
				cornerj.push_back(onenode);
			}
			oj["corners"] = cornerj;
		}

		ofs.open(inputprefix.substr(0, inputprefix.length() - 10) + "geom_refine.json");
		ofs << oj;
		ofs.close();
		return 1;

	}
	catch (const cxxopts::OptionException& e)
	{
		std::cout << "error parsing options: " << e.what() << std::endl;
		exit(1);
	}

	return 0;
}