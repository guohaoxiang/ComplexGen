#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <math.h>
#include "Helper.h"
#include "TinyMatrix.h"
#include "happly.h"
#define DENOMINATOR_EPS 1e-6
#define epsilon 0.0001

#define TH_PATCH_CLOSE 0.5

double safe_acos(double u)
{
	u = std::min(std::max(u, -0.999999), 0.999999);
	return std::acos(u);
}

double safe_asin(double u)
{
	u = std::min(std::max(u, -0.999999), 0.999999);
	return std::asin(u);
}

void update_minmax(const std::vector<double>& vs, double& vmin, double& vmax, int split)
{
	//split [0, 2 * M_PI] to split number of intervals
	std::vector<int> bins(split, 0);
	double inter = 2 * M_PI / split;
	for (auto v : vs)
	{
		int tmp = std::floor(v / inter);
		while (tmp < 0)
		{
			tmp = tmp + split;
		}
		bins[tmp % split]++;
	}
	bins.insert(bins.end(), bins.begin(), bins.end());
	int longest_inter = -1, start_id = -1, iter = 0;
	while (iter < bins.size())
	{
		if (bins[iter] > 0)
		{
			//get interval
			int tmp_start_id = iter;
			while (iter < bins.size() && bins[iter])
			{
				iter++;
			}
			if (longest_inter == -1 || longest_inter < iter - tmp_start_id)
			{
				start_id = tmp_start_id;
				longest_inter = iter - tmp_start_id;
			}
		}
		else
		{
			iter++;
		}
	}
	//std::cout << "longest inter: " << longest_inter << std::endl;
	if (longest_inter * inter > 2 * M_PI - TH_CIRCLE)
	{
		vmin = 0.0;
		vmax = 2 * M_PI;
	}
	else
	{
		vmin = start_id * inter;
		vmax = (start_id + longest_inter) * inter;
	}
}

void get_pcs_bb(const std::vector<vec3d>& pcs, vec3d& bb_min, vec3d& bb_max)
{
	if (pcs.size() > 0)
	{
		for (size_t i = 0; i < 3; i++)
		{
			bb_min[i] = pcs[0][i];
			bb_max[i] = pcs[0][i];
		}
		
		for (size_t i = 1; i < pcs.size(); i++)
		{
			for (size_t j = 0; j < 3; j++)
			{
				if (bb_min[j] > pcs[i][j])
					bb_min[j] = pcs[i][j];
				if (bb_max[j] < pcs[i][j])
					bb_max[j] = pcs[i][j];
			}
		}
	}
	else
	{
		for (size_t i = 0; i < 3; i++)
		{
			bb_min[i] = 0.0;
			bb_max[i] = 0.0;
		}
	}
	
}

vec3d get_tri_normal(const vec3d& p1, const vec3d& p2, const vec3d& p3)
{
	vec3d diff1 = p2 - p1;
	vec3d diff2 = p3 - p1;
	vec3d normal = diff1.Cross(diff2);
	normal.Normalize();
	return normal;
}

void save_pts_normal_xyz(const std::string& fn, const std::vector<vec3d>& pts, const std::vector<vec3d>& normals)
{
	assert(pts.size() == normals.size());
	std::ofstream ofs(fn);
	for (size_t i = 0; i < pts.size(); i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			ofs << pts[i][j] << " ";
		}
		for (size_t j = 0; j < 3; j++)
		{
			ofs << normals[i][j] << " ";
		}
		ofs << std::endl;
	}

	ofs.close();
}


void save_pts_xyz(const std::string& fn, const std::vector<vec3d>& pts)
{
	std::ofstream ofs(fn);
	for (size_t i = 0; i < pts.size(); i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			ofs << pts[i][j] << " ";
		}
		ofs << std::endl;
	}
	ofs.close();
}

vec3d get_nn_xyz(const vec3d& dir1, double max_angle_diff_degree)
{
	std::vector<vec3d> all_xyzs;
	all_xyzs.push_back(vec3d(1, 0, 0));
	all_xyzs.push_back(vec3d(-1, 0, 0));
	all_xyzs.push_back(vec3d(0, 1, 0));
	all_xyzs.push_back(vec3d(0, -1, 0));
	all_xyzs.push_back(vec3d(0, 0, 1));
	all_xyzs.push_back(vec3d(0, 0, -1));

	double min_angle_diff = -1.0;
	int min_id = -1;
	for (size_t i = 0; i < 6; i++)
	{
		double cosvalue = dir1.Dot(all_xyzs[i]);
		double angle = 180.0 / M_PI * std::acos(cosvalue);
		if (min_angle_diff < 0 || min_angle_diff > angle)
		{
			min_id = i;
			min_angle_diff = angle;
		}
	}
	
	//std::cout << "!!! min angle error: " << min_angle_diff << std::endl;
	if (min_angle_diff < max_angle_diff_degree)
	{
		std::cout << "find nearest xyz axis!" << std::endl;
		return all_xyzs[min_id];
	}
	else
	{
		return dir1;
	}
}

//bool check_patch_uv_order(std::vector<std::vector<vec3d>>& input, std::vector<std::vector<vec3d>>& output)
bool check_patch_uv_order(const std::vector<vec3d>& input, int udim, int vdim, std::vector<vec3d>& output)
{
	output.clear();
	//int udim = input.size(), vdim = input.front().size();
	double u_angle_diff = 0.0, v_angle_diff = 0.0;
	for (int i = 1; i < udim - 1; i++)
	{
		vec3d v0 = input[i * vdim] - input[(i - 1) * vdim];
		vec3d v1 = input[(i + 1) * vdim] - input[i * vdim];
		v0.Normalize();
		v1.Normalize();
		double dot = v0.Dot(v1);
		dot = std::max(-0.999999, std::min(dot, 0.999999));
		u_angle_diff += std::acos(dot);

		v0 = input[i * vdim + vdim - 1] - input[(i - 1) * vdim + vdim - 1];
		v1 = input[(i + 1) * vdim + vdim - 1] - input[i * vdim + vdim - 1];
		v0.Normalize();
		v1.Normalize();
		dot = v0.Dot(v1);
		dot = std::max(-0.999999, std::min(dot, 0.999999));
		u_angle_diff += std::acos(dot);
	}

	for (int i = 1; i < vdim - 1; i++)
	{
		vec3d v0 = input[i] - input[i - 1];
		vec3d v1 = input[i + 1] - input[i];
		v0.Normalize();
		v1.Normalize();
		double dot = v0.Dot(v1);
		dot = std::max(-0.999999, std::min(dot, 0.999999));
		v_angle_diff += std::acos(dot);

		
		v0 = input[(udim - 1) * vdim + i] - input[(udim - 1) * vdim + i - 1];
		v1 = input[(udim - 1) * vdim + i + 1] - input[(udim - 1) * vdim + i];
		v0.Normalize();
		v1.Normalize();
		dot = v0.Dot(v1);
		dot = std::max(-0.999999, std::min(dot, 0.999999));
		v_angle_diff += std::acos(dot);
	}

	u_angle_diff /= 2 * (udim - 2);
	v_angle_diff /= 2 * (vdim - 2);
	//std::cout << "udiff: " << u_angle_diff << " vdiff: " << v_angle_diff << std::endl;
	if (u_angle_diff < v_angle_diff)
	{
		//output.resize(udim * vdim);
		output.clear();
		for (size_t i = 0; i < vdim; i++)
		{
			for (size_t j = 0; j < udim; j++)
			{
				output.push_back(input[j * vdim + i]);
			}
		}
		return false;
	}
	return true;
}

void get_vertical_vectors(const vec3d& normal, vec3d& u, vec3d& v)
{
	//rotate z direction of stadard frame to input
	//normal = normal / normal.L2Norm();
	//double sin_theta = sqrt(1 - normal[2] * normal[2]);
	vec3d xdir(1.0, 0.0, 0.0), ydir(0.0, 1.0, 0.0);

	v = normal.Cross(xdir);
	if (u.Length() < epsilon)
	{
		v = normal.Cross(xdir);
	}
	v.Normalize();
	u = v.Cross(normal);
	
}

void get_tri_area(const std::vector<vec3d>& pts, const std::vector<std::vector<size_t>>& faces, std::vector<double>& area)
{
	area.clear();
	//compute area using heron's formula
	for (size_t i = 0; i < faces.size(); i++)
	{
		double a = (pts[faces[i][0]] - pts[faces[i][1]]).Length();
		double b = (pts[faces[i][0]] - pts[faces[i][2]]).Length();
		double c = (pts[faces[i][2]] - pts[faces[i][1]]).Length();
		double s = (a + b + c) / 2.0;
		double tmp = (s * (s - a) * (s - b) * (s - c));
		if (tmp > 0.0)
			area.push_back(sqrt(tmp));
		else
			area.push_back(0.0);
	}
}

void get_tri_area_normal(const std::vector<vec3d>& pts, const std::vector<std::vector<size_t>>& faces, std::vector<double>& area, std::vector<vec3d>& normals)
{
	area.clear();
	normals.clear();
	//compute area using heron's formula
	for (size_t i = 0; i < faces.size(); i++)
	{
		double a = (pts[faces[i][0]] - pts[faces[i][1]]).Length();
		double b = (pts[faces[i][0]] - pts[faces[i][2]]).Length();
		double c = (pts[faces[i][2]] - pts[faces[i][1]]).Length();
		double s = (a + b + c) / 2.0;
		double tmp = (s * (s - a) * (s - b) * (s - c));
		if (tmp > 0.0)
			area.push_back(sqrt(tmp));
		else
			area.push_back(0.0);
		//area.push_back(sqrt(s * (s - a) * (s - b) * (s - c)));
		vec3d v1 = pts[faces[i][1]] - pts[faces[i][0]];
		vec3d v2 = pts[faces[i][2]] - pts[faces[i][0]];
		vec3d normal = v1.Cross(v2);
		normal.Normalize();
		normals.push_back(normal);
	}
}

void convert_grid_to_trimesh(const std::vector<vec3d>& pts, int u_split, int v_split, bool u_closed, bool v_closed, std::vector<std::vector<size_t>>& faces)
{
	//faces are output
	assert(pts.size() == u_split * v_split);
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

void get_rotmat_from_normal(const vec3d& normal, ColumnMatrix3d& mat)
{
	//get mat so that it map the input normal to z axis
	double sin_theta = sqrt(1 - normal[2] * normal[2]);
	if (abs(sin_theta) < epsilon)
	{
		//std::cout << "vertical normal" << std::endl;
		mat.SetEntries(1, 0, 0, 0, 1, 0, 0, 0, 1, false); //identity
	}
	else
	{
		double theta = safe_acos(normal[2]);

		double phi = acos(std::max(std::min(normal[0] / (sqrt(1 - normal[2] * normal[2])),0.999999),-0.999999));
		double phi_ref = asin(std::max(std::min(normal[1] / (sqrt(1 - normal[2] * normal[2])),0.999999),-0.999999));
		if (phi_ref < 0)
			phi = 2 * M_PI - phi;
		double cos_theta = normal[2];

		ColumnMatrix3d R_z(cos(phi), sin(phi), 0, -sin(phi), cos(phi), 0, 0, 0, 1, false); //set using row format
		ColumnMatrix3d R_y(cos_theta, 0, -sin_theta, 0, 1, 0, sin_theta, 0, cos_theta, false);
		mat = R_y * R_z;
	}
}

void save_obj(const std::string& fn, const std::vector<vec3d>& pts, const std::vector<std::vector<size_t>>& faces)
{
	//a face may contains more than 3 verts
	std::ofstream ofs(fn);
	for (const auto& pt : pts)
	{
		ofs << "v " << pt << std::endl;
	}

	for (size_t i = 0; i < faces.size(); i++)
	{
		ofs << "f ";
		for (size_t j = 0; j < faces[i].size(); j++)
		{
			ofs << faces[i][j] + 1 << " ";
		}
		ofs << std::endl;
	}

	ofs.close();
}

#include <random>
void save_obj_grouped(const std::string& fn, const std::vector<std::vector<vec3d>>& pts, const std::vector<std::vector<std::vector<size_t>>>& faces)
{
	//save material file
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<double> unif_dist(0, 1);
	/*std::ofstream ofs("complexgen.mtl");
	for (size_t i = 0; i < 50; i++)
	{
		ofs << "newmtl m" << i << std::endl;
		ofs << "Kd " << unif_dist(e2) << " " << unif_dist(e2) << " " << unif_dist(e2) << std::endl;
		ofs << "Ka 0 0 0" <<  std::endl;
	}
	ofs.close();*/

	//mtl
	std::ofstream ofs;
	ofs.open(fn);
	ofs << "mtllib complexgen.mtl" << std::endl;
	for (size_t i = 0; i < pts.size(); i++)
	{
		ofs << "g group" << i << std::endl;
		ofs << "usemtl m" << i%50 << std::endl;
		for (const auto& pt : pts[i])
		{
			ofs << "v " << pt << std::endl;
		}

		for (size_t k = 0; k < faces[i].size(); k++)
		{
			ofs << "f ";
			for (size_t j = 0; j < faces[i][k].size(); j++)
			{
				ofs << faces[i][k][j] + 1 << " ";
			}
			ofs << std::endl;
		}
	}

	ofs.close();
	
}

//void split_quad(const std::vector<std::vector<size_t>>& quad_faces, std::vector<std::vector<size_t>>& tri_faces)
void split_quad(const std::vector<std::vector<size_t>>& quad_faces, std::vector<std::vector<size_t>>& tri_faces, std::vector<size_t>& face2patch)
{
	std::vector<size_t> face2patch_new;
	tri_faces.clear();
	for (size_t i = 0; i < quad_faces.size(); i++)
	{
		assert(quad_faces[i].size() == 4);
		tri_faces.push_back(std::vector<size_t>({ quad_faces[i][0], quad_faces[i][1], quad_faces[i][2] }));
		tri_faces.push_back(std::vector<size_t>({ quad_faces[i][0], quad_faces[i][2], quad_faces[i][3] }));
		face2patch_new.push_back(face2patch[i]);
		face2patch_new.push_back(face2patch[i]);
	}

	face2patch = face2patch_new;
}

void save_pts_ply(const std::string& fn, const std::vector<vec3d>& pts)
{
	std::vector<std::array<double, 3>> meshVertexPositions;
	happly::PLYData plyOut;
	for (size_t i = 0; i < pts.size(); i++)
	{
		meshVertexPositions.push_back(std::array<double, 3>{ {pts[i][0], pts[i][1], pts[i][2]}});
	}

	// Add mesh data (elements are created automatically)
	plyOut.addVertexPositions(meshVertexPositions);
	// Write the object to file
	plyOut.write(fn, happly::DataFormat::ASCII);
}

void save_pts_ply(const std::string& fn, const std::vector<std::vector<vec3d>>& pts)
{
	std::vector<vec3d> concate;
	for (size_t i = 0; i < pts.size(); i++)
	{
		concate.insert(concate.end(), pts[i].begin(), pts[i].end());
	}
	save_pts_ply(fn, concate);
}

void save_valid_curves_obj(const std::string& fn, const std::vector<std::vector<vec3d>>& pts, const std::vector<bool>& curve_valid, const std::vector<bool>& curve_close)
{
	assert(pts.size() == curve_close.size());
	int vert_counter = 0;
	std::ofstream ofs(fn);
	for (size_t i = 0; i < pts.size(); i++)
	{
		if (!curve_valid[i])
			continue;
		int cur_counter = 0;
		for (auto& pt : pts[i])
		{
			ofs << "v " << pt << std::endl;
			cur_counter++;
		}

		if (cur_counter > 0)
		{
			////ofs << "l";
			//for (size_t j = 0; j < cur_counter - 1; j++)
			//{
			//	ofs << "l " << j + vert_counter + 1 << " " << j + vert_counter + 2 << std::endl;
			//}
			//if (curve_close[i])
			//{
			//	ofs << "l " << vert_counter + cur_counter << " " << vert_counter + 1 << std::endl;
			//}
			////ofs << std::endl;

			ofs << "l";
			for (size_t j = 0; j < cur_counter; j++)
			{
				ofs << " " << j + vert_counter + 1;
			}
			if (curve_close[i])
			{
				//ofs << "l " << vert_counter + cur_counter << " " << vert_counter + 1 << std::endl;
				ofs << " " << vert_counter + 1;
			}
			ofs << std::endl;
		}

		vert_counter += cur_counter;
	}

	ofs.close();
}


void save_curves_obj(const std::string& fn, const std::vector<std::vector<vec3d>>& pts, const std::vector<bool>& curve_close)
{
	assert(pts.size() == curve_close.size());
	int vert_counter = 0;
	std::ofstream ofs(fn);
	for (size_t i = 0; i < pts.size(); i++)
	{
		int cur_counter = 0;
		for (auto& pt : pts[i])
		{
			ofs << "v " << pt << std::endl;
			cur_counter++;
		}
		
		if (cur_counter > 0)
		{
			////ofs << "l";
			//for (size_t j = 0; j < cur_counter - 1; j++)
			//{
			//	ofs << "l " << j + vert_counter + 1 << " " << j + vert_counter + 2 << std::endl;
			//}
			//if (curve_close[i])
			//{
			//	ofs << "l " << vert_counter + cur_counter << " " << vert_counter + 1 << std::endl;
			//}
			////ofs << std::endl;

			ofs << "l";
			for (size_t j = 0; j < cur_counter; j++)
			{
				ofs << " " << j + vert_counter + 1;
			}
			if (curve_close[i])
			{
				//ofs << "l " << vert_counter + cur_counter << " " << vert_counter + 1 << std::endl;
				ofs << " " << vert_counter + 1;
			}
			ofs << std::endl;
		}

		vert_counter += cur_counter;
	}

	ofs.close();
}

void save_pts_color_ply(const std::string& fn, const std::vector<vec3d>& pts, const std::vector<int>& pts_mask)
{
	if (pts.empty())
	{
		std::cout << "empty point cloud" << std::endl;
		return;
	}

	assert(pts.size() == pts_mask.size());
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<double> unif_dist(0, 1);
	std::vector<std::array<double, 3>> meshVertexPositions;
	std::vector<std::array<double, 3>> meshVertexColors;
	happly::PLYData plyOut;
	//pts color starting from 0
	int n_color = *(std::max_element(pts_mask.begin(), pts_mask.end())) + 1;
	std::vector<std::array<double, 3>> mask2color;
	for (size_t i = 0; i < n_color; i++)
	{
		mask2color.push_back(std::array<double, 3>{ {unif_dist(e2), unif_dist(e2), unif_dist(e2)}});
	}

	for (size_t i = 0; i < pts.size(); i++)
	{
		meshVertexPositions.push_back(std::array<double, 3>{ {pts[i][0], pts[i][1], pts[i][2]}});
		meshVertexColors.push_back(mask2color[pts_mask[i]]);
	}

	// Add mesh data (elements are created automatically)
	plyOut.addVertexPositions(meshVertexPositions);
	plyOut.addVertexColors(meshVertexColors);
	// Write the object to file
	plyOut.write(fn, happly::DataFormat::ASCII);
}

void estimate_normal_from_grid(const std::vector<vec3d>& input_pts, std::vector<vec3d>& input_normals, int xdim, int ydim, bool flag_xclose)
{
	input_normals.clear();
	//int xdim = 10, ydim = 10;
	assert(input_pts.size() == xdim * ydim);
	int dirs[][4] = { {1,0,0,1},{0,1,-1,0},{-1,0,0,-1},{0,-1,1,0} };
	int x1(-1), y1(-1), x2(-1), y2(-1);
	for (int i = 0; i < xdim; i++)
	{
		for (int j = 0; j < ydim; j++)
		{
			const vec3d& curp = input_pts[ydim * i + j];
			std::vector<vec3d> normals;
			for (size_t k = 0; k < 4; k++)
			{
				x1 = i + dirs[k][0];
				y1 = j + dirs[k][1];
				x2 = i + dirs[k][2];
				y2 = j + dirs[k][3];
				if (flag_xclose && x1 == xdim)
				{
					x1 = 0;
				}
				if (flag_xclose && x2 == xdim)
				{
					x2 = 0;
				}

				if (flag_xclose && x1 == -1)
				{
					x1 = xdim - 1;
				}
				
				if (flag_xclose && x2 == -1)
				{
					x2 = xdim - 1;
				}

				if (x1 > -1 && x1 < xdim && y1 > -1 && y1 < ydim && x2 > -1 && x2 < xdim && y2 > -1 && y2 < ydim)
				{
					vec3d dir1 = input_pts[x1 * ydim + y1] - curp;
					vec3d dir2 = input_pts[x2 * ydim + y2] - curp;
					vec3d tmpnormal = dir1.Cross(dir2);
					tmpnormal.Normalize();
					normals.push_back(tmpnormal);
				}
			}
			assert(!normals.empty());
			vec3d sum_normal(0, 0, 0);
			for (size_t k = 0; k < normals.size(); k++)
			{
				sum_normal = sum_normal + normals[k];
			}
			input_normals.push_back(sum_normal / (1.0 * normals.size()));
		}
	}

}

void expand_grid_points(const std::vector<vec3d>& input_pts, int dim_u_input, int dim_v_input, double dist, bool u_closed, std::vector<vec3d>& output_pts)
{
	output_pts.clear();
	std::vector<vec3d> u_neg(dim_u_input), u_pos(dim_u_input);
	for (size_t i = 0; i < dim_u_input; i++)
	{
		vec3d dir_neg = input_pts[i * dim_v_input] - input_pts[i * dim_v_input + 1];
		dir_neg.Normalize();
		u_neg[i] = input_pts[i * dim_v_input] + dist * dir_neg;
		vec3d dir_pos = input_pts[i * dim_v_input + dim_v_input - 1] - input_pts[i * dim_v_input + dim_v_input - 2];
		dir_pos.Normalize();
		u_pos[i] = input_pts[i * dim_v_input + dim_v_input - 1] + dir_pos * dist;
	}

	if (!u_closed)
	{
		std::vector<vec3d> v_neg(dim_v_input), v_pos(dim_v_input);
		for (size_t i = 0; i < dim_v_input; i++)
		{
			vec3d dir_neg = input_pts[i] - input_pts[i + dim_v_input];
			dir_neg.Normalize();
			v_neg[i] = input_pts[i] + dist * dir_neg;
			vec3d dir_pos = input_pts[(dim_u_input - 1) * dim_v_input + i] - input_pts[(dim_u_input - 2) * dim_v_input + i];
			dir_pos.Normalize();
			v_pos[i] = input_pts[(dim_u_input - 1) * dim_v_input + i] + dist * dir_pos;
		}

		//four corners
		vec3d v_neg_neg, v_neg_pos, v_pos_neg, v_pos_pos;
		vec3d dir_neg = v_neg[0] - v_neg[1];
		dir_neg.Normalize();
		v_neg_neg = v_neg[0] + dist * dir_neg;
		vec3d dir_pos = v_neg[dim_v_input - 1] - v_neg[dim_v_input - 2];
		dir_pos.Normalize();
		v_neg_pos = v_neg[dim_v_input - 1] + dist * dir_pos;

		dir_neg = v_pos[0] - v_pos[1];
		dir_neg.Normalize();
		v_pos_neg = v_pos[0] + dist * dir_neg;
		dir_pos = v_pos[dim_v_input - 1] - v_pos[dim_v_input - 2];
		dir_pos.Normalize();
		v_pos_pos = v_pos[dim_v_input - 1] + dist * dir_pos;

		std::vector<vec3d> input_pts_new;
		input_pts_new.push_back(v_neg_neg);
		input_pts_new.insert(input_pts_new.end(), v_neg.begin(), v_neg.end());
		input_pts_new.push_back(v_neg_pos);
		for (size_t i = 0; i < dim_u_input; i++)
		{
			input_pts_new.push_back(u_neg[i]);
			for (size_t j = 0; j < dim_v_input; j++)
			{
				input_pts_new.push_back(input_pts[i * dim_v_input + j]);
			}
			input_pts_new.push_back(u_pos[i]);
		}
		input_pts_new.push_back(v_pos_neg);
		input_pts_new.insert(input_pts_new.end(), v_pos.begin(), v_pos.end());
		input_pts_new.push_back(v_pos_pos);

		//input_pts = input_pts_new;
		output_pts = input_pts_new;

		dim_u_input = dim_u_input + 2;
	}
	else
	{
		/*input_pts.insert(input_pts.begin(), u_neg.begin(), u_neg.end());
		input_pts.insert(input_pts.end(), u_pos.begin(), u_pos.end());*/
		std::vector<vec3d> input_pts_new;
		for (size_t i = 0; i < dim_u_input; i++)
		{
			input_pts_new.push_back(u_neg[i]);
			for (size_t j = 0; j < dim_v_input; j++)
			{
				input_pts_new.push_back(input_pts[i * dim_v_input + j]);
			}
			input_pts_new.push_back(u_pos[i]);
		}
		output_pts = input_pts_new;
	}
}

void complex_loader(const char* filename, std::vector<vec3d>& corners, std::vector<std::vector<vec3d>>& curves,
	std::vector< std::vector<vec3d>>& patches, std::vector<string>& curve_type, std::vector<string>& patch_type,
	Eigen::VectorXd& is_curve_closed, Eigen::MatrixXd& curve_corner_corres, Eigen::MatrixXd& patch_curve_corres, std::vector< std::vector<vec3d>>& patch_normals, std::vector<bool> &patch_close)
{
	FILE* rf = fopen(filename, "r");
	int n_corners, n_curves, n_patches;
	fscanf(rf, "%d%d%d", &n_corners, &n_curves, &n_patches);
	
	int grid_dim = 10;
	if (n_corners == -1)
	{
		grid_dim = 20;
		n_corners = n_curves;
		n_curves = n_patches;
		fscanf(rf, "%d", &n_patches);
	}

	std::cout << "corner: " << n_corners << " curve: " << n_curves << " patch: " << n_patches << std::endl;
	
	corners.resize(n_corners);
	double value3d[3];
	char str[1024];
	double valuei;
	for (int i = 0; i < n_corners; i++)
	{
		fscanf(rf, "%lf%lf%lf", value3d, value3d + 1, value3d + 2);
		corners[i] = vec3d(value3d[0], value3d[1], value3d[2]);
	}

	is_curve_closed.resize(n_curves);
	curve_type.resize(n_curves);
	patch_type.resize(n_patches);
	curves.resize(n_curves);
	for (int i = 0; i < n_curves; i++)
	{
		fscanf(rf, "%s%lf", str, &valuei);
		//myassert(valuei == 0 || valuei == 1);
		is_curve_closed(i) = valuei;
		curve_type[i] = string(str);
		curves[i].resize(34);
		for (int j = 0; j < curves[i].size(); j++)
		{
			fscanf(rf, "%lf%lf%lf", value3d, value3d + 1, value3d + 2);
			curves[i][j] = vec3d(value3d[0], value3d[1], value3d[2]);
		}
	}

	patches.resize(n_patches);
	for (int i = 0; i < n_patches; i++)
	{
		fscanf(rf, "%s", str);
		patch_type[i] = string(str);
		patches[i].resize(grid_dim * grid_dim);
		for (int j = 0; j < patches[i].size(); j++)
		{
			fscanf(rf, "%lf%lf%lf", value3d, value3d + 1, value3d + 2);
			patches[i][j] = vec3d(value3d[0], value3d[1], value3d[2]);
		}
	}

	patch_close.resize(n_patches, false);

	curve_corner_corres.resize(n_curves, n_corners);
	for (int i = 0; i < n_curves; i++)
		for (int j = 0; j < n_corners; j++)
		{
			fscanf(rf, "%lf", value3d);
			curve_corner_corres(i, j) = value3d[0];
		}

	patch_curve_corres.resize(n_patches, n_curves);
	for (int i = 0; i < n_patches; i++)
		for (int j = 0; j < n_curves; j++)
		{
			fscanf(rf, "%lf", value3d);
			patch_curve_corres(i, j) = value3d[0];
		}
	
	bool flag_normal_input = false;
	if (flag_normal_input)
	{
		int flag_with_normal = 0;
		if (fscanf(rf, "%d", &flag_with_normal) == 1)
		{
			patch_normals.resize(n_patches);
			for (int i = 0; i < n_patches; i++)
			{
				fscanf(rf, "%s", str);
				//patch_type[i] = string(str);
				patch_normals[i].resize(grid_dim * grid_dim);
				for (int j = 0; j < patch_normals[i].size(); j++)
				{
					fscanf(rf, "%lf%lf%lf", value3d, value3d + 1, value3d + 2);
					patch_normals[i][j] = vec3d(value3d[0], value3d[1], value3d[2]);
				}
			}
		}
	}

	bool flag_patch_close = true;
	if (flag_patch_close)
	{
		double prob_close;
		for (size_t i = 0; i < n_patches; i++)
		{
			fscanf(rf, "%lf", &prob_close);
			if (prob_close > TH_PATCH_CLOSE)
			{
				patch_close[i] = true;
			}
		}
	}

	fclose(rf);
}

void complex_loader(const char* filename, std::vector<vec3d>& corners, std::vector<std::vector<vec3d>>& curves,
	std::vector< std::vector<vec3d>>& patches, std::vector<string>& curve_type, std::vector<string>& patch_type,
	Eigen::VectorXd& is_curve_closed, Eigen::MatrixXd& curve_corner_corres, Eigen::MatrixXd& patch_curve_corres, Eigen::MatrixXd& patch_corner_corres, std::vector< std::vector<vec3d>>& patch_normals, std::vector<bool>& patch_close)
{
	FILE* rf = fopen(filename, "r");
	int n_corners, n_curves, n_patches;
	fscanf(rf, "%d%d%d", &n_corners, &n_curves, &n_patches);

	int grid_dim = 10;
	if (n_corners == -1)
	{
		grid_dim = 20;
		n_corners = n_curves;
		n_curves = n_patches;
		fscanf(rf, "%d", &n_patches);
	}

	std::cout << "corner: " << n_corners << " curve: " << n_curves << " patch: " << n_patches << std::endl;

	corners.resize(n_corners);
	double value3d[3];
	char str[1024];
	double valuei;
	for (int i = 0; i < n_corners; i++)
	{
		fscanf(rf, "%lf%lf%lf", value3d, value3d + 1, value3d + 2);
		corners[i] = vec3d(value3d[0], value3d[1], value3d[2]);
	}

	is_curve_closed.resize(n_curves);
	curve_type.resize(n_curves);
	patch_type.resize(n_patches);
	curves.resize(n_curves);
	for (int i = 0; i < n_curves; i++)
	{
		fscanf(rf, "%s%lf", str, &valuei);
		//myassert(valuei == 0 || valuei == 1);
		is_curve_closed(i) = valuei;
		curve_type[i] = string(str);
		curves[i].resize(34);
		for (int j = 0; j < curves[i].size(); j++)
		{
			fscanf(rf, "%lf%lf%lf", value3d, value3d + 1, value3d + 2);
			curves[i][j] = vec3d(value3d[0], value3d[1], value3d[2]);
		}
	}

	patches.resize(n_patches);
	for (int i = 0; i < n_patches; i++)
	{
		fscanf(rf, "%s", str);
		patch_type[i] = string(str);
		patches[i].resize(grid_dim * grid_dim);
		for (int j = 0; j < patches[i].size(); j++)
		{
			fscanf(rf, "%lf%lf%lf", value3d, value3d + 1, value3d + 2);
			patches[i][j] = vec3d(value3d[0], value3d[1], value3d[2]);
		}
	}

	patch_close.resize(n_patches, false);

	curve_corner_corres.resize(n_curves, n_corners);
	for (int i = 0; i < n_curves; i++)
		for (int j = 0; j < n_corners; j++)
		{
			fscanf(rf, "%lf", value3d);
			curve_corner_corres(i, j) = value3d[0];
		}

	patch_curve_corres.resize(n_patches, n_curves);
	for (int i = 0; i < n_patches; i++)
		for (int j = 0; j < n_curves; j++)
		{
			fscanf(rf, "%lf", value3d);
			patch_curve_corres(i, j) = value3d[0];
		}

	bool flag_normal_input = false;
	if (flag_normal_input)
	{
		int flag_with_normal = 0;
		if (fscanf(rf, "%d", &flag_with_normal) == 1)
		{
			patch_normals.resize(n_patches);
			for (int i = 0; i < n_patches; i++)
			{
				fscanf(rf, "%s", str);
				//patch_type[i] = string(str);
				patch_normals[i].resize(grid_dim * grid_dim);
				for (int j = 0; j < patch_normals[i].size(); j++)
				{
					fscanf(rf, "%lf%lf%lf", value3d, value3d + 1, value3d + 2);
					patch_normals[i][j] = vec3d(value3d[0], value3d[1], value3d[2]);
				}
			}
		}
	}

	bool flag_patch_close = true;
	if (flag_patch_close)
	{
		double prob_close;
		for (size_t i = 0; i < n_patches; i++)
		{
			fscanf(rf, "%lf", &prob_close);
			if (prob_close > TH_PATCH_CLOSE)
			{
				patch_close[i] = true;
			}
		}
	}

	patch_corner_corres.resize(n_patches, n_corners);
	for (int i = 0; i < n_patches; i++)
		for (int j = 0; j < n_corners; j++)
		{
			fscanf(rf, "%lf", value3d);
			patch_corner_corres(i, j) = value3d[0];
		}

	fclose(rf);
}