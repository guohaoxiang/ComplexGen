#pragma once
#include <vector>
#include <string>
#include "Eigen/Dense"
#include "TinyVector.h"
#include "TinyMatrix.h"

#define TH_CIRCLE (M_PI / 12.0)

typedef TinyVector<double, 3> vec3d;
using std::string;

//void update_minmax(const std::vector<double>& vs, double& vmin, double& vmax, int split = 12);
double safe_acos(double u);

double safe_asin(double u);

void update_minmax(const std::vector<double>& vs, double& vmin, double& vmax, int split = 20);

void get_pcs_bb(const std::vector<vec3d>& pcs, vec3d& bb_min, vec3d& bb_max);

bool check_patch_uv_order(const std::vector<vec3d>& input, int udim, int vdim, std::vector<vec3d>& output);

vec3d get_tri_normal(const vec3d& p1, const vec3d& p2, const vec3d& p3);

void save_pts_normal_xyz(const std::string& fn, const std::vector<vec3d>& pts, const std::vector<vec3d>& normals);

void save_pts_xyz(const std::string& fn, const std::vector<vec3d>& pts);

vec3d get_nn_xyz(const vec3d& dir1, double max_angle_diff_degree);

void save_pts_ply(const std::string& fn, const std::vector<vec3d>& pts);

void save_pts_ply(const std::string& fn, const std::vector<std::vector<vec3d>>& pts);

void save_valid_curves_obj(const std::string& fn, const std::vector<std::vector<vec3d>>& pts, const std::vector<bool>& curve_valid, const std::vector<bool>& curve_close);

void save_curves_obj(const std::string& fn, const std::vector<std::vector<vec3d>>& pts, const std::vector<bool>& curve_close);

void save_pts_color_ply(const std::string& fn, const std::vector<vec3d>& pts, const std::vector<int>& pts_mask);

void save_obj(const std::string& fn, const std::vector<vec3d>& pts, const std::vector<std::vector<size_t>>& faces);

void save_obj_grouped(const std::string& fn, const std::vector<std::vector<vec3d>>& pts, const std::vector<std::vector<std::vector<size_t>>>& faces);

void split_quad(const std::vector<std::vector<size_t>>& quad_faces, std::vector<std::vector<size_t>>& tri_faces, std::vector<size_t> &face2patch);

void get_vertical_vectors(const vec3d& input, vec3d& u, vec3d& v);

void get_rotmat_from_normal(const vec3d& normal, ColumnMatrix3d& mat);

void get_tri_area(const std::vector<vec3d>& pts, const std::vector<std::vector<size_t>>& faces, std::vector<double>& area);

void get_tri_area_normal(const std::vector<vec3d>& pts, const std::vector<std::vector<size_t>>& faces, std::vector<double>& area, std::vector<vec3d> &normals);

void convert_grid_to_trimesh(const std::vector<vec3d>& pts, int u_split, int v_split, bool u_closed, bool v_closed, std::vector<std::vector<size_t>>& output_faces);

void estimate_normal_from_grid(const std::vector<vec3d>& input_pts, std::vector<vec3d>& input_normals, int xdim, int ydim, bool flag_xclose = false);

void expand_grid_points(const std::vector<vec3d>& input_pts, int dim_u_input, int dim_v_input, double dist, bool u_closed, std::vector<vec3d>& output_pts);

void complex_loader(const char* filename, std::vector<vec3d>& corners, std::vector<std::vector<vec3d>>& curves,
	std::vector< std::vector<vec3d>>& patches, std::vector<string>& curve_type, std::vector<string>& patch_type,
	Eigen::VectorXd& is_curve_closed, Eigen::MatrixXd& curve_corner_corres, Eigen::MatrixXd& patch_curve_corres, std::vector< std::vector<vec3d>>& patch_normals, std::vector<bool>& patch_close);

void complex_loader(const char* filename, std::vector<vec3d>& corners, std::vector<std::vector<vec3d>>& curves,
	std::vector< std::vector<vec3d>>& patches, std::vector<string>& curve_type, std::vector<string>& patch_type,
	Eigen::VectorXd& is_curve_closed, Eigen::MatrixXd& curve_corner_corres, Eigen::MatrixXd& patch_curve_corres, Eigen::MatrixXd& patch_corner_corres, std::vector< std::vector<vec3d>>& patch_normals, std::vector<bool>& patch_close);

bool sample_pts_from_mesh_parametric(const std::vector<TinyVector<double, 3>>& tri_verts, const std::vector<TinyVector<size_t, 3>>& tri_faces, const std::vector<TinyVector<double, 3>>& tri_normals, int n_sample, std::vector<TinyVector<double, 3>>& output_pts, std::vector<TinyVector<double, 3>>& output_normals, std::vector<int>& output_pts_faces);