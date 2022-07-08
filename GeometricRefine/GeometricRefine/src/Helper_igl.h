#pragma once
#include <vector>
#include "Intersector.h"
#include "TinyVector.h"
#include "Helper.h"
#include "Primal_Dual_graph.h"


typedef TinyVector<double, 3> vec3d;

int construct_mesh_intersection(const std::vector<vec3d>& input_pts, const std::vector<std::vector<size_t>>& input_faces, BlackMesh::BlackMesh<double>& output_mesh, std::vector<size_t>& facemap_n2o);

void compute_shortest_dist_AABB(const Eigen::MatrixXd &input_pts , const Eigen::MatrixXi input_faces, const std::vector<vec3d>& queries, std::vector<vec3d>& res, std::vector<double>& dist, std::vector<size_t> &faceid);

void compute_points_to_segs_closest(const std::vector<std::pair<vec3d, vec3d>>& segs, const std::vector<vec3d>& queries, std::vector<vec3d>& closest, std::vector<double>& dist, std::vector<size_t> &closest_id);

void update_comp_flag(const Primal_Dual_graph &pd_graph, std::vector<bool> &flag_keep_comp);

bool reorder_edge(std::vector<std::pair<int, int>>& one_edge);

void extract_curve_corner(const Primal_Dual_graph &pd_graph, const BlackMesh::BlackMesh<double> &m, const std::vector<size_t> &comp2patch, const std::set<std::pair<int, int>> &patch_pairs, std::map<std::pair<int, int> ,std::vector<int>> &pp2curvevid, std::vector<bool> &flag_curve_close, std::vector<int> &corner_vid);

void save_black_mesh_all_patch(const BlackMesh::BlackMesh<double>& output_mesh, const std::string& prefix);
//int construct_mesh_intersection(const std::vector<vec3d>& input_pts, const std::vector<std::vector<size_t>>& input_faces, BlackMesh::BlackMesh<double>& output_mesh);
//int construct_mesh_intersection(const std::vector<vec3d>& input_pts, const std::vector<std::vector<size_t>>& input_faces, std::vector<vec3d> &output_pts, std::vector<std::vector<size_t>> &output_faces, std::vector<size_t>& output_labels);