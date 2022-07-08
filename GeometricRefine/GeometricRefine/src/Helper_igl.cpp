#include <iostream>
#include "Helper_igl.h"
#include <igl/AABB.h>
#include <igl/point_mesh_squared_distance.h>

//int construct_mesh_intersection()
int construct_mesh_intersection(const std::vector<vec3d>& input_pts, const std::vector<std::vector<size_t>>& input_faces, BlackMesh::BlackMesh<double>& output_mesh, std::vector<size_t>& facemap_n2o)
{
	BlackMesh::BlackMesh<double>* input_mesh = new BlackMesh::BlackMesh<double>();
	//bool succ = MeshIO::readMesh("tri_mesh.obj", *input_mesh);
	//BlackMesh::BlackMesh<double>* output_mesh = new BlackMesh::BlackMesh<double>();
	/*shadowMesh->construct_from<double>(*input_mesh);*/

	//construct input mesh
	for (size_t i = 0; i < input_pts.size(); i++)
	{
		std::vector<double> tmp;
		tmp.push_back(input_pts[i][0]);
		tmp.push_back(input_pts[i][1]);
		tmp.push_back(input_pts[i][2]);
		input_mesh->insert_vtx(tmp);
	}

	for (size_t i = 0; i < input_faces.size(); i++)
	{
		std::vector<int> oneface;
		for (size_t j = 0; j < input_faces[i].size(); j++)
		{
			oneface.push_back(input_faces[i][j]);
		}
		input_mesh->insert_face(oneface);
	}
	
	input_mesh->update_mesh_properties();

	Intersector<double, double> intsc;
	intsc.run(input_mesh, &output_mesh);

	//MeshIO::writeOBJ("inter_mesh.obj", output_mesh);

	int num_comp = output_mesh.mark_component_with_coherence();
	facemap_n2o = intsc.facemap_n2o;
	//output_mesh->generate_face_color();

	//debug code
	//output_mesh.mNumComponents = input_faces.size();
	//std::vector<size_t> facemap_n2o = intsc.facemap_n2o;
	//for (size_t i = 0; i < facemap_n2o.size(); i++)
	//{
	//	output_mesh.mTriangles[i].component_id = facemap_n2o[i];
	//}

	MeshIO::writeOBJgrouped(output_mesh);

	return num_comp;

}

#include <list>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_segment_primitive.h>

void compute_points_to_segs_closest(const std::vector<std::pair<vec3d, vec3d>>& segs, const std::vector<vec3d>& queries, std::vector<vec3d>& closest_pts, std::vector<double>& dist, std::vector<size_t>& closest_id)
{
	closest_pts.clear();
	dist.clear();
	closest_id.clear();

	typedef CGAL::Simple_cartesian<double> K;
	typedef K::FT FT;
	typedef K::Point_3 Point;
	typedef K::Plane_3 Plane;
	typedef K::Segment_3 Segment;
	typedef K::Triangle_3 Triangle;
	typedef std::vector<Segment>::iterator Iterator;
	typedef CGAL::AABB_segment_primitive<K, Iterator> Primitive;
	typedef CGAL::AABB_traits<K, Primitive> Traits;
	typedef CGAL::AABB_tree<Traits> Tree;
	typedef Tree::Point_and_primitive_id Point_and_primitive_id;

	std::vector<Segment> segments;
	for (size_t i = 0; i < segs.size(); i++)
	{
		Point first(segs[i].first[0], segs[i].first[1], segs[i].first[2]);
		Point second(segs[i].second[0], segs[i].second[1], segs[i].second[2]);
		segments.push_back(Segment(first, second));
	}

	Tree tree(segments.begin(), segments.end());

	for (size_t i = 0; i < queries.size(); i++)
	{
		Point point_query(queries[i][0], queries[i][1], queries[i][2]);
		Point_and_primitive_id pp = tree.closest_point_and_primitive(point_query);
		vec3d p(pp.first[0], pp.first[1], pp.first[2]);
		//size_t id = pp.second;//
		//auto s = *pp.second - segments.begin();
		//Iterator it = pp.second;
		//Iterator begin = segments.begin();
		//size_t id = begin - it;

		closest_pts.push_back(p);
		dist.push_back((p - queries[i]).Length());
		closest_id.push_back(pp.second - segments.begin());
	}
	

	//Point a(1.0, 0.0, 0.0);
	//Point b(0.0, 1.0, 0.0);
	//Point c(0.0, 0.0, 1.0);
	//Point d(0.0, 0.0, 0.0);
	//segments.push_back(Segment(a, b));
	//segments.push_back(Segment(a, c));
	//segments.push_back(Segment(c, d));
	//// constructs the AABB tree and the internal search tree for
	//// efficient distance computations.
	//// counts #intersections with a plane query
	//Plane plane_query(a, b, d);
	//std::cout << tree.number_of_intersected_primitives(plane_query)
	//	<< " intersections(s) with plane" << std::endl;
	//// counts #intersections with a triangle query
	//Triangle triangle_query(a, b, c);
	//std::cout << tree.number_of_intersected_primitives(triangle_query)
	//	<< " intersections(s) with triangle" << std::endl;
	//// computes the closest point from a point query
	//Point point_query(2.0, 2.0, 2.0);
	//Point closest = tree.closest_point(point_query);
	//Point_and_primitive_id pp = tree.closest_point_and_primitive(point_query);
	//
	////std::cerr << "closest point is: " << closest << std::endl;
	//size_t id = pp.second;
}

void AABB_test()
{
	Eigen::MatrixXd V(3, 3);
	Eigen::MatrixXi F(1, 3);
	V << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 , 0.0;
	F << 0, 1, 2;

	Eigen::MatrixXd Q(1, 3);
	Q << 1.0, 1.0, 0.0;
	
	Eigen::VectorXd sqrD;
	Eigen::VectorXi I;
	Eigen::MatrixXd C;
	igl::point_mesh_squared_distance(V, V, F, sqrD, I, C);
	double max_distance = sqrt(sqrD.maxCoeff());
	std::cout << "dist shape: " << sqrD.size() << std::endl;
	igl::point_mesh_squared_distance(Q, V, F, sqrD, I, C);
	max_distance = sqrt(sqrD.maxCoeff());

	std::cout << "dist shape: " << sqrD.size() << std::endl;
}

//void compute_shortest_dist_AABB(const std::vector<vec3d>& input_pts, const std::vector<std::vector<size_t>>& input_faces, const std::vector<vec3d>& queries, std::vector<vec3d>& res, std::vector<double>& dist)
void compute_shortest_dist_AABB(const Eigen::MatrixXd& input_pts, const Eigen::MatrixXi input_faces, const std::vector<vec3d>& queries, std::vector<vec3d>& res, std::vector<double>& dist, std::vector<size_t>& faceid)
{
	res.clear();
	dist.clear();
	faceid.clear();
	Eigen::MatrixXd all_queries(queries.size(), 3);
	//for (size_t i = 0; i < queries.size(); i++)
	for (size_t i = 0; i < queries.size(); i++)
	{
		for (size_t j = 0; j < 3; j++)
		{
			all_queries(i, j) = queries[i][j];
		}
	}

	Eigen::VectorXi I;
	Eigen::MatrixXd C;
	Eigen::VectorXd sqrD;
	//tree.squared_distance(input_pts, input_faces, all_queries, sqrD, I, C);
	igl::point_mesh_squared_distance(all_queries, input_pts, input_faces, sqrD, I, C);
	//igl::point_mesh_squared_distance(input_pts, input_pts, input_faces, sqrD, I, C);
	for (size_t i = 0; i < queries.size(); i++)
	{
		res.push_back(vec3d(C(i, 0), C(i, 1), C(i, 2)));
		dist.push_back(std::sqrt(sqrD(i)));
		faceid.push_back(I(i));
	}
}

void update_comp_flag(const Primal_Dual_graph& pd_graph, std::vector<bool>& flag_keep_comp)
{
	//update flag_keep_comp according to pd_graph
	flag_keep_comp.clear();
	flag_keep_comp.resize(pd_graph.node_list.size(), false);
	for (size_t i = 0; i < pd_graph.node_list.size(); i++)
	{
		if (pd_graph.node_list[i].sub_nodes.empty())
		{
			//no son patch
			if (pd_graph.flag_node_valid[i])
			{
				flag_keep_comp[i] = true;
			}
		}
		else
		{
			if (pd_graph.flag_node_valid[i])
			{
				for (auto cid : pd_graph.node_list[i].sub_nodes)
				{
					flag_keep_comp[cid] = true;
				}
				flag_keep_comp[i] = true;
			}
			else
			{
				for (auto cid : pd_graph.node_list[i].sub_nodes)
				{
					flag_keep_comp[cid] = false;
				}
			}
		}
	}
}

bool reorder_edge(std::vector<std::pair<int, int>>& one_edge)
{
	//return true if edges are close, else return false
	std::map<int, std::vector<int>> vert2edgeid;
	for (int i = 0; i < one_edge.size(); i++)
	{
		vert2edgeid[one_edge[i].first].push_back(i);
		vert2edgeid[one_edge[i].second].push_back(i);
	}


	bool flag_open = false;

	int start = -1;

	if (start == -1)
	{
		//no start
		for (auto vepair : vert2edgeid)
		{
			if (vepair.second.size() == 1)
			{
				start = vepair.first;
				flag_open = true;
				break;
			}
		}
	}

	if (start == -1)
	{
		//close case
		start = one_edge[0].first;
	}

	

	auto findstart = vert2edgeid.find(start);
	assert(findstart != vert2edgeid.end());
	//assert(findstart->second.size() == 1);
	/*if (findstart == vert2edgeid.end() || findstart->second.size() != 1)
	{
		return;
	}*/

	int start_eid = findstart->second[0];

	std::vector<std::pair<int, int>> one_edge_new;
	int cur = start, prev = -1, next = one_edge[start_eid].first;
	next = next == cur ? one_edge[start_eid].second : next;

	std::map<int, int> vert_color;
	for (auto e : one_edge)
	{
		vert_color[e.first] = -1;
		vert_color[e.second] = -1;
	}

	while (next != -1)
	{
		one_edge_new.push_back(std::pair<int, int>(cur, next));
		vert_color[cur] = 1;
		vert_color[next] = 1;
		prev = cur;
		cur = next;
		next = -1;
		//find next
		auto it = vert2edgeid.find(cur);
		assert(it != vert2edgeid.end());
		for (size_t i = 0; i < it->second.size(); i++)
		{
			int eid = it->second[i];
			int diffv = one_edge[eid].first == cur ? one_edge[eid].second : one_edge[eid].first;
			if (diffv != prev && vert_color[diffv] == -1)
			{
				next = diffv;
				break;
			}
		}

	}
	//assert(one_edge_new.size() == one_edge.size());
	one_edge = one_edge_new;
	return flag_open;
}


void extract_curve_corner(const Primal_Dual_graph& pd_graph, const BlackMesh::BlackMesh<double>& m, const std::vector<size_t>& comp2patch, const std::set<std::pair<int, int>>& patch_pairs, std::map<std::pair<int, int>, std::vector<int>>& pp2curvevid, std::vector<bool>& flag_curve_close, std::vector<int>& corner_vid)
{
	//curve_vid.clear();
	pp2curvevid.clear();
	flag_curve_close.clear();
	corner_vid.clear();
	std::map<std::pair<int, int>, std::vector<std::pair<int, int>>> pp2edges;
	const std::vector<bool>& flag_node = pd_graph.flag_node_valid;
	//componnet_id of triangles not changed.
	//init pp2edges
	for (const auto& pp : patch_pairs)
	{
		pp2edges[std::pair<int, int>(pp.first, pp.second)] = std::vector<std::pair<int, int>>();
		pp2curvevid[std::pair<int, int>(pp.first, pp.second)] = std::vector<int>();
	}

	for (const auto& e : m.mEdges)
	{
		std::set<int> valid_patches_set;
		for (const auto& f : e.incident_faces)
		{
			if (flag_node[m.mTriangles[f].component_id])
			{
				valid_patches_set.insert(comp2patch[m.mTriangles[f].component_id]);
			}
		}
		assert(valid_patches_set.size() <= 2);
		if (valid_patches_set.size() == 2)
		{
			std::vector<int> valid_patches(valid_patches_set.begin(), valid_patches_set.end());
			if (valid_patches[0] != valid_patches[1])
			{
				std::pair<int, int> tmppair(std::min(valid_patches[0], valid_patches[1]), std::max(valid_patches[0], valid_patches[1]));
				assert(pp2edges.find(tmppair) != pp2edges.end());
				pp2edges[tmppair].push_back(std::pair<int, int>(e.vertices.first, e.vertices.second));
			}
			
		}
	}

	std::set<int> cornerset;
	for (auto it : pp2edges)
	{
		if (it.second.size() > 0)
		{
			//get edge vertices 
			bool flag_open = reorder_edge(it.second);
			std::vector<int> oneedge;
			for (auto p : it.second)
			{
				oneedge.push_back(p.first);
			}
			oneedge.push_back(it.second.back().second);
			//curve_vid.push_back(oneedge);
			pp2curvevid[it.first] = oneedge;
			flag_curve_close.push_back(!flag_open);
			if (flag_open)
			{
				cornerset.insert(oneedge.front());
				cornerset.insert(oneedge.back());
			}
		}
	}

	corner_vid.assign(cornerset.begin(), cornerset.end());

	return;
}

void save_black_mesh_all_patch(const BlackMesh::BlackMesh<double>& output_mesh, const std::string& prefix)
{
	int ncomp = output_mesh.GetNumComponents();
	std::vector<vec3d> output_mesh_pts;
	for (size_t i = 0; i < output_mesh.GetNumVertices(); i++)
	{
		output_mesh_pts.push_back(vec3d(output_mesh.mVertices[i].pos[0], output_mesh.mVertices[i].pos[1], output_mesh.mVertices[i].pos[2]));
	}
	std::vector<std::vector<size_t>> output_mesh_faces;
	for (size_t i = 0; i < output_mesh.GetNumTriangles(); i++)
	{
		std::vector<size_t> oneface;
		for (size_t j = 0; j < output_mesh.mTriangles[i].vertices.size(); j++)
		{
			oneface.push_back(output_mesh.mTriangles[i].vertices[j]);
		}
		output_mesh_faces.push_back(oneface);
		//output_valid_faces.push_back(oneface);
	}

	std::vector<std::vector<size_t>> comp2faces(ncomp);
	for (size_t i = 0; i < output_mesh.GetNumTriangles(); i++)
	{
		comp2faces[output_mesh.mTriangles[i].component_id].push_back(i);
	}

	for (size_t i = 0; i < ncomp; i++)
	{
		std::vector<std::vector<size_t>> curface;
		for (size_t j = 0; j < comp2faces[i].size(); j++)
		{
			curface.push_back(output_mesh_faces[comp2faces[i][j]]);
		}

		save_obj( prefix + std::to_string(i) + ".obj", output_mesh_pts, curface);
	}
}