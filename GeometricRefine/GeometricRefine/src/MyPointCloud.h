#pragma once
#include <ostream>
#include <cmath>
#include <vector>
#include "happly.h"
#include "TinyVector.h"
#include "KD_Tree.h"

typedef TinyVector<double, 3> vec3d;

using ig::KD_Tree;

class MyPointCloud
{
public:
	MyPointCloud();
	~MyPointCloud();
	void set_points(const std::vector<vec3d>& input_pts)
	{
		pts = input_pts;
	}

	void load_ply(const char* fn, bool with_normals = false)
	{
		pts.clear();
		if (kd_tree) delete kd_tree;

		happly::PLYData plyIn(fn);
		std::vector<double> x, y, z;
		x = plyIn.getElement("vertex").getProperty<double>("x");
		y = plyIn.getElement("vertex").getProperty<double>("y");
		z = plyIn.getElement("vertex").getProperty<double>("z");

		for (size_t i = 0; i < x.size(); i++)
		{
			pts.push_back(vec3d(x[i], y[i], z[i]));
		}

		if (with_normals)
		{
			normals.clear();
			std::vector<double> nx, ny, nz;
			nx = plyIn.getElement("vertex").getProperty<double>("nx");
			ny = plyIn.getElement("vertex").getProperty<double>("ny");
			nz = plyIn.getElement("vertex").getProperty<double>("nz");

			for (size_t i = 0; i < nx.size(); i++)
			{
				normals.push_back(vec3d(nx[i], ny[i], nz[i]));
			}
		}
	}
	
	void save_ply(const char* fn)
	{
		//no normals
		assert(pts.size() != 0);
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
	
	void build_kdtree()
	{
		//only build once
		if (kd_tree) return;
		assert(pts.size() != 0);
		kd_tree = new KD_Tree<double, 3>(pts);
	}

	vec3d find_nearest_point(const vec3d& query)
	{
		if (!kd_tree) build_kdtree();
		vec3d nn;
		ptrdiff_t pid;
		kd_tree->find_Nearest_Point(query, nn, &pid);
		return nn;
	}

	void find_nearest_point(const std::vector<vec3d>& queries, std::vector<vec3d>& nns)
	{
		if (!kd_tree) build_kdtree();
		nns.clear();
		nns.resize(queries.size());
		ptrdiff_t pid;
		for (size_t i = 0; i < queries.size(); i++)
		{
			kd_tree->find_Nearest_Point(queries[i], nns[i], &pid);
		}
	}
	
	KD_Tree<double, 3>* kd_tree;
	std::vector<vec3d> pts;
	std::vector<vec3d> normals; //not used
};

MyPointCloud::MyPointCloud()
{
	kd_tree = NULL;
}

MyPointCloud::~MyPointCloud()
{
	if (kd_tree) delete kd_tree;
}