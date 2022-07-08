#pragma once
#include "BlackMesh.h"
#include "MeshIO.h"

#include "igl/readOBJ.h"
//#undef IGL_STATIC_LIBRARY
#include "igl/copyleft/cgal/mesh_boolean.h"
//#include <igl/opengl/glfw/Viewer.h>

#include <Eigen/Core>
#include <iostream>
#include <vector>


template <class RealI, class RealO>
class Intersector
{
public:
	std::vector<size_t> facemap_n2o;
private:
	BlackMesh::BlackMesh<RealI> *_msh;
	BlackMesh::BlackMesh<RealO> *_mshO;

	typedef Eigen::Matrix<
		RealI,
		Eigen::Dynamic,
		Eigen::Dynamic,
		Eigen::MatrixXd::IsRowMajor> MatrixXRealI;

	typedef Eigen::Matrix<
		RealO,
		Eigen::Dynamic,
		Eigen::Dynamic,
		Eigen::MatrixXd::IsRowMajor> MatrixXRealO;

	//libigl internal use
	MatrixXRealO /*_VA, _VB,*/ _VC;
	Eigen::MatrixXi /*_FA, _FB,*/ _FC;
	Eigen::VectorXi _J;
	MatrixXRealI _VV;
	Eigen::MatrixXi _FF;

	void assemble_data()//;
	{

		std::vector<std::vector<int>> FAr;

		//vtx
		_VV.resize(_msh->GetNumVertices(), 3);
		for (int i = 0; i < _msh->GetNumVertices(); i++) {
			auto vh = _msh->GetVertices()[i];

			Utils::type_conv(vh.pos[0], _VV(i, 0));
			Utils::type_conv(vh.pos[1], _VV(i, 1));
			Utils::type_conv(vh.pos[2], _VV(i, 2));
		}

		//idx
		for (int i = 0; i < _msh->GetNumTriangles(); i++) {
			auto th = _msh->GetTriangles()[i];

			FAr.push_back(th.vertices);

		}

		//pass idx to eigen
		_FF.resize(FAr.size(), 3);

		for (int i = 0; i < FAr.size(); i++) {
			_FF.row(i) = Eigen::Vector3i(FAr[i].data());
		}

	}
	void do_the_union()//;
	{
		typedef typename Eigen::MatrixXd::Scalar Scalar;
		typedef CGAL::Epeck Kernel;
		typedef Kernel::FT ExactScalar;
		typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> MatrixX3S;
		typedef Eigen::Matrix<typename Eigen::VectorXi::Scalar, Eigen::Dynamic, 1> VectorXJ;
		typedef Eigen::Matrix<
			ExactScalar,
			Eigen::Dynamic,
			Eigen::Dynamic,
			Eigen::MatrixXd::IsRowMajor> MatrixXES;
		MatrixXES V;
		Eigen::MatrixXi F;
		VectorXJ  CJ; //the mapping from new faces to old faces
		{
			Eigen::VectorXi I;
			igl::copyleft::cgal::RemeshSelfIntersectionsParam params;
			params.stitch_all = true;
			MatrixXES Vr;
			Eigen::MatrixXi Fr;
			Eigen::MatrixXi IF;
			igl::copyleft::cgal::remesh_self_intersections(
				_VV, _FF, params, Vr, Fr, IF, CJ, I);
			assert(I.size() == Vr.rows());

			// Merge coinciding vertices into non-manifold vertices.
			std::for_each(Fr.data(), Fr.data() + Fr.size(),
				[&I](typename Eigen::MatrixXi::Scalar& a) { a = I[a]; });
			// Remove unreferenced vertices.
			Eigen::VectorXi UIM;
			igl::remove_unreferenced(Vr, Fr, V, _FC, UIM);

			//MatrixX3S Vs;
			igl::copyleft::cgal::assign(V, _VC);
			
			//update facemap_n2o
			assert(Fr.rows() == CJ.rows());
			facemap_n2o.clear();
			for (size_t i = 0; i < CJ.rows(); i++)
			{
				facemap_n2o.push_back(CJ(i));
			}
		}
	}
	void restore_the_mesh()//;
	{
		_mshO->clear();
	for (int i = 0; i < _VC.rows(); i++) {
		_mshO->insert_vtx(std::vector<RealO>{_VC(i, 0), _VC(i, 1), _VC(i, 2)});
	}
	for (int i = 0; i < _FC.rows(); i++) {
		_mshO->insert_face(std::vector<int>{_FC(i, 0), _FC(i, 1), _FC(i, 2)});
	}
	_mshO->update_mesh_properties();
	
}


public:
	Intersector() {}
	~Intersector() {}

	void run(BlackMesh::BlackMesh<RealI> *msh, BlackMesh::BlackMesh<RealO> *mshO)//;
	{
		std::cout << "-----------Resolving Mesh---------------\n";

		_msh = msh;
		_mshO = mshO;
		assemble_data();
		do_the_union();
		restore_the_mesh();
	}

	void output_mesh(const char* filename)//;
	{
		MeshIO::writeOBJ<Real>(filename, *_mshO);
	}


};

