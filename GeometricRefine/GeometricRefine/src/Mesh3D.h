#ifndef MESH3D_H
#define MESH3D_H

#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <map>
#include <iostream>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <queue>
#include <array>
#include <cstdlib>

#include "TinyVector.h"

/** \defgroup MeshCore Mesh data structure */

namespace MeshLib
{
	//declare classes for the compiler
	template <typename Real> class HE_vert;
	template <typename Real> class HE_face;
	template <typename Real> class HE_edge;

	//! vertex class \ingroup MeshCore
	/*!
	*	The basic vertex class for half-edge structure.
	*/
	template <typename Real>
	class HE_vert
	{
	public:
		TinyVector<Real, 3> pos;		//!< 3D coordinate
		HE_edge<Real> *edge;			//!< one of the half-edges_list emanating from the vertex
		TinyVector<Real, 3> normal;	  //!< vertex normal
		ptrdiff_t id;					//!< index
		unsigned int degree;			//!< the degree of vertex
		bool tag;				//!< tag for programming easily
	public:
		//! constructor
		HE_vert(const TinyVector<Real, 3> &v)
			: pos(v), edge(0), id(-1), degree(0), tag(false)
		{
		}
		//destructor
		~HE_vert()
		{
		}
	};

	//! edge class \ingroup MeshCore
	/*!
	*	The basic edge class for half-edge structure.
	*/
	template <typename Real>
	class HE_edge
	{
	public:
		HE_vert<Real> *vert;	//!< vertex at the end of the half-edge
		HE_edge<Real> *pair;	//!< oppositely oriented adjacent half-edge
		HE_face<Real> *face;	//!< face the half-edge borders
		HE_edge<Real> *next;	//!< next half-edge around the face
		HE_edge<Real> *prev;	//!< prev half-edge around the face
		ptrdiff_t id;			//!< index
		bool tag;				//!< tag for programming easily
	public:

		//!constructor
		HE_edge()
			: vert(0), pair(0), face(0), next(0), prev(0), id(-1), tag(false)
		{
		}
		//!destructor
		~HE_edge()
		{
		}

		//! compute the middle point
		inline TinyVector<Real, 3> GetMidPoint()
		{
			return (Real)0.5 * (vert->pos + pair->vert->pos);
		}
	};

	//! face class \ingroup MeshCore
	/*!
	*	The basic face class for half-edge structure.
	*/
	template <typename Real>
	class HE_face
	{
	public:
		HE_edge<Real> *edge;		//!< one of the half-edges_list bordering the face
		unsigned int valence;		//!< the number of edges_list
		TinyVector<Real, 3> normal;	//!< face normal
		int id;					  //!< index
		bool tag;				  //!< tag for programming easily
		std::vector<ptrdiff_t> texture_indices; //! texture indices
		std::vector<ptrdiff_t> normal_indices; //! texture indices
		int groupid;
	public:
		//!constructor
		HE_face()
			: edge(0), id(-1), tag(false), groupid(-1)
		{
		}
		//!destructor
		~HE_face()
		{
		}
		//! compute the barycenter
		inline TinyVector<Real, 3> GetCentroid()
		{
			TinyVector<Real, 3> V(0, 0, 0);
			HE_edge<Real> *he = edge;
			int i = 0;
			do
			{
				V += he->vert->pos;
				he = he->next;
				i++;
			}
			while(he != edge);
			return V / Real(i);
		}
		//! whether texture_indices exists
		bool has_texture_map()
		{
			return (!texture_indices.empty()) && (texture_indices.size() == (size_t)valence);
		}
		//! whether normal_indices exists
		bool has_normal_map()
		{
			return (!normal_indices.empty()) && (normal_indices.size() == (size_t)valence);
		}
	};

	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	bool CompareEdgeID (HE_edge<Real> *he1, HE_edge<Real> *he2 )
	{
		return he1->id < he2->id;
	}

	template <typename Real>
	bool CompareVertexID (HE_vert<Real> *hv1, HE_vert<Real> *hv2 )
	{
		return hv1->id < hv2->id;
	}

	template <typename Real>
	bool CompareFaceID (HE_face<Real> *hf1, HE_face<Real> *hf2 )
	{
		return hf1->id < hf2->id;
	}

	//////////////////////////////////////////////////////////////////////////

	//! Mesh3D class: Half edge data structure \ingroup MeshCore
	/*!
	* a half-edge based mesh data structure
	* For understanding half-edge structure,
	* please read the article in http://www.flipcode.com/articles/article_halfedge.shtml
	*/
	template <typename Real>
	class Mesh3D
	{
	public:

		// type definition
		typedef std::vector<HE_vert<Real>* > VERTEX_LIST;
		typedef std::vector<HE_face<Real>* > FACE_LIST;
		typedef std::vector<HE_edge<Real>* > EDGE_LIST;

		typedef VERTEX_LIST *PTR_VERTEX_LIST;
		typedef FACE_LIST *PTR_FACE_LIST;
		typedef EDGE_LIST *PTR_EDGE_LIST;

		typedef typename VERTEX_LIST::iterator VERTEX_ITER;
		typedef typename FACE_LIST::iterator FACE_ITER;
		typedef typename EDGE_LIST::iterator EDGE_ITER;

		typedef typename VERTEX_LIST::reverse_iterator VERTEX_RITER;
		typedef typename FACE_LIST::reverse_iterator FACE_RITER;
		typedef typename EDGE_LIST::reverse_iterator EDGE_RITER;
		typedef std::pair<HE_vert<Real>*, HE_vert<Real>* > PAIR_VERTEX;

	protected:

		// mesh data

		PTR_VERTEX_LIST vertices_list;		//!< store vertices
		PTR_EDGE_LIST edges_list;			//!< store edges
		PTR_FACE_LIST faces_list;			//!< store faces

		std::vector<TinyVector<float, 2> > texture_array;
		std::vector<TinyVector<float, 3> > normal_array;
		// mesh type

		bool m_closed;						//!< indicate whether the mesh is closed
		bool m_quad;						//!< indicate whether the mesh is quadrilateral
		bool m_tri;							//!< indicate whether the mesh is triangular
		bool m_hex;							//!< indicate whether the mesh is hexagonal
		bool m_pentagon;					//!< indicate whether the mesh is pentagonal

		//! associate two end vertices with its edge: only useful in creating mesh
		std::map<PAIR_VERTEX, HE_edge<Real>*> m_edgemap;

		// mesh info

		int m_num_components;				//!< number of components
		int m_num_boundaries;				//!< number of boundaries
		int m_genus;						//!< the genus value

		bool m_encounter_non_manifold;

	public:

		//! values for the bounding box
		Real xmax, xmin, ymax, ymin, zmax, zmin;

		//! store all the boundary vertices, each vector corresponds to one boundary
		std::vector<std::vector<HE_vert<Real>*> > boundaryvertices;
		std::vector<std::string> groupname;
	public:
		//! constructor
		Mesh3D(void);

		//! destructor
		~Mesh3D(void);

		//! get the pointer of vertices list
		inline PTR_VERTEX_LIST get_vertices_list()
		{
			return vertices_list;
		}

		//! get the pointer of edges list
		inline PTR_EDGE_LIST get_edges_list()
		{
			return edges_list;
		}

		//! get the pointer of faces list
		inline PTR_FACE_LIST get_faces_list()
		{
			return faces_list;
		}

		//! get the total number of vertices
		inline ptrdiff_t get_num_of_vertices()
		{
			return vertices_list ? (ptrdiff_t)vertices_list->size() : 0;
		}

		//! get the total number of faces
		inline ptrdiff_t get_num_of_faces()
		{
			return faces_list ? (ptrdiff_t)faces_list->size() : 0;
		}

		//! get the total number of half-edges
		inline ptrdiff_t get_num_of_edges()
		{
			return edges_list ? (ptrdiff_t)edges_list->size() : 0;
		}

		//! get the pointer of the id-th vertex
		inline HE_vert<Real> *get_vertex(ptrdiff_t id)
		{
			return id >= get_num_of_vertices() || id < 0 ? NULL : (*vertices_list)[id];
		}

		//! get the pointer of the id-th edge
		inline HE_edge<Real> *get_edge(ptrdiff_t id)
		{
			return id >= get_num_of_edges() || id < 0 ? NULL : (*edges_list)[id];
		}

		//! get the pointer of the id-th face
		inline HE_face<Real> *get_face(ptrdiff_t id)
		{
			return id >= get_num_of_faces() || id < 0 ? NULL : (*faces_list)[id];
		}

		//! get the number of components
		inline int get_num_of_components()
		{
			return m_num_components;
		}

		//! get the number of boundaries
		inline int get_num_of_boundaries()
		{
			return m_num_boundaries;
		}

		//! get the genus
		inline int genus()
		{
			return m_genus;
		}

		//! check whether the mesh is valid
		inline bool is_valid()
		{
			if( get_num_of_vertices() == 0 || get_num_of_faces() == 0 )
			{ return false; }
			return true;
		}

		//! check whether the mesh is closed
		inline bool is_closed()
		{
			return m_closed;
		}

		//! check whether the mesh is triangular
		inline bool is_tri()
		{
			return m_tri;
		}

		//! check whether the mesh is quadrilateral
		inline bool is_quad()
		{
			return m_quad;
		}

		//! check whether the mesh is hexgaonal
		inline bool is_hex()
		{
			return m_hex;
		}

		//! check whether the mesh is pentagonal
		inline bool is_pentagon()
		{
			return m_pentagon;
		}

		//! insert a vertex
		/*!
		*	\param v a 3d point
		*	\return a pointer to the created vertex
		*/
		HE_vert<Real> *insert_vertex(const TinyVector<Real, 3> &v);

		//! insert a face
		/*!
		*	\param vec_hv the vertices list of a face
		*	\param texture the pointer of texture vector
		*	\param normal the normal of texture vector
		*	\return a pointer to the created face
		*/
		HE_face<Real> *insert_face(VERTEX_LIST &vec_hv, std::vector<ptrdiff_t> *texture = 0, std::vector<ptrdiff_t> *normal = 0);

		//! check whether the vertex is on border
		bool is_on_boundary(HE_vert<Real> *hv);
		//! check whether the face is on border
		bool is_on_boundary(HE_face<Real> *hf);
		//! check whether the edge is on border
		bool is_on_boundary(HE_edge<Real> *he);

		//FILE IO

		//! load a 3D mesh from an OFF format file
		bool load_off(const char *fins);
		
		//modified by haoxiang on 03/10/2021
		void load_mesh(const std::vector<std::array<double, 3>>& pos, const std::vector<std::vector<size_t>>& indices);
		void load_mesh(const std::vector<TinyVector<double, 3>>& pos, const std::vector<std::vector<size_t>>& indices);

		//! export the current mesh to an OFF format file
		void write_off(const char *fouts);
		//! load a 3D mesh from an OBJ format file
		bool load_obj(const char *fins);
		//! export the current mesh to an OBJ format file
		void write_obj(const char *fouts);
		//! export to a VTK format file
		void write_vtk(const char *fouts);

		//! update mesh:
		/*!
		*	call it when you have created the mesh
		*/
		void update_mesh();

		//! update normal
		/*!
		*	compute all the normals of vertices and faces
		*/
		void update_normal(bool onlyupdate_facenormal = false);

		//! compute the bounding box
		void compute_boundingbox();

		//! get a copy of the current
		Mesh3D<Real> *make_copy();

		//! return a face-orientation-changed mesh
		Mesh3D<Real> *reverse_orientation();

		//! init edge tags
		/*!
		*	for a pair of edges, only one of them is tagged to be true.
		*/
		void init_edge_tag();

		//! reset all the vertices' tag
		void reset_vertices_tag(bool tag_status);
		//! reset all the faces' tag
		void reset_faces_tag(bool tag_status);
		//! reset all the edges' tag
		void reset_edges_tag(bool tag_status);
		//! reset all tag exclude edges' tag2
		void reset_all_tag(bool tag_status);

		//! translate the mesh with tran_V
		void translate(const TinyVector<Real, 3> &tran_V);
		//! scale the mesh
		void scale(Real factorx, Real factory, Real factorz);

		//! check whether there is any non-manifold case
		inline bool is_encounter_nonmanifold()
		{
			return m_encounter_non_manifold;
		}

		//! set texture array
		void set_texture_array(const std::vector< TinyVector<float, 2> > &textures)
		{
			texture_array.assign(textures.begin(), textures.end());
		}
		//! set normal array
		void set_normal_array(const std::vector< TinyVector<float, 3> > &normals)
		{
			normal_array.assign(normals.begin(), normals.end());
		}
		//! get texture array
		std::vector< TinyVector<float, 2> > &get_texture_array()
		{
			return texture_array;
		}
		//! get texture array
		const std::vector< TinyVector<float, 2> > &get_texture_array() const
		{
			return texture_array;
		}
		//! get normal array
		std::vector< TinyVector<float, 3> > &get_normal_array()
		{
			return normal_array;
		}
		//! get normal array
		const std::vector< TinyVector<float, 3> > &get_normal_array() const
		{
			return normal_array;
		}

		//! swap edge
		/*!
		*	\param triedge an edge between two triangular faces
		*/
		void swap_edge(HE_edge<Real> *triedge, bool update_vertex_normal = true);

	private:

		//! insert an edge
		HE_edge<Real> *insert_edge(HE_vert<Real> *vstart, HE_vert<Real> *vend);

		//! clear all the data
		void clear_data();

		//! clear vertices
		void clear_vertices();
		//! clear edges
		void clear_edges();
		//! clear faces
		void clear_faces();

		//! check whether the mesh is closed
		void check_closed();
		//! check the mesh type
		void check_meshtype();

		//! compute all the normals of faces
		void compute_faces_list_normal();
		//! compute the normal of a face
		void compute_perface_normal(HE_face<Real> *hf);
		//! compute all the normals of vertices
		void compute_vertices_list_normal();
		//! compute the normal of a vertex
		void compute_pervertex_normal(HE_vert<Real> *hv);

		//! compute the number of components
		void compute_num_components();
		//! compute the number of boundaries
		void compute_num_boundaries();
		//! compute the genus
		void compute_genus();

		//! handle the boundary half edges specially
		void set_nextedge_for_border_vertices();

		//! remove the vertices which have no connection to others.
		void remove_hanged_vertices();

		//! align edges's id
		/*!
		set mesh's edge(vertex(startid), vertex(endid)->id = edgeid.
		only used in make_copy
		*/
		void copy_edge_id(ptrdiff_t edgeid, ptrdiff_t startid, ptrdiff_t endid, Mesh3D<Real> *mesh);
	};
} //end of namespace

//typedef MeshLib::Mesh3D<float> Mesh3f;
typedef MeshLib::Mesh3D<double> Mesh3d;

#endif //MESH3D_H
