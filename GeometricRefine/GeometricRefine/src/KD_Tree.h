#pragma once

#include "TinyVector.h"
#include<vector>
#include<cassert>

namespace ig
{
	/****************************************************************************
	This class KD_Tree is for 3d data nearest point query.

	There is two constructors for KD_TREE.

	KD_Tree(const std::vector<Vector>& point_set);
	KD_Tree(const std::vector<Vector>& point_set, const std::vector<ptrdiff_t>& primitive_id_array);

	You can use the latter constructor to attach a id to each point for further processing or other purposes.

	There is only one public function for kd_tree -- to find nearest point of the query point:

	Vectorf find_Nearst_Point(const Vector &query,ptrdiff_t *primitive_id=nullptr);

	If you use the second constructor to construct class, you can pass a int-pointer argument to function to get the primitive_id of the corresponding nearest point.

	*****************************************************************************
	Other options:

	If input data has many repeating items, you can define SAME_VERTICES_REDUCTION, then data items distance less than 1e-8(and with same primitive id)
	will not be inserted into tree repeatedly.

	****************************************************************************/

	//#define SAME_VERTICES_REDUCTION

#define vertices_num_threshold 8
	// When sub-tree node number < vertices_num_threshold, algorithm with traverse the entire sub-tree to find nearest point
	// You can define vertices_num_threshold = 0 or 1 to get original kd-tree implementation

	//#define SIMPLE_ANN
#ifdef SIMPLE_ANN
#define max_compare_times 36
#endif

	namespace kdtree_impl
	{
		template <typename T, int Dim>
		class KD_Tree_Node
		{
			typedef TinyVector<T, Dim> Vector;

		public:

			KD_Tree_Node();

			KD_Tree_Node(const std::vector<Vector>& vertexList, const std::vector<ptrdiff_t>& face_id_list, int depth, KD_Tree_Node<T, Dim>* parentNode);

			KD_Tree_Node(const std::vector<Vector>& vertexList, int depth, KD_Tree_Node<T, Dim>* parentNode);

			~KD_Tree_Node();

		protected:
			size_t getnum(std::vector<T>& num_list, std::vector<ptrdiff_t>& seq, ptrdiff_t beg, ptrdiff_t end, int n);

			ptrdiff_t findnum(const std::vector<T>& num_list, int n);

			int FindBestAxis(const std::vector<Vector>& vertexList);

			void BuildTree(const std::vector<Vector>& vertexList, const std::vector<ptrdiff_t>& face_id_list, int depth);

			void BuildTree(const std::vector<Vector>& vertexList, int depth);

		public:
			Vector vertex; //vertices stored within this node
			ptrdiff_t faceID; //primitive id for AABB query(face id, line segment id, etc)
			KD_Tree_Node<T, Dim>* left, *right, *parent;
			int node_depth;
			int splitAxis;
			size_t m_nNumVerts; //num of vertices stored (Dim->n)
		};

		template <typename T, int Dim>
		class Search_Info
		{
		public:
			Search_Info()
			{
				available_Branch_Num = compare_count = 0;
			}
		public:
			std::vector<T> cur_min_distance;
			std::vector<kdtree_impl::KD_Tree_Node<T, Dim>*> cur_nearest_vertex;
			int available_Branch_Num;
			int compare_count;
		};
	}

	template <typename T, int Dim>
	class KD_Tree
	{
	public:
		typedef T Real;
		typedef TinyVector<T, Dim> Vector;
		KD_Tree(const std::vector<Vector>& vertex_list, const std::vector<ptrdiff_t>& face_id_list);
		KD_Tree(const std::vector<Vector>& vertex_list);
		~KD_Tree();
		void find_Nearest_Point(const Vector& query, Vector& nearestP, ptrdiff_t* primitive_id = nullptr);
		void find_Nearest_Point_by_traverse(const Vector& query, Vector& nearestP, ptrdiff_t* primitive_id = nullptr);

	protected:
		kdtree_impl::KD_Tree_Node<T, Dim>* locate(kdtree_impl::KD_Tree_Node<T, Dim>* cur_root, const Vector& query,
			kdtree_impl::Search_Info<T, Dim>& search_info);
		void search_for_nearest(kdtree_impl::KD_Tree_Node<T, Dim>* cur_node, const Vector& query, kdtree_impl::KD_Tree_Node<T, Dim>* root, int branch_num,
			kdtree_impl::Search_Info<T, Dim>& search_info);
		kdtree_impl::KD_Tree_Node<T, Dim>* traverse_find_nearest(kdtree_impl::KD_Tree_Node<T, Dim>* root, const Vector& query, T& rst);

	private:
		bool primitive_query;
		kdtree_impl::KD_Tree_Node<T, Dim>* tree_root;
	};
}
#include "KD_Tree.inl"