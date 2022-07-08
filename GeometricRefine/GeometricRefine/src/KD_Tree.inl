namespace ig
{
	namespace kdtree_impl
	{
		//////////////////////////////////////////////////////////////
		template <typename T, int Dim>
		KD_Tree_Node<T, Dim>::KD_Tree_Node()
		{
			left = nullptr;
			right = nullptr;
			parent = nullptr;
		}
		//////////////////////////////////////////////////////////////
		template <typename T, int Dim>
		KD_Tree_Node<T, Dim>::~KD_Tree_Node()
		{
			if (left) delete left;
			if (right) delete right;
		}
		//////////////////////////////////////////////////////////////
		template <typename T, int Dim>
		KD_Tree_Node<T, Dim>::KD_Tree_Node(const std::vector<Vector>& vertexList, const std::vector<ptrdiff_t>& face_id_list, int depth, KD_Tree_Node<T, Dim>* parentNode)
		{
			left = nullptr;
			right = nullptr;
			parent = parentNode;

			node_depth = depth;
			//printf("depth=%d\n", depth);
			m_nNumVerts = vertexList.size();
			BuildTree(vertexList, face_id_list, depth);
		}
		//////////////////////////////////////////////////////////////
		template <typename T, int Dim>
		KD_Tree_Node<T, Dim>::KD_Tree_Node(const std::vector<Vector>& vertexList, int depth, KD_Tree_Node<T, Dim>* parentNode)
		{
			left = nullptr;
			right = nullptr;
			parent = parentNode;

			node_depth = depth;
			//printf("depth=%d\n", depth);
			m_nNumVerts = vertexList.size();
			BuildTree(vertexList, depth);
		}
		//////////////////////////////////////////////////////////////
		template <typename T, int Dim>
		size_t KD_Tree_Node<T, Dim>::getnum(std::vector<T>& num_list, std::vector<ptrdiff_t>& seq, ptrdiff_t beg, ptrdiff_t end, int n)
		{
		fun_beg:
			//this function only be called by findnum()
			//simplified quick sort for find k-th largest number in linear time
			if (beg == n && beg == end)
				return seq[beg];
			ptrdiff_t number = end - beg + 1;
			ptrdiff_t random_index = beg + (ptrdiff_t)floor(0.5*number);
			T value = num_list[random_index];
			ptrdiff_t position = seq[random_index];
			num_list[random_index] = num_list[beg];
			seq[random_index] = seq[beg];

			ptrdiff_t cur_beg, cur_end;
			cur_beg = beg;
			cur_end = end;
			do
			{
				while (num_list[cur_end] >= value && cur_end > cur_beg) cur_end--;
				if (cur_end > cur_beg)
				{
					num_list[cur_beg] = num_list[cur_end];
					seq[cur_beg] = seq[cur_end];
				}
				while (num_list[cur_beg] <= value && cur_end > cur_beg) cur_beg++;
				if (cur_end > cur_beg)
				{
					num_list[cur_end] = num_list[cur_beg];
					seq[cur_end] = seq[cur_beg];
				}
			} while (cur_beg < cur_end);

			//num_list[cur_beg] = value;
			//seq[cur_beg] = position;

			if (cur_beg == n)
				return position;
			if (cur_beg > n)
			{
				end = cur_beg - 1;
			}
			//return getnum(num_list, seq, beg, cur_beg - 1, n);
			else
			{
				beg = cur_end + 1;
			}
			//return getnum(num_list, seq, cur_end + 1, end, n);
			goto fun_beg;
			//avoid stack overflow when number > 1e6, and another solution is to use while(1)
		}
		//////////////////////////////////////////////////////////////
		template <typename T, int Dim>
		ptrdiff_t KD_Tree_Node<T, Dim>::findnum(const std::vector<T>& num_list, int n)
		{
			//find k-th largest number in num_list in linear time
			std::vector<ptrdiff_t> seq;
			std::vector<T> num_list_copy;
			seq.resize(num_list.size());
			num_list_copy.resize(num_list.size());
			for (int i = 0; i < num_list.size(); i++)
			{
				seq[i] = i;
				num_list_copy[i] = num_list[i];
			}
			return getnum(num_list_copy, seq, 0, (ptrdiff_t)num_list.size() - 1, n);
		}
		//////////////////////////////////////////////////////////////
		template <typename T, int Dim>
		int KD_Tree_Node<T, Dim>::FindBestAxis(const std::vector<Vector>& vertexList)
		{
			size_t vertexNum = vertexList.size();

			//divide vertexList into two List - pick a better axis
			int iAxis = 0;
			double iAxisResult[3]; //variance

			for (iAxis = 0; iAxis < 3; iAxis++)
			{
				double average = 0;
				double variance = 0;
				for (int i = 0; i < vertexNum; i++)
				{
					average += vertexList[i][iAxis];
				} // vertices
				average /= vertexNum;
				variance = 0;
				for (int i = 0; i < vertexNum; i++)
				{
					variance += (vertexList[i][iAxis] - average)*(vertexList[i][iAxis] - average);
				}

				iAxisResult[iAxis] = variance;
			} //axis

			int index = 0;
			double result = iAxisResult[0];
			for (int i = 1; i < 3; i++)
			{
				if (iAxisResult[i] > result)
				{
					result = iAxisResult[i];
					index = i;
				}
			}

			return index;
		}
		//////////////////////////////////////////////////////////////
		template <typename T, int Dim>
		void KD_Tree_Node<T, Dim>::BuildTree(const std::vector<Vector>& vertexList, const std::vector<ptrdiff_t>& face_id_list, int depth)
		{
			int vertexNum = (int)vertexList.size();

			bool bMakeChildren = false;

			if (vertexNum > 1)
			{
				bMakeChildren = true;
			}
			else if (vertexNum == 1)
			{
				vertex = vertexList[0];
				faceID = face_id_list[0];
			}

			if (bMakeChildren)
			{
				// Find best axis with largest variance
				int iAxis = FindBestAxis(vertexList);
				splitAxis = iAxis;

				std::vector<T> num_list;
				num_list.clear();
				for (int i = 0; i < vertexNum; i++)
					num_list.push_back(vertexList[i][iAxis]);

				ptrdiff_t curNodeVector = findnum(num_list, (int)ceil(vertexNum / 2.0));
				vertex = vertexList[curNodeVector];
				faceID = face_id_list[curNodeVector];

				std::vector<T>().swap(num_list);
				//free space

				Vector center = vertexList[curNodeVector];
				//Log("split: %f\n", fSplit);

				std::vector<Vector> leftSide;
				std::vector<Vector> rightSide;
				std::vector<ptrdiff_t> leftSide_faceID;
				std::vector<ptrdiff_t> rightSide_faceID;

				int leftCount = 0, rightCount = 0; //debug

				for (int i = 0; i < vertexNum; i++)
				{
					if (i == curNodeVector)
						continue;
#ifdef SAME_VERTICES_REDUCTION
					if ((vertex - vertexList[i]).Length() < 1e-8 && face_id_list[i] == faceID)
					{
						continue;
					}
#endif
					if (vertexList[i][iAxis] < center[iAxis])
					{
						leftCount++;
						leftSide.push_back(vertexList[i]);
						leftSide_faceID.push_back(face_id_list[i]);
					}
					else
					{
						rightCount++;
						rightSide.push_back(vertexList[i]);
						rightSide_faceID.push_back(face_id_list[i]);
					}
				}

				if (depth > 1)
				{
					//free space
					//std::vector<Vector>().swap(vertexList);
					//std::vector<int>().swap(face_id_list);
				}

				//vertexNum = leftSide.size() + rightSide.size();
				if (leftSide.size() > 0 || rightSide.size() > 0)
				{
					//Build child nodes
					if (leftSide.size() > 0)
					{
						left = new KD_Tree_Node(leftSide, leftSide_faceID, depth + 1, this);
					}
					if (rightSide.size() > 0)
					{
						right = new KD_Tree_Node(rightSide, rightSide_faceID, depth + 1, this);
					}
				}
				else
				{
					//should never happen
					bMakeChildren = false;
				}
			}
		}
		//////////////////////////////////////////////////////////////
		template <typename T, int Dim>
		void KD_Tree_Node<T, Dim>::BuildTree(const std::vector<Vector>& vertexList, int depth)
		{
			int vertexNum = (int)vertexList.size();

			bool bMakeChildren = false;

			if (vertexNum > 1)
			{
				bMakeChildren = true;
			}
			else if (vertexNum == 1)
			{
				vertex = vertexList[0];
			}

			if (bMakeChildren)
			{
				// Find best axis with largest variance
				int iAxis = FindBestAxis(vertexList);
				splitAxis = iAxis;

				std::vector<T> num_list;
				num_list.clear();
				for (int i = 0; i < vertexNum; i++)
					num_list.push_back((T)vertexList[i][iAxis]);

				ptrdiff_t curNodeVector = findnum(num_list, (int)ceil(vertexNum / (T)2.0));
				vertex = vertexList[curNodeVector];
				Vector center = vertexList[curNodeVector];
				std::vector<T>().swap(num_list);
				//free space

				std::vector<Vector> leftSide;
				std::vector<Vector> rightSide;

				int leftCount = 0, rightCount = 0; //debug

				for (int i = 0; i < vertexNum; i++)
				{
					if (i == curNodeVector)
						continue;
#ifdef SAME_VERTICES_REDUCTION
					if ((vertex - vertexList[i]).Length() < 1e-8)
					{
						continue;
					}
#endif
					if (vertexList[i][iAxis] < center[iAxis])
					{
						leftCount++;
						leftSide.push_back(vertexList[i]);
					}
					else
					{
						rightCount++;
						rightSide.push_back(vertexList[i]);
					}
				}

				if (depth > 1)
				{
					//free space
					//std::vector<Vector>().swap(vertexList);
				}

				//vertexNum = leftSide.size() + rightSide.size();
				if (leftSide.size() > 0 || rightSide.size() > 0)
				{
					//Build child nodes
					if (leftSide.size() > 0)
					{
						left = new KD_Tree_Node(leftSide, depth + 1, this);
					}
					if (rightSide.size() > 0)
					{
						right = new KD_Tree_Node(rightSide, depth + 1, this);
					}
				}
				else
				{
					//should never happen
					bMakeChildren = false;
				}
			}
		}
	}
	//////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////
	template <typename T, int Dim>
	KD_Tree<T, Dim>::KD_Tree(const std::vector<Vector>& vertex_list, const std::vector<ptrdiff_t>& face_id_list)
	{
		tree_root = new kdtree_impl::KD_Tree_Node<T, Dim>(vertex_list, face_id_list, 1, nullptr);
		primitive_query = true;
	}
	//////////////////////////////////////////////////////////////
	template <typename T, int Dim>
	KD_Tree<T, Dim>::KD_Tree(const std::vector<Vector>& vertex_list)
	{
		tree_root = new kdtree_impl::KD_Tree_Node<T, Dim>(vertex_list, 1, nullptr);
		primitive_query = false;
	}
	//////////////////////////////////////////////////////////////
	template <typename T, int Dim>
	KD_Tree<T, Dim>::~KD_Tree()
	{
		if (tree_root) delete tree_root;
	}
	//////////////////////////////////////////////////////////////
	template <typename T, int Dim>
	void KD_Tree<T, Dim>::find_Nearest_Point(const Vector& query, TinyVector<T, Dim>& nearestP, ptrdiff_t* primitive_id)
	{
		kdtree_impl::Search_Info<T, Dim> search_info;
		kdtree_impl::KD_Tree_Node<T, Dim>* cur_leaf_node = locate(tree_root, query, search_info);
		search_info.cur_min_distance.push_back((cur_leaf_node->vertex - query).Length());
		search_info.cur_nearest_vertex.push_back(cur_leaf_node);
		search_for_nearest(cur_leaf_node, query, nullptr, 0, search_info);

		if (primitive_query)
		{
			if (primitive_id)
				*primitive_id = search_info.cur_nearest_vertex[0]->faceID;
			else
				assert(0); // faceid should be provided as input
		}
		nearestP = search_info.cur_nearest_vertex[0]->vertex;
	}
	//////////////////////////////////////////////////////////////
	template <typename T, int Dim>
	void KD_Tree<T, Dim>::find_Nearest_Point_by_traverse(const Vector& query, TinyVector<T, Dim>& nearestP, ptrdiff_t* primitive_id)
	{
		T rst;
		kdtree_impl::KD_Tree_Node<T, Dim>* best = traverse_find_nearest(tree_root, query, rst);
		if (primitive_query && primitive_id != nullptr)
			*primitive_id = best->faceID;
		nearestP = best->vertex;
	}
	//////////////////////////////////////////////////////////////
	template <typename T, int Dim>
	kdtree_impl::KD_Tree_Node<T, Dim>* KD_Tree<T, Dim>::locate(kdtree_impl::KD_Tree_Node<T, Dim>* cur_root, const Vector& query,
		kdtree_impl::Search_Info<T, Dim>& search_info)
	{
		//follow the build tree split rule to locate query point
		//leaf
		if (cur_root->left == nullptr && cur_root->right == nullptr)
		{
			return cur_root;
		}

		Vector vertex = cur_root->vertex;
		int iAxis = cur_root->splitAxis;
		if (query[iAxis] < vertex[iAxis])
		{
			//go left
			if (cur_root->left == nullptr)
				return cur_root;
			return locate(cur_root->left, query, search_info);
		}
		else
		{
			//go right
			if (cur_root->right == nullptr)
				return cur_root;
			return locate(cur_root->right, query, search_info);
		}
	}
	//////////////////////////////////////////////////////////////
	template <typename T, int Dim>
	void KD_Tree<T, Dim>::search_for_nearest(kdtree_impl::KD_Tree_Node<T, Dim>* cur_node, const Vector& query, kdtree_impl::KD_Tree_Node<T, Dim>* root, int branch_num,
		kdtree_impl::Search_Info<T, Dim>& search_info)
	{
		//leafnode
		if (cur_node->left == nullptr && cur_node->right == nullptr)
		{
			if (cur_node->parent != root)
				search_for_nearest(cur_node->parent, query, root, branch_num, search_info);
			return;
		}

		int iAxis = cur_node->splitAxis;
		Vector vertex = cur_node->vertex;
		double dist_to_hyperPlane = fabs(query[iAxis] - vertex[iAxis]);
		bool left;
		if (query[iAxis] < vertex[iAxis])
			left = true;
		else
			left = false;

		if ((vertex - query).Length() < search_info.cur_min_distance[branch_num])
		{
			search_info.cur_min_distance[branch_num] = (vertex - query).Length();
			search_info.cur_nearest_vertex[branch_num] = cur_node;
		}
		search_info.compare_count++;
		if (dist_to_hyperPlane > search_info.cur_min_distance[branch_num] || (left && cur_node->right == nullptr) || (!left && cur_node->left == nullptr))
		{
			//do not need to check other side
#ifdef SIMPLE_ANN
			if (cur_node->parent != root && search_info.compare_count < max_compare_times)
				search_for_nearest(cur_node->parent, query, root, branch_num);
#else
			if (cur_node->parent != root)
				search_for_nearest(cur_node->parent, query, root, branch_num, search_info);
#endif
		}
		else
		{
			if (left)
			{
				if (cur_node->right->m_nNumVerts < vertices_num_threshold)
				{
					T local_rst;
					kdtree_impl::KD_Tree_Node<T, Dim>* best_vertex = traverse_find_nearest(cur_node->right, query, local_rst);
					if (local_rst < search_info.cur_min_distance[branch_num])
					{
						search_info.cur_min_distance[branch_num] = local_rst;
						search_info.cur_nearest_vertex[branch_num] = best_vertex;
					}
					search_info.compare_count++;
				}
				else
				{
					kdtree_impl::KD_Tree_Node<T, Dim>* cur_leaf_node = locate(cur_node->right, query, search_info);
					search_info.available_Branch_Num++;
					int cur_branch = search_info.available_Branch_Num;
					search_info.cur_min_distance.push_back((cur_leaf_node->vertex - query).Length());
					search_info.cur_nearest_vertex.push_back(cur_leaf_node);
					search_for_nearest(cur_leaf_node, query, cur_node, search_info.available_Branch_Num, search_info);
					if (search_info.cur_min_distance[cur_branch] < search_info.cur_min_distance[branch_num])
					{
						search_info.cur_min_distance[branch_num] = search_info.cur_min_distance[cur_branch];
						search_info.cur_nearest_vertex[branch_num] = search_info.cur_nearest_vertex[cur_branch];
					}
					search_info.available_Branch_Num--;
					search_info.cur_min_distance.pop_back();
					search_info.cur_nearest_vertex.pop_back();
				}
			}
			else
			{
				if (cur_node->left->m_nNumVerts < vertices_num_threshold)
				{
					T local_rst;
					kdtree_impl::KD_Tree_Node<T, Dim>* best_vertex = traverse_find_nearest(cur_node->left, query, local_rst);
					if (local_rst < search_info.cur_min_distance[branch_num])
					{
						search_info.cur_min_distance[branch_num] = local_rst;
						search_info.cur_nearest_vertex[branch_num] = best_vertex;
					}
					search_info.compare_count++;
				}
				else
				{
					kdtree_impl::KD_Tree_Node<T, Dim>* cur_leaf_node = locate(cur_node->left, query, search_info);
					search_info.available_Branch_Num++;
					int cur_branch = search_info.available_Branch_Num;
					search_info.cur_min_distance.push_back((cur_leaf_node->vertex - query).Length());
					search_info.cur_nearest_vertex.push_back(cur_leaf_node);
					search_for_nearest(cur_leaf_node, query, cur_node, search_info.available_Branch_Num, search_info);
					if (search_info.cur_min_distance[cur_branch] < search_info.cur_min_distance[branch_num])
					{
						search_info.cur_min_distance[branch_num] = search_info.cur_min_distance[cur_branch];
						search_info.cur_nearest_vertex[branch_num] = search_info.cur_nearest_vertex[cur_branch];
					}
					search_info.available_Branch_Num--;
					search_info.cur_min_distance.pop_back();
					search_info.cur_nearest_vertex.pop_back();
				}
			}
#ifdef SIMPLE_ANN
			if (cur_node->parent != root && search_info.compare_count < max_compare_times)
				search_for_nearest(cur_node->parent, query, root, branch_num, search_info);
#else
			if (cur_node->parent != root)
				search_for_nearest(cur_node->parent, query, root, branch_num, search_info);
#endif
		}
	}
	//////////////////////////////////////////////////////////////
	template <typename T, int Dim>
	kdtree_impl::KD_Tree_Node<T, Dim>* KD_Tree<T, Dim>::traverse_find_nearest(kdtree_impl::KD_Tree_Node<T, Dim>* root, const Vector& query, T& rst)
	{
		//traverse entire sub-tree to find nearest point
		T cur_node_rst;
		kdtree_impl::KD_Tree_Node<T, Dim>* cur_best = root;
		cur_node_rst = (root->vertex - query).Length();

		T local_rst;
		kdtree_impl::KD_Tree_Node<T, Dim>* local_vertex;
		if (root->left != nullptr)
		{
			local_vertex = traverse_find_nearest(root->left, query, local_rst);
			if (local_rst < cur_node_rst)
			{
				cur_node_rst = local_rst;
				cur_best = local_vertex;
			}
		}

		if (root->right != nullptr)
		{
			local_vertex = traverse_find_nearest(root->right, query, local_rst);
			if (local_rst < cur_node_rst)
			{
				cur_node_rst = local_rst;
				cur_best = local_vertex;
			}
		}
		rst = cur_node_rst;
		return cur_best;
	}
}