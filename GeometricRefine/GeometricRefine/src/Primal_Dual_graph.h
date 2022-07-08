#pragma once
#include <set>
#include "BlackMesh.h"
#include "stdafx.h"


class Primal_Dual_graph {
public:
	struct Node {
		//the index of node should be the same as component id
		std::set<int> incident_edges;//the edge id in this class;
										//int connected_component_id;
										//int id_in_compoennt;
		std::vector<int> sub_nodes; //only used if the node is a merged one
	};

	struct Nonmanifold_edge {
		std::set<int> incident_components;//should follow the order in the Edge.incident_face
		int original_edge_id;
		//int connected_component_id;
		//int degree;
	};

	std::vector<Node> node_list;
	std::vector<Nonmanifold_edge> edge_list;
	std::vector<std::set<int>> node2node;
	std::vector<bool> flag_node_valid; // not considering the connectivity if set to false
	std::vector<bool> flag_edge_valid;
	//std::vector<std::vector<int>> non_manifold_curves;//empty
	//std::vector<vector<int>> connected_component_list_edge;
	//std::vector<vector<int>> connected_component_list_node;//empty

	inline void clear() {
		node_list.clear();
		edge_list.clear();
		//non_manifold_curves.clear();
		//connected_component_list_node.clear();
		//mNumConnectedComponents = 0;
	}

	inline void build_connection_graph(BlackMesh::BlackMesh<double> *_msh)
	{
		if(_msh->GetNumComponents()==0)
		_msh->mark_component_with_coherence();

		node_list.clear();
		node_list.resize(_msh->GetNumComponents());

		_msh->update_edge_manifold_flag();
		for (int i = 0; i < _msh->GetNumEdges(); i++) {
			if (_msh->GetEdges()[i].is_nonmanifold_edge == 1) {
				Primal_Dual_graph::Nonmanifold_edge ed;
				for (int j = 0; j < _msh->GetEdges()[i].incident_faces.size(); j++) {
					int id =
						_msh->GetTriangles()[_msh->GetEdges()[i].incident_faces[j]].component_id;
					ed.incident_components.insert(id);
					node_list[id].incident_edges.insert(edge_list.size());
				}
				ed.original_edge_id = i;
				edge_list.push_back(ed);
			}
		}

		flag_node_valid.clear();
		flag_node_valid.resize(node_list.size(), true);
		flag_edge_valid.clear();
		flag_edge_valid.resize(edge_list.size(), true);
		//define node2node here
		node2node.clear();
		node2node.resize(node_list.size());
		for (int i = 0; i < node_list.size(); i++)
		{
			for (auto eid : node_list[i].incident_edges)
			{
				for (auto en : edge_list[eid].incident_components)
				{
					if (en != i)
					{
						node2node[i].insert(en);
					}
				}
			}
		}
	}

	void del_node(int id)
	{
		flag_node_valid[id] = false;
		//update nb edges
		for (auto eid : node_list[id].incident_edges)
		{
			edge_list[eid].incident_components.erase(id);
		}
		//update node2node
		for (auto fid : node2node[id])
		{
			node2node[fid].erase(id);
		}
	}

	void del_edge(int eid)
	{
		//simply set flag_edge_valid[eid] = false;
		flag_edge_valid[eid] = false;
		//change node_list.incident_edges and node 2 node
		for (auto cid : edge_list[eid].incident_components)
		{
			node_list[cid].incident_edges.erase(eid);
			for (auto ncid : edge_list[eid].incident_components)
			{
				if (ncid != cid)
				{
					node2node[cid].erase(ncid);
				}
			}
		}
	}
};

