#define _CRT_SECURE_NO_WARNINGS

#include "Mesh3D.h"
#include <list>

namespace MeshLib
{
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	Mesh3D<Real>::Mesh3D(void)
	{
		//initialization
		vertices_list = NULL;
		faces_list = NULL;
		edges_list = NULL;

		xmax = ymax = zmax = (Real)1.0;
		xmin = ymin = zmin = (Real) - 1.0;

		m_closed = true;
		m_quad = false;
		m_tri = false;
		m_hex = false;
		m_pentagon = false;

		m_num_components = 0;
		m_num_boundaries = 0;
		m_genus = 0;
		m_encounter_non_manifold = false;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	Mesh3D<Real>::~Mesh3D(void)
	{
		clear_data();
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::clear_data()
	{
		clear_vertices();
		clear_edges();
		clear_faces();
		normal_array.resize(0);
		texture_array.resize(0);
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::clear_vertices()
	{
		if (vertices_list == NULL)
		{ return; }

		for (VERTEX_ITER viter = vertices_list->begin(); viter != vertices_list->end(); viter++)
		{ delete *viter; }

		delete vertices_list;
		vertices_list = NULL;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::clear_edges()
	{
		m_edgemap.clear();
		if (edges_list == NULL)
		{ return; }

		for (EDGE_ITER eiter = edges_list->begin(); eiter != edges_list->end(); eiter++)
		{ delete *eiter; }

		delete edges_list;
		edges_list = NULL;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::clear_faces()
	{
		if (faces_list == NULL)
		{ return; }

		for (FACE_ITER fiter = faces_list->begin(); fiter != faces_list->end(); fiter++)
		{ delete *fiter; }

		delete faces_list;
		faces_list = NULL;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	HE_vert<Real> *Mesh3D<Real>::insert_vertex(const TinyVector<Real, 3> &v)
	{
		HE_vert<Real> *hv = new HE_vert<Real>(v);
		if (vertices_list == NULL)
		{ vertices_list = new VERTEX_LIST; }

		hv->id = (ptrdiff_t)vertices_list->size();
		vertices_list->push_back(hv);

		return hv;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	HE_face<Real> *Mesh3D<Real>::insert_face(VERTEX_LIST &vec_hv, std::vector<ptrdiff_t> *texture, std::vector<ptrdiff_t> *normal)
	{
		int vsize = (int)vec_hv.size();
		if (vsize < 3)
		{ return NULL; }

		//////////////////////////////////////////////////////////////////////////
		//detect non-manifold
		bool b_find = false;
		for (int i = 0; i < vsize; i++)
		{
			HE_edge<Real> *c_he = m_edgemap[PAIR_VERTEX(vec_hv[i], vec_hv[(i + 1) % vsize])];
			if (c_he && c_he->face)
			{
				//detect nonmanifold
				b_find = true;
				break;
			}
		}

		if (b_find)
		{
			m_encounter_non_manifold = true;

			//guess there are faces with reverse orientation, try to reverse them.
			//If these faces are inserted after their neighbor faces which have correct orientation,
			//probably we can obtain correct meshes.
			for (int i = 0; i < vsize; i++)
			{
				HE_edge<Real> *c_he = m_edgemap[PAIR_VERTEX(vec_hv[(i + 1) % vsize], vec_hv[i])];
				if (c_he && c_he->face)
				{
					return NULL;
				}
			}
			//it seems fine, reverse the vector.
			std::reverse(vec_hv.begin(), vec_hv.end());
			if (texture)
			{
				std::reverse(texture->begin(), texture->end());
			}
			if (normal)
			{
				std::reverse(normal->begin(), normal->end());
			}
		}

		//////////////////////////////////////////////////////////////////////////

		if (faces_list == NULL)
		{ faces_list = new FACE_LIST; }

		HE_face<Real> *hf = new HE_face<Real>;
		hf->valence = vsize;
		VERTEX_ITER viter = vec_hv.begin();
		VERTEX_ITER nviter = vec_hv.begin();
		nviter++;

		HE_edge<Real> *he1, *he2;
		std::vector<HE_edge<Real>* > v_he;
		int i;
		for (i = 0; i < vsize - 1; i++)
		{
			he1 = insert_edge( *viter, *nviter);
			he2 = insert_edge( *nviter, *viter);

			if (hf->edge == NULL)
			{ hf->edge = he1; }

			he1->face = hf;
			he1->pair = he2;
			he2->pair = he1;
			v_he.push_back(he1);
			viter++, nviter++;
		}

		nviter = vec_hv.begin();

		he1 = insert_edge(*viter, *nviter);
		he2 = insert_edge(*nviter , *viter);
		he1->face = hf;
		if (hf->edge == NULL)
		{ hf->edge = he1; }

		he1->pair = he2;
		he2->pair = he1;
		v_he.push_back(he1);

		for (i = 0; i < vsize - 1; i++)
		{
			v_he[i]->next = v_he[i + 1];
			v_he[i + 1]->prev = v_he[i];
		}
		v_he[i]->next = v_he[0];
		v_he[0]->prev = v_he[i];

		hf->id = (int)faces_list->size();
		faces_list->push_back(hf);

		if (texture)
		{
			hf->texture_indices.resize(texture->size());
			std::copy(texture->begin(), texture->end(), hf->texture_indices.begin());
		}
		if (normal)
		{
			hf->normal_indices.resize(normal->size());
			std::copy(normal->begin(), normal->end(), hf->normal_indices.begin());
		}

		return hf;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	HE_edge<Real> *Mesh3D<Real>::insert_edge(HE_vert<Real> *vstart, HE_vert<Real> *vend)
	{
		if (vstart == NULL || vend == NULL)
		{
			return NULL;
		}

		if (edges_list == NULL)
		{ edges_list = new EDGE_LIST; }

		if( m_edgemap[PAIR_VERTEX(vstart, vend)] != NULL )
		{ return m_edgemap[PAIR_VERTEX(vstart, vend)]; }

		HE_edge<Real> *he = new HE_edge<Real>;

		he->vert = vend;
		he->vert->degree++;
		vstart->edge = he;
		m_edgemap[PAIR_VERTEX(vstart, vend)] = he;

		he->id = (int)edges_list->size();
		edges_list->push_back(he);

		return he;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::set_nextedge_for_border_vertices()
	{
		if (!is_valid())
		{ return; }

		EDGE_ITER eiter = edges_list->begin();

		for (; eiter != edges_list->end(); eiter++)
		{
			if ((*eiter)->next == NULL && (*eiter)->face == NULL)
			{ (*eiter)->pair->vert->edge = *eiter; }
		}

		for (eiter = edges_list->begin(); eiter != edges_list->end(); eiter++)
		{
			if ( (*eiter)->next == NULL )
			{
				HE_vert<Real> *hv = (*eiter)->vert;
				if (hv->edge != (*eiter)->pair)
				{
					(*eiter)->next = hv->edge;
					hv->edge->prev = *eiter;
				}
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	bool Mesh3D<Real>::is_on_boundary(HE_vert<Real> *hv)
	{
		HE_edge<Real> *edge = hv->edge;
		do
		{
			if (edge == NULL || edge->pair->face == NULL || edge->face == NULL)
			{ return true; }

			edge = edge->pair->next;
		}
		while (edge != hv->edge);

		return false;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	bool Mesh3D<Real>::is_on_boundary(HE_face<Real> *hf)
	{
		HE_edge<Real> *edge = hf->edge;

		do
		{
			if (is_on_boundary(edge))
			{ return true; }
			edge = edge->next;
		}
		while (edge != hf->edge);

		return false;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	bool Mesh3D<Real>::is_on_boundary(HE_edge<Real> *he)
	{
		if(he->face == NULL || he->pair->face == NULL)
		{ return true; }
		return false;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::check_closed()
	{
		m_closed = true;
		VERTEX_ITER viter = vertices_list->begin();
		for (; viter != vertices_list->end(); viter++)
		{
			if (is_on_boundary(*viter))
			{
				m_closed = false;
				return;
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::check_meshtype()
	{
		if (get_num_of_faces() == 0)
		{
			m_tri = false;
			m_quad = false;
			m_hex = false;
			return;
		}

		FACE_ITER fiter = faces_list->begin();
		m_tri = true;
		m_quad = true;
		m_hex = true;
		m_pentagon = true;
		for (; fiter != faces_list->end(); fiter++)
		{
			int d = (*fiter)->valence;
			if (d != 3)
			{
				m_tri = false;
			}
			if (d != 4)
			{
				m_quad = false;
			}
			if (d != 5)
			{
				m_pentagon = false;
			}
			if (d != 6)
			{
				m_hex = false;
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::update_mesh()
	{
		if (!is_valid())
		{ return; }

		//comment on 20200921
		//remove_hanged_vertices();
		set_nextedge_for_border_vertices();
		check_closed();
		check_meshtype();
		m_edgemap.clear();
		update_normal();
		compute_boundingbox();
		//comment on 20200921
		//compute_genus();
	}
	//////////////////////////////////////////////////////////////////////////
	//FILE IO
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	bool Mesh3D<Real>::load_off(const char *fins)
	{
		std::ifstream fin(fins);

		try
		{
			clear_data();

			int vsize, fsize, esize;

			std::string head;

			fin >> head;

			if (head == "OFF")
			{
			}
			else
			{ return false; }

			fin >> vsize >> fsize >> esize;

			Real x, y, z;
			for (int i = 0; i < vsize; i++)
			{
				fin >> x >> y >> z;
				TinyVector<Real, 3> nvv(x, y, z) ;
				insert_vertex(nvv);
			}

			for (int i = 0; i < fsize; i++)
			{
				VERTEX_LIST v_list;
				int valence;
				fin >> valence;

				for (int j = 0; j < valence; j++)
				{
					int id;
					fin >> id;
					HE_vert<Real> *hv = get_vertex(id);

					bool findit = false;
					for (int i = 0; i < (int) v_list.size(); i++)
					{
						if (hv == v_list[i])
						{
							findit = true;
							break;
						}
					}
					if (findit == false && hv != NULL)
					{
						v_list.push_back(hv);
					}
				}

				if ((int)v_list.size() >= 3)
				{
					insert_face(v_list);
				}
			}

			update_mesh();
		}
		catch (...)
		{
			//catch any error
			clear_data();
			xmax = ymax = zmax = (Real)1.0;
			xmin = ymin = zmin = (Real) - 1.0;
			fin.close();
			return false;
		}
		fin.close();
		return is_valid();
	}

	template <typename Real>
	void Mesh3D<Real>::load_mesh(const std::vector<std::array<double, 3>>& pos, const std::vector<std::vector<size_t>>& indices)
	{
		clear_data();

		int vsize, fsize, esize;


		//fin >> vsize >> fsize >> esize;
		vsize = pos.size();
		fsize = indices.size();

		Real x, y, z;
		for (int i = 0; i < vsize; i++)
		{
			//fin >> x >> y >> z;
			x = pos[i][0];
			y = pos[i][1];
			z = pos[i][2];
			TinyVector<Real, 3> nvv(x, y, z);
			insert_vertex(nvv);
		}

		for (int i = 0; i < fsize; i++)
		{
			VERTEX_LIST v_list;
			int valence = indices[i].size(); 
			//fin >> valence;

			for (int j = 0; j < valence; j++)
			{
				int id = indices[i][j];
				HE_vert<Real>* hv = get_vertex(id);

				bool findit = false;
				for (int i = 0; i < (int)v_list.size(); i++)
				{
					if (hv == v_list[i])
					{
						findit = true;
						break;
					}
				}
				if (findit == false && hv != NULL)
				{
					v_list.push_back(hv);
				}
			}

			if ((int)v_list.size() >= 3)
			{
				insert_face(v_list);
			}
		}

		update_mesh();
	}

	template <typename Real>
	void Mesh3D<Real>::load_mesh(const std::vector<TinyVector<double, 3>>& pos, const std::vector<std::vector<size_t>>& indices)
	{
		clear_data();
		int vsize, fsize, esize;

		//fin >> vsize >> fsize >> esize;
		vsize = pos.size();
		fsize = indices.size();

		Real x, y, z;
		for (int i = 0; i < vsize; i++)
		{
			insert_vertex(pos[i]);
		}

		for (int i = 0; i < fsize; i++)
		{
			VERTEX_LIST v_list;
			int valence = indices[i].size();
			//fin >> valence;

			for (int j = 0; j < valence; j++)
			{
				int id = indices[i][j];
				HE_vert<Real>* hv = get_vertex(id);

				bool findit = false;
				for (int i = 0; i < (int)v_list.size(); i++)
				{
					if (hv == v_list[i])
					{
						findit = true;
						break;
					}
				}
				if (findit == false && hv != NULL)
				{
					v_list.push_back(hv);
				}
			}

			if ((int)v_list.size() >= 3)
			{
				insert_face(v_list);
			}
		}

		update_mesh();
	}


	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::write_off(const char *fouts)
	{
		std::ofstream fout(fouts);
		fout.precision(16);
		fout << "OFF\n";
		//output the number of vertices_list, faces_list-> edges_list
		fout << (int)vertices_list->size() << " " << (int)faces_list->size() << " " << (int)edges_list->size() / 2 << "\n";

		//output coordinates of each vertex
		VERTEX_ITER viter = vertices_list->begin();
		for (; viter != vertices_list->end(); viter++)
		{
			fout << std::scientific << (*viter)->pos << "\n";
		}

		//output the valence of each face and its vertices_list' id

		FACE_ITER fiter = faces_list->begin();

		for (; fiter != faces_list->end(); fiter++)
		{
			fout << (*fiter)->valence;

			HE_edge<Real> *edge = (*fiter)->edge;

			do
			{
				fout << " " << edge->pair->vert->id;
				edge = edge->next;
			}
			while (edge != (*fiter)->edge);
			fout << "\n";
		}

		fout.close();
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	bool Mesh3D<Real>::load_obj(const char *fins)
	{
		FILE *m_pFile = fopen(fins, "r");

		char *tok;
		char temp[128];
		groupname.clear();
		std::vector<int> group_idlist;
		try
		{
			clear_data();
			//read vertices
			fseek(m_pFile, 0, SEEK_SET);
			char pLine[512];
			while(fgets(pLine, 512, m_pFile))
			{
				if(pLine[0] == 'v' && pLine[1] == ' ')
				{
					TinyVector<Real, 3> nvv;
					tok = strtok(pLine, " ");
					for (int i = 0; i < 3; i++)
					{
						tok = strtok(NULL, " ");
						strcpy(temp, tok);
						temp[strcspn(temp, " ")] = 0;
						nvv[i] = (Real)atof(temp);
					}
					insert_vertex(nvv);
				}
				else if(pLine[0] == 'v' && pLine[1] == 't')
				{
					TinyVector<float, 2> nvv;
					tok = strtok(pLine, " ");
					for (int i = 0; i < 2; i++)
					{
						tok = strtok(NULL, " ");
						strcpy(temp, tok);
						temp[strcspn(temp, " ")] = 0;
						nvv[i] = (float)atof(temp);
					}
					texture_array.push_back(nvv);
				}
				else if(pLine[0] == 'v' && pLine[1] == 'n')
				{
					TinyVector<float, 3> nvv;
					tok = strtok(pLine, " ");
					for (int i = 0; i < 3; i++)
					{
						tok = strtok(NULL, " ");
						strcpy(temp, tok);
						temp[strcspn(temp, " ")] = 0;
						nvv[i] = (float)atof(temp);
					}
					normal_array.push_back(nvv);
				}
				else if (pLine[0] == 'g')
				{
					tok = strtok(&pLine[1], " ");
					strcpy(temp, tok);
					std::string gname(temp, strlen(temp) - 1);
					bool find = false;
					for (size_t i = 0; i < groupname.size(); i++)
					{
						if (gname.compare(groupname[i]) == 0)
						{
							group_idlist.push_back((int)i);
							find = true;
							break;
						}
					}
					if (!find)
					{
						group_idlist.push_back((int)groupname.size());
						groupname.push_back(gname);
					}
				}
			}

			//read facets

			fseek(m_pFile, 0, SEEK_SET);

			int group_counter = -1;

			while(fgets(pLine, 512, m_pFile))
			{
				char *pTmp = pLine;
				if(pTmp[0] == 'f')
				{
					VERTEX_LIST s_faceid;
					std::vector<ptrdiff_t> normal_ind, texture_ind;
					tok = strtok(pLine, " ");
					while ((tok = strtok(NULL, " ")) != NULL)
					{
						strcpy(temp, tok);
						size_t len = strlen(temp);
						int start_pos = 0;
						int pos[2] = { -1, -1};
						for (int k = 0; k < (int)len; k++)
						{
							if (temp[k] == '/')
							{
								pos[start_pos] = k;
								start_pos++;
							}
						}
						size_t end_pos = len;
						if (start_pos == 1)
						{
							end_pos = pos[0];
							std::string mstr(&temp[pos[0] + 1], len - pos[0] - 1);
							int id = (int)strtol(mstr.c_str(), NULL, 10) - 1;
							texture_ind.push_back(id);
						}
						else if (start_pos == 2)
						{
							end_pos = pos[0];
							if (pos[0] + 1 != pos[1])
							{
								std::string mstr(&temp[pos[0] + 1], pos[1] - pos[0] - 1);
								int id = (int)strtol(mstr.c_str(), NULL, 10) - 1;
								texture_ind.push_back(id);
							}
							std::string mstr2(&temp[pos[1] + 1], len - pos[1] - 1);
							int id2 = (int)strtol(mstr2.c_str(), NULL, 10) - 1;
							normal_ind.push_back(id2);
						}
						std::string mstr(&temp[0], end_pos);
						int id = (int)strtol(mstr.c_str(), NULL, 10) - 1;
						HE_vert<Real> *hv = get_vertex(id);
						bool findit = false;
						for (int i = 0; i < (int) s_faceid.size(); i++)
						{
							if (hv == s_faceid[i])	//remove the redundant vertex id if it exists
							{
								findit = true;
								break;
							}
						}
						if (findit == false && hv != NULL)
						{
							s_faceid.push_back(hv);
						}
					}
					if (s_faceid.size() >= 3)
					{
						HE_face<Real> *hf = insert_face(s_faceid, texture_ind.size() == s_faceid.size() ? &texture_ind : 0, normal_ind.size() == s_faceid.size() ? &normal_ind : 0);
						if (hf)
							hf->groupid = group_counter == -1 ? -1 : group_idlist[group_counter];
					}
				}
				else if(pTmp[0] == 'g')
				{
					group_counter++;
				}
			}
			update_mesh();
		}
		catch (...)
		{
			clear_data();
			xmax = ymax = zmax = (Real)1.0;
			xmin = ymin = zmin = (Real) - 1.0;

			fclose(m_pFile);
			return false;
		}

		fclose(m_pFile);
		return is_valid();
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::write_obj(const char *fouts)
	{
		std::ofstream fout(fouts);

		if (groupname.empty())
		{ fout << "g object\n"; }
		//fout.precision(16);
		//output coordinates of each vertex
		VERTEX_ITER viter = vertices_list->begin();
		std::map<ptrdiff_t, ptrdiff_t> vmap;
		ptrdiff_t count = 1;
		for (; viter != vertices_list->end(); viter++)
		{
			fout << "v "/*<< std::scientific*/ << (*viter)->pos << "\n";
			vmap[(*viter)->id] = count;
			count++;
		}

		if (!normal_array.empty())
		{
			for (size_t i = 0; i < normal_array.size(); i++)
			{
				fout << "vn "/*<< std::scientific*/ << normal_array[i] << "\n";
			}
		}
		else
		{
			for (viter = vertices_list->begin(); viter != vertices_list->end(); viter++)
			{
				fout << "vn "/*<< std::scientific*/ << (*viter)->normal << "\n";
			}
		}

		if (!texture_array.empty())
		{
			for (size_t i = 0; i < texture_array.size(); i++)
			{
				fout << "vt " << std::scientific << texture_array[i] << "\n";
			}
		}

		//output the valence of each face and its vertices_list' id

		int last_groupid = -1;

		FACE_ITER fiter = faces_list->begin();

		for (; fiter != faces_list->end(); fiter++)
		{
			if ((*fiter)->groupid != -1)
			{
				if (last_groupid != (*fiter)->groupid )
				{
					last_groupid = (*fiter)->groupid;
					fout << "g " << groupname[last_groupid] << std::endl;
				}
			}

			fout << "f";

			HE_edge<Real> *edge = (*fiter)->edge;
			unsigned int count = 0;
			do
			{
				fout << " " << vmap[edge->pair->vert->id];
				if ((*fiter)->has_texture_map())
				{
					fout << "/" << (*fiter)->texture_indices[count] + 1;
				}
				else
				{ fout << "/"; }

				if ((*fiter)->has_normal_map())
				{
					fout << "/" << (*fiter)->normal_indices[count] + 1;
				}
				else
				{ fout << "/" << edge->pair->vert->id + 1; }

				edge = edge->next;
			}
			while (edge != (*fiter)->edge);
			fout << "\n";
		}
		fout.close();
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::write_vtk(const char *fouts)
	{
		std::ofstream fout(fouts);
		fout.precision(16);
		fout << "# vtk DataFile Version 3.0\n"
			<< "Polygonal Mesh\n"
			<< "ASCII\n"
			<< "DATASET POLYDATA\n"
			<< "POINTS " << vertices_list->size() << " double\n";

		//output coordinates of each vertex
		VERTEX_ITER viter = vertices_list->begin();
		for (; viter != vertices_list->end(); viter++)
		{
			fout << std::scientific << (*viter)->pos << "\n";
		}

		FACE_ITER fiter = faces_list->begin();
		size_t count = 0;
		for (; fiter != faces_list->end(); fiter++)
		{
			count += (*fiter)->valence + 1;
		}

		fout << "POLYGONS " << faces_list->size() << " " << count << "\n";

		//output the valence of each face and its vertices_list' id

		fiter = faces_list->begin();

		for (; fiter != faces_list->end(); fiter++)
		{
			fout << (*fiter)->valence;

			HE_edge<Real> *edge = (*fiter)->edge;

			do
			{
				fout << " " << edge->pair->vert->id;
				edge = edge->next;
			}
			while (edge != (*fiter)->edge);
			fout << "\n";
		}

		fout.close();
	}
	//////////////////////////////////////////////////////////////////////////
	//For rendering
	template <typename Real>
	void Mesh3D<Real>::compute_boundingbox()
	{
		if (vertices_list->size() < 3)
		{ return; }

		xmax = ymax = zmax = (Real) - 10e10;
		xmin = ymin = zmin = (Real)10e10;

		VERTEX_ITER viter = vertices_list->begin();
		for (; viter != vertices_list->end(); viter++)
		{
			xmin = (*viter)->pos[0] < xmin ? (*viter)->pos[0] : xmin;
			ymin = (*viter)->pos[1] < ymin ? (*viter)->pos[1] : ymin;
			zmin = (*viter)->pos[2] < zmin ? (*viter)->pos[2] : zmin;
			xmax = (*viter)->pos[0] > xmax ? (*viter)->pos[0] : xmax;
			ymax = (*viter)->pos[1] > ymax ? (*viter)->pos[1] : ymax;
			zmax = (*viter)->pos[2] > zmax ? (*viter)->pos[2] : zmax;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::compute_faces_list_normal()
	{
		for (FACE_ITER fiter = faces_list->begin(); fiter != faces_list->end(); fiter++)
		{
			compute_perface_normal(*fiter);
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::compute_perface_normal(HE_face<Real> *hf)
	{
		size_t i = 0;
		HE_edge<Real> *pedge = hf->edge;
		HE_edge<Real> *nedge = hf->edge->next;

		hf->normal = TinyVector<Real, 3>(0, 0, 0);
		for (i = 0; i < hf->valence; i++)
		{
			//cross product
			HE_vert<Real> *p = pedge->vert;
			HE_vert<Real> *c = pedge->next->vert;
			HE_vert<Real> *n = nedge->next->vert;
			TinyVector<Real, 3> pc, nc;
			pc = p->pos - c->pos;
			nc = n->pos - c->pos;

			hf->normal -= pc.Cross(nc);
			pedge = nedge;
			nedge = nedge->next;
			if (hf->valence == 3)
			{
				break;
			}
		}
		hf->normal.Normalize();
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::compute_vertices_list_normal()
	{
		VERTEX_ITER viter = vertices_list->begin();

		for (; viter != vertices_list->end(); viter++)
		{
			compute_pervertex_normal(*viter);
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::compute_pervertex_normal(HE_vert<Real> *hv)
	{
		HE_edge<Real> *edge = hv->edge;

		if (edge == NULL)
		{
			hv->normal = TinyVector<Real, 3>(0, 0, 0);
			return;
		}
		hv->normal = TinyVector<Real, 3>(0, 0, 0);

		do
		{
			if (edge->face != NULL)
			{
				hv->normal += edge->face->normal;
			}
			edge = edge->pair->next;
		}
		while (edge != hv->edge);

		hv->normal.Normalize();
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::update_normal(bool onlyupdate_facenormal)
	{
		compute_faces_list_normal();
		if (onlyupdate_facenormal == false)
		{
			compute_vertices_list_normal();
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::copy_edge_id(ptrdiff_t edgeid, ptrdiff_t startid, ptrdiff_t endid, Mesh3D<Real> *mesh)
	{
		mesh->m_edgemap[PAIR_VERTEX( mesh->get_vertex(startid), mesh->get_vertex(endid))]->id = edgeid;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	Mesh3D<Real> *Mesh3D<Real>::make_copy()
	{
		Mesh3D<Real> *new_mesh = new Mesh3D<Real>;

		VERTEX_ITER viter = vertices_list->begin();
		for (; viter != vertices_list->end(); viter++)
		{
			new_mesh->insert_vertex((*viter)->pos);
		}

		FACE_ITER fiter = faces_list->begin();

		for (; fiter != faces_list->end(); fiter++)
		{
			HE_face<Real> *hf = *fiter;
			HE_edge<Real> *edge = hf->edge;
			VERTEX_LIST mvlist;
			do
			{
				mvlist.push_back(new_mesh->get_vertex(edge->pair->vert->id));
				edge = edge->next;
			}
			while(edge != hf->edge);

			new_mesh->insert_face(mvlist, hf->texture_indices.empty() ? 0 : &hf->texture_indices, hf->normal_indices.empty() ? 0 : &hf->normal_indices);
		}

		for (EDGE_ITER eiter = edges_list->begin(); eiter != edges_list->end(); eiter++)
		{
			HE_edge<Real> *he = *eiter;
			copy_edge_id(he->id, he->pair->vert->id, he->vert->id, new_mesh);
		}

		PTR_EDGE_LIST edgelist = new_mesh->get_edges_list();

		std::sort(edgelist->begin(), edgelist->end(), CompareEdgeID<Real>);

		new_mesh->update_mesh();

		VERTEX_ITER cviter = new_mesh->get_vertices_list()->begin();
		for (viter = vertices_list->begin(); viter != vertices_list->end(); viter++, cviter++)
		{
			(*cviter)->edge = new_mesh->get_edge((*viter)->edge->id);
		}

		FACE_ITER cfiter = new_mesh->get_faces_list()->begin();
		for (fiter = faces_list->begin(); fiter != faces_list->end(); fiter++ , cfiter++)
		{
			(*cfiter)->edge = new_mesh->get_edge((*fiter)->edge->id);
		}

		new_mesh->set_normal_array(normal_array);
		new_mesh->set_texture_array(texture_array);

		return new_mesh;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	Mesh3D<Real> *Mesh3D<Real>::reverse_orientation()
	{
		Mesh3D<Real> *new_mesh = new Mesh3D<Real>;

		VERTEX_ITER viter = vertices_list->begin();
		for (; viter != vertices_list->end(); viter++)
		{
			new_mesh->insert_vertex((*viter)->pos);
		}

		FACE_ITER fiter = faces_list->begin();

		for (; fiter != faces_list->end(); fiter++)
		{
			HE_face<Real> *hf = *fiter;
			HE_edge<Real> *edge = hf->edge;
			VERTEX_LIST mvlist;
			do
			{
				mvlist.push_back(new_mesh->get_vertex(edge->pair->vert->id));
				edge = edge->next;
			}
			while(edge != hf->edge);
			std::reverse(mvlist.begin(), mvlist.end());
			new_mesh->insert_face(mvlist);
		}

		new_mesh->update_mesh();
		return new_mesh;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::compute_num_components()
	{
		FACE_ITER fiter = faces_list->begin();
		for (; fiter != faces_list->end(); fiter++ )
		{
			(*fiter)->tag = false;
		}

		m_num_components = 0;

		for (fiter = faces_list->begin(); fiter != faces_list->end(); fiter++ )
		{
			HE_face<Real> *hf = *fiter;
			if (hf->tag == false)
			{
				m_num_components++;

				std::queue<HE_face<Real>* > facets;
				facets.push(hf);
				hf->tag = true;

				while (!facets.empty())
				{
					HE_face<Real> *pFacet = facets.front();
					facets.pop();
					pFacet->tag = true;

					HE_edge<Real> *he = pFacet->edge;
					do
					{
						if(he->pair->face != NULL && he->pair->face->tag == false)
						{
							facets.push(he->pair->face);
							he->pair->face->tag = true;

							HE_edge<Real> *vhe = he->vert->edge;

							do
							{
								if (vhe->face && vhe->face->tag == false)
								{
									facets.push(vhe->face);
									vhe->face->tag = true;

									HE_edge<Real> *mvhe = vhe->vert->edge;
									do
									{
										if (mvhe->face && mvhe->face->tag == false)
										{
											facets.push(mvhe->face);
											mvhe->face->tag = true;
										}
										mvhe = mvhe->pair->next;
									}
									while(mvhe != vhe->vert->edge);
								}
								vhe = vhe->pair->next;
							}
							while(vhe != he->vert->edge);
						}
						he = he->next;
					}
					while(he != pFacet->edge);

					he = pFacet->edge;
				}
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::compute_num_boundaries()
	{
		//if the mesh is not manifold, this function may report the wrong number

		VERTEX_ITER viter = vertices_list->begin();
		for (; viter != vertices_list->end(); viter++ )
		{
			(*viter)->tag = false;
		}

		boundaryvertices.clear();

		m_num_boundaries = 0;

		for(viter = vertices_list->begin(); viter != vertices_list->end(); viter++)
		{
			HE_vert<Real> *hv = *viter;

			if (is_on_boundary(hv) && hv->tag == false)
			{
				std::vector<HE_vert<Real>*> vedges;
				vedges.push_back(hv);

				m_num_boundaries++;
				std::list<HE_vert<Real>* > vertices;
				hv->tag = true;
				vertices.push_front(hv);

				while (!vertices.empty())
				{
					HE_vert<Real> *pVertex = vertices.front();
					pVertex->tag = true;
					vertices.pop_front();

					HE_edge<Real> *he = pVertex->edge;

					do
					{
						if (is_on_boundary(he) && is_on_boundary(he->vert) && he->vert->tag == false)
						{
							he->vert->tag = true;
							vertices.push_front(he->vert);
							vedges.push_back(he->vert);
						}
						he = he->pair->next;
					}
					while(he != pVertex->edge);
				}

				HE_vert<Real> *hvtmp = vedges[0];
				vedges[0] = vedges[1];
				vedges[1] = hvtmp;
				boundaryvertices.push_back(vedges);
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::compute_genus()
	{
		compute_num_components();
		compute_num_boundaries();
		int c = m_num_components;
		int b = m_num_boundaries;
		ptrdiff_t v = get_num_of_vertices();
		ptrdiff_t e = get_num_of_edges() / 2;
		ptrdiff_t f = get_num_of_faces();

		m_genus = (int)(2 * c + e - b - f - v) / 2;
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::remove_hanged_vertices()
	{
		bool find = false;
		VERTEX_ITER viter = vertices_list->begin();
		for (; viter != vertices_list->end(); viter++)
		{
			if ((*viter)->edge == NULL)
			{
				find = true;
				break;
			}
		}

		if (find)
		{
			VERTEX_LIST *new_vertices_list = new VERTEX_LIST;
			int i = 0;
			for (viter = vertices_list->begin(); viter != vertices_list->end(); viter++)
			{
				if ((*viter)->edge != NULL)
				{
					new_vertices_list->push_back(*viter);
					(*viter)->id = i;
					i++;
				}
				else
				{ delete *viter; }
			}
			delete vertices_list;
			vertices_list = new_vertices_list;

			EDGE_LIST *new_edges_list = new EDGE_LIST;
			i = 0;
			for (EDGE_ITER eiter = edges_list->begin(); eiter != edges_list->end(); eiter++)
			{
				if ((*eiter)->face != NULL || (*eiter)->pair->face != NULL)
				{
					new_edges_list->push_back(*eiter);
					(*eiter)->id = i;
					i++;
				}
				else
				{ delete *eiter; }
			}
			delete edges_list;
			edges_list = new_edges_list;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::reset_vertices_tag(bool tag_status)
	{
		VERTEX_ITER viter = vertices_list->begin();
		for (; viter != vertices_list->end(); viter++)
		{ (*viter)->tag = tag_status; }
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::reset_faces_tag(bool tag_status)
	{
		FACE_ITER fiter = faces_list->begin();
		for (; fiter != faces_list->end(); fiter++)
		{ (*fiter)->tag = tag_status; }
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::reset_edges_tag(bool tag_status)
	{
		EDGE_ITER eiter = edges_list->begin();
		for (; eiter != edges_list->end(); eiter++)
		{ (*eiter)->tag = tag_status; }
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::reset_all_tag(bool tag_status)
	{
		reset_edges_tag(tag_status);
		reset_faces_tag(tag_status);
		reset_vertices_tag(tag_status);
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::translate(const TinyVector<Real, 3> &tran_V)
	{
		VERTEX_ITER viter = vertices_list->begin();
		for (; viter != vertices_list->end(); viter++)
		{
			HE_vert<Real> *hv = *viter;
			hv->pos += tran_V;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::scale(Real factorx, Real factory, Real factorz)
	{
		VERTEX_ITER viter = vertices_list->begin();
		for (; viter != vertices_list->end(); viter++)
		{
			HE_vert<Real> *hv = *viter;
			hv->pos[0] *= factorx;
			hv->pos[1] *= factory;
			hv->pos[2] *= factorz;
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::init_edge_tag()
	{
		reset_edges_tag(false);
		EDGE_ITER eiter = edges_list->begin();
		for (; eiter != edges_list->end(); eiter++)
		{
			HE_edge<Real> *he = *eiter;
			if (he->tag == false && he->pair->tag == false)
			{
				he->tag = true;
			}
		}
	}
	//////////////////////////////////////////////////////////////////////////
	template <typename Real>
	void Mesh3D<Real>::swap_edge(HE_edge<Real> *triedge, bool update_vertex_normal)
	{
		if (triedge == NULL || triedge->face == NULL || triedge->pair->face == NULL || triedge->face->valence != 3 || triedge->pair->face->valence != 3)
		{
			return;
		}

		HE_vert<Real> *hv2 = triedge->vert;
		HE_vert<Real> *hv1 = triedge->pair->vert;
		HE_vert<Real> *hv3 = triedge->next->vert;
		HE_vert<Real> *hv4 = triedge->pair->next->vert;

		//check whether the swap causes non_manifold
		HE_edge<Real> *che = hv3->edge;
		do
		{
			if (che->vert == hv4)
			{
				return;
			}
			che = che->pair->next;
		}
		while (che != hv3->edge);

		HE_face<Real> *hf1 = triedge->face;
		HE_face<Real> *hf2 = triedge->pair->face;

		HE_edge<Real> *he1 = triedge->next->next;
		HE_edge<Real> *he2 = triedge->next;
		HE_edge<Real> *he3 = triedge->pair->next->next;
		HE_edge<Real> *he4 = triedge->pair->next;

		triedge->vert = hv3;
		triedge->next = he1;
		triedge->prev = he4;

		triedge->pair->vert = hv4;
		triedge->pair->next = he3;
		triedge->pair->prev = he2;

		hf1->edge = triedge;
		hf2->edge = triedge->pair;

		he1->face = hf1;
		he2->face = hf2;
		he3->face = hf2;
		he4->face = hf1;

		he1->next = he4;
		he1->prev = triedge;
		he2->next = triedge->pair;
		he2->prev = he3;
		he3->next = he2;
		he3->prev = triedge->pair;
		he4->next = triedge;
		he4->prev = he1;

		if (hv1->edge == triedge)
		{
			hv1->edge = he1->pair;
		}
		if (hv2->edge == triedge->pair)
		{
			hv2->edge = he3->pair;
		}

		hv1->degree--;
		hv2->degree--;
		hv3->degree++;
		hv4->degree++;

		compute_perface_normal(hf1);
		compute_perface_normal(hf2);
		if (update_vertex_normal)
		{
			compute_pervertex_normal(hv1);
			compute_pervertex_normal(hv2);
			compute_pervertex_normal(hv3);
			compute_pervertex_normal(hv4);
		}
	}
} //end of namespace

template class MeshLib::Mesh3D<double>;
//template class MeshLib::Mesh3D<float>;

#undef _CRT_SECURE_NO_WARNINGS