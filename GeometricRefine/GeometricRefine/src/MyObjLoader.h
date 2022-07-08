#pragma once

#include "TinyVector.h"
#include <vector>
#include <set>
#include <map>
#include <cstdio>
#include <fstream>
#include <queue>
#include "MyTuple.h"

using namespace std;
typedef TinyVector<double, 3> vec3d;

bool obj_loader(const char filename[], vector<vec3d> &vertices,
                std::set<MySortedTuple<2>> &sharp_edges, vector<vector<size_t>> &facets, double sharp_angle_in_degree = 60)
{
    vertices.resize(0);
    facets.resize(0);
    double threshold = cos(sharp_angle_in_degree / 180 * M_PI);
    std::map<MySortedTuple<2>, std::vector<vec3d>> edge2facenormals;
    FILE *m_pFile = fopen(filename, "r");
    if (!m_pFile)
        return false;
    char temp[128];
    char *tok;

    fseek(m_pFile, 0, SEEK_SET);
    char pLine[1024];
    vec3d v;
    while (fgets(pLine, 1024, m_pFile))
    {
        if (pLine[0] == 'v' && pLine[1] == ' ')
        {

            tok = strtok(pLine, " ");
            for (int i = 0; i < 3; i++)
            {
                tok = strtok(NULL, " ");
                strcpy(temp, tok);
                temp[strcspn(temp, " ")] = 0;
                v[i] = (double)atof(temp);
            }
            vertices.push_back(v);
        }
    }
    fseek(m_pFile, 0, SEEK_SET);
    std::vector<size_t> s_faceid;
    while (fgets(pLine, 1024, m_pFile))
    {
        if (pLine[0] == 'f')
        {
            s_faceid.resize(0);
            tok = strtok(pLine, " ");
            while ((tok = strtok(NULL, " ")) != NULL)
            {
                strcpy(temp, tok);
                temp[strcspn(temp, "/")] = 0;
                int id = (int)strtol(temp, NULL, 10) - 1;
                s_faceid.push_back(id);
            }
            facets.push_back(s_faceid);
            vec3d normal(0, 0, 0);
            for (size_t j = 0; j < s_faceid.size(); j++)
                normal += vertices[s_faceid[j]].Cross(vertices[s_faceid[(j + 1) % s_faceid.size()]]);
            normal.Normalize();
            for (size_t j = 0; j < s_faceid.size(); j++)
                edge2facenormals[MySortedTuple<2>(s_faceid[j], s_faceid[(j + 1) % s_faceid.size()])].push_back(normal);
        }
    }
    fclose(m_pFile);
    // ofstream sharpfeature("sharp.obj");
    // size_t sharpcounter = 1;
    for (auto iter = edge2facenormals.begin(); iter != edge2facenormals.end(); iter++)
    {
        if (iter->second.size() < 2)
            continue;
        for (size_t j = 0; j < iter->second.size(); j++)
        {
            if (fabs(iter->second[j].Dot(iter->second[(j + 1) % iter->second.size()])) < threshold)
            {
                sharp_edges.insert(iter->first);
                // sharpfeature << "v " << vertices[iter->first.sorted_vert[0]] << std::endl;
                // sharpfeature << "v " << vertices[iter->first.sorted_vert[1]] << std::endl;
                // sharpfeature << "l " << sharpcounter << ' ' << sharpcounter + 1 << std::endl;
                // sharpcounter += 2;
                break;
            }
        }
    }
    // sharpfeature.close();

    return true;
}

bool is_sharp_edge_list(const std::vector<size_t> &edge_list, const std::set<MySortedTuple<2>> &sharp_edges)
{
    for (size_t i = 0; i + 1 < edge_list.size(); i++)
    {
        if (sharp_edges.count(MySortedTuple<2>(edge_list[i], edge_list[i + 1])) > 0)
            return true;
    }
    return false;
}

int detect_num_loops(const std::vector<std::vector<size_t>> &facets, const std::vector<size_t> &selected_facets)
{
    std::map<MySortedTuple<2>, int> edges;
    for (size_t i = 0; i < selected_facets.size(); i++)
    {
        const size_t &fid = selected_facets[i];
        for (int j = 0; j < facets[fid].size(); j++)
        {
            MySortedTuple<2> e(facets[fid][j], facets[fid][(j + 1) % facets[fid].size()]);
            auto iter = edges.find(e);
            if (iter != edges.end())
                iter->second++;
            else
                edges[e] = 1;
        }
    }
    std::map<size_t, std::set<size_t>> vertex_links;
    for (auto &iter : edges)
    {
        if (iter.second == 1)
        {
            vertex_links[iter.first.sorted_vert[0]].insert(iter.first.sorted_vert[1]);
            vertex_links[iter.first.sorted_vert[1]].insert(iter.first.sorted_vert[0]);
        }
    }
    std::set<size_t> used_vertices;
    int num_loop = 0;
    for (auto &iter : vertex_links)
    {
        if (used_vertices.count(iter.first) > 0)
            continue;

        std::queue<size_t> myqueue;
        myqueue.push(iter.first);
        used_vertices.insert(iter.first);

        while (!myqueue.empty())
        {
            size_t vid = myqueue.front();
            myqueue.pop();
            for (auto &n : vertex_links[vid])
            {
                if (used_vertices.count(n) > 0)
                    continue;
                myqueue.push(n);
                used_vertices.insert(n);
            }
        }
        num_loop++;
    }
    std::cout << "loop: " << num_loop << std::endl;
    return num_loop;
}

