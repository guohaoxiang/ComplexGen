#pragma once
//#include "Public_header.h"
#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <iterator>

namespace MeshIO {



	template <typename Scalar, typename Index>
	inline bool readOBJ(const std::string obj_file_name,
		std::vector<std::vector<Scalar > > & V,
		std::vector<std::vector<Scalar > > & TC,
		std::vector<std::vector<Scalar > > & N,
		std::vector<std::vector<Index > > & F,
		std::vector<std::vector<Index > > & FTC,
		std::vector<std::vector<Index > > & FN)
{
		// Open file, and check for error
		FILE * obj_file = fopen(obj_file_name.c_str(), "r");
		if (NULL == obj_file)
		{
			fprintf(stderr, "IOError: %s could not be opened...\n",
				obj_file_name.c_str());
			return false;
		}
		return MeshIO::readOBJ(obj_file, V, TC, N, F, FTC, FN);
	}

	template <typename Scalar, typename Index>
	inline bool readOBJ(
		FILE * obj_file,
		std::vector<std::vector<Scalar > > & V,
		std::vector<std::vector<Scalar > > & TC,
		std::vector<std::vector<Scalar > > & N,
		std::vector<std::vector<Index > > & F,
		std::vector<std::vector<Index > > & FTC,
		std::vector<std::vector<Index > > & FN)// File open was successful so clear outputs
	{
	V.clear();
	TC.clear();
	N.clear();
	F.clear();
	FTC.clear();
	FN.clear();

	// variables and constants to assist parsing the .obj file
	// Constant strings to compare against
	std::string v("v");
	std::string vn("vn");
	std::string vt("vt");
	std::string f("f");
	std::string tic_tac_toe("#");
#ifndef IGL_LINE_MAX
#  define IGL_LINE_MAX 2048
#endif

	char line[IGL_LINE_MAX];
	int line_no = 1;
	while (fgets(line, IGL_LINE_MAX, obj_file) != NULL)
	{
		char type[IGL_LINE_MAX];
		// Read first word containing type
		if (sscanf(line, "%s", type) == 1)
		{
			// Get pointer to rest of line right after type
			char * l = &line[strlen(type)];
			if (type == v)
			{
				std::istringstream ls(&line[1]);
				std::vector<Scalar > vertex{ std::istream_iterator<Scalar >(ls), std::istream_iterator<Scalar >() };

				if (vertex.size() < 3)
				{
					fprintf(stderr,
						"Error: readOBJ() vertex on line %d should have at least 3 coordinates",
						line_no);
					fclose(obj_file);
					return false;
				}

				V.push_back(vertex);
			}
			else if (type == vn)
			{
				double x[3];
				int count =
					sscanf(l, "%lf %lf %lf\n", &x[0], &x[1], &x[2]);
				if (count != 3)
				{
					fprintf(stderr,
						"Error: readOBJ() normal on line %d should have 3 coordinates",
						line_no);
					fclose(obj_file);
					return false;
				}
				std::vector<Scalar > normal(count);
				for (int i = 0; i < count; i++)
				{
					normal[i] = x[i];
				}
				N.push_back(normal);
			}
			else if (type == vt)
			{
				double x[3];
				int count =
					sscanf(l, "%lf %lf %lf\n", &x[0], &x[1], &x[2]);
				if (count != 2 && count != 3)
				{
					fprintf(stderr,
						"Error: readOBJ() texture coords on line %d should have 2 "
						"or 3 coordinates (%d)",
						line_no, count);
					fclose(obj_file);
					return false;
				}
				std::vector<Scalar > tex(count);
				for (int i = 0; i < count; i++)
				{
					tex[i] = x[i];
				}
				TC.push_back(tex);
			}
			else if (type == f)
			{
				const auto & shift = [&V](const int i)->int
				{
					return i < 0 ? i + V.size() : i - 1;
				};
				const auto & shift_t = [&TC](const int i)->int
				{
					return i < 0 ? i + TC.size() : i - 1;
				};
				const auto & shift_n = [&N](const int i)->int
				{
					return i < 0 ? i + N.size() : i - 1;
				};
				std::vector<Index > f;
				std::vector<Index > ftc;
				std::vector<Index > fn;
				// Read each "word" after type
				char word[IGL_LINE_MAX];
				int offset;
				while (sscanf(l, "%s%n", word, &offset) == 1)
				{
					// adjust offset
					l += offset;
					// Process word
					long int i, it, in;
					if (sscanf(word, "%ld/%ld/%ld", &i, &it, &in) == 3)
					{
						f.push_back(shift(i));
						ftc.push_back(shift_t(it));
						fn.push_back(shift_n(in));
					}
					else if (sscanf(word, "%ld/%ld", &i, &it) == 2)
					{
						f.push_back(shift(i));
						ftc.push_back(shift_t(it));
					}
					else if (sscanf(word, "%ld//%ld", &i, &in) == 2)
					{
						f.push_back(shift(i));
						fn.push_back(shift_n(in));
					}
					else if (sscanf(word, "%ld", &i) == 1)
					{
						f.push_back(shift(i));
					}
					else
					{
						fprintf(stderr,
							"Error: readOBJ() face on line %d has invalid element format\n",
							line_no);
						fclose(obj_file);
						return false;
					}
				}
				if (
					(f.size() > 0 && fn.size() == 0 && ftc.size() == 0) ||
					(f.size() > 0 && fn.size() == f.size() && ftc.size() == 0) ||
					(f.size() > 0 && fn.size() == 0 && ftc.size() == f.size()) ||
					(f.size() > 0 && fn.size() == f.size() && ftc.size() == f.size()))
				{
					// No matter what add each type to lists so that lists are the
					// correct lengths
					F.push_back(f);
					FTC.push_back(ftc);
					FN.push_back(fn);
				}
				else
				{
					fprintf(stderr,
						"Error: readOBJ() face on line %d has invalid format\n", line_no);
					fclose(obj_file);
					return false;
				}
			}
			else if (strlen(type) >= 1 && (type[0] == '#' ||
				type[0] == 'g' ||
				type[0] == 's' ||
				strcmp("usemtl", type) == 0 ||
				strcmp("mtllib", type) == 0))
			{
				//ignore comments or other shit
			}
			else
			{

			}
		}
		else
		{
			// ignore empty line
		}
		line_no++;
	}
	fclose(obj_file);

	assert(F.size() == FN.size());
	assert(F.size() == FTC.size());

	return true;
}


		template <class Real>
		inline bool writeOBJ(const std::string obj_file_name, BlackMesh::BlackMesh<Real>& msh) {
			std::ofstream off(obj_file_name);
			for (int i = 0; i < msh.GetNumVertices(); i++) {
				std::vector<Real> pt = msh.GetVertices()[i].pos;
				off << std::setprecision(18) <<"v " << Utils::to_double(pt[0]) << "  " 
					<< Utils::to_double(pt[1]) << "  " << Utils::to_double(pt[2]) << "\n";
			}

			for (int i = 0; i < msh.GetNumTriangles(); i++)
			{
				off << "f " << msh.GetTriangles()[i].vertices[0] + 1
					<< "  " << msh.GetTriangles()[i].vertices[1] + 1
					<< "  " << msh.GetTriangles()[i].vertices[2] + 1 << "\n";
			}
			off.close();
			std::cout << "Write OBJ Finished, " << msh.GetNumVertices() << " vertices, " << msh.GetNumTriangles() << " faces\n";
			return true;
		}

		template <typename Scalar, typename Index>
		inline bool readOFF(
			const std::string off_file_name,
			std::vector<std::vector<Scalar > > & V,
			std::vector<std::vector<Index > > & F,
			std::vector<std::vector<Scalar > > & N,
			std::vector<std::vector<Scalar > > & C)
		{
			using namespace std;
			FILE * off_file = fopen(off_file_name.c_str(), "r");
			if (NULL == off_file)
			{
				printf("IOError: %s could not be opened...\n", off_file_name.c_str());
				return false;
			}
			return readOFF(off_file, V, F, N, C);
		}

		template <typename Scalar, typename Index>
		inline bool readOFF(
			FILE * off_file,
			std::vector<std::vector<Scalar > > & V,
			std::vector<std::vector<Index > > & F,
			std::vector<std::vector<Scalar > > & N,
			std::vector<std::vector<Scalar > > & C)
		{
			using namespace std;
			V.clear();
			F.clear();
			N.clear();
			C.clear();

			// First line is always OFF
			char header[1000];
			const std::string OFF("OFF");
			const std::string NOFF("NOFF");
			const std::string COFF("COFF");
			if (fscanf(off_file, "%s\n", header) != 1
				|| !(
					string(header).compare(0, OFF.length(), OFF) == 0 ||
					string(header).compare(0, COFF.length(), COFF) == 0 ||
					string(header).compare(0, NOFF.length(), NOFF) == 0))
			{
				printf("Error: readOFF() first line should be OFF or NOFF or COFF, not %s...", header);
				fclose(off_file);
				return false;
			}
			bool has_normals = string(header).compare(0, NOFF.length(), NOFF) == 0;
			bool has_vertexColors = string(header).compare(0, COFF.length(), COFF) == 0;
			// Second line is #vertices #faces #edges
			int number_of_vertices;
			int number_of_faces;
			int number_of_edges;
			char tic_tac_toe;
			char line[1000];
			bool still_comments = true;
			while (still_comments)
			{
				fgets(line, 1000, off_file);
				still_comments = (line[0] == '#' || line[0] == '\n');
			}
			sscanf(line, "%d %d %d", &number_of_vertices, &number_of_faces, &number_of_edges);
			V.resize(number_of_vertices);
			if (has_normals)
				N.resize(number_of_vertices);
			if (has_vertexColors)
				C.resize(number_of_vertices);
			F.resize(number_of_faces);
			//printf("%s %d %d %d\n",(has_normals ? "NOFF" : "OFF"),number_of_vertices,number_of_faces,number_of_edges);
			// Read vertices
			for (int i = 0; i<number_of_vertices;)
			{
				fgets(line, 1000, off_file);
				double x, y, z, nx, ny, nz;
				if (sscanf(line, "%lg %lg %lg %lg %lg %lg", &x, &y, &z, &nx, &ny, &nz) >= 3)
				{
					std::vector<Scalar > vertex;
					vertex.resize(3);
					vertex[0] = x;
					vertex[1] = y;
					vertex[2] = z;
					V[i] = vertex;

					if (has_normals)
					{
						std::vector<Scalar > normal;
						normal.resize(3);
						normal[0] = nx;
						normal[1] = ny;
						normal[2] = nz;
						N[i] = normal;
					}

					if (has_vertexColors)
					{
						C[i].resize(3);
						C[i][0] = nx / 255.0;
						C[i][1] = ny / 255.0;
						C[i][2] = nz / 255.0;
					}
					i++;
				}
				else if (
					fscanf(off_file, "%[#]", &tic_tac_toe) == 1)
				{
					char comment[1000];
					fscanf(off_file, "%[^\n]", comment);
				}
				else
				{
					printf("Error: bad line (%d)\n", i);
					if (feof(off_file))
					{
						fclose(off_file);
						return false;
					}
				}
			}
			// Read faces
			for (int i = 0; i<number_of_faces;)
			{
				std::vector<Index > face;
				int valence;
				if (fscanf(off_file, "%d", &valence) == 1)
				{
					face.resize(valence);
					for (int j = 0; j<valence; j++)
					{
						int index;
						if (j<valence - 1)
						{
							fscanf(off_file, "%d", &index);
						}
						else {
							fscanf(off_file, "%d%*[^\n]", &index);
						}

						face[j] = index;
					}
					F[i] = face;
					i++;
				}
				else if (
					fscanf(off_file, "%[#]", &tic_tac_toe) == 1)
				{
					char comment[1000];
					fscanf(off_file, "%[^\n]", comment);
				}
				else
				{
					printf("Error: bad line\n");
					fclose(off_file);
					return false;
				}
			}
			fclose(off_file);
			return true;
		}


		template <class Real>
		inline bool readMesh(const std::string obj_file_name, BlackMesh::BlackMesh<Real>& msh) {
			std::string file_format = obj_file_name.substr(obj_file_name.size() - 3, 3);

			std::vector<std::vector<double > >  V;
			std::vector<std::vector<double > >  TC;
			std::vector<std::vector<double > >  N;
			std::vector<std::vector<int > >  F;
			std::vector<std::vector<int > > FTC;
			std::vector<std::vector<int > >  FN;

			bool succ;

			if (file_format == "off" || file_format == "OFF") {
				succ = MeshIO::readOFF(obj_file_name, V, F, N, TC);
			}
			else if (file_format == "obj" || file_format == "OBJ") {
				succ = MeshIO::readOBJ(obj_file_name, V, TC, N, F, FTC, FN);
			}


			std::cout << "Read Finished, " << V.size() << " vertices, " << F.size() << " faces\n";

			if (!succ) return false;

			msh.clear();

			
			for (int i = 0; i < V.size(); i++) {
				
				msh.insert_vtx(std::vector<Real>{V[i][0], V[i][1], V[i][2]});
			}

			
			for (int i = 0; i < F.size(); i++) {
				std::vector<int> face;
				for (int j = 0; j < 3; j++)
					face.push_back(F[i][j]);
				msh.insert_face(face);
			}

			return true;
		}

		template <class Real>
		inline bool writeOBJ(const std::string obj_file_name, BlackMesh::Dual_graph<Real>& msh) {
			std::ofstream off(obj_file_name);
			for (int i = 0; i < msh.GetNumVertices(); i++) {
				std::vector<Real> pt = msh.GetVertices()[i].pos;
				off << std::setprecision(18) << "v " << Utils::to_double(pt[0]) << "  "
					<< Utils::to_double(pt[1]) << "  " << Utils::to_double(pt[2]) << "\n";
			}

			for (int i = 0; i < msh.GetNumEdges(); i++)
			{


				off << "l " << msh.GetEdges()[i].vertices.first+ 1
					<< "  " << msh.GetEdges()[i].vertices.second + 1
				 << "\n";
			}
			off.close();
			std::cout << "Write OBJ Finished, " << msh.GetNumVertices() << " vertices, " << msh.GetNumTriangles() << " faces\n";
			return true;
		}

		template <class Real>
		inline void writeOBJgrouped(BlackMesh::BlackMesh<Real>& msh) {
			if (msh.GetNumComponents() == 0)msh.mark_component_with_coherence();

			ofstream off("_debug_colored_output.obj");
			off << "mtllib _debug_colored_output.mtl\n";
			ofstream offm("_debug_colored_output.mtl");
			for (int i = 0; i < msh.GetNumVertices(); i++) {
				off <<std::setprecision(12)<< "v " << Utils::to_double(msh.GetVertices()[i].pos[0])
					<< " " << Utils::to_double(msh.GetVertices()[i].pos[1])
					<< " " << Utils::to_double(msh.GetVertices()[i].pos[2] )<< "\n";
			}

			std::vector<std::vector<int>> cc_contained_tris;
			cc_contained_tris.resize(msh.GetNumComponents(), std::vector<int>{});
			for (int i = 0; i < msh.GetNumTriangles(); i++) {
				int ccid = msh.GetTriangles()[i].component_id;
				cc_contained_tris[ccid].push_back(i);
			}

			for (int j = 0; j < cc_contained_tris.size(); j++) {
				off << "g cc" << j << "\n";

				off << "usemtl mcc" << j%17 << "\n";


				int ccid = j;

				for (int m = 0; m < cc_contained_tris[ccid].size(); m++) {
					int triid = cc_contained_tris[ccid][m];

					off << "f " << msh.GetTriangles()[triid].vertices[0] + 1 << " "
						<< msh.GetTriangles()[triid].vertices[1] + 1 << " "
						<< msh.GetTriangles()[triid].vertices[2] + 1 << " \n";

				}


				if (j < 17) {

					offm << "newmtl mcc" << j << "\n";
					offm << "\tNs 10.0\n\tNi 1.5\n\td 1.0\n\tTr 0.0\n\tillum 2\n\tKa 1.0 1.0 1.0\n\tKd "
						<< color_table[j % 17][0] << " " << color_table[j % 17][1] << " " << color_table[j % 17][2] << "\n\tKs 0.0 0.0 0.0\n\tKe 0.0 0.0 0.0\n";

				}

				}
			off.close();
			offm.close();


		}

		template <class Real>
		inline void writeOBJgrouped_Flipped(BlackMesh::BlackMesh<Real>& msh) {
			if (msh.GetNumComponents() == 0)msh.mark_component_with_coherence();

			ofstream off("_debug_colored_output_flip.obj");
			off << "mtllib _debug_colored_output.mtl\n";
			
			for (int i = 0; i < msh.GetNumVertices(); i++) {
				off << std::setprecision(12) << "v " << msh.GetVertices()[i].pos[0]
					<< " " << msh.GetVertices()[i].pos[1]
					<< " " << msh.GetVertices()[i].pos[2] << "\n";
			}

			std::vector<std::vector<int>> cc_contained_tris;
			cc_contained_tris.resize(msh.GetNumComponents(), std::vector<int>{});
			for (int i = 0; i < msh.GetNumTriangles(); i++) {
				int ccid = msh.GetTriangles()[i].component_id;
				cc_contained_tris[ccid].push_back(i);
			}

			for (int j = 0; j < cc_contained_tris.size(); j++) {
				off << "g cc" << j << "\n";
				off << "usemtl mcc" << j << "\n";

				int ccid = j;

				for (int m = 0; m < cc_contained_tris[ccid].size(); m++) {
					int triid = cc_contained_tris[ccid][m];

					off << "f " << msh.GetTriangles()[triid].vertices[2] + 1 << " "
						<< msh.GetTriangles()[triid].vertices[1] + 1 << " "
						<< msh.GetTriangles()[triid].vertices[0] + 1 << " \n";

				}


		
			}
			off.close();
		}
}
