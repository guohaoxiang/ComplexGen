// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once
#include <stdio.h>
#include <time.h>

//#define _DIR_MODE

//#define _DISABLE_VD
//#define _DISABLE_OPT

#define _MAX_DEL_ITER_NUM 2

#define SINGLE_PRECISION
#define _USE_MATH_DEFINES

#define _ITERATE_ONLY_ONE

//#define _CHECK_EACH_ENERGY_TERM

//debug//
//#define _RECORD_ERROR_LOGS
//#define _OUTPUT_VD_DEL_AS_MESH
//#define _OUTPUT_OPT_DEL_AS_MESH
//#define _OUTPUT_MOSEK_LOG
//#define _OUTPUT_MRF_LOG
//#define _OUTPUT_BRANCH_BOUND_LOG
#define _OUTPUT_DEBUG_LOGS//should add debug swtich
//#define _OUTPUT_RESOLVED_MESH
//#define _OUTPUT_COHERENT_MESH
//#define _OUTPUT_MERGED_NONMANIFOLD_EDGES
//#define _SAVE_OFFRENDER_TO_IMG

#define _USE_TIMER

#define _TREAT_PIX_SMALLER_THAN_AS_ZERO 10

#define _PUNISH_CONNECTION_ON_CURV_SMALLER_THAN 0.03// curvlen/sqrt(areai+areaj)
#define _PUNISH_SCORE -1e-5

#define _DISABLE_FILLING_WARNING

#define _USE_ZNEAR_CUT_SCENE
#define _SEARCH_NEIB_ON_DEGENERATE 2
#define _MAX_TIME_FOR_MOSEK 5.0//mins
//#define _MAX_CC_NM_FOR_PREPROCESSING 10000

enum POSTPRO_TYPE { KEEP_ALL_IN, DELETE_ALL_IN };


#include <stdio.h>
#include <tchar.h>
#include <iomanip>
#include <io.h>
#include <vector>
#include <queue>
#include <list>
#include <string>
#include <iostream>
//#include <Windows.h>
#include <math.h>



// freeglut
#include <GL/glew.h>
#include <GL/freeglut.h>

// anttweakbar
//#include <AntTweakBar.h>

//// openmesh
// #include <OpenMesh/Core/IO/MeshIO.hh>
// #include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

// eigen
#include <Eigen/Sparse>
#include <Eigen/Dense>


using namespace std;
/* Application precision -- can be set to single or Float precision */
#if defined SINGLE_PRECISION
typedef float Float;
#else
typedef double Float;
#endif

const Float EPS = 1.0e-2;

/*std vector */
typedef std::vector<std::vector<int>>							vecveci;

/* Useful Eigen typedefs based on the current precision */
typedef Eigen::Matrix<int, 2, 1>  								Vector2i;
typedef Eigen::Matrix<int, 3, 1>  								Vector3i;
typedef Eigen::Matrix<int, 4, 1>								Vector4i;
typedef Eigen::Matrix<Float, 2, 1>								Vector2f;
typedef Eigen::Matrix<Float, 3, 1>								Vector3F;
typedef Eigen::Matrix<Float, 4, 1>								Vector4F;
typedef Eigen::Matrix<Float, 2, 2>								Matrix2F;
typedef Eigen::Matrix<Float, 3, 3>								Matrix3F;
typedef Eigen::Matrix<Float, 4, 4>								Matrix4F;
typedef Eigen::Matrix<int, Eigen::Dynamic, 1>					VectorXi;
typedef Eigen::Matrix<bool, Eigen::Dynamic, 1>  				VectorXb;
typedef Eigen::Matrix<Float, Eigen::Dynamic, 1> 				VectorXF;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>		MatrixXi;
typedef Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic>	MatrixXF;
typedef Eigen::Matrix<UINT8, Eigen::Dynamic, Eigen::Dynamic>	MatrixXu8;
typedef Eigen::SparseMatrix<Float>								SMatrixF;
typedef Eigen::Triplet<Float>									TripletF;

//double c1 = 0.35;
//double c2 = 0.5;
//double c3 = 1.0;

const double color_table[17][3] =
{
	{ 1.0, 0.5, 0.5 },
	{ 0.5, 1.0, 0.5 },
	{ 0.5, 0.5, 1.0 },
	{ 0.5, 1.0, 1.0 },
	{ 1.0, 0.5, 1.0 },
	{ 1.0, 1.0, 0.5 },

	{ 0.35, 0.5, 0.5 },
	{ 0.5, 0.35, 0.5 },
	{ 0.5, 0.5, 0.35 },
	{ 0.5, 0.35, 0.35 },
	{ 0.35, 0.5, 0.35 },
	{ 0.35, 0.35, 0.5 },

	{ 0.7, 0.7, 0.7 },
	{ 0.8125, 0.664, 0.246 },
	{ 0.85, 0.53, 0.09 },
	{ 89 / 255.0, 163 / 255.0, 255 / 255.0 },
	{ 18 / 255.0, 94 / 255.0, 234 / 255.0 }

};


extern double _eps_vis;
extern double _sw, _uw, _bw;
extern int _tmer;
extern int _opt_max_iter;

extern ofstream *_log_off;
extern char* _file_under_processing;
extern int _num_threads;
//inline void save_error_logs(const char* filename, const char* message) {
//	char buffer[128];
//	string fm(filename);
//	auto found = fm.find_last_of("/\\");
//	string filem = fm.substr(found + 1);
//	found = filem.find_last_of(".");
//	filem = filem.substr(0, found);
//	sprintf(buffer, "_error_logs_%s.txt", filem.c_str());
//
//	ofstream off(buffer);
//	off << message;
//	off.close();
//}
//#endif

