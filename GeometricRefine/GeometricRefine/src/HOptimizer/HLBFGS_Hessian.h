#pragma once

#include <vector>
#include "HLBFGS_Sparse_Matrix.h"

//! ICFS_INFO stores ICFS's working arrays
class ICFS_INFO
{
public:
	ICFS_INFO();
	void allocate_mem(const size_t N);
	size_t * get_lcol_ptr();
	size_t * get_lrow_ind();
	double * get_ldiag();
	double * get_l();
	size_t * get_iwa();
	double * get_wa1();
	double * get_wa2();
	size_t & get_p();
	double * get_r();
	double * get_CGP();
	double * get_CGQ();
	double * get_CGR();
	double * get_CGZ();
	double & get_icfs_alpha();
	void set_lrow_ind_size(const size_t size);
	void set_l_size(const size_t size);
private:
	std::vector<size_t> lcol_ptr;
	std::vector<size_t> lrow_ind;
	std::vector<double> ldiag;
	std::vector<double> l;
	std::vector<size_t> iwa;
	std::vector<double> wa1;
	std::vector<double> wa2;
	size_t p;
	std::vector<double> r;
	std::vector<double> CGP;
	std::vector<double> CGQ;
	std::vector<double> CGR;
	std::vector<double> CGZ;
	double icfs_alpha;
};
//////////////////////////////////////////////////////////////////////////
//! Stores the pointers of hessian matrix
class HESSIAN_MATRIX
{
public:
	HESSIAN_MATRIX();
	~HESSIAN_MATRIX();
	size_t get_dimension();
	size_t get_nonzeros();
	double * get_values();
	size_t * get_rowind();
	size_t * get_colptr();
	double * get_diag();
	ICFS_INFO& get_icfs_info();
	HLBFGS_Sparse_Matrix * get_mat(const size_t dim);
	void build_hessian_info(size_t icfs_parameter = 15);
	bool solve_task(size_t dim, double *q, char &task);
	bool solve(size_t dim, double *q);
private:
	ICFS_INFO l_info;
	HLBFGS_Sparse_Matrix *mat;
};
