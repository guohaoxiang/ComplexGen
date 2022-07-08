#include "HLBFGS_Hessian.h"
#include "ICFS.h"

ICFS_INFO::ICFS_INFO()
	:p(15)
{
}

void ICFS_INFO::allocate_mem(const size_t N)
{
	if (N > 0)
	{
		lcol_ptr.resize(N + 1);
		ldiag.resize(N);
		iwa.resize(3 * N);
		wa1.resize(N);
		wa2.resize(N);
		r.resize(N);
		p = 15;
		CGP.resize(N);
		CGR.resize(N);
		CGQ.resize(N);
		CGZ.resize(N);
	}
}
size_t * ICFS_INFO::get_lcol_ptr()
{
	return &lcol_ptr[0];
}
size_t * ICFS_INFO::get_lrow_ind()
{
	return &lrow_ind[0];
}
double * ICFS_INFO::get_ldiag()
{
	return &ldiag[0];
}
double * ICFS_INFO::get_l()
{
	return &l[0];
}
size_t * ICFS_INFO::get_iwa()
{
	return &iwa[0];
}
double *ICFS_INFO::get_wa1()
{
	return &wa1[0];
}
double * ICFS_INFO::get_wa2()
{
	return &wa2[0];
}
size_t & ICFS_INFO::get_p()
{
	return p;
}
double * ICFS_INFO::get_r()
{
	return &r[0];
}
double * ICFS_INFO::get_CGP()
{
	return &CGP[0];
}
double * ICFS_INFO::get_CGQ()
{
	return &CGQ[0];
}
double * ICFS_INFO::get_CGR()
{
	return &CGR[0];
}
double * ICFS_INFO::get_CGZ()
{
	return &CGZ[0];
}
double & ICFS_INFO::get_icfs_alpha()
{
	return icfs_alpha;
}
void ICFS_INFO::set_lrow_ind_size(const size_t size)
{
	lrow_ind.resize(size);
}
void ICFS_INFO::set_l_size(const size_t size)
{
	l.resize(size);
}
//////////////////////////////////////////////////////////////////////////
HESSIAN_MATRIX::HESSIAN_MATRIX()
{
	mat = 0;
}
HESSIAN_MATRIX::~HESSIAN_MATRIX()
{
	if (mat)
	{
		delete mat; mat = 0;
	}
}
size_t HESSIAN_MATRIX::get_dimension()
{
	return mat->rows();
}
size_t HESSIAN_MATRIX::get_nonzeros()
{
	return mat->get_nonzero();
}
double * HESSIAN_MATRIX::get_values()
{
	return mat->get_values();
}
size_t * HESSIAN_MATRIX::get_rowind()
{
	return mat->get_rowind();
}
size_t * HESSIAN_MATRIX::get_colptr()
{
	return mat->get_colptr();
}
double * HESSIAN_MATRIX::get_diag()
{
	return mat->get_diag();
}
ICFS_INFO& HESSIAN_MATRIX::get_icfs_info()
{
	return l_info;
}
HLBFGS_Sparse_Matrix * HESSIAN_MATRIX::get_mat(const size_t dim)
{
	if (mat)
	{
		mat->reset();
		//delete mat; mat = 0;
	}
	else
	{
		mat = new HLBFGS_Sparse_Matrix(dim, dim, SPARSE_SYM_LOWER, SPARSE_CCS, SPARSE_FORTRAN_TYPE, true);
	}
	return mat;
}
void HESSIAN_MATRIX::build_hessian_info(size_t icfs_parameter/* = 15*/)
{
	l_info.get_p() = icfs_parameter;
	l_info.set_lrow_ind_size(get_nonzeros()
		+ get_dimension() * l_info.get_p());
	l_info.set_l_size(get_nonzeros() + get_dimension()
		* l_info.get_p());
	l_info.get_icfs_alpha() = 0;
	size_t n = get_dimension();
	size_t nnz = get_nonzeros();
	dicfs_(&n, &nnz, get_values(), get_diag(),
		get_colptr(), get_rowind(), l_info.get_l(),
		l_info.get_ldiag(), l_info.get_lcol_ptr(), l_info.get_lrow_ind(),
		&l_info.get_p(), &l_info.get_icfs_alpha(), l_info.get_iwa(),
		l_info.get_wa1(), l_info.get_wa2());
}
bool HESSIAN_MATRIX::solve_task(size_t dim, double *q, char &task)
{
	if (dim != get_dimension()) return false;
	dstrsol_(&dim, l_info.get_l(), l_info.get_ldiag(),
		l_info.get_lcol_ptr(), l_info.get_lrow_ind(), q, &task);
	return true;
}

bool HESSIAN_MATRIX::solve(size_t dim, double *q)
{
	if (dim != get_dimension()) return false;
	char task1 = 'N';
	char task2 = 'T';
	dstrsol_(&dim, l_info.get_l(), l_info.get_ldiag(),
		l_info.get_lcol_ptr(), l_info.get_lrow_ind(), q, &task1);
	dstrsol_(&dim, l_info.get_l(), l_info.get_ldiag(),
		l_info.get_lcol_ptr(), l_info.get_lrow_ind(), q, &task2);
	return true;
}