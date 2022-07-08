#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <ostream>
#include <cstring>

// \addtogroup MathSuite
//@{
//! HLBFGS Sparse Entry class \ingroup MathSuite
class HLBFGS_Sparse_Entry
{
public:

	//! Index ID
	size_t index;

	//! Real value
	double value;

public:
	//! constructor
	HLBFGS_Sparse_Entry(size_t ind = 0, double v = 0);

	//! The compare function for sorting
	bool operator<(const HLBFGS_Sparse_Entry & m_r) const;
};

//! Symmetric status
enum HLBFGS_SPARSE_SYMMETRIC_STATE
{
	SPARSE_NOSYM, /*!< general case */
	SPARSE_SYM_UPPER, /*!< symmetric (store upper triangular part) */
	SPARSE_SYM_LOWER, /*!< symmetric (store lower triangular part) */
	SPARSE_SYM_BOTH   /*!< symmetric (store both upper and lower triangular part) */
};

//!     Storage
enum HLBFGS_SPARSE_STORAGE_TYPE
{
	SPARSE_CCS, /*!< compress column format */
	SPARSE_CRS, /*!< compress row format */
	SPARSE_TRIPLE /*!< row-wise coordinate format */
};

//! Array type
enum HLBFGS_SPARSE_ARRAY_TYPE
{
	SPARSE_FORTRAN_TYPE, /*!< the index starts from 1 */
	SPARSE_C_TYPE /*!< the index starts from 0 */
};

//////////////////////////////////////////////////////////////////////////
//! HLBFGS Sparse Matrix Class
class HLBFGS_Sparse_Matrix
{
private:

	//! Status for creating sparse solver
	enum STORE_STATE
	{
		ENABLE, DISABLE, LOCK
	};

	STORE_STATE state_fill_entry;
	HLBFGS_SPARSE_SYMMETRIC_STATE sym_state;
	HLBFGS_SPARSE_STORAGE_TYPE s_store;
	HLBFGS_SPARSE_ARRAY_TYPE arraytype;

	size_t nrows; //!< number of rows
	size_t ncols; //!< number of columns
	size_t nonzero; //!< number of nonzeros
	//! pointers to where columns begin in rowind and values 0-based, length is (col+1)
	/*!
	 * When s_store is CRS, colptr stores column indices;
	 */
	std::vector<size_t> colptr;
	//! row indices, 0-based
	/*!
	 * When s_store is CRS, rowind stores row-pointers
	 */
	std::vector<size_t> rowind;
	std::vector<double> values; //!< nonzero values of the sparse matrix

	std::vector<std::vector<HLBFGS_Sparse_Entry> > entryset; //!< store temporary sparse entries

	std::vector<double> diag; //! special usage for some libraries

	bool save_diag_separetely;

public:

	//! Sparse matrix constructor
	/*!
	 * \param m row dimension
	 * \param n column dimension
	 * \param symmetric_state
	 * \param m_store the storage format
	 * \param atype Fortran or C type of array
	 */
	HLBFGS_Sparse_Matrix(size_t m, size_t n, HLBFGS_SPARSE_SYMMETRIC_STATE symmetric_state =
		SPARSE_NOSYM, HLBFGS_SPARSE_STORAGE_TYPE m_store = SPARSE_CCS, HLBFGS_SPARSE_ARRAY_TYPE atype = SPARSE_C_TYPE,
		bool save_diag = false);

	//! Sparse matrix destructor
	~HLBFGS_Sparse_Matrix();

	//! reset
	void reset();

	//! Start to build sparse matrix pattern
	void begin_fill_entry();

	//! Construct sparse pattern
	void end_fill_entry();

	//! Fill matrix entry \f$  Mat_{row_index, col_index} += val \f$
	void fill_entry(size_t row_index, size_t col_index, double val = 0);

	//fill the diagonal entry
	void fill_diag(size_t diagid, double v = 0);

	//! get the number of nonzeros
	size_t get_nonzero();

	//! get the row dimension
	size_t rows();

	//! get the column dimension
	size_t cols();

	//! return the symmetric state
	bool issymmetric();

	//! tell whether the matrix is upper or lower symmetric
	bool issym_store_upper_or_lower();

	//! return symmetric state
	HLBFGS_SPARSE_SYMMETRIC_STATE symmetric_state();

	//! tell whether the matrix is square
	bool issquare();

	//! return the storage format
	HLBFGS_SPARSE_STORAGE_TYPE storage();

	//! return array type
	HLBFGS_SPARSE_ARRAY_TYPE get_arraytype();

	//! get rowind
	size_t *get_rowind();

	const size_t *get_rowind() const;

	//! get colptr
	size_t *get_colptr();

	const size_t *get_colptr() const;

	//! get the values array
	double *get_values();

	const double *get_values() const;

	//! get the diagonal array
	double *get_diag();

	const double *get_diag() const;

	const bool is_diag_saved() const;

	//////////////////////////////////////////////////////////////////////////
private:
	//! Clear memory
	void clear_mem();

	//! fill matrix entry (internal) \f$ Mat[rowid][colid] += val \f$
	bool fill_entry_internal(size_t row_index, size_t col_index, double val = 0);
	//////////////////////////////////////////////////////////////////////////
};

//! print sparse matrix
std::ostream & operator<<(std::ostream & s, const HLBFGS_Sparse_Matrix * A);

//@}
