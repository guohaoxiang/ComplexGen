#include "HLBFGS_Sparse_Matrix.h"

HLBFGS_Sparse_Entry::HLBFGS_Sparse_Entry(size_t ind/* = 0*/, double v/* = 0*/)
	:index(ind), value(v)
{
}

bool HLBFGS_Sparse_Entry::operator<(const HLBFGS_Sparse_Entry & m_r) const
{
	return index < m_r.index;
}
////////////////////////////////////////////////////////////////////

HLBFGS_Sparse_Matrix::HLBFGS_Sparse_Matrix(size_t m, size_t n, HLBFGS_SPARSE_SYMMETRIC_STATE symmetric_state /*=
	SPARSE_NOSYM*/, HLBFGS_SPARSE_STORAGE_TYPE m_store /*= SPARSE_CCS*/, HLBFGS_SPARSE_ARRAY_TYPE atype /*= SPARSE_C_TYPE*/,
	bool save_diag /*= false*/) :
	state_fill_entry(DISABLE), sym_state(symmetric_state), s_store(m_store), arraytype(atype),
	nrows(m), ncols(n), nonzero(0), save_diag_separetely(save_diag)
{
	if (m != n)
	{
		symmetric_state = SPARSE_NOSYM;
	}

	size_t nn = (m_store == SPARSE_CCS ? ncols : nrows);
	entryset.resize(nn);
	if (save_diag_separetely)
	{
		diag.resize(nrows < ncols ? nrows : ncols);
		//std::fill(diag.begin(), diag.end(), 0.0);
		memset(&diag[0], 0, sizeof(double) * diag.size());
	}
}

HLBFGS_Sparse_Matrix::~HLBFGS_Sparse_Matrix()
{
	clear_mem();
}

void HLBFGS_Sparse_Matrix::reset()
{
	diag.assign(diag.size(), 0);
	size_t nn = (s_store == SPARSE_CCS ? ncols : nrows);
	entryset.resize(nn);
	for (size_t i = 0; i < entryset.size(); i++)
		entryset[i].resize(0);
}

void HLBFGS_Sparse_Matrix::begin_fill_entry()
{
	state_fill_entry = ENABLE;
}

void HLBFGS_Sparse_Matrix::end_fill_entry()
{
	assert(state_fill_entry == ENABLE);

	clear_mem();

	state_fill_entry = LOCK;

	int inc = (arraytype == SPARSE_FORTRAN_TYPE ? 1 : 0);

	if (s_store == SPARSE_CCS)
	{
		//construct map and ccs matrix
		size_t i, j, k = 0;
		colptr.resize(ncols + 1);
		colptr[0] = inc;
		for (j = 1; j < ncols + 1; j++)
		{
			colptr[j] = (int)entryset[j - 1].size() + colptr[j - 1];
		}

		nonzero = colptr[ncols];

		if (nonzero > 0)
		{
			rowind.resize(nonzero);
			values.resize(nonzero);

			for (j = 0; j < ncols; j++)
			{
				for (i = 0; i < colptr[j + 1] - colptr[j]; i++)
				{
					rowind[k] = entryset[j][i].index + inc;
					values[k] = entryset[j][i].value;
					k++;
				}
			}
		}
	}
	else if (s_store == SPARSE_CRS)
	{
		//construct map and crs matrix
		size_t i, j, k = 0;
		rowind.resize(nrows + 1);
		rowind[0] = inc;
		for (j = 1; j < nrows + 1; j++)
		{
			rowind[j] = (int)entryset[j - 1].size() + rowind[j - 1];
		}
		nonzero = rowind[nrows];
		if (nonzero > 0)
		{
			colptr.resize(nonzero);
			values.resize(nonzero);

			for (j = 0; j < nrows; j++)
			{
				for (i = 0; i < rowind[j + 1] - rowind[j]; i++)
				{
					colptr[k] = entryset[j][i].index + inc;
					values[k] = entryset[j][i].value;
					k++;
				}
			}
		}
	}
	else if (s_store == SPARSE_TRIPLE)
	{
		size_t i, j, k = 0;
		nonzero = 0;
		for (i = 0; i < nrows; i++)
		{
			nonzero += (int)entryset[i].size();
		}

		if (nonzero > 0)
		{
			rowind.resize(nonzero);
			colptr.resize(nonzero);
			values.resize(nonzero);

			for (i = 0; i < nrows; i++)
			{
				size_t jsize = entryset[i].size();
				for (j = 0; j < jsize; j++)
				{
					rowind[k] = i + inc;
					colptr[k] = entryset[i][j].index + inc;
					values[k] = entryset[i][j].value;
					k++;
				}
			}
		}
	}
	entryset.clear();
}

void HLBFGS_Sparse_Matrix::fill_entry(size_t row_index, size_t col_index, double val/* = 0*/)
{
	if (row_index >= nrows || col_index >= ncols)
		return;

	if (save_diag_separetely && row_index == col_index)
	{
		diag[row_index] += val;
		return;
	}

	if (sym_state == SPARSE_NOSYM)
	{
		fill_entry_internal(row_index, col_index, val);
	}
	else if (sym_state == SPARSE_SYM_UPPER)
	{
		if (row_index <= col_index)
		{
			fill_entry_internal(row_index, col_index, val);
		}
		else
		{
			fill_entry_internal(col_index, row_index, val);
		}
	}
	else if (sym_state == SPARSE_SYM_LOWER)
	{
		if (row_index <= col_index)
		{
			fill_entry_internal(col_index, row_index, val);
		}
		else
		{
			fill_entry_internal(row_index, col_index, val);
		}
	}
	else if (sym_state == SPARSE_SYM_BOTH)
	{
		fill_entry_internal(row_index, col_index, val);

		if (row_index != col_index)
		{
			fill_entry_internal(col_index, row_index, val);
		}
	}
}

void HLBFGS_Sparse_Matrix::fill_diag(size_t diagid, double v/* = 0*/)
{
	if (save_diag_separetely)
	{
		if (diag.size() == 0)
		{
			diag.resize(nrows < ncols ? nrows : ncols);
			//std::fill(diag.begin(), diag.end(), 0.0);
			memset(&diag[0], 0, sizeof(double) * diag.size());
		}
		diag[diagid] += v;
	}
}

size_t HLBFGS_Sparse_Matrix::get_nonzero()
{
	return nonzero;
}

size_t HLBFGS_Sparse_Matrix::rows()
{
	return nrows;
}

size_t HLBFGS_Sparse_Matrix::cols()
{
	return ncols;
}

bool HLBFGS_Sparse_Matrix::issymmetric()
{
	return sym_state != SPARSE_NOSYM;
}

bool HLBFGS_Sparse_Matrix::issym_store_upper_or_lower()
{
	return (sym_state == SPARSE_SYM_LOWER) || (sym_state == SPARSE_SYM_UPPER);
}

HLBFGS_SPARSE_SYMMETRIC_STATE HLBFGS_Sparse_Matrix::symmetric_state()
{
	return sym_state;
}

bool HLBFGS_Sparse_Matrix::issquare()
{
	return nrows == ncols;
}

HLBFGS_SPARSE_STORAGE_TYPE HLBFGS_Sparse_Matrix::storage()
{
	return s_store;
}

HLBFGS_SPARSE_ARRAY_TYPE HLBFGS_Sparse_Matrix::get_arraytype()
{
	return arraytype;
}

size_t* HLBFGS_Sparse_Matrix::get_rowind()
{
	return rowind.empty() ? nullptr : &rowind[0];
}

const size_t * HLBFGS_Sparse_Matrix::get_rowind() const
{
	return rowind.empty() ? nullptr : &rowind[0];
}

size_t* HLBFGS_Sparse_Matrix::get_colptr()
{
	return colptr.empty() ? nullptr : &colptr[0];
}

const size_t* HLBFGS_Sparse_Matrix::get_colptr() const
{
	return colptr.empty() ? nullptr : &colptr[0];
}

double* HLBFGS_Sparse_Matrix::get_values()
{
	return values.empty() ? nullptr : &values[0];
}

const double* HLBFGS_Sparse_Matrix::get_values() const
{
	return values.empty() ? nullptr : &values[0];
}

double* HLBFGS_Sparse_Matrix::get_diag()
{
	return diag.empty() ? nullptr : &diag[0];
}

const double* HLBFGS_Sparse_Matrix::get_diag() const
{
	return diag.empty() ? nullptr : &diag[0];
}

const bool HLBFGS_Sparse_Matrix::is_diag_saved() const
{
	return save_diag_separetely;
}

void HLBFGS_Sparse_Matrix::clear_mem()
{
	colptr.clear();
	rowind.clear();
	values.clear();
}

bool HLBFGS_Sparse_Matrix::fill_entry_internal(size_t row_index, size_t col_index, double val/* = 0*/)
{
	assert(state_fill_entry == ENABLE);

	size_t search_index = (s_store == SPARSE_CCS ? row_index : col_index);
	size_t pos_index = (s_store == SPARSE_CCS ? col_index : row_index);

	HLBFGS_Sparse_Entry forcompare(search_index);

	std::vector<HLBFGS_Sparse_Entry>::iterator iter = std::lower_bound(
		entryset[pos_index].begin(), entryset[pos_index].end(),
		forcompare);
	if (iter != entryset[pos_index].end())
	{
		if (iter->index == search_index)
		{
			iter->value += val;
		}
		else
			entryset[pos_index].insert(iter,
				HLBFGS_Sparse_Entry(search_index, val));
	}
	else
	{
		entryset[pos_index].push_back(HLBFGS_Sparse_Entry(search_index, val));
	}
	return true;
}

/////////////////////////////////////////////////////////////////////

std::ostream & operator<<(std::ostream & s, HLBFGS_Sparse_Matrix * A)
{
	s.precision(16);
	if (A == nullptr)
	{
		s << "the matrix does not exist !\n ";
	}

	const size_t row = A->rows();
	const size_t col = A->cols();
	const size_t nonzero = A->get_nonzero();
	const size_t *rowind = A->get_rowind();
	const size_t *colptr = A->get_colptr();
	const double *values = A->get_values();

	s << "row :" << row << " col :" << col << " Nonzero: " << nonzero << "\n\n";

	s << "matrix --- (i, j, value)\n\n";

	HLBFGS_SPARSE_STORAGE_TYPE s_store = A->storage();
	int inc = (A->get_arraytype() == SPARSE_FORTRAN_TYPE ? -1 : 0);
	if (s_store == SPARSE_CCS)
	{
		size_t k = 0;
		for (size_t i = 1; i < col + 1; i++)
		{
			for (size_t j = 0; j < colptr[i] - colptr[i - 1]; j++)
			{
				s << rowind[k] + inc << " " << i - 1 << " " << std::scientific
					<< values[k] << "\n";
				k++;
			}
		}
	}
	else if (s_store == SPARSE_CRS)
	{
		size_t k = 0;
		for (size_t i = 1; i < row + 1; i++)
		{
			for (size_t j = 0; j < rowind[i] - rowind[i - 1]; j++)
			{
				s << i - 1 << " " << colptr[k] + inc << " " << std::scientific
					<< values[k] << "\n";
				k++;
			}
		}
	}
	else if (s_store == SPARSE_TRIPLE)
	{
		for (size_t k = 0; k < nonzero; k++)
		{
			s << rowind[k] + inc << " " << colptr[k] + inc << " "
				<< std::scientific << values[k] << "\n";
		}
	}

	if (A->is_diag_saved())
	{
		double *diag = A->get_diag();
		const size_t diag_size = std::min(row, col);
		for (size_t k = 0; k < diag_size; k++)
		{
			s << k << " " << k << " " << diag[k] << "\n";
		}
	}

	return s;
}