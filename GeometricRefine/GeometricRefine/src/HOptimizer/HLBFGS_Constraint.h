#pragma once

#include <vector>
#include <cassert>

#include "HLBFGS_Sparse_Matrix.h"

class ConstraintJacobian
{
public:
	ConstraintJacobian(size_t num_funcs = 0, size_t num_vars = 0);
	void resize(size_t num_funcs, size_t num_vars);
	void reset();
	void add_value(size_t func_id, double value);
	void add_differential(size_t func_id, size_t var_id, double value);
	std::vector< std::vector<std::pair<size_t, double> > > & get_Jacobian();
	const std::vector< std::vector<std::pair<size_t, double> > > & get_Jacobian() const;
	std::vector<double> & get_constraints();
	const std::vector<double> & get_constraints() const;
public:
	size_t m_num_funcs, m_num_vars;
	std::vector< std::vector<std::pair<size_t, double> > > constraint_jacobian_;
	std::vector<double> constraint_values_;
};

class ConstraintHessian
{
public:
	ConstraintHessian(size_t num_eq, size_t num_ieq, HLBFGS_Sparse_Matrix &mat,
		const std::vector<double> & lambda, const std::vector<double> & sigma,
		size_t info, const ConstraintJacobian &jacobian);
	void add_differential(size_t i, size_t var_id0, size_t var_id1, double diff_value);

public:
	size_t n_E_, n_IE_;
	HLBFGS_Sparse_Matrix& hessian;
	const ConstraintJacobian &m_jacobian;
	const std::vector<double> & m_lambda, &m_sigma;
	size_t info14;
};
